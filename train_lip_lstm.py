"""Train a visual LSTM with frame-level phoneme supervision on aligned lip crops.

The training pairs come from:
 - lip_crop_results_full: cropped lip videos
 - phonemes_s1_aligned/phoneme_predictions.csv: frame-aligned phoneme labels

Each video is treated as a variable-length frame sequence. A per-frame visual
encoder extracts features, a bidirectional LSTM aggregates them through time,
and a linear projection produces frame-wise phoneme logits. The training loss
is a graph-aware frame cross entropy that uses the aligned labels directly and
penalizes blank predictions more heavily than mistakes within a similarity
cluster of related phonemes.

Changes in this version
-----------------------
1. blank_penalty raised from 0.5 → 2.0 (also exposed as --blank-penalty CLI arg).
   The previous run produced 70/52/50/44 spurious insertions of ɪ/n/ɛ/s,
   indicating the model learned that spamming high-frequency tokens incurs lower
   expected loss than committing to correct but rarer ones.  The blank_penalty
   term adds `blank_penalty * p(blank) * I[target ≠ blank]` to each frame's
   loss, directly taxing confident non-blank predictions on non-blank frames.

2. Dropout raised from 0.2 → 0.45 in both the projector head (FrameEncoder)
   and the inter-layer LSTM dropout (VisualLSTMFrameCE).  Weight-decay default
   raised from 1e-4 → 5e-4.  Together these target the large train/val gap
   observed in the previous run (train ~0.55, val ~1.11 at epoch 22).

3. blank_penalty reverted 2.0 → 0.5.  Doubling it did not reduce insertions:
   ɪ/ɛ/n insertions went from 70/50/52 to 87/83/69 in the run that used 2.0,
   and val_loss diverged from val_ter after epoch ~15 (train_loss kept
   dropping to ~0.34 while val_loss rose to ~1.2), i.e. severe overfitting
   rather than a blank-dominance problem.  early_stopping_patience lowered
   from 60 → 8 so training stops shortly after val performance stops
   improving instead of training ~45 extra overfitting epochs.  Checkpoint
   selection ("best_model.pt") now tracks val_ter instead of val_loss, since
   val_ter is the metric we actually care about and the two diverge in the
   later epochs.

Usage:
    python train_lip_lstm.py --dry-run
    python train_lip_lstm.py --epochs 20 --batch-size 4
    python train_lip_lstm.py --blank-penalty 1.0  # raise if blank-dominance returns
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import matplotlib
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_BLANK_TOKEN = "<blank>"
MISSING_PREDICTION_TOKEN = "<missed>"
INSERTION_TOKEN = "<inserted>"

PHONEME_SIMILARITY_GROUPS = {
    "bilabial_stop": {"p", "b", "m", "w"},
    "labiodental_fricative": {"f", "v"},
    "dental_fricative": {"θ", "ð"},
    "alveolar_stop": {"t", "d"},
    "alveolar_fricative": {"s", "z"},
    "alveolar_nasal_liquid": {"n", "l", "ɹ", "r"},
    "postalveolar": {"ʃ", "ʒ", "tʃ", "dʒ"},
    "velar": {"k", "g", "ŋ"},
    "glottal": {"h"},
    "high_front_vowel": {"i", "ɪ", "e", "eɪ"},
    "mid_front_vowel": {"ɛ", "æ"},
    "central_vowel": {"ə", "ɐ", "ʌ", "ɜ", "ɚ"},
    "back_vowel": {"u", "ʊ", "ɔ", "ɑ", "ɒ", "o", "oʊ"},
    "low_vowel": {"a", "æ", "ɑ", "ɐ"},
}


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 20
    learning_rate: float = 1e-4
    # Raised from 1e-4 → 5e-4 to close the train/val overfitting gap.
    weight_decay: float = 5e-4
    hidden_dim: int = 256
    lstm_layers: int = 2
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    # Lowered from 60 → 8. The previous run's val_loss bottomed out around
    # epoch 15 and then rose monotonically through epoch 60 while train_loss
    # kept falling — a patience of 60 let training run ~45 epochs purely
    # overfitting after the useful minimum had already passed.
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3


@dataclass
class Sample:
    video_path: Path
    frame_tokens: List[str]
    sentence: str = ""
    canonical_tokens: List[str] = field(default_factory=list)


@dataclass
class AlignedPhonemeRecord:
    sentence: str
    canonical_tokens: List[str]
    frame_tokens: List[str]
    num_frames: int


def find_lip_videos(data_dir: Path) -> List[Path]:
    return [path for path in sorted(data_dir.rglob("*")) if path.is_file() and path.suffix.lower() in VIDEO_EXTS]


def normalize_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_lipcrop"):
        return stem[: -len("_lipcrop")]
    return stem


def normalize_frame_token(token: Optional[str]) -> str:
    if token is None:
        return DEFAULT_BLANK_TOKEN
    normalized = str(token).strip()
    if not normalized:
        return DEFAULT_BLANK_TOKEN
    return normalized


def load_aligned_phoneme_targets(csv_path: Path) -> Dict[str, AlignedPhonemeRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Phoneme CSV not found: {csv_path}")

    targets: Dict[str, AlignedPhonemeRecord] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stem = row.get("stem", "").strip()
            frame_labels_field = row.get("per_frame_labels", "").strip()
            if not stem or not frame_labels_field:
                continue

            try:
                raw_frame_labels = json.loads(frame_labels_field)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid per_frame_labels JSON for stem={stem}: {exc}") from exc

            frame_tokens = [normalize_frame_token(token) for token in raw_frame_labels]
            if not frame_tokens:
                continue

            canonical_field = row.get("canonical_phonemes", "").strip()
            canonical_tokens = canonical_field.split() if canonical_field else []
            sentence = row.get("sentence", "").strip()
            num_frames_field = row.get("num_frames", "").strip()
            num_frames = int(num_frames_field) if num_frames_field.isdigit() else len(frame_tokens)

            targets[stem] = AlignedPhonemeRecord(
                sentence=sentence,
                canonical_tokens=canonical_tokens,
                frame_tokens=frame_tokens,
                num_frames=num_frames,
            )

    if not targets:
        raise ValueError(f"No phoneme targets could be read from {csv_path}")

    return targets


def build_vocab(samples: Sequence[Sample]) -> List[str]:
    counter: Counter[str] = Counter()
    for sample in samples:
        counter.update(token for token in sample.frame_tokens if token != DEFAULT_BLANK_TOKEN)
    vocab = sorted(counter.keys())
    if not vocab:
        raise ValueError("Cannot build an empty phoneme vocabulary")
    return vocab


def split_samples(
    samples: List[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train : n_train + n_val]
    test_samples = shuffled[n_train + n_val :]
    return train_samples, val_samples, test_samples


class LipPhonemeDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        vocab: Sequence[str],
        frame_size: int = 224,
        frame_stride: int = 1,
        augment: bool = False,
    ):
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

        self.samples = list(samples)
        self.frame_size = int(frame_size)
        self.frame_stride = int(frame_stride)
        self.token_to_id = {token: index + 1 for index, token in enumerate(vocab)}
        self.blank_id = 0
        self.is_training = bool(augment)
        self.train_transforms, self.val_transforms = self._build_transforms()

    def _build_transforms(self) -> Tuple[v2.Compose, v2.Compose]:
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        train_transforms = v2.Compose([
            v2.Resize((self.frame_size, self.frame_size), interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05)], p=0.4),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        val_transforms = v2.Compose([
            v2.Resize((self.frame_size, self.frame_size), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        return train_transforms, val_transforms

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_path: Path) -> torch.Tensor:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        frames: List[torch.Tensor] = []
        try:
            while True:
                success, frame_bgr = capture.read()
                if not success:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
                frames.append(frame_tensor)
        finally:
            capture.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")

        return torch.stack(frames, dim=0)  # [T, C, H, W]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        video = self._load_video(sample.video_path)
        frame_tokens = list(sample.frame_tokens)

        # Synchronize raw frames and labels before stride
        matched_length = min(len(frame_tokens), video.shape[0])
        video = video[:matched_length]
        frame_tokens = frame_tokens[:matched_length]

        if self.frame_stride > 1:
            video = video[::self.frame_stride]
            frame_tokens = frame_tokens[::self.frame_stride]

        if not frame_tokens:
            raise RuntimeError(f"Empty frame target sequence for {sample.video_path}")

        if self.is_training:
            video = self.train_transforms(video)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
            b_factor = 1.0 + (random.random() * 0.2 - 0.1)
            c_factor = 1.0 + (random.random() * 0.2 - 0.1)
            video = (video * std + mean) * float(b_factor)
            video = (video - mean) * float(c_factor) + mean
            video = (video - mean) / std

            if random.random() < 0.2:
                t, c, h, w = video.shape
                rect_h = max(1, int(h * (0.08 + random.random() * 0.15)))
                rect_w = max(1, int(w * (0.08 + random.random() * 0.15)))
                y = random.randint(0, max(0, h - rect_h))
                x = random.randint(0, max(0, w - rect_w))
                video[:, :, y : y + rect_h, x : x + rect_w] = 0.0
        else:
            video = self.val_transforms(video)

        target_ids = [
            self.blank_id if token == DEFAULT_BLANK_TOKEN else self.token_to_id[token]
            for token in frame_tokens
        ]

        return {
            "video": video,
            "video_length": torch.tensor(video.shape[0], dtype=torch.long),
            "frame_targets": torch.tensor(target_ids, dtype=torch.long),
            "stem": sample.video_path.stem,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    video_lengths = torch.tensor([int(item["video_length"]) for item in batch], dtype=torch.long)
    max_frames = int(video_lengths.max().item())

    channels, height, width = batch[0]["video"].shape[1:]
    videos = torch.zeros(batch_size, max_frames, channels, height, width, dtype=batch[0]["video"].dtype)
    frame_targets = torch.zeros(batch_size, max_frames, dtype=torch.long)

    stems: List[str] = []
    for index, item in enumerate(batch):
        v = item["video"]
        t = item["frame_targets"]
        videos[index, : v.shape[0]] = v
        frame_targets[index, : t.shape[0]] = t
        stems.append(str(item["stem"]))

    return {
        "videos": videos,
        "video_lengths": video_lengths,
        "frame_targets": frame_targets,
        "stems": stems,
    }


def collapse_frame_ids(
    frame_ids: Sequence[int],
    id_to_token: Dict[int, str],
    blank_id: int = 0,
) -> List[str]:
    collapsed: List[str] = []
    previous_token: Optional[str] = None
    for token_id in frame_ids:
        if token_id == blank_id:
            previous_token = None
            continue
        token = id_to_token.get(int(token_id))
        if token is None or token == DEFAULT_BLANK_TOKEN:
            previous_token = None
            continue
        if token == previous_token:
            continue
        collapsed.append(token)
        previous_token = token
    return collapsed


def frame_ids_to_text(frame_ids: Sequence[int], id_to_token: Dict[int, str], blank_id: int = 0) -> str:
    return " ".join(collapse_frame_ids(frame_ids, id_to_token, blank_id=blank_id)).strip()


def phoneme_similarity_score(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if left == DEFAULT_BLANK_TOKEN or right == DEFAULT_BLANK_TOKEN:
        return 0.0

    shared_groups = [
        group_name
        for group_name, group_tokens in PHONEME_SIMILARITY_GROUPS.items()
        if left in group_tokens and right in group_tokens
    ]
    if shared_groups:
        if any(group_name.endswith("_vowel") for group_name in shared_groups):
            return 0.75
        if len(shared_groups) >= 2:
            return 0.85
        return 0.65

    left_is_vowel = any(
        left in group_tokens
        for group_name, group_tokens in PHONEME_SIMILARITY_GROUPS.items()
        if group_name.endswith("_vowel")
    )
    right_is_vowel = any(
        right in group_tokens
        for group_name, group_tokens in PHONEME_SIMILARITY_GROUPS.items()
        if group_name.endswith("_vowel")
    )
    if left_is_vowel and right_is_vowel:
        return 0.2
    if (not left_is_vowel) and (not right_is_vowel):
        return 0.15
    return 0.0


def build_similarity_distribution(vocab: Sequence[str]) -> torch.Tensor:
    labels = [DEFAULT_BLANK_TOKEN, *list(vocab)]
    label_count = len(labels)
    similarity_matrix = torch.zeros(label_count, label_count, dtype=torch.float32)

    for left_index, left_label in enumerate(labels):
        for right_index, right_label in enumerate(labels):
            similarity_matrix[left_index, right_index] = phoneme_similarity_score(left_label, right_label)

    distribution = torch.zeros_like(similarity_matrix)
    for label_index, label in enumerate(labels):
        if label == DEFAULT_BLANK_TOKEN:
            distribution[label_index, label_index] = 1.0
            continue

        neighbor_scores = similarity_matrix[label_index].clone()
        neighbor_scores[0] = 0.0
        neighbor_scores[label_index] = 0.0
        neighbor_total = float(neighbor_scores.sum().item())

        distribution[label_index, label_index] = 1.0
        if neighbor_total > 0.0:
            distribution[label_index] = 0.85 * distribution[label_index]
            distribution[label_index] = distribution[label_index] + 0.15 * (neighbor_scores / neighbor_total)

    return distribution


class GraphAwareFrameCrossEntropyLoss(nn.Module):
    """Frame-level cross-entropy with phoneme-similarity soft targets.

    The blank_penalty term taxes the model for predicting any non-blank token
    on frames whose target IS a non-blank token but the model is still
    assigning probability mass to the blank class.  More precisely it adds

        blank_penalty * p(blank | frame) * 1[target ≠ blank]

    to the per-frame loss.

    Default blank_penalty is 0.5.  A previous run raised this to 2.0 to try to
    suppress over-insertion of high-frequency phonemes (ɪ, n, ɛ, s), but that
    run actually produced *more* insertions of those tokens (87/69/83 vs.
    70/52/50) and a much larger train/val loss gap, indicating the insertions
    are driven by frame-level prediction flicker / overfitting rather than
    blank-dominance.  blank_penalty was reverted to 0.5 accordingly.  Pass
    --blank-penalty to tune: increase toward 1.0-2.0 only if deletions
    (frames where the model predicts blank but the target is a real phoneme)
    become the dominant error mode.
    """

    def __init__(self, vocab: Sequence[str], blank_id: int = 0, blank_penalty: float = 0.5):
        super().__init__()
        self.blank_id = int(blank_id)
        self.blank_penalty = float(blank_penalty)
        self.register_buffer("target_distributions", build_similarity_distribution(vocab))

    def forward(
        self,
        logits: torch.Tensor,
        frame_targets: torch.Tensor,
        frame_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        max_length = log_probs.shape[1]
        valid_mask = (
            torch.arange(max_length, device=frame_lengths.device).unsqueeze(0)
            < frame_lengths.unsqueeze(1)
        )

        flat_log_probs = log_probs[valid_mask]
        flat_targets = frame_targets[valid_mask].clamp(min=0, max=self.target_distributions.shape[0] - 1)
        target_distributions = self.target_distributions.to(logits.device)[flat_targets]

        loss = -(target_distributions * flat_log_probs).sum(dim=-1)

        if self.blank_penalty > 0.0:
            blank_prob = flat_log_probs.exp()[:, self.blank_id]
            nonblank_mask = (flat_targets != self.blank_id).float()
            loss = loss + self.blank_penalty * blank_prob * nonblank_mask

        return loss.mean()


class FrameEncoder(nn.Module):
    """Per-frame ResNet-18 visual encoder.

    Projector dropout raised from 0.2 → 0.45 to reduce overfitting.
    """

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True, finetune: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            # Raised 0.2 → 0.45: targets the large train/val gap in the
            # previous run.  The projector is the narrowest bottleneck before
            # the sequence model, making it the least disruptive place to add
            # regularisation without interfering with LSTM dynamics.
            nn.Dropout(0.45),
        )

        if not finetune:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.backbone(frames)
        return self.projector(features)


class VisualLSTMFrameCE(nn.Module):
    """Bidirectional LSTM over per-frame ResNet-18 features.

    Inter-layer LSTM dropout raised from 0.2 → 0.45 to complement the
    projector dropout increase.  Has no effect when lstm_layers=1 (PyTorch
    silently ignores the dropout parameter for single-layer LSTMs).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        pretrained_backbone: bool = True,
        finetune_backbone: bool = True,
    ):
        super().__init__()
        self.frame_encoder = FrameEncoder(
            hidden_dim=hidden_dim,
            pretrained=pretrained_backbone,
            finetune=finetune_backbone,
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            # Raised 0.2 → 0.45 to match projector dropout.
            dropout=0.45 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, videos: torch.Tensor, video_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = videos.shape
        flat_frames = videos.reshape(batch_size * time_steps, channels, height, width)
        frame_features = self.frame_encoder(flat_frames)
        sequence_features = frame_features.reshape(batch_size, time_steps, -1)

        packed = nn.utils.rnn.pack_padded_sequence(
            sequence_features, video_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return self.classifier(lstm_output)


def decode_greedy(logits: torch.Tensor, blank_id: int = 0) -> List[int]:
    return torch.argmax(logits, dim=-1).tolist()


def tokens_to_text(token_ids: List[int], id_to_token: Dict[int, str], blank_id: int = 0) -> str:
    tokens = [
        id_to_token[token_id]
        for token_id in token_ids
        if token_id != blank_id and token_id in id_to_token
    ]
    return " ".join(tokens).strip()


def align_token_sequences(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> List[Tuple[Optional[str], Optional[str]]]:
    reference_length = len(reference)
    hypothesis_length = len(hypothesis)

    distances = [[0] * (hypothesis_length + 1) for _ in range(reference_length + 1)]
    backpointers: List[List[str]] = [[""] * (hypothesis_length + 1) for _ in range(reference_length + 1)]

    for ref_index in range(1, reference_length + 1):
        distances[ref_index][0] = ref_index
        backpointers[ref_index][0] = "delete"
    for hyp_index in range(1, hypothesis_length + 1):
        distances[0][hyp_index] = hyp_index
        backpointers[0][hyp_index] = "insert"

    for ref_index in range(1, reference_length + 1):
        for hyp_index in range(1, hypothesis_length + 1):
            substitute_cost = distances[ref_index - 1][hyp_index - 1] + int(
                reference[ref_index - 1] != hypothesis[hyp_index - 1]
            )
            delete_cost = distances[ref_index - 1][hyp_index] + 1
            insert_cost = distances[ref_index][hyp_index - 1] + 1

            best_cost, _, best_op = min(
                (substitute_cost, 0, "substitute"),
                (delete_cost, 1, "delete"),
                (insert_cost, 2, "insert"),
                key=lambda item: (item[0], item[1]),
            )

            distances[ref_index][hyp_index] = best_cost
            backpointers[ref_index][hyp_index] = best_op

    aligned_pairs: List[Tuple[Optional[str], Optional[str]]] = []
    ref_index = reference_length
    hyp_index = hypothesis_length
    while ref_index > 0 or hyp_index > 0:
        operation = backpointers[ref_index][hyp_index]
        if operation == "substitute":
            aligned_pairs.append((reference[ref_index - 1], hypothesis[hyp_index - 1]))
            ref_index -= 1
            hyp_index -= 1
        elif operation == "delete":
            aligned_pairs.append((reference[ref_index - 1], None))
            ref_index -= 1
        else:
            aligned_pairs.append((None, hypothesis[hyp_index - 1]))
            hyp_index -= 1

    aligned_pairs.reverse()
    return aligned_pairs


@torch.no_grad()
def evaluate_batch(
    model: VisualLSTMFrameCE,
    batch: Dict[str, Any],
    criterion: GraphAwareFrameCrossEntropyLoss,
    device: torch.device,
    id_to_token: Dict[int, str],
) -> Tuple[float, List[str], List[str]]:
    videos = batch["videos"].to(device)
    video_lengths = batch["video_lengths"].to(device)
    frame_targets = batch["frame_targets"].to(device)

    logits = model(videos, video_lengths)
    loss = criterion(logits, frame_targets, video_lengths)

    predicted_texts: List[str] = []
    target_texts: List[str] = []
    for item_index in range(logits.shape[0]):
        length = int(video_lengths[item_index].item())
        pred_ids = decode_greedy(logits[item_index, :length].cpu(), blank_id=0)
        target_ids = frame_targets[item_index, :length].cpu().tolist()
        predicted_texts.append(tokens_to_text(pred_ids, id_to_token))
        target_texts.append(tokens_to_text(target_ids, id_to_token))

    return float(loss.item()), predicted_texts, target_texts


@torch.no_grad()
def evaluate_loader_with_sequences(
    model: VisualLSTMFrameCE,
    loader: DataLoader,
    criterion: GraphAwareFrameCrossEntropyLoss,
    device: torch.device,
    id_to_token: Dict[int, str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_tokens = 0
    total_token_errors = 0
    all_predictions: List[str] = []
    all_targets: List[str] = []

    for batch in tqdm(loader, desc="Test", leave=False):
        loss, predictions, targets = evaluate_batch(model, batch, criterion, device, id_to_token)
        total_loss += loss
        num_batches += 1
        all_predictions.extend(predictions)
        all_targets.extend(targets)

        for prediction, target in zip(predictions, targets):
            pred_tokens = prediction.split() if prediction else []
            target_tokens = target.split() if target else []
            total_token_errors += levenshtein_distance(pred_tokens, target_tokens)
            total_tokens += max(1, len(target_tokens))

    metrics = {
        "loss": total_loss / max(1, num_batches),
        "token_error_rate": total_token_errors / max(1, total_tokens),
    }
    return metrics, all_predictions, all_targets


def train_epoch(
    model: VisualLSTMFrameCE,
    loader: DataLoader,
    criterion: GraphAwareFrameCrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        videos = batch["videos"].to(device)
        video_lengths = batch["video_lengths"].to(device)
        frame_targets = batch["frame_targets"].to(device)

        logits = model(videos, video_lengths)
        loss = criterion(logits, frame_targets, video_lengths)

        if not torch.isfinite(loss):
            logger.warning("Skipping non-finite loss batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    model: VisualLSTMFrameCE,
    loader: DataLoader,
    criterion: GraphAwareFrameCrossEntropyLoss,
    device: torch.device,
    id_to_token: Dict[int, str],
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_tokens = 0
    total_token_errors = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        loss, predictions, targets = evaluate_batch(model, batch, criterion, device, id_to_token)
        total_loss += loss
        num_batches += 1

        for prediction, target in zip(predictions, targets):
            pred_tokens = prediction.split() if prediction else []
            target_tokens = target.split() if target else []
            total_token_errors += levenshtein_distance(pred_tokens, target_tokens)
            total_tokens += max(1, len(target_tokens))

    return {
        "loss": total_loss / max(1, num_batches),
        "token_error_rate": total_token_errors / max(1, total_tokens),
    }


def levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current_row = [i]
        for j, right_token in enumerate(right, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (left_token != right_token)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def build_phoneme_diagnostics(
    target_texts: Sequence[str],
    predicted_texts: Sequence[str],
    labels: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    label_list = list(labels)
    label_to_index = {label: index for index, label in enumerate(label_list)}
    matrix = np.zeros((len(label_list) + 1, len(label_list) + 1), dtype=np.int64)

    stats: Dict[str, Dict[str, float]] = {
        label: {"support": 0.0, "predicted": 0.0, "correct": 0.0, "missed": 0.0}
        for label in label_list
    }
    top_confusions: Counter[Tuple[str, str]] = Counter()

    for target_text, predicted_text in zip(target_texts, predicted_texts):
        target_tokens = target_text.split() if target_text else []
        predicted_tokens = predicted_text.split() if predicted_text else []

        for actual_token, predicted_token in align_token_sequences(target_tokens, predicted_tokens):
            if actual_token is None:
                if predicted_token is None:
                    continue
                predicted_index = label_to_index.get(predicted_token)
                if predicted_index is None:
                    continue
                matrix[len(label_list), predicted_index] += 1
                stats[predicted_token]["predicted"] += 1
                top_confusions[(INSERTION_TOKEN, predicted_token)] += 1
                continue

            actual_index = label_to_index.get(actual_token)
            if actual_index is None:
                continue

            stats[actual_token]["support"] += 1

            if predicted_token is None:
                matrix[actual_index, len(label_list)] += 1
                stats[actual_token]["missed"] += 1
                top_confusions[(actual_token, MISSING_PREDICTION_TOKEN)] += 1
                continue

            predicted_index = label_to_index.get(predicted_token)
            if predicted_index is None:
                continue

            matrix[actual_index, predicted_index] += 1
            stats[predicted_token]["predicted"] += 1
            if actual_token == predicted_token:
                stats[actual_token]["correct"] += 1
            else:
                top_confusions[(actual_token, predicted_token)] += 1

    per_phoneme_report: List[Dict[str, Any]] = []
    for label in label_list:
        support = int(stats[label]["support"])
        predicted = int(stats[label]["predicted"])
        correct = int(stats[label]["correct"])
        missed = int(stats[label]["missed"])
        recall = correct / support if support else 0.0
        precision = correct / predicted if predicted else 0.0
        per_phoneme_report.append({
            "phoneme": label,
            "support": support,
            "predicted": predicted,
            "correct": correct,
            "missed": missed,
            "recall": recall,
            "precision": precision,
        })

    per_phoneme_report.sort(key=lambda item: (item["recall"], -item["support"], str(item["phoneme"])))
    top_confusion_report = [
        {"actual": actual, "predicted": predicted, "count": int(count)}
        for (actual, predicted), count in top_confusions.most_common(20)
    ]
    return matrix, stats, per_phoneme_report, top_confusion_report


def save_confusion_matrix_csv(matrix: np.ndarray, labels: Sequence[str], output_path: Path) -> None:
    row_labels = list(labels) + [INSERTION_TOKEN]
    column_labels = list(labels) + [MISSING_PREDICTION_TOKEN]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *column_labels])
        for row_label, row_values in zip(row_labels, matrix):
            writer.writerow([row_label, *[int(v) for v in row_values.tolist()]])


def save_confusion_matrix_plot(matrix: np.ndarray, labels: Sequence[str], output_path: Path) -> None:
    display_labels = list(labels) + [MISSING_PREDICTION_TOKEN]
    row_labels = list(labels) + [INSERTION_TOKEN]
    size = max(8.0, 0.45 * len(display_labels))

    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted phoneme")
    ax.set_ylabel("Actual phoneme")
    ax.set_title("Phoneme confusion matrix")

    if matrix.size <= 400:
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = int(matrix[row_index, column_index])
                if value:
                    ax.text(
                        column_index, row_index, str(value),
                        ha="center", va="center", color="black", fontsize=8,
                    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_samples(video_dir: Path, phoneme_csv: Path) -> List[Sample]:
    targets = load_aligned_phoneme_targets(phoneme_csv)
    samples: List[Sample] = []

    for video_path in find_lip_videos(video_dir):
        key = normalize_stem(video_path)
        record = targets.get(key)
        if not record:
            continue
        samples.append(Sample(
            video_path=video_path,
            frame_tokens=record.frame_tokens,
            sentence=record.sentence,
            canonical_tokens=record.canonical_tokens,
        ))

    if not samples:
        raise FileNotFoundError(
            f"No paired samples found. Check that {video_dir} matches stems in {phoneme_csv}"
        )

    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a visual LSTM Frame CE model on lip crops and frame phoneme targets"
    )
    parser.add_argument("--video-dir", type=str, default="s1_lip_crops")
    parser.add_argument("--phoneme-csv", type=str, default="phonemes_s1_aligned/phoneme_predictions.csv")
    parser.add_argument("--output-dir", type=str, default="visual_lstm_ce_results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Raised from 1e-4 → 5e-4; pass --weight-decay to override.
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="L2 weight decay (raised from 1e-4 → 5e-4 to reduce overfitting)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--blank-penalty", type=float, default=0.5,
        help=(
            "Blank-penalty coefficient for GraphAwareFrameCrossEntropyLoss. "
            "Reverted from 2.0 back to 0.5: raising it to 2.0 in a previous run "
            "did not reduce insertion errors (ɪ/ɛ/n insertions rose from "
            "70/50/52 to 87/83/69) and coincided with much worse overfitting. "
            "Increase toward 1.0-2.0 only if frame-level deletions (predicting "
            "blank on non-blank frames) become the dominant error mode again."
        ),
    )
    parser.add_argument(
        "--class-aware-sampling", action="store_true",
        help="Use class-aware sampling based on inverse token frequency",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info(
        "Regularisation: blank_penalty=%.2f  weight_decay=%.2e  "
        "projector_dropout=0.45  lstm_dropout=0.45",
        args.blank_penalty, args.weight_decay,
    )
    logger.info(
        "Early stopping: patience=%d epochs, min_delta=%.4f, selection metric=val_ter",
        TrainingConfig.early_stopping_patience, TrainingConfig.early_stopping_min_delta,
    )

    video_dir = Path(args.video_dir)
    phoneme_csv = Path(args.phoneme_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(video_dir, phoneme_csv)
    train_samples, val_samples, test_samples = split_samples(
        samples,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=args.seed,
    )

    vocab = build_vocab(samples)
    id_to_token = {index + 1: token for index, token in enumerate(vocab)}
    id_to_token[0] = DEFAULT_BLANK_TOKEN

    with (output_dir / "phoneme_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"blank_id": 0, "blank_token": DEFAULT_BLANK_TOKEN, "tokens": vocab},
            handle, ensure_ascii=False, indent=2,
        )

    train_dataset = LipPhonemeDataset(
        train_samples, vocab=vocab, frame_size=args.frame_size,
        frame_stride=args.frame_stride, augment=True,
    )
    val_dataset = LipPhonemeDataset(
        val_samples, vocab=vocab, frame_size=args.frame_size,
        frame_stride=args.frame_stride, augment=False,
    )
    test_dataset = LipPhonemeDataset(
        test_samples, vocab=vocab, frame_size=args.frame_size,
        frame_stride=args.frame_stride, augment=False,
    )

    if args.class_aware_sampling:
        phoneme_counts: Counter = Counter()
        for s in train_samples:
            phoneme_counts.update(s.frame_tokens)

        inv_freq = {token: 1.0 / max(1, count) for token, count in phoneme_counts.items()}
        sample_weights: List[float] = []
        for s in train_samples:
            toks = [t for t in s.frame_tokens if t in inv_freq]
            w = float(sum(inv_freq[t] for t in toks) / len(toks)) if toks else 1.0
            sample_weights.append(w)

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=4, collate_fn=collate_batch,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_batch,
        )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = VisualLSTMFrameCE(
        vocab_size=len(vocab) + 1,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        pretrained_backbone=not args.no_pretrained,
        finetune_backbone=not args.freeze_backbone,
    ).to(device)

    criterion = GraphAwareFrameCrossEntropyLoss(vocab=vocab, blank_id=0, blank_penalty=args.blank_penalty)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=TrainingConfig.early_stopping_min_delta,
        min_lr=args.min_lr,
    )

    logger.info(
        "Loaded %d samples (%d train / %d val / %d test)",
        len(samples), len(train_samples), len(val_samples), len(test_samples),
    )
    logger.info("Phoneme vocabulary size: %d (+ blank)", len(vocab))

    if args.dry_run:
        batch = next(iter(train_loader))
        videos = batch["videos"].to(device)
        video_lengths = batch["video_lengths"].to(device)
        frame_targets = batch["frame_targets"].to(device)
        with torch.no_grad():
            logits = model(videos, video_lengths)
            loss = criterion(logits, frame_targets, video_lengths)
        logger.info("Dry run videos shape: %s", tuple(videos.shape))
        logger.info("Dry run logits shape: %s", tuple(logits.shape))
        logger.info("Dry run Frame Graph CE loss: %.6f", float(loss.item()))
        return

    best_val_ter = float("inf")
    epochs_without_improvement = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_ter": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device, id_to_token)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_ter"].append(val_metrics["token_error_rate"])
        scheduler.step(val_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | val_TER=%.4f | lr=%.2e",
            epoch, args.epochs, train_loss,
            val_metrics["loss"], val_metrics["token_error_rate"], current_lr,
        )

        # Checkpoint selection is based on val_ter (not val_loss): a previous
        # run showed val_loss and val_ter diverging after ~epoch 15 (val_loss
        # kept rising while val_ter still improved for several more epochs),
        # and TER is the metric we actually care about.
        improved = val_metrics["token_error_rate"] < (best_val_ter - TrainingConfig.early_stopping_min_delta)
        if improved:
            best_val_ter = val_metrics["token_error_rate"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_ter": best_val_ter,
                    "vocab": vocab,
                    "config": vars(args),
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= TrainingConfig.early_stopping_patience:
            logger.info("Early stopping triggered")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": vars(args),
        },
        output_dir / "final_model.pt",
    )

    test_metrics, test_predictions, test_targets = evaluate_loader_with_sequences(
        model, test_loader, criterion, device, id_to_token,
    )
    matrix, _, per_phoneme_report, top_confusions = build_phoneme_diagnostics(
        test_targets, test_predictions, vocab,
    )

    save_confusion_matrix_csv(matrix, vocab, output_dir / "test_confusion_matrix.csv")
    save_confusion_matrix_plot(matrix, vocab, output_dir / "test_confusion_matrix.png")

    completely_missed_phonemes = [
        item["phoneme"]
        for item in per_phoneme_report
        if int(item["support"]) > 0 and int(item["correct"]) == 0
    ]

    diagnostics = {
        "completely_missed_phonemes": completely_missed_phonemes,
        "per_phoneme_report": per_phoneme_report,
        "top_confusions": top_confusions,
    }
    with (output_dir / "test_phoneme_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)

    with (output_dir / "test_phoneme_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["phoneme", "support", "predicted", "correct", "missed", "recall", "precision"],
        )
        writer.writeheader()
        writer.writerows(per_phoneme_report)

    with (output_dir / "test_metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"test_loss={test_metrics['loss']:.6f}\n")
        handle.write(f"test_token_error_rate={test_metrics['token_error_rate']:.6f}\n")
        handle.write(f"completely_missed_phonemes={len(completely_missed_phonemes)}\n")
        if completely_missed_phonemes:
            handle.write("missed_phonemes=" + ", ".join(completely_missed_phonemes) + "\n")
        if top_confusions:
            handle.write("top_confusions=\n")
            for item in top_confusions[:10]:
                handle.write(f"  {item['actual']} -> {item['predicted']}: {item['count']}\n")

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    logger.info("Training complete. Best val TER: %.4f", best_val_ter)
    logger.info("Test loss: %.4f | Test TER: %.4f", test_metrics["loss"], test_metrics["token_error_rate"])


if __name__ == "__main__":
    main()