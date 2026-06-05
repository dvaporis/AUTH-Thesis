#!/usr/bin/env python3
"""Train a visual LSTM + CTC model on lip crops and wav2vec2 phoneme targets.

The training pairs come from:
 - lip_crop_results_full: cropped lip videos
 - phoneme_results/phoneme_predictions.csv: phoneme strings per source clip

Each video is treated as a variable-length frame sequence. A per-frame visual
encoder extracts features, a bidirectional LSTM aggregates them through time,
and a linear projection produces frame-wise phoneme logits. `nn.CTCLoss` is used
to align the frame emissions with the phoneme target sequence.

Usage:
    python train_lip_lstm_ctc.py --dry-run
    python train_lip_lstm_ctc.py --epochs 20 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    lstm_layers: int = 2
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-3


@dataclass
class Sample:
    video_path: Path
    target_tokens: List[str]


def find_lip_videos(data_dir: Path) -> List[Path]:
    files = [path for path in sorted(data_dir.rglob("*")) if path.is_file() and path.suffix.lower() in VIDEO_EXTS]
    return files


def normalize_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_lipcrop"):
        return stem[: -len("_lipcrop")]
    return stem


def normalize_csv_file_field(file_field: str) -> str:
    filename = file_field.replace("\\", "/").rsplit("/", 1)[-1]
    return Path(filename).stem


def load_phoneme_targets(csv_path: Path) -> Dict[str, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Phoneme CSV not found: {csv_path}")

    targets: Dict[str, List[str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_field = row.get("file", "").strip()
            phoneme_field = row.get("phonemes", "").strip()
            if not file_field or not phoneme_field:
                continue

            key = normalize_csv_file_field(file_field)
            targets[key] = phoneme_field.split()

    if not targets:
        raise ValueError(f"No phoneme targets could be read from {csv_path}")

    return targets


def build_vocab(samples: Sequence[Sample]) -> List[str]:
    counter: Counter[str] = Counter()
    for sample in samples:
        counter.update(sample.target_tokens)
    vocab = sorted(counter.keys())
    if not vocab:
        raise ValueError("Cannot build an empty phoneme vocabulary")
    return vocab


def split_samples(samples: List[Sample], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample], List[Sample]]:
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
    def __init__(self, samples: Sequence[Sample], vocab: Sequence[str], frame_size: int = 224, frame_stride: int = 1, augment: bool = False):
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
        train_transforms = v2.Compose([
            v2.Resize((self.frame_size, self.frame_size), interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05)], p=0.4),
            v2.ToDtype(torch.float32, scale=True),
        ])
        val_transforms = v2.Compose([
            v2.Resize((self.frame_size, self.frame_size), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ])
        return train_transforms, val_transforms

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_path: Path) -> torch.Tensor:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        frames: List[torch.Tensor] = []
        frame_index = 0
        try:
            while True:
                success, frame_bgr = capture.read()
                if not success:
                    break
                if frame_index % self.frame_stride != 0:
                    frame_index += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
                frames.append(frame_tensor)
                frame_index += 1
        finally:
            capture.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")

        video = torch.stack(frames, dim=0)  # [T, C, H, W]
        # Apply spatial/image transforms first (resizing, flip, color jitter)
        if self.is_training:
            video = self.train_transforms(video)
        else:
            video = self.val_transforms(video)

        # Additional lightweight augmentations applied per-video during training
        if self.is_training:
            # brightness jitter
            b_factor = 1.0 + (random.random() * 0.2 - 0.1)
            video = video * float(b_factor)

            # contrast jitter
            mean = video.mean(dim=(1, 2, 3), keepdim=True)
            c_factor = 1.0 + (random.random() * 0.2 - 0.1)
            video = (video - mean) * float(c_factor) + mean

            # random erasing (simple rectangular mask)
            if random.random() < 0.2:
                t, c, h, w = video.shape
                rect_h = max(1, int(h * (0.08 + random.random() * 0.15)))
                rect_w = max(1, int(w * (0.08 + random.random() * 0.15)))
                y = random.randint(0, max(0, h - rect_h))
                x = random.randint(0, max(0, w - rect_w))
                video[:, :, y : y + rect_h, x : x + rect_w] = 0.0
        return video

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        video = self._load_video(sample.video_path)
        target_ids = [self.token_to_id[token] for token in sample.target_tokens if token in self.token_to_id]

        if not target_ids:
            raise RuntimeError(f"Empty target sequence for {sample.video_path}")

        return {
            "video": video,
            "video_length": torch.tensor(video.shape[0], dtype=torch.long),
            "targets": torch.tensor(target_ids, dtype=torch.long),
            "target_length": torch.tensor(len(target_ids), dtype=torch.long),
            "stem": sample.video_path.stem,
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[str]]:
    batch_size = len(batch)
    video_lengths = torch.tensor([int(item["video_length"]) for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([int(item["target_length"]) for item in batch], dtype=torch.long)

    max_frames = int(video_lengths.max().item())
    channels, height, width = batch[0]["video"].shape[1:]
    videos = torch.zeros(batch_size, max_frames, channels, height, width, dtype=batch[0]["video"].dtype)

    targets: List[torch.Tensor] = []
    stems: List[str] = []
    for index, item in enumerate(batch):
        video = item["video"]
        videos[index, : video.shape[0]] = video
        targets.append(item["targets"])
        stems.append(str(item["stem"]))

    flat_targets = torch.cat(targets, dim=0)

    return {
        "videos": videos,
        "video_lengths": video_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "stems": stems,
    }


class FrameEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, pretrained: bool = True, finetune: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        if not finetune:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.backbone(frames)
        return self.projector(features)


class VisualLSTMCTC(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 256, lstm_layers: int = 2, pretrained_backbone: bool = True, finetune_backbone: bool = True):
        super().__init__()
        self.frame_encoder = FrameEncoder(hidden_dim=hidden_dim, pretrained=pretrained_backbone, finetune=finetune_backbone)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, videos: torch.Tensor, video_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = videos.shape
        flat_frames = videos.reshape(batch_size * time_steps, channels, height, width)
        frame_features = self.frame_encoder(flat_frames)
        sequence_features = frame_features.reshape(batch_size, time_steps, -1)

        packed = nn.utils.rnn.pack_padded_sequence(sequence_features, video_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        logits = self.classifier(lstm_output)
        return logits


def decode_greedy(logits: torch.Tensor, blank_id: int = 0) -> List[int]:
    predicted = torch.argmax(logits, dim=-1).tolist()
    collapsed: List[int] = []
    previous_id: Optional[int] = None
    for token_id in predicted:
        if token_id == blank_id:
            previous_id = token_id
            continue
        if previous_id == token_id:
            continue
        collapsed.append(int(token_id))
        previous_id = token_id
    return collapsed


def tokens_to_text(token_ids: List[int], id_to_token: Dict[int, str], blank_id: int = 0) -> str:
    tokens = [id_to_token[token_id] for token_id in token_ids if token_id != blank_id and token_id in id_to_token]
    return " ".join(tokens).strip()


def split_target_and_lengths(targets: torch.Tensor, target_lengths: torch.Tensor) -> List[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    offset = 0
    for length in target_lengths.tolist():
        next_offset = offset + int(length)
        chunks.append(targets[offset:next_offset])
        offset = next_offset
    return chunks


def align_token_sequences(reference: Sequence[str], hypothesis: Sequence[str]) -> List[Tuple[Optional[str], Optional[str]]]:
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
            substitute_cost = distances[ref_index - 1][hyp_index - 1] + int(reference[ref_index - 1] != hypothesis[hyp_index - 1])
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
def evaluate_batch(model: VisualLSTMCTC, batch: Dict[str, torch.Tensor | List[str]], criterion: nn.CTCLoss, device: torch.device, id_to_token: Dict[int, str]) -> Tuple[float, List[str], List[str]]:
    videos = batch["videos"].to(device)
    video_lengths = batch["video_lengths"].to(device)
    targets = batch["targets"].to(device)
    target_lengths = batch["target_lengths"].to(device)

    logits = model(videos, video_lengths)
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    loss = criterion(log_probs, targets, video_lengths, target_lengths)

    target_chunks = split_target_and_lengths(targets.cpu(), target_lengths.cpu())
    predicted_texts: List[str] = []
    target_texts: List[str] = []
    for item_index in range(logits.shape[0]):
        pred_ids = decode_greedy(logits[item_index, : int(video_lengths[item_index])].cpu(), blank_id=0)
        predicted_texts.append(tokens_to_text(pred_ids, id_to_token))
        target_texts.append(tokens_to_text(target_chunks[item_index].tolist(), id_to_token))

    return float(loss.item()), predicted_texts, target_texts


@torch.no_grad()
def evaluate_loader_with_sequences(
    model: VisualLSTMCTC,
    loader: DataLoader,
    criterion: nn.CTCLoss,
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


def train_epoch(model: VisualLSTMCTC, loader: DataLoader, criterion: nn.CTCLoss, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        videos = batch["videos"].to(device)
        video_lengths = batch["video_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        logits = model(videos, video_lengths)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        loss = criterion(log_probs, targets, video_lengths, target_lengths)

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
def validate(model: VisualLSTMCTC, loader: DataLoader, criterion: nn.CTCLoss, device: torch.device, id_to_token: Dict[int, str]) -> Dict[str, float]:
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
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]], List[Dict[str, object]], List[Dict[str, object]]]:
    label_list = list(labels)
    label_to_index = {label: index for index, label in enumerate(label_list)}
    matrix = np.zeros((len(label_list) + 1, len(label_list) + 1), dtype=np.int64)

    stats: Dict[str, Dict[str, float]] = {
        label: {"support": 0.0, "predicted": 0.0, "correct": 0.0, "missed": 0.0} for label in label_list
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

    per_phoneme_report: List[Dict[str, object]] = []
    for label in label_list:
        support = int(stats[label]["support"])
        predicted = int(stats[label]["predicted"])
        correct = int(stats[label]["correct"])
        missed = int(stats[label]["missed"])
        recall = correct / support if support else 0.0
        precision = correct / predicted if predicted else 0.0
        per_phoneme_report.append(
            {
                "phoneme": label,
                "support": support,
                "predicted": predicted,
                "correct": correct,
                "missed": missed,
                "recall": recall,
                "precision": precision,
            }
        )

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
            writer.writerow([row_label, *[int(value) for value in row_values.tolist()]])


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
                    ax.text(column_index, row_index, str(value), ha="center", va="center", color="black", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_samples(video_dir: Path, phoneme_csv: Path) -> List[Sample]:
    targets = load_phoneme_targets(phoneme_csv)
    samples: List[Sample] = []

    for video_path in find_lip_videos(video_dir):
        key = normalize_stem(video_path)
        target_tokens = targets.get(key)
        if not target_tokens:
            continue
        samples.append(Sample(video_path=video_path, target_tokens=target_tokens))

    if not samples:
        raise FileNotFoundError(
            f"No paired samples found. Check that {video_dir} matches stems in {phoneme_csv}"
        )

    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a visual LSTM CTC model on lip crops and phoneme targets")
    parser.add_argument("--video-dir", type=str, default="lip_crop_results_full")
    parser.add_argument("--phoneme-csv", type=str, default="phoneme_results/phoneme_predictions.csv")
    parser.add_argument("--output-dir", type=str, default="visual_lstm_ctc_results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    parser.add_argument("--class-aware-sampling", action="store_true", help="Use class-aware (weighted) sampling based on phoneme inverse frequency for training")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

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
        json.dump({"blank_id": 0, "blank_token": DEFAULT_BLANK_TOKEN, "tokens": vocab}, handle, ensure_ascii=False, indent=2)

    train_dataset = LipPhonemeDataset(train_samples, vocab=vocab, frame_size=args.frame_size, frame_stride=args.frame_stride, augment=True)
    val_dataset = LipPhonemeDataset(val_samples, vocab=vocab, frame_size=args.frame_size, frame_stride=args.frame_stride, augment=False)
    test_dataset = LipPhonemeDataset(test_samples, vocab=vocab, frame_size=args.frame_size, frame_stride=args.frame_stride, augment=False)

    # Optionally use class-aware sampling based on inverse phoneme frequency
    if args.class_aware_sampling:
        phoneme_counts: Counter = Counter()
        for s in train_samples:
            phoneme_counts.update(s.target_tokens)

        inv_freq = {token: 1.0 / max(1, count) for token, count in phoneme_counts.items()}
        sample_weights: List[float] = []
        for s in train_samples:
            toks = [t for t in s.target_tokens if t in inv_freq]
            if toks:
                w = float(sum(inv_freq[t] for t in toks) / len(toks))
            else:
                w = 1.0
            sample_weights.append(w)

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, collate_fn=collate_batch)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = VisualLSTMCTC(
        vocab_size=len(vocab) + 1,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        pretrained_backbone=not args.no_pretrained,
        finetune_backbone=not args.freeze_backbone,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=TrainingConfig.early_stopping_min_delta,
        min_lr=args.min_lr,
    )

    logger.info("Loaded %d samples (%d train / %d val / %d test)", len(samples), len(train_samples), len(val_samples), len(test_samples))
    logger.info("Phoneme vocabulary size: %d (+ blank)", len(vocab))

    if args.dry_run:
        batch = next(iter(train_loader))
        videos = batch["videos"].to(device)
        video_lengths = batch["video_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        with torch.no_grad():
            logits = model(videos, video_lengths)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, video_lengths, target_lengths)
        logger.info("Dry run videos shape: %s", tuple(videos.shape))
        logger.info("Dry run logits shape: %s", tuple(logits.shape))
        logger.info("Dry run CTC loss: %.6f", float(loss.item()))
        return

    best_val_loss = float("inf")
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
            epoch,
            args.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["token_error_rate"],
            current_lr,
        )

        improved = val_metrics["loss"] < (best_val_loss - TrainingConfig.early_stopping_min_delta)
        if improved:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
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
        model,
        test_loader,
        criterion,
        device,
        id_to_token,
    )
    matrix, _, per_phoneme_report, top_confusions = build_phoneme_diagnostics(test_targets, test_predictions, vocab)

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
        writer = csv.DictWriter(handle, fieldnames=["phoneme", "support", "predicted", "correct", "missed", "recall", "precision"])
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

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Test loss: %.4f | Test TER: %.4f", test_metrics["loss"], test_metrics["token_error_rate"])


if __name__ == "__main__":
    main()