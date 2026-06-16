#!/usr/bin/env python3
"""Train a visual lip-reading model on GRID corpus lip crops.

Architecture
------------
- 3D Conv frontend  : captures inter-frame motion at low parameter cost
- Positional encoding + 2-layer BiLSTM (or Transformer, switchable via --encoder)
- Frame-wise linear classifier

Loss
----
Primary   : Frame-level cross-entropy with viseme-aware soft targets
            (model is penalised less for confusing visually identical phonemes)
Auxiliary : CTC on canonical phoneme sequence (sequence-level regularisation)
Weighting : Per-frame confidence derived from audio-alignment scores in spans_json

Data contract (phoneme_predictions.csv columns)
-----------------------------------------------
  stem               : video filename without extension
  sentence           : text transcript (unused in training)
  canonical_phonemes : space-separated ground-truth phoneme sequence
  num_frames         : total video frame count
  per_frame_labels   : JSON array of per-frame phoneme strings ('' = silence)
  spans_json         : JSON array of {token, start_sec, end_sec, score} dicts

Usage
-----
  python train_lip_viseme.py \
      --video-dir   s1_lip_crops \
      --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
      --output-dir  viseme_results

  # Quick smoke-test (1 batch, no weight saving)
  python train_lip_viseme.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
BLANK_TOKEN = "<blank>"
SILENCE_TOKEN = ""          # raw value used in per_frame_labels CSV
MISSING_TOKEN = "<missed>"
INSERTION_TOKEN = "<inserted>"

# ---------------------------------------------------------------------------
# Viseme groups
#
# Phonemes within the same viseme class are visually indistinguishable.
# The similarity penalty is strongest here (score → 1.0 within a class).
# Groups are ordered from most to least visually distinctive.
# References: Bozkurt et al. 2007; Montgomery & Jackson 1983 (GRID-relevant).
# ---------------------------------------------------------------------------

VISEME_GROUPS: Dict[str, List[str]] = {
    # Completely indistinguishable on lips
    "bilabial":       ["p", "b", "m"],
    "labiodental":    ["f", "v"],
    "dental":         ["θ", "ð"],
    # Tongue-tip / alveolar — blade hidden behind teeth
    "alveolar":       ["t", "d", "n", "s", "z", "l"],
    # Post-alveolar / palato-alveolar
    "postalveolar":   ["ʃ", "ʒ", "tʃ", "dʒ"],
    # Velar — entirely invisible
    "velar":          ["k", "g", "ŋ"],
    # Glottal
    "glottal":        ["h"],
    # Approximants (partially visible rounding for /w/)
    "approximant":    ["w", "j", "ɹ", "r"],
    # Vowels: grouped by lip aperture / rounding
    "close_front":    ["iː", "i", "ɪ"],
    "close_back":     ["uː", "u", "ʊ"],
    "mid_front":      ["e", "eɪ", "ɛ"],
    "mid_central":    ["ə", "ɐ", "ɜ", "ɚ", "ʌ"],
    "mid_back":       ["ɔ", "ɔɪ", "oʊ", "o"],
    "open_front":     ["æ", "a", "aɪ", "aʊ"],
    "open_back":      ["ɑ", "ɒ"],
}

# Articulatory feature vectors [place, manner, voiced]
# Used for a secondary, finer-grained distance between non-viseme phonemes.
# Values are integers; distance = normalised Hamming distance over features.
_PLACE   = {"bilabial": 0, "labiodental": 1, "dental": 2, "alveolar": 3,
            "postalveolar": 4, "palatal": 5, "velar": 6, "glottal": 7,
            "vowel": 8}
_MANNER  = {"plosive": 0, "fricative": 1, "affricate": 2, "nasal": 3,
            "approximant": 4, "lateral": 5, "vowel": 6}
_VOICED  = {"voiced": 1, "unvoiced": 0}

ARTICULATORY_FEATURES: Dict[str, Tuple[int, int, int]] = {
    # (place_idx, manner_idx, voiced_idx)
    "p": (0, 0, 0), "b": (0, 0, 1), "m": (0, 3, 1),
    "f": (1, 1, 0), "v": (1, 1, 1),
    "θ": (2, 1, 0), "ð": (2, 1, 1),
    "t": (3, 0, 0), "d": (3, 0, 1), "n": (3, 3, 1),
    "s": (3, 1, 0), "z": (3, 1, 1), "l": (3, 5, 1),
    "ʃ": (4, 1, 0), "ʒ": (4, 1, 1), "tʃ": (4, 2, 0), "dʒ": (4, 2, 1),
    "k": (6, 0, 0), "g": (6, 0, 1), "ŋ": (6, 3, 1),
    "h": (7, 1, 0),
    "w": (0, 4, 1), "j": (5, 4, 1), "ɹ": (3, 4, 1), "r": (3, 4, 1),
    # Vowels — all share place=vowel, manner=vowel, voiced=1
    **{k: (8, 6, 1) for k in [
        "iː","i","ɪ","uː","u","ʊ","e","eɪ","ɛ","ə","ɐ","ɜ","ɚ","ʌ",
        "ɔ","ɔɪ","oʊ","o","æ","a","aɪ","aʊ","ɑ","ɒ",
    ]},
}


# ---------------------------------------------------------------------------
# Phoneme → viseme index map  (built once from VISEME_GROUPS)
# ---------------------------------------------------------------------------

def _build_viseme_map() -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for group_idx, phonemes in enumerate(VISEME_GROUPS.values()):
        for ph in phonemes:
            mapping[ph] = group_idx
    return mapping

PHONEME_TO_VISEME: Dict[str, int] = _build_viseme_map()
NUM_VISEME_GROUPS = len(VISEME_GROUPS)


# ---------------------------------------------------------------------------
# Similarity matrix construction
# ---------------------------------------------------------------------------

def _articulatory_similarity(a: str, b: str) -> float:
    """Normalised articulatory similarity in [0, 1]."""
    fa = ARTICULATORY_FEATURES.get(a)
    fb = ARTICULATORY_FEATURES.get(b)
    if fa is None or fb is None:
        return 0.0
    # 3 binary/categorical features; normalise by max possible distance
    diffs = sum(int(x != y) for x, y in zip(fa, fb))
    return 1.0 - diffs / 3.0


def build_similarity_matrix(vocab: Sequence[str]) -> torch.Tensor:
    """
    Build an (N+1) × (N+1) similarity matrix where index 0 is BLANK_TOKEN.

    Entry [i, j] ∈ [0, 1]:
      - 1.0  : identical
      - 0.95 : same viseme class (visually indistinguishable)
      - articulatory_similarity * 0.4 : same consonant class, different viseme
      - 0.0  : blank vs. anything
    """
    labels = [BLANK_TOKEN] + list(vocab)
    n = len(labels)
    mat = torch.zeros(n, n, dtype=torch.float32)

    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                mat[i, j] = 1.0
                continue
            if a == BLANK_TOKEN or b == BLANK_TOKEN:
                mat[i, j] = 0.0
                continue

            va = PHONEME_TO_VISEME.get(a, -1)
            vb = PHONEME_TO_VISEME.get(b, -1)

            if va != -1 and va == vb:
                # Same viseme class → visually indistinguishable
                mat[i, j] = 0.95
            else:
                # Fall back to articulatory distance
                mat[i, j] = _articulatory_similarity(a, b) * 0.4

    return mat


def similarity_matrix_to_soft_targets(sim: torch.Tensor) -> torch.Tensor:
    """
    Convert raw similarity scores to proper probability distributions.

    For each row (true label), mass is split:
      85% on the true label, 15% distributed over neighbours
      proportionally to their similarity scores (excluding self).

    Blank rows keep 100% mass on blank (we do not want the model
    to ever predict a phoneme when the target is silence).
    """
    n = sim.shape[0]
    dist = torch.zeros_like(sim)

    for i in range(n):
        if i == 0:
            # Blank: hard target — model must learn when to be silent
            dist[i, 0] = 1.0
            continue

        neighbour = sim[i].clone()
        neighbour[i] = 0.0          # exclude self
        neighbour[0] = 0.0          # exclude blank from soft mass
        total = neighbour.sum()

        dist[i, i] = 0.85
        if total > 0:
            dist[i] += 0.15 * (neighbour / total)

    return dist


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    video_path: Path
    frame_tokens: List[str]           # per-frame phoneme, '' = silence
    canonical_tokens: List[str]       # collapsed sequence for CTC
    frame_weights: List[float]        # per-frame confidence weight
    sentence: str = ""


@dataclass
class AlignedRecord:
    frame_tokens: List[str]
    canonical_tokens: List[str]
    frame_weights: List[float]
    sentence: str
    num_frames: int


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _parse_frame_weights(
    spans_json_str: str,
    num_frames: int,
    fps: float = 25.0,
) -> List[float]:
    """
    Derive a per-frame confidence weight from spans_json.

    Each span has a log-prob `score`.  We convert to a linear confidence,
    then propagate it to the frames that fall within [start_sec, end_sec].
    Frames not covered by any span (silence) get weight 1.0 (neutral).
    """
    weights = [1.0] * num_frames

    if not spans_json_str.strip():
        return weights

    try:
        spans = json.loads(spans_json_str)
    except json.JSONDecodeError:
        return weights

    for span in spans:
        score = float(span.get("score", 0.0))  # log-prob, ≤ 0
        # Map log-prob to [0.5, 1.5]: well-aligned frames get up-weight,
        # poorly-aligned frames (very negative score) get down-weight.
        # Clamp score to [-10, 0] to avoid extreme rescaling.
        clamped = max(-10.0, min(0.0, score))
        confidence = 1.0 + clamped / 10.0   # in [0.0, 1.0] → remap to [0.5, 1.5]
        confidence = 0.5 + confidence        # shift to [0.5, 1.5]

        start_frame = int(span.get("start_sec", 0.0) * fps)
        end_frame   = int(span.get("end_sec",   0.0) * fps)
        for f in range(start_frame, min(end_frame + 1, num_frames)):
            weights[f] = confidence

    return weights


def load_phoneme_csv(csv_path: Path, fps: float = 25.0) -> Dict[str, AlignedRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Phoneme CSV not found: {csv_path}")

    records: Dict[str, AlignedRecord] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stem = row.get("stem", "").strip()
            frame_labels_raw = row.get("per_frame_labels", "").strip()
            if not stem or not frame_labels_raw:
                continue

            try:
                raw_labels: List[Optional[str]] = json.loads(frame_labels_raw)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping %s: bad per_frame_labels JSON (%s)", stem, exc)
                continue

            # Normalise: None / whitespace → '' (silence)
            frame_tokens = [
                (lbl.strip() if isinstance(lbl, str) else "")
                for lbl in raw_labels
            ]
            if not frame_tokens:
                continue

            # Canonical sequence: collapse runs and drop silence
            canonical: List[str] = []
            prev: Optional[str] = None
            for tok in frame_tokens:
                if tok == "":
                    prev = None
                    continue
                if tok != prev:
                    canonical.append(tok)
                    prev = tok

            if not canonical:
                continue

            num_frames_str = row.get("num_frames", "").strip()
            num_frames = int(num_frames_str) if num_frames_str.isdigit() else len(frame_tokens)

            frame_weights = _parse_frame_weights(
                row.get("spans_json", ""), num_frames=len(frame_tokens), fps=fps
            )

            records[stem] = AlignedRecord(
                frame_tokens=frame_tokens,
                canonical_tokens=canonical,
                frame_weights=frame_weights,
                sentence=row.get("sentence", "").strip(),
                num_frames=num_frames,
            )

    if not records:
        raise ValueError(f"No valid records found in {csv_path}")

    logger.info("Loaded %d records from %s", len(records), csv_path)
    return records


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(samples: Sequence[Sample], min_count: int = 2) -> List[str]:
    """
    Build sorted phoneme vocabulary from training samples.
    Phonemes appearing fewer than `min_count` times are merged into their
    closest viseme-class neighbour that does meet the threshold.
    """
    counter: Counter[str] = Counter()
    for s in samples:
        counter.update(t for t in s.frame_tokens if t != "")

    # First pass: keep tokens above threshold
    kept = {tok for tok, cnt in counter.items() if cnt >= min_count}

    # Second pass: remap rare tokens to nearest frequent viseme neighbour
    remap: Dict[str, str] = {}
    for tok, cnt in counter.items():
        if tok in kept:
            remap[tok] = tok
            continue
        # Find a kept token in the same viseme group
        my_viseme = PHONEME_TO_VISEME.get(tok, -1)
        best: Optional[str] = None
        best_count = -1
        for candidate in kept:
            if PHONEME_TO_VISEME.get(candidate, -2) == my_viseme and counter[candidate] > best_count:
                best = candidate
                best_count = counter[candidate]
        if best is not None:
            remap[tok] = best
        else:
            # Fall back to most frequent token overall
            remap[tok] = counter.most_common(1)[0][0]

    return sorted(kept), remap


def apply_vocab_remap(samples: List[Sample], remap: Dict[str, str]) -> List[Sample]:
    remapped = []
    for s in samples:
        new_frame = [remap.get(t, t) if t != "" else "" for t in s.frame_tokens]
        new_canon = [remap.get(t, t) for t in s.canonical_tokens]
        remapped.append(Sample(
            video_path=s.video_path,
            frame_tokens=new_frame,
            canonical_tokens=new_canon,
            frame_weights=s.frame_weights,
            sentence=s.sentence,
        ))
    return remapped


# ---------------------------------------------------------------------------
# Dataset & collation
# ---------------------------------------------------------------------------

class LipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        vocab: Sequence[str],
        frame_size: int = 96,
        frame_stride: int = 1,
        augment: bool = False,
    ):
        self.samples     = list(samples)
        self.frame_size  = frame_size
        self.frame_stride = frame_stride
        self.augment     = augment
        # token → id mapping: blank=0, phonemes=1..V
        self.tok2id = {tok: idx + 1 for idx, tok in enumerate(vocab)}
        self.blank_id = 0

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.transform = v2.Compose([
            v2.Resize((frame_size, frame_size),
                      interpolation=v2.InterpolationMode.BILINEAR,
                      antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
        self._mean = torch.tensor(mean).view(1, 3, 1, 1)
        self._std  = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        frames: List[torch.Tensor] = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
        finally:
            cap.release()
        if not frames:
            raise RuntimeError(f"No frames in {path}")
        return torch.stack(frames)          # (T, C, H, W)  uint8

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        video  = self._load_frames(sample.video_path)       # (T, C, H, W)

        T_vid  = video.shape[0]
        T_tok  = len(sample.frame_tokens)
        T      = min(T_vid, T_tok)

        video        = video[:T]
        frame_tokens = sample.frame_tokens[:T]
        frame_weights = sample.frame_weights[:T]

        if self.frame_stride > 1:
            video         = video[::self.frame_stride]
            frame_tokens  = frame_tokens[::self.frame_stride]
            frame_weights = frame_weights[::self.frame_stride]

        # --- Spatial transforms (applied consistently across all frames) ---
        video = self.transform(video)                        # (T, C, H, W) float32

        if self.augment:
            # Horizontal flip (consistent across frames)
            if random.random() < 0.5:
                video = v2.functional.horizontal_flip(video)

            # Colour jitter (same factors for all frames → no temporal flicker)
            if random.random() < 0.4:
                video = video * self._std + self._mean      # de-normalise
                video = v2.functional.adjust_brightness(video, random.uniform(0.8, 1.2))
                video = v2.functional.adjust_contrast(video,   random.uniform(0.8, 1.2))
                video = v2.functional.adjust_saturation(video, random.uniform(0.9, 1.1))
                video = (video - self._mean) / self._std    # re-normalise

            # Random cutout (same region for all frames)
            if random.random() < 0.25:
                _, _, H, W = video.shape
                ch = max(1, int(H * random.uniform(0.08, 0.20)))
                cw = max(1, int(W * random.uniform(0.08, 0.20)))
                y  = random.randint(0, max(0, H - ch))
                x  = random.randint(0, max(0, W - cw))
                video[:, :, y:y+ch, x:x+cw] = 0.0

        # --- Target ids ---
        frame_ids = [
            self.tok2id.get(tok, self.blank_id) if tok != "" else self.blank_id
            for tok in frame_tokens
        ]

        # --- CTC targets: canonical sequence ---
        ctc_ids = [
            self.tok2id.get(tok, self.blank_id)
            for tok in sample.canonical_tokens
            if tok in self.tok2id
        ]
        if not ctc_ids:
            # Fallback: collapse frame ids
            ctc_ids = [
                fid for fid in frame_ids if fid != self.blank_id
            ]
            if not ctc_ids:
                ctc_ids = [self.blank_id]

        return {
            "video":          video,                                             # (T, C, H, W)
            "video_length":   torch.tensor(video.shape[0], dtype=torch.long),
            "frame_targets":  torch.tensor(frame_ids,     dtype=torch.long),    # (T,)
            "frame_weights":  torch.tensor(frame_weights, dtype=torch.float32), # (T,)
            "ctc_targets":    torch.tensor(ctc_ids,       dtype=torch.long),    # (S,)
            "ctc_length":     torch.tensor(len(ctc_ids),  dtype=torch.long),
            "stem":           sample.video_path.stem,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B           = len(batch)
    vid_lens    = torch.tensor([b["video_length"] for b in batch], dtype=torch.long)
    T_max       = int(vid_lens.max().item())
    C, H, W     = batch[0]["video"].shape[1:]

    videos      = torch.zeros(B, T_max, C, H, W)
    frame_tgts  = torch.zeros(B, T_max, dtype=torch.long)
    frame_wts   = torch.zeros(B, T_max)

    ctc_lens    = torch.tensor([b["ctc_length"] for b in batch], dtype=torch.long)
    ctc_tgts    = torch.cat([b["ctc_targets"] for b in batch])   # (sum_S,)

    stems: List[str] = []
    for i, b in enumerate(batch):
        t = int(b["video_length"])
        videos[i, :t]     = b["video"]
        frame_tgts[i, :t] = b["frame_targets"]
        frame_wts[i, :t]  = b["frame_weights"]
        stems.append(b["stem"])

    return {
        "videos":       videos,
        "video_lengths": vid_lens,
        "frame_targets": frame_tgts,
        "frame_weights": frame_wts,
        "ctc_targets":   ctc_tgts,
        "ctc_lengths":   ctc_lens,
        "stems":         stems,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Conv3DFrontEnd(nn.Module):
    """
    Lightweight 3D convolutional frontend.

    Accepts  : (B, T, C, H, W)
    Produces : (B, T, hidden_dim)

    Uses depthwise-separable 3D convolutions to keep parameter count low.
    Temporal stride = 1 throughout so frame count is preserved for frame-CE.
    Spatial downsampling via stride-2 convolutions; AdaptiveAvgPool collapses
    the remaining spatial dimensions.
    """
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        # Three 3D conv blocks.  Temporal kernel = 3, spatial kernel = 3.
        # Groups = in_channels for depthwise step.
        def _block(in_c: int, out_c: int, spatial_stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                # Depthwise 3D
                nn.Conv3d(in_c, in_c,
                          kernel_size=(3, 3, 3),
                          stride=(1, spatial_stride, spatial_stride),
                          padding=(1, 1, 1),
                          groups=in_c,
                          bias=False),
                # Pointwise
                nn.Conv3d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        self.layer1 = _block(3,   32, spatial_stride=2)   # H/2, W/2
        self.layer2 = _block(32,  64, spatial_stride=2)   # H/4, W/4
        self.layer3 = _block(64, 128, spatial_stride=2)   # H/8, W/8

        self.pool = nn.AdaptiveAvgPool3d((None, 3, 3))     # keep T, collapse spatial to 3×3

        self.projector = nn.Sequential(
            nn.Linear(128 * 9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, C, H, W)  →  (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)                        # (B, 128, T, 3, 3)
        B, C, T, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, C * h * w)
        return self.projector(x)                # (B, T, hidden_dim)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LipReadingModel(nn.Module):
    """
    3D-Conv frontend → positional encoding → BiLSTM encoder → frame classifier.

    forward() returns:
        logits        : (B, T, vocab_size+1)   — for frame-CE loss
        log_probs_ctc : (T, B, vocab_size+1)   — for CTC loss (time-first)
        out_lengths   : (B,)                   — same as input lengths (stride=1)
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        frontend_dropout: float = 0.3,
        encoder: str = "bilstm",            # "bilstm" | "transformer"
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()
        self.frontend = Conv3DFrontEnd(hidden_dim=hidden_dim, dropout=frontend_dropout)
        self.pos_enc  = SinusoidalPositionalEncoding(hidden_dim, dropout=0.1)

        self.encoder_type = encoder
        if encoder == "bilstm":
            self.seq_encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            )
            enc_dim = hidden_dim * 2
        elif encoder == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=lstm_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
            enc_dim = hidden_dim
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        self.norm       = nn.LayerNorm(enc_dim)
        self.dropout    = nn.Dropout(lstm_dropout)
        self.classifier = nn.Linear(enc_dim, vocab_size + 1)   # +1 for blank at index 0

    def forward(
        self,
        videos: torch.Tensor,           # (B, T, C, H, W)
        video_lengths: torch.Tensor,    # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.frontend(videos)            # (B, T, hidden_dim)
        feat = self.pos_enc(feat)

        if self.encoder_type == "bilstm":
            packed = nn.utils.rnn.pack_padded_sequence(
                feat, video_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            enc_out, _ = self.seq_encoder(packed)
            enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)
        else:
            T_max = feat.size(1)
            pad_mask = (
                torch.arange(T_max, device=video_lengths.device).unsqueeze(0)
                >= video_lengths.unsqueeze(1)
            )
            enc_out = self.seq_encoder(feat, src_key_padding_mask=pad_mask)

        enc_out = self.norm(enc_out)
        enc_out = self.dropout(enc_out)
        logits  = self.classifier(enc_out)                 # (B, T, V+1)
        log_probs_ctc = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V+1)

        return logits, log_probs_ctc, video_lengths


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class VisemeAwareFrameLoss(nn.Module):
    """
    Combined frame-CE (with viseme soft targets) + auxiliary CTC loss.

    Frame-CE
    --------
    For each frame:
      - Compute KL(soft_target || model_log_prob)
      - Multiply by per-frame confidence weight from audio alignment
      - Apply an insertion penalty when target=blank but model is confident
        about a phoneme (prevents spurious insertions in silence)

    CTC (auxiliary)
    ---------------
    Standard CTC on canonical phoneme sequences. Weighted by ctc_weight.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        blank_id: int = 0,
        ctc_weight: float = 0.3,
        insertion_penalty: float = 0.5,
    ):
        super().__init__()
        self.blank_id          = blank_id
        self.ctc_weight        = ctc_weight
        self.insertion_penalty = insertion_penalty
        self.ctc_loss          = nn.CTCLoss(blank=blank_id, zero_infinity=True)

        sim  = build_similarity_matrix(vocab)
        soft = similarity_matrix_to_soft_targets(sim)
        self.register_buffer("soft_targets", soft)      # (V+1, V+1)

    def forward(
        self,
        logits: torch.Tensor,           # (B, T, V+1)
        log_probs_ctc: torch.Tensor,    # (T, B, V+1)
        frame_targets: torch.Tensor,    # (B, T)   int
        frame_weights: torch.Tensor,    # (B, T)   float
        video_lengths: torch.Tensor,    # (B,)
        ctc_targets: torch.Tensor,      # (sum_S,) int
        ctc_lengths: torch.Tensor,      # (B,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        B, T_max, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)           # (B, T, V+1)

        # Valid-frame mask
        mask = (
            torch.arange(T_max, device=video_lengths.device).unsqueeze(0)
            < video_lengths.unsqueeze(1)
        )                                                    # (B, T)

        # --- Frame-CE with soft targets ---
        flat_lp  = log_probs[mask]                          # (N, V+1)
        flat_tgt = frame_targets[mask]                      # (N,)
        flat_wt  = frame_weights[mask]                      # (N,)

        tgt_clamped  = flat_tgt.clamp(0, self.soft_targets.shape[0] - 1)
        soft_dist    = self.soft_targets[tgt_clamped]       # (N, V+1)
        frame_ce     = -(soft_dist * flat_lp).sum(dim=-1)   # (N,)

        # Insertion penalty: target=blank, model mass on phonemes
        is_silence   = (flat_tgt == self.blank_id).float()
        blank_prob   = flat_lp[:, self.blank_id].exp()
        ins_penalty  = self.insertion_penalty * is_silence * (1.0 - blank_prob)

        frame_loss   = ((frame_ce + ins_penalty) * flat_wt).mean()

        # --- Auxiliary CTC loss ---
        # CTC needs input_lengths ≤ T_max
        ctc_input_lengths = video_lengths.clamp(max=T_max)
        ctc_l = self.ctc_loss(
            log_probs_ctc,          # (T, B, V+1)
            ctc_targets,            # (sum_S,)
            ctc_input_lengths,      # (B,)
            ctc_lengths,            # (B,)
        )

        total = frame_loss + self.ctc_weight * ctc_l

        return total, {
            "frame_loss": float(frame_loss),
            "ctc_loss":   float(ctc_l) if ctc_l.isfinite() else 0.0,
            "total_loss": float(total),
        }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def greedy_decode(logits: torch.Tensor, length: int, blank_id: int = 0) -> List[str]:
    """CTC-style greedy collapse of frame logits."""
    ids     = torch.argmax(logits[:length], dim=-1).tolist()
    tokens: List[str] = []
    prev    = None
    for i in ids:
        if i == blank_id:
            prev = None
            continue
        if i == prev:
            continue
        tokens.append(i)
        prev = i
    return tokens


def levenshtein(a: Sequence, b: Sequence) -> int:
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        curr = [i]
        for j, bj in enumerate(b, 1):
            curr.append(min(curr[j-1]+1, prev[j]+1, prev[j-1]+(ai != bj)))
        prev = curr
    return prev[-1]


def token_error_rate(preds: List[List[int]], refs: List[List[int]]) -> float:
    errs  = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    total = sum(max(1, len(r)) for r in refs)
    return errs / total


# ---------------------------------------------------------------------------
# Train / validate / test loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: LipReadingModel,
    loader: DataLoader,
    criterion: VisemeAwareFrameLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {"frame_loss": 0.0, "ctc_loss": 0.0, "total_loss": 0.0}
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        videos        = batch["videos"].to(device)
        vid_lens      = batch["video_lengths"].to(device)
        frame_tgts    = batch["frame_targets"].to(device)
        frame_wts     = batch["frame_weights"].to(device)
        ctc_tgts      = batch["ctc_targets"].to(device)
        ctc_lens      = batch["ctc_lengths"].to(device)

        logits, lp_ctc, out_lens = model(videos, vid_lens)
        loss, breakdown = criterion(
            logits, lp_ctc,
            frame_tgts, frame_wts, vid_lens,
            ctc_tgts, ctc_lens,
        )

        if not loss.isfinite():
            logger.warning("Non-finite loss — skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in breakdown.items():
            totals[k] += v
        n += 1

    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: LipReadingModel,
    loader: DataLoader,
    criterion: VisemeAwareFrameLoss,
    device: torch.device,
    id2tok: Dict[int, str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    model.eval()
    totals: Dict[str, float] = {"frame_loss": 0.0, "ctc_loss": 0.0, "total_loss": 0.0}
    n = 0
    all_preds: List[str] = []
    all_refs:  List[str] = []

    for batch in tqdm(loader, desc="eval ", leave=False):
        videos     = batch["videos"].to(device)
        vid_lens   = batch["video_lengths"].to(device)
        frame_tgts = batch["frame_targets"].to(device)
        frame_wts  = batch["frame_weights"].to(device)
        ctc_tgts   = batch["ctc_targets"].to(device)
        ctc_lens   = batch["ctc_lengths"].to(device)

        logits, lp_ctc, _ = model(videos, vid_lens)
        _, breakdown = criterion(
            logits, lp_ctc,
            frame_tgts, frame_wts, vid_lens,
            ctc_tgts, ctc_lens,
        )
        for k, v in breakdown.items():
            totals[k] += v
        n += 1

        # Decode predictions & references per sample
        for i in range(logits.shape[0]):
            L    = int(vid_lens[i].item())
            pred = greedy_decode(logits[i].cpu(), L)
            ref  = [int(x) for x in frame_tgts[i, :L].cpu().tolist() if x != 0]

            # Collapse consecutive identical tokens in reference too
            ref_collapsed: List[int] = []
            prev = None
            for tok in ref:
                if tok != prev:
                    ref_collapsed.append(tok)
                    prev = tok

            pred_str = " ".join(id2tok.get(t, "?") for t in pred)
            ref_str  = " ".join(id2tok.get(t, "?") for t in ref_collapsed)
            all_preds.append(pred_str)
            all_refs.append(ref_str)

    avg = {k: v / max(1, n) for k, v in totals.items()}

    # Compute TER
    pred_seqs = [p.split() for p in all_preds]
    ref_seqs  = [r.split() for r in all_refs]
    errs  = sum(levenshtein(p, r) for p, r in zip(pred_seqs, ref_seqs))
    total = sum(max(1, len(r)) for r in ref_seqs)
    avg["ter"] = errs / total

    return avg, all_preds, all_refs


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def phoneme_report(
    all_preds: List[str],
    all_refs:  List[str],
    vocab: Sequence[str],
    output_dir: Path,
) -> None:
    """Save per-phoneme precision/recall CSV and confusion matrix plot."""
    label_list  = list(vocab)
    l2i         = {l: i for i, l in enumerate(label_list)}
    n           = len(label_list)
    matrix      = np.zeros((n + 1, n + 1), dtype=np.int64)
    stats       = {l: {"support": 0, "correct": 0, "predicted": 0} for l in label_list}

    def _align(ref: List[str], hyp: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
        R, H = len(ref), len(hyp)
        dp = [[0]*(H+1) for _ in range(R+1)]
        bp: List[List[str]] = [[""]*(H+1) for _ in range(R+1)]
        for r in range(1, R+1): dp[r][0] = r; bp[r][0] = "D"
        for h in range(1, H+1): dp[0][h] = h; bp[0][h] = "I"
        for r in range(1, R+1):
            for h in range(1, H+1):
                sc = dp[r-1][h-1] + int(ref[r-1] != hyp[h-1])
                dc = dp[r-1][h] + 1
                ic = dp[r][h-1] + 1
                best, op = min((sc,"S"),(dc,"D"),(ic,"I"), key=lambda x: x[0])
                dp[r][h] = best; bp[r][h] = op
        pairs: List[Tuple[Optional[str], Optional[str]]] = []
        r, h = R, H
        while r > 0 or h > 0:
            op = bp[r][h]
            if op == "S": pairs.append((ref[r-1], hyp[h-1])); r -= 1; h -= 1
            elif op == "D": pairs.append((ref[r-1], None)); r -= 1
            else: pairs.append((None, hyp[h-1])); h -= 1
        pairs.reverse()
        return pairs

    for ref_str, pred_str in zip(all_refs, all_preds):
        ref  = ref_str.split()  if ref_str  else []
        pred = pred_str.split() if pred_str else []
        for a, p in _align(ref, pred):
            ai = l2i.get(a) if a else None
            pi = l2i.get(p) if p else None
            if a is not None and a in stats:
                stats[a]["support"] += 1
            if a is not None and p is not None and a in stats:
                stats[p]["predicted"] = stats[p].get("predicted", 0) + 1
                ri = ai if ai is not None else n
                ci = pi if pi is not None else n
                matrix[ri, ci] += 1
                if a == p:
                    stats[a]["correct"] += 1
            elif a is not None and p is None:
                matrix[ai if ai is not None else n, n] += 1
            elif a is None and p is not None:
                matrix[n, pi if pi is not None else n] += 1

    rows = []
    for l in label_list:
        s   = stats[l]["support"]
        c   = stats[l]["correct"]
        pr  = stats[l].get("predicted", 0)
        rows.append({
            "phoneme":   l,
            "support":   s,
            "correct":   c,
            "recall":    round(c / s, 4) if s else 0.0,
            "precision": round(c / pr, 4) if pr else 0.0,
        })
    rows.sort(key=lambda r: r["recall"])

    csv_path = output_dir / "phoneme_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["phoneme","support","correct","recall","precision"])
        w.writeheader(); w.writerows(rows)

    # Confusion matrix plot
    size = max(8.0, 0.4 * (n + 1))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = label_list + [MISSING_TOKEN]
    ax.set_xticks(range(n+1)); ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n+1)); ax.set_yticklabels(label_list + [INSERTION_TOKEN], fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Phoneme confusion matrix")
    if (n+1)**2 <= 400:
        for r in range(n+1):
            for c in range(n+1):
                v = int(matrix[r, c])
                if v:
                    ax.text(c, r, str(v), ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Phoneme report → %s", csv_path)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def find_videos(video_dir: Path) -> Dict[str, Path]:
    return {
        p.stem: p
        for p in sorted(video_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    }


def build_samples(video_dir: Path, phoneme_csv: Path) -> List[Sample]:
    records   = load_phoneme_csv(phoneme_csv)
    video_map = find_videos(video_dir)

    # Allow stems like "bbaf2n_lipcrop" → "bbaf2n"
    normalised: Dict[str, Path] = {}
    for stem, path in video_map.items():
        key = stem[: -len("_lipcrop")] if stem.endswith("_lipcrop") else stem
        normalised[key] = path

    samples: List[Sample] = []
    missing_video = 0
    for stem, rec in records.items():
        path = normalised.get(stem)
        if path is None:
            missing_video += 1
            continue
        samples.append(Sample(
            video_path       = path,
            frame_tokens     = rec.frame_tokens,
            canonical_tokens = rec.canonical_tokens,
            frame_weights    = rec.frame_weights,
            sentence         = rec.sentence,
        ))

    if missing_video:
        logger.warning("%d records had no matching video file", missing_video)
    if not samples:
        raise FileNotFoundError(
            f"No paired samples found.\n"
            f"  video_dir  : {video_dir}  ({len(video_map)} videos)\n"
            f"  phoneme CSV: {phoneme_csv} ({len(records)} records)"
        )
    logger.info("Paired %d samples (video + phoneme labels)", len(samples))
    return samples


def split_samples(
    samples: List[Sample],
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    rng = random.Random(seed)
    s   = samples.copy()
    rng.shuffle(s)
    n       = len(s)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return s[:n_train], s[n_train:n_train+n_val], s[n_train+n_val:]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train lip-reading model on GRID corpus with viseme-aware loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--video-dir",   default="s1_lip_crops")
    p.add_argument("--phoneme-csv", default="phonemes_s1_aligned/phoneme_predictions.csv")
    p.add_argument("--output-dir",  default="viseme_results")
    p.add_argument("--fps",         type=float, default=25.0,
                   help="Video frame rate (used for spans_json confidence mapping)")
    # Training
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--min-phoneme-count", type=int, default=3,
                   help="Merge phonemes below this count into nearest viseme neighbour")
    # Model
    p.add_argument("--hidden-dim",   type=int,   default=256)
    p.add_argument("--frame-size",   type=int,   default=96,
                   help="Spatial resolution fed to the 3D conv frontend")
    p.add_argument("--frame-stride", type=int,   default=1)
    p.add_argument("--encoder",      choices=["bilstm", "transformer"], default="bilstm")
    p.add_argument("--lstm-layers",  type=int,   default=2)
    p.add_argument("--lstm-dropout", type=float, default=0.3)
    p.add_argument("--transformer-heads",  type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=2)
    # Loss
    p.add_argument("--ctc-weight",        type=float, default=0.3,
                   help="Weight of auxiliary CTC loss (0 to disable)")
    p.add_argument("--insertion-penalty", type=float, default=0.5,
                   help="Extra penalty for predicting phonemes during silence frames")
    # LR schedule
    p.add_argument("--warmup-epochs",  type=int,   default=5)
    p.add_argument("--lr-patience",    type=int,   default=6)
    p.add_argument("--lr-factor",      type=float, default=0.5)
    p.add_argument("--min-lr",         type=float, default=1e-6)
    p.add_argument("--early-stop",     type=int,   default=12,
                   help="Stop after this many epochs without val TER improvement")
    # Misc
    p.add_argument("--class-aware-sampling", action="store_true",
                   help="Upsample rare-phoneme clips during training")
    p.add_argument("--dry-run", action="store_true",
                   help="Run one batch forward pass and exit")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = make_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    all_samples = build_samples(Path(args.video_dir), Path(args.phoneme_csv))
    train_s, val_s, test_s = split_samples(
        all_samples, train_ratio=0.7, val_ratio=0.15, seed=args.seed
    )

    # Vocabulary from training split only
    vocab, remap = build_vocab(train_s, min_count=args.min_phoneme_count)
    train_s = apply_vocab_remap(train_s, remap)
    val_s   = apply_vocab_remap(val_s,   remap)
    test_s  = apply_vocab_remap(test_s,  remap)

    id2tok = {0: BLANK_TOKEN}
    id2tok.update({i+1: tok for i, tok in enumerate(vocab)})

    vocab_info = {"blank_id": 0, "tokens": vocab, "remap": remap}
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as fh:
        json.dump(vocab_info, fh, ensure_ascii=False, indent=2)

    logger.info(
        "Split — train:%d  val:%d  test:%d  |  vocab size: %d",
        len(train_s), len(val_s), len(test_s), len(vocab),
    )

    # Datasets
    train_ds = LipDataset(train_s, vocab, frame_size=args.frame_size,
                          frame_stride=args.frame_stride, augment=True)
    val_ds   = LipDataset(val_s,   vocab, frame_size=args.frame_size,
                          frame_stride=args.frame_stride, augment=False)
    test_ds  = LipDataset(test_s,  vocab, frame_size=args.frame_size,
                          frame_stride=args.frame_stride, augment=False)

    # Optional class-aware sampler
    if args.class_aware_sampling:
        ph_counts: Counter[str] = Counter()
        for s in train_s:
            ph_counts.update(t for t in s.frame_tokens if t != "")
        inv = {t: 1.0 / max(1, c) for t, c in ph_counts.items()}
        weights = [
            float(np.mean([inv.get(t, 1.0) for t in s.frame_tokens if t != ""])
                  if any(t != "" for t in s.frame_tokens) else 1.0)
            for s in train_s
        ]
        sampler    = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers,
                                  collate_fn=collate_fn, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn,
                                  pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # Model, criterion, optimiser
    # ------------------------------------------------------------------
    model = LipReadingModel(
        vocab_size       = len(vocab),
        hidden_dim       = args.hidden_dim,
        lstm_layers      = args.lstm_layers,
        lstm_dropout     = args.lstm_dropout,
        frontend_dropout = args.lstm_dropout,
        encoder          = args.encoder,
        transformer_heads  = args.transformer_heads,
        transformer_layers = args.transformer_layers,
    ).to(device)

    criterion = VisemeAwareFrameLoss(
        vocab             = vocab,
        blank_id          = 0,
        ctc_weight        = args.ctc_weight,
        insertion_penalty = args.insertion_penalty,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    if args.dry_run:
        batch = next(iter(train_loader))
        videos   = batch["videos"].to(device)
        vid_lens = batch["video_lengths"].to(device)
        logits, lp_ctc, _ = model(videos, vid_lens)
        loss, bd = criterion(
            logits, lp_ctc,
            batch["frame_targets"].to(device),
            batch["frame_weights"].to(device),
            vid_lens,
            batch["ctc_targets"].to(device),
            batch["ctc_lengths"].to(device),
        )
        logger.info("Dry run — videos: %s  logits: %s", tuple(videos.shape), tuple(logits.shape))
        logger.info("Dry run — loss breakdown: %s", bd)
        return

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_ter   = float("inf")
    no_improve = 0
    history: Dict[str, List] = {
        "train_total": [], "train_frame": [], "train_ctc": [],
        "val_total":   [], "val_ter":   [], "lr": [],
    }

    for epoch in range(1, args.epochs + 1):
        # Linear LR warmup
        if epoch <= args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * epoch / args.warmup_epochs

        train_m = train_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_m, _, _ = evaluate(model, val_loader, criterion, device, id2tok)

        # Scheduler step (after warmup)
        if epoch > args.warmup_epochs:
            scheduler.step(val_m["ter"])

        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_total"].append(train_m["total_loss"])
        history["train_frame"].append(train_m["frame_loss"])
        history["train_ctc"].append(train_m["ctc_loss"])
        history["val_total"].append(val_m["total_loss"])
        history["val_ter"].append(val_m["ter"])
        history["lr"].append(cur_lr)

        logger.info(
            "Epoch %03d/%03d | "
            "train loss=%.4f (frame=%.4f ctc=%.4f) | "
            "val loss=%.4f TER=%.4f | lr=%.2e",
            epoch, args.epochs,
            train_m["total_loss"], train_m["frame_loss"], train_m["ctc_loss"],
            val_m["total_loss"], val_m["ter"], cur_lr,
        )

        if val_m["ter"] < best_ter - 1e-4:
            best_ter   = val_m["ter"]
            no_improve = 0
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": epoch, "val_ter": best_ter,
                 "vocab": vocab, "config": vars(args)},
                output_dir / "best_model.pt",
            )
            logger.info("  ✓ Saved best model (val TER %.4f)", best_ter)
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

    torch.save(
        {"model_state_dict": model.state_dict(), "vocab": vocab, "config": vars(args)},
        output_dir / "final_model.pt",
    )
    with (output_dir / "history.json").open("w") as fh:
        json.dump(history, fh, indent=2)

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    # Load best checkpoint
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_m, test_preds, test_refs = evaluate(model, test_loader, criterion, device, id2tok)
    logger.info(
        "Test — loss=%.4f  TER=%.4f",
        test_m["total_loss"], test_m["ter"],
    )

    phoneme_report(test_preds, test_refs, vocab, output_dir)

    with (output_dir / "test_metrics.json").open("w") as fh:
        json.dump(test_m, fh, indent=2)

    # Save raw predictions
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["reference", "prediction"])
        w.writeheader()
        w.writerows({"reference": r, "prediction": p}
                    for r, p in zip(test_refs, test_preds))

    logger.info("Done.  Best val TER: %.4f  |  Test TER: %.4f", best_ter, test_m["ter"])


if __name__ == "__main__":
    main()