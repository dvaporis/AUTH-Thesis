#!/usr/bin/env python3
"""Ablation study for lip-reading models based on train_lip_viseme.py.

Six model variants:
  1. conv3d + bi-lstm + full loss (baseline)
  2. conv3d + transformer + full loss
  3. conv3d + projection head only (no sequence encoder) + full loss
  4. conv3d + bi-lstm + CTC loss only
  5. conv3d + bi-lstm + frame-CE loss only (no CTC)
  6. conv3d + bi-lstm + hard cross-entropy loss (no phoneme similarity, all mistakes equal)

Outputs per run: history.json, test_metrics.json, phoneme_report.csv
A final ablation_summary.json aggregates results across all runs.

Usage:
    python ablation_experiment.py --video-dir s1_lip_crops \
                             --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
                             --output-dir ablation_results \
                             --epochs 40
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from collections import Counter
from copy import deepcopy
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv"}
BLANK_TOKEN  = "<blank>"
MISSING_TOKEN    = "<missed>"
INSERTION_TOKEN  = "<inserted>"

VISEME_GROUPS: Dict[str, List[str]] = {
    "bilabial":     ["p", "b", "m"],
    "labiodental":  ["f", "v"],
    "dental":       ["θ", "ð"],
    "alveolar":     ["t", "d", "n", "s", "z", "l"],
    "postalveolar": ["ʃ", "ʒ", "tʃ", "dʒ"],
    "velar":        ["k", "g", "ŋ"],
    "glottal":      ["h"],
    "approximant":  ["w", "j", "ɹ", "r"],
    "close_front":  ["iː", "i", "ɪ"],
    "close_back":   ["uː", "u", "ʊ"],
    "mid_front":    ["e", "eɪ", "ɛ"],
    "mid_central":  ["ə", "ɐ", "ɜ", "ɚ", "ʌ"],
    "mid_back":     ["ɔ", "ɔɪ", "oʊ", "o"],
    "open_front":   ["æ", "a", "aɪ", "aʊ"],
    "open_back":    ["ɑ", "ɒ"],
}

ARTICULATORY_FEATURES: Dict[str, Tuple[int, int, int]] = {
    "p": (0,0,0),"b": (0,0,1),"m": (0,3,1),
    "f": (1,1,0),"v": (1,1,1),
    "θ": (2,1,0),"ð": (2,1,1),
    "t": (3,0,0),"d": (3,0,1),"n": (3,3,1),
    "s": (3,1,0),"z": (3,1,1),"l": (3,5,1),
    "ʃ": (4,1,0),"ʒ": (4,1,1),"tʃ": (4,2,0),"dʒ": (4,2,1),
    "k": (6,0,0),"g": (6,0,1),"ŋ": (6,3,1),
    "h": (7,1,0),
    "w": (0,4,1),"j": (5,4,1),"ɹ": (3,4,1),"r": (3,4,1),
    **{k: (8,6,1) for k in [
        "iː","i","ɪ","uː","u","ʊ","e","eɪ","ɛ","ə","ɐ","ɜ","ɚ","ʌ",
        "ɔ","ɔɪ","oʊ","o","æ","a","aɪ","aʊ","ɑ","ɒ",
    ]},
}

def _build_viseme_map() -> Dict[str, int]:
    m: Dict[str, int] = {}
    for idx, phonemes in enumerate(VISEME_GROUPS.values()):
        for ph in phonemes:
            m[ph] = idx
    return m

PHONEME_TO_VISEME = _build_viseme_map()


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _articulatory_similarity(a: str, b: str) -> float:
    fa = ARTICULATORY_FEATURES.get(a)
    fb = ARTICULATORY_FEATURES.get(b)
    if fa is None or fb is None:
        return 0.0
    return 1.0 - sum(int(x != y) for x, y in zip(fa, fb)) / 3.0


def build_similarity_matrix(vocab: Sequence[str]) -> torch.Tensor:
    labels = [BLANK_TOKEN] + list(vocab)
    n = len(labels)
    mat = torch.zeros(n, n, dtype=torch.float32)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                mat[i, j] = 1.0
            elif a == BLANK_TOKEN or b == BLANK_TOKEN:
                mat[i, j] = 0.0
            else:
                va = PHONEME_TO_VISEME.get(a, -1)
                vb = PHONEME_TO_VISEME.get(b, -1)
                if va != -1 and va == vb:
                    mat[i, j] = 0.95
                else:
                    mat[i, j] = _articulatory_similarity(a, b) * 0.4
    return mat


def similarity_to_soft_targets(sim: torch.Tensor) -> torch.Tensor:
    n = sim.shape[0]
    dist = torch.zeros_like(sim)
    for i in range(n):
        if i == 0:
            dist[i, 0] = 1.0
            continue
        nb = sim[i].clone()
        nb[i] = 0.0
        nb[0] = 0.0
        total = nb.sum()
        dist[i, i] = 0.85
        if total > 0:
            dist[i] += 0.15 * (nb / total)
    return dist


def hard_targets(n_vocab_with_blank: int) -> torch.Tensor:
    """Identity matrix — each label maps only to itself (no phoneme similarity)."""
    return torch.eye(n_vocab_with_blank, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    video_path: Path
    frame_tokens: List[str]
    canonical_tokens: List[str]
    frame_weights: List[float]
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

def _parse_frame_weights(spans_str: str, num_frames: int, fps: float = 25.0) -> List[float]:
    weights = [1.0] * num_frames
    if not spans_str.strip():
        return weights
    try:
        spans = json.loads(spans_str)
    except json.JSONDecodeError:
        return weights
    for span in spans:
        score = float(span.get("score", 0.0))
        clamped = max(-10.0, min(0.0, score))
        confidence = 0.5 + (1.0 + clamped / 10.0)
        sf = int(span.get("start_sec", 0.0) * fps)
        ef = int(span.get("end_sec", 0.0) * fps)
        for f in range(sf, min(ef + 1, num_frames)):
            weights[f] = confidence
    return weights


def load_phoneme_csv(csv_path: Path, fps: float = 25.0) -> Dict[str, AlignedRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    records: Dict[str, AlignedRecord] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stem = row.get("stem", "").strip()
            raw  = row.get("per_frame_labels", "").strip()
            if not stem or not raw:
                continue
            try:
                raw_labels: List[Optional[str]] = json.loads(raw)
            except json.JSONDecodeError:
                continue
            frame_tokens = [(lbl.strip() if isinstance(lbl, str) else "") for lbl in raw_labels]
            if not frame_tokens:
                continue
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
            nf_str = row.get("num_frames", "").strip()
            num_frames = int(nf_str) if nf_str.isdigit() else len(frame_tokens)
            weights = _parse_frame_weights(row.get("spans_json", ""), len(frame_tokens), fps)
            records[stem] = AlignedRecord(
                frame_tokens=frame_tokens,
                canonical_tokens=canonical,
                frame_weights=weights,
                sentence=row.get("sentence", "").strip(),
                num_frames=num_frames,
            )
    if not records:
        raise ValueError(f"No valid records in {csv_path}")
    return records


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(samples: Sequence[Sample], min_count: int = 2) -> Tuple[List[str], Dict[str, str]]:
    counter: Counter[str] = Counter()
    for s in samples:
        counter.update(t for t in s.frame_tokens if t != "")
    kept = {tok for tok, cnt in counter.items() if cnt >= min_count}
    remap: Dict[str, str] = {}
    for tok, cnt in counter.items():
        if tok in kept:
            remap[tok] = tok
            continue
        my_v = PHONEME_TO_VISEME.get(tok, -1)
        best: Optional[str] = None
        best_c = -1
        for c in kept:
            if PHONEME_TO_VISEME.get(c, -2) == my_v and counter[c] > best_c:
                best, best_c = c, counter[c]
        remap[tok] = best if best else counter.most_common(1)[0][0]
    return sorted(kept), remap


def apply_remap(samples: List[Sample], remap: Dict[str, str]) -> List[Sample]:
    out = []
    for s in samples:
        out.append(Sample(
            video_path=s.video_path,
            frame_tokens=[remap.get(t, t) if t != "" else "" for t in s.frame_tokens],
            canonical_tokens=[remap.get(t, t) for t in s.canonical_tokens],
            frame_weights=s.frame_weights,
            sentence=s.sentence,
        ))
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LipDataset(torch.utils.data.Dataset):
    def __init__(self, samples, vocab, frame_size=96, frame_stride=1, augment=False):
        self.samples = list(samples)
        self.frame_stride = frame_stride
        self.augment = augment
        self.tok2id = {tok: idx + 1 for idx, tok in enumerate(vocab)}
        self.blank_id = 0
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.transform = v2.Compose([
            v2.Resize((frame_size, frame_size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
        self._mean = torch.tensor(mean).view(1, 3, 1, 1)
        self._std  = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def _load(self, path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        frames = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok: break
                frames.append(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2,0,1))
        finally:
            cap.release()
        if not frames:
            raise RuntimeError(f"No frames: {path}")
        return torch.stack(frames)

    def __getitem__(self, idx):
        s = self.samples[idx]
        video = self._load(s.video_path)
        T = min(video.shape[0], len(s.frame_tokens))
        video = video[:T]
        ft    = s.frame_tokens[:T]
        fw    = s.frame_weights[:T]
        if self.frame_stride > 1:
            video = video[::self.frame_stride]
            ft    = ft[::self.frame_stride]
            fw    = fw[::self.frame_stride]
        video = self.transform(video)
        if self.augment:
            if random.random() < 0.5:
                video = v2.functional.horizontal_flip(video)
            if random.random() < 0.4:
                video = video * self._std + self._mean
                video = v2.functional.adjust_brightness(video, random.uniform(0.8, 1.2))
                video = v2.functional.adjust_contrast(video,   random.uniform(0.8, 1.2))
                video = v2.functional.adjust_saturation(video, random.uniform(0.9, 1.1))
                video = (video - self._mean) / self._std
            if random.random() < 0.25:
                _, _, H, W = video.shape
                ch = max(1, int(H * random.uniform(0.08, 0.20)))
                cw = max(1, int(W * random.uniform(0.08, 0.20)))
                y  = random.randint(0, max(0, H - ch))
                x  = random.randint(0, max(0, W - cw))
                video[:, :, y:y+ch, x:x+cw] = 0.0
        frame_ids = [self.tok2id.get(t, 0) if t != "" else 0 for t in ft]
        ctc_ids = [self.tok2id.get(t, 0) for t in s.canonical_tokens if t in self.tok2id] or [0]
        return {
            "video":         video,
            "video_length":  torch.tensor(video.shape[0], dtype=torch.long),
            "frame_targets": torch.tensor(frame_ids, dtype=torch.long),
            "frame_weights": torch.tensor(fw, dtype=torch.float32),
            "ctc_targets":   torch.tensor(ctc_ids, dtype=torch.long),
            "ctc_length":    torch.tensor(len(ctc_ids), dtype=torch.long),
            "stem":          s.video_path.stem,
        }


def collate_fn(batch):
    B = len(batch)
    lens = torch.tensor([b["video_length"] for b in batch], dtype=torch.long)
    T = int(lens.max())
    C, H, W = batch[0]["video"].shape[1:]
    videos = torch.zeros(B, T, C, H, W)
    ftgts  = torch.zeros(B, T, dtype=torch.long)
    fwts   = torch.zeros(B, T)
    ctc_lens = torch.tensor([b["ctc_length"] for b in batch], dtype=torch.long)
    ctc_tgts = torch.cat([b["ctc_targets"] for b in batch])
    stems = []
    for i, b in enumerate(batch):
        t = int(b["video_length"])
        videos[i, :t] = b["video"]
        ftgts[i, :t]  = b["frame_targets"]
        fwts[i, :t]   = b["frame_weights"]
        stems.append(b["stem"])
    return {
        "videos": videos, "video_lengths": lens,
        "frame_targets": ftgts, "frame_weights": fwts,
        "ctc_targets": ctc_tgts, "ctc_lengths": ctc_lens,
        "stems": stems,
    }


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class Conv3DFrontEnd(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        def _block(ic, oc, ss=1):
            return nn.Sequential(
                nn.Conv3d(ic, ic, (3,3,3), stride=(1,ss,ss), padding=(1,1,1), groups=ic, bias=False),
                nn.Conv3d(ic, oc, 1, bias=False),
                nn.BatchNorm3d(oc),
                nn.ReLU(inplace=True),
            )
        self.layer1 = _block(3,  32, 2)
        self.layer2 = _block(32, 64, 2)
        self.layer3 = _block(64, 128, 2)
        self.pool   = nn.AdaptiveAvgPool3d((None, 3, 3))
        self.projector = nn.Sequential(
            nn.Linear(128*9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x)
        B, C, T, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, C*h*w)
        return self.projector(x)


class SinPE(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class LipModel(nn.Module):
    """
    Flexible lip-reading model supporting 3 encoder variants:
      - 'bilstm'      : BiLSTM sequence encoder
      - 'transformer' : Transformer encoder
      - 'none'        : No sequence encoder (projection head only)
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        encoder: str = "bilstm",
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        frontend_dropout: float = 0.3,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()
        self.encoder_type = encoder
        self.frontend = Conv3DFrontEnd(hidden_dim=hidden_dim, dropout=frontend_dropout)
        self.pos_enc  = SinPE(hidden_dim, dropout=0.1)

        if encoder == "bilstm":
            self.seq_encoder = nn.LSTM(
                hidden_dim, hidden_dim, num_layers=lstm_layers,
                batch_first=True, bidirectional=True,
                dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            )
            enc_dim = hidden_dim * 2
        elif encoder == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=transformer_heads,
                dim_feedforward=hidden_dim*4, dropout=lstm_dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.seq_encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
            enc_dim = hidden_dim
        elif encoder == "none":
            self.seq_encoder = None
            enc_dim = hidden_dim
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        self.norm       = nn.LayerNorm(enc_dim)
        self.dropout    = nn.Dropout(lstm_dropout)
        self.classifier = nn.Linear(enc_dim, vocab_size + 1)

    def forward(self, videos, video_lengths):
        feat = self.frontend(videos)
        feat = self.pos_enc(feat)

        if self.encoder_type == "bilstm":
            packed = nn.utils.rnn.pack_padded_sequence(
                feat, video_lengths.cpu(), batch_first=True, enforce_sorted=False)
            enc, _ = self.seq_encoder(packed)
            enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True)
        elif self.encoder_type == "transformer":
            T = feat.size(1)
            pad_mask = (
                torch.arange(T, device=video_lengths.device).unsqueeze(0)
                >= video_lengths.unsqueeze(1)
            )
            enc = self.seq_encoder(feat, src_key_padding_mask=pad_mask)
        else:
            enc = feat

        enc    = self.norm(enc)
        enc    = self.dropout(enc)
        logits = self.classifier(enc)
        log_probs_ctc = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return logits, log_probs_ctc, video_lengths


# ---------------------------------------------------------------------------
# Loss variants
# ---------------------------------------------------------------------------

class AblationLoss(nn.Module):
    """
    Configurable loss supporting:
      - soft_targets : viseme-aware soft labels vs hard identity labels
      - use_ctc      : include auxiliary CTC term
      - use_frame_ce : include frame-level CE term
      - insertion_penalty : penalise phoneme prediction on silence frames
    """
    def __init__(
        self,
        vocab: Sequence[str],
        blank_id: int = 0,
        ctc_weight: float = 0.3,
        insertion_penalty: float = 0.5,
        use_soft_targets: bool = True,
        use_ctc: bool = True,
        use_frame_ce: bool = True,
    ):
        super().__init__()
        self.blank_id         = blank_id
        self.ctc_weight       = ctc_weight
        self.ins_penalty      = insertion_penalty
        self.use_ctc          = use_ctc
        self.use_frame_ce     = use_frame_ce
        self.ctc_loss_fn      = nn.CTCLoss(blank=blank_id, zero_infinity=True)

        n = len(vocab) + 1
        if use_soft_targets:
            sim  = build_similarity_matrix(vocab)
            soft = similarity_to_soft_targets(sim)
        else:
            soft = hard_targets(n)
        self.register_buffer("soft_targets", soft)

    def forward(self, logits, log_probs_ctc, frame_targets, frame_weights,
                video_lengths, ctc_targets, ctc_lengths):
        B, T_max, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        mask = (
            torch.arange(T_max, device=video_lengths.device).unsqueeze(0)
            < video_lengths.unsqueeze(1)
        )
        breakdown: Dict[str, float] = {}
        total = torch.tensor(0.0, device=logits.device)

        if self.use_frame_ce:
            flat_lp  = log_probs[mask]
            flat_tgt = frame_targets[mask]
            flat_wt  = frame_weights[mask]
            clamped  = flat_tgt.clamp(0, self.soft_targets.shape[0] - 1)
            soft     = self.soft_targets[clamped]
            ce       = -(soft * flat_lp).sum(dim=-1)
            is_sil   = (flat_tgt == self.blank_id).float()
            bp       = flat_lp[:, self.blank_id].exp()
            ins      = self.ins_penalty * is_sil * (1.0 - bp)
            frame_loss = ((ce + ins) * flat_wt).mean()
            breakdown["frame_loss"] = float(frame_loss)
            total = total + frame_loss

        if self.use_ctc:
            ctc_in_lens = video_lengths.clamp(max=T_max)
            ctc_l = self.ctc_loss_fn(log_probs_ctc, ctc_targets, ctc_in_lens, ctc_lengths)
            breakdown["ctc_loss"] = float(ctc_l) if ctc_l.isfinite() else 0.0
            total = total + self.ctc_weight * ctc_l

        if not breakdown:
            raise ValueError("At least one of use_frame_ce or use_ctc must be True")

        breakdown["total_loss"] = float(total)
        return total, breakdown


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


def greedy_decode(logits: torch.Tensor, length: int, blank_id: int = 0) -> List[int]:
    ids = torch.argmax(logits[:length], dim=-1).tolist()
    tokens, prev = [], None
    for i in ids:
        if i == blank_id: prev = None; continue
        if i == prev: continue
        tokens.append(i); prev = i
    return tokens


def frame_accuracy(logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> float:
    """Per-frame accuracy (including blank frames)."""
    preds = torch.argmax(logits, dim=-1)
    T = logits.shape[1]
    mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    correct = ((preds == targets) & mask).sum().item()
    total   = mask.sum().item()
    return correct / max(1, total)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    tots: Dict[str, float] = {}
    n_acc, n_batches = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        videos   = batch["videos"].to(device)
        vlens    = batch["video_lengths"].to(device)
        ftgts    = batch["frame_targets"].to(device)
        fwts     = batch["frame_weights"].to(device)
        ctc_tgts = batch["ctc_targets"].to(device)
        ctc_lens = batch["ctc_lengths"].to(device)
        logits, lp_ctc, _ = model(videos, vlens)
        loss, bd = criterion(logits, lp_ctc, ftgts, fwts, vlens, ctc_tgts, ctc_lens)
        if not loss.isfinite():
            continue
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for k, v in bd.items():
            tots[k] = tots.get(k, 0.0) + v
        n_acc += frame_accuracy(logits.detach(), ftgts, vlens)
        n_batches += 1
    avg = {k: v / max(1, n_batches) for k, v in tots.items()}
    avg["frame_acc"] = n_acc / max(1, n_batches)
    return avg


@torch.no_grad()
def eval_loop(model, loader, criterion, device, id2tok):
    model.eval()
    tots: Dict[str, float] = {}
    n_acc, n_batches = 0.0, 0
    all_preds, all_refs = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        videos   = batch["videos"].to(device)
        vlens    = batch["video_lengths"].to(device)
        ftgts    = batch["frame_targets"].to(device)
        fwts     = batch["frame_weights"].to(device)
        ctc_tgts = batch["ctc_targets"].to(device)
        ctc_lens = batch["ctc_lengths"].to(device)
        logits, lp_ctc, _ = model(videos, vlens)
        _, bd = criterion(logits, lp_ctc, ftgts, fwts, vlens, ctc_tgts, ctc_lens)
        for k, v in bd.items():
            tots[k] = tots.get(k, 0.0) + v
        n_acc += frame_accuracy(logits, ftgts, vlens)
        n_batches += 1
        for i in range(logits.shape[0]):
            L    = int(vlens[i].item())
            pred = greedy_decode(logits[i].cpu(), L)
            ref  = [int(x) for x in ftgts[i, :L].cpu().tolist() if x != 0]
            rc: List[int] = []
            prev = None
            for t in ref:
                if t != prev: rc.append(t); prev = t
            all_preds.append(" ".join(id2tok.get(t, "?") for t in pred))
            all_refs.append(" ".join(id2tok.get(t, "?") for t in rc))
    avg = {k: v / max(1, n_batches) for k, v in tots.items()}
    avg["frame_acc"] = n_acc / max(1, n_batches)
    # TER
    errs  = sum(levenshtein(p.split(), r.split()) for p, r in zip(all_preds, all_refs))
    total = sum(max(1, len(r.split())) for r in all_refs)
    avg["ter"] = errs / total
    return avg, all_preds, all_refs


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_videos(video_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(video_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            k = p.stem
            if k.endswith("_lipcrop"):
                k = k[:-len("_lipcrop")]
            out[k] = p
    return out


def build_all_samples(video_dir: Path, phoneme_csv: Path) -> List[Sample]:
    records   = load_phoneme_csv(phoneme_csv)
    video_map = find_videos(video_dir)
    samples   = []
    for stem, rec in records.items():
        p = video_map.get(stem)
        if p is None:
            continue
        samples.append(Sample(
            video_path=p,
            frame_tokens=rec.frame_tokens,
            canonical_tokens=rec.canonical_tokens,
            frame_weights=rec.frame_weights,
            sentence=rec.sentence,
        ))
    if not samples:
        raise FileNotFoundError(f"No paired samples in {video_dir} / {phoneme_csv}")
    return samples


def split_samples(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    s   = samples.copy()
    rng.shuffle(s)
    n       = len(s)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return s[:n_train], s[n_train:n_train+n_val], s[n_train+n_val:]


# ---------------------------------------------------------------------------
# Ablation configs
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = [
    {
        "name": "1_bilstm_full_loss",
        "label": "BiLSTM + Full Loss",
        "encoder": "bilstm",
        "use_soft_targets": True,
        "use_ctc": True,
        "use_frame_ce": True,
    },
    {
        "name": "2_transformer_full_loss",
        "label": "Transformer + Full Loss",
        "encoder": "transformer",
        "use_soft_targets": True,
        "use_ctc": True,
        "use_frame_ce": True,
    },
    {
        "name": "3_no_seq_encoder",
        "label": "No Seq. Encoder (Proj. Only)",
        "encoder": "none",
        "use_soft_targets": True,
        "use_ctc": True,
        "use_frame_ce": True,
    },
    {
        "name": "4_bilstm_ctc_only",
        "label": "BiLSTM + CTC Only",
        "encoder": "bilstm",
        "use_soft_targets": False,   # CTC only has no concept of soft targets
        "use_ctc": True,
        "use_frame_ce": False,
    },
    {
        "name": "5_bilstm_framece_only",
        "label": "BiLSTM + Frame-CE Only",
        "encoder": "bilstm",
        "use_soft_targets": True,
        "use_ctc": False,
        "use_frame_ce": True,
    },
    {
        "name": "6_bilstm_hard_targets",
        "label": "BiLSTM + Hard Targets (No Similarity)",
        "encoder": "bilstm",
        "use_soft_targets": False,
        "use_ctc": True,
        "use_frame_ce": True,
    },
]


# ---------------------------------------------------------------------------
# Single-run trainer
# ---------------------------------------------------------------------------

def run_variant(variant: Dict[str, Any], args, train_s, val_s, test_s, vocab, id2tok, device) -> Dict[str, Any]:
    run_dir = Path(args.output_dir) / variant["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Running variant: %s", variant["label"])

    # Datasets
    train_ds = LipDataset(train_s, vocab, frame_size=args.frame_size,
                          frame_stride=args.frame_stride, augment=True)
    val_ds   = LipDataset(val_s,   vocab, frame_size=args.frame_size,
                          frame_stride=args.frame_stride, augment=False)
    test_ds  = LipDataset(test_s,  vocab, frame_size=args.frame_size,
                          frame_stride=args.frame_stride, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    model = LipModel(
        vocab_size=len(vocab),
        hidden_dim=args.hidden_dim,
        encoder=variant["encoder"],
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        frontend_dropout=args.lstm_dropout,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
    ).to(device)

    criterion = AblationLoss(
        vocab=vocab,
        blank_id=0,
        ctc_weight=args.ctc_weight,
        insertion_penalty=args.insertion_penalty,
        use_soft_targets=variant["use_soft_targets"],
        use_ctc=variant["use_ctc"],
        use_frame_ce=variant["use_frame_ce"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )

    history: Dict[str, List] = {
        "train_total": [], "train_frame": [], "train_ctc": [], "train_frame_acc": [],
        "val_total":   [], "val_frame":   [], "val_ctc":   [], "val_frame_acc":   [],
        "val_ter": [], "lr": [],
    }

    best_ter   = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * epoch / args.warmup_epochs

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m, _, _ = eval_loop(model, val_loader, criterion, device, id2tok)

        if epoch > args.warmup_epochs:
            scheduler.step(val_m.get("ter", val_m.get("total_loss", 1.0)))

        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_total"].append(train_m.get("total_loss", 0.0))
        history["train_frame"].append(train_m.get("frame_loss", 0.0))
        history["train_ctc"].append(train_m.get("ctc_loss", 0.0))
        history["train_frame_acc"].append(train_m.get("frame_acc", 0.0))
        history["val_total"].append(val_m.get("total_loss", 0.0))
        history["val_frame"].append(val_m.get("frame_loss", 0.0))
        history["val_ctc"].append(val_m.get("ctc_loss", 0.0))
        history["val_frame_acc"].append(val_m.get("frame_acc", 0.0))
        history["val_ter"].append(val_m.get("ter", 1.0))
        history["lr"].append(cur_lr)

        logger.info(
            "[%s] Ep %03d/%03d | train_loss=%.4f acc=%.3f | val_loss=%.4f TER=%.4f acc=%.3f | lr=%.2e",
            variant["name"], epoch, args.epochs,
            train_m.get("total_loss", 0), train_m.get("frame_acc", 0),
            val_m.get("total_loss", 0), val_m.get("ter", 1), val_m.get("frame_acc", 0),
            cur_lr,
        )

        ter = val_m.get("ter", 1.0)
        if ter < best_ter - 1e-4:
            best_ter   = ter
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Save history
    with (run_dir / "history.json").open("w") as fh:
        json.dump(history, fh, indent=2)

    # Test evaluation with best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    test_m, test_preds, test_refs = eval_loop(model, test_loader, criterion, device, id2tok)
    wall_time = time.time() - t0

    logger.info("[%s] Test TER=%.4f loss=%.4f in %.1fs",
                variant["name"], test_m.get("ter", 1), test_m.get("total_loss", 0), wall_time)

    test_m["wall_time_sec"] = wall_time
    with (run_dir / "test_metrics.json").open("w") as fh:
        json.dump(test_m, fh, indent=2)

    # AUC of val TER curve (lower is better, so AUC of 1-TER)
    val_acc_curve = [1.0 - v for v in history["val_ter"]]
    auc_val_acc   = float(np.trapz(val_acc_curve) / max(1, len(val_acc_curve)))

    return {
        "name":    variant["name"],
        "label":   variant["label"],
        "history": history,
        "test":    test_m,
        "auc_val_acc": auc_val_acc,
        "epochs_run": len(history["val_ter"]),
        "best_val_ter": best_ter,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"
]

def _save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(results: List[Dict[str, Any]], output_dir: Path) -> None:
    labels   = [r["label"] for r in results]
    colors   = COLORS[:len(results)]
    n_epochs = max(r["epochs_run"] for r in results)

    def _epochs(r):
        return list(range(1, r["epochs_run"] + 1))

    # ---- 1. Train loss curves ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        ax.plot(_epochs(r), r["history"]["train_total"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss vs Epoch"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_plot(fig, output_dir / "01_train_loss.png")

    # ---- 2. Val loss curves ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        ax.plot(_epochs(r), r["history"]["val_total"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss vs Epoch"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_plot(fig, output_dir / "02_val_loss.png")

    # ---- 3. Train + Val loss side by side (one subplot per model) ----
    ncols = 3
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows), sharex=False)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, r, c in zip(axes_flat, results, colors):
        ep = _epochs(r)
        ax.plot(ep, r["history"]["train_total"], color=c, lw=2, label="Train")
        ax.plot(ep, r["history"]["val_total"],   color=c, lw=2, ls="--", label="Val")
        ax.set_title(r["label"], fontsize=9)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(results):]:
        ax.set_visible(False)
    fig.suptitle("Train vs Validation Loss per Model", fontsize=12, fontweight="bold")
    _save_plot(fig, output_dir / "03_train_val_loss_grid.png")

    # ---- 4. Val TER vs Epoch ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        ax.plot(_epochs(r), r["history"]["val_ter"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Token Error Rate (lower = better)")
    ax.set_title("Validation TER vs Epoch"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_plot(fig, output_dir / "04_val_ter.png")

    # ---- 5. Val Frame Accuracy vs Epoch ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        ax.plot(_epochs(r), r["history"]["val_frame_acc"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Frame Accuracy (higher = better)")
    ax.set_title("Validation Frame Accuracy vs Epoch"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_plot(fig, output_dir / "05_val_frame_acc.png")

    # ---- 6. Train Frame Accuracy vs Epoch ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        ax.plot(_epochs(r), r["history"]["train_frame_acc"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Frame Accuracy (higher = better)")
    ax.set_title("Training Frame Accuracy vs Epoch"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_plot(fig, output_dir / "06_train_frame_acc.png")

    # ---- 7. LR schedule ----
    fig, ax = plt.subplots(figsize=(10, 4))
    for r, c in zip(results, colors):
        ax.plot(_epochs(r), r["history"]["lr"], color=c, lw=1.5, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule"); ax.legend(fontsize=8)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    _save_plot(fig, output_dir / "07_lr_schedule.png")

    # ---- 8. AUC bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    short_labels = [r["label"].replace(" + ", "\n+\n") for r in results]
    aucs = [r["auc_val_acc"] for r in results]
    bars = ax.bar(short_labels, aucs, color=colors, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("AUC of Val (1 - TER)")
    ax.set_title("Area Under Curve – Validation Accuracy\n(higher is better)")
    ax.set_ylim(0, max(aucs)*1.15 + 0.01)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(fontsize=8)
    _save_plot(fig, output_dir / "08_auc_bar.png")

    # ---- 9. Test TER bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    test_ters = [r["test"].get("ter", 1.0) for r in results]
    bars = ax.bar(short_labels, test_ters, color=colors, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, test_ters):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Token Error Rate (lower is better)")
    ax.set_title("Test Token Error Rate per Ablation Variant")
    ax.set_ylim(0, max(test_ters)*1.15 + 0.02)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(fontsize=8)
    _save_plot(fig, output_dir / "09_test_ter_bar.png")

    # ---- 10. Best Val TER bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    best_ters = [r["best_val_ter"] for r in results]
    bars = ax.bar(short_labels, best_ters, color=colors, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, best_ters):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Token Error Rate (lower is better)")
    ax.set_title("Best Validation TER per Ablation Variant")
    ax.set_ylim(0, max(best_ters)*1.15 + 0.02)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(fontsize=8)
    _save_plot(fig, output_dir / "10_best_val_ter_bar.png")

    # ---- 11. Summary dashboard (combined 4-panel) ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # Panel A: val loss
    for r, c in zip(results, colors):
        axes[0, 0].plot(_epochs(r), r["history"]["val_total"], color=c, lw=2, label=r["label"])
    axes[0, 0].set_title("Val Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend(fontsize=7); axes[0, 0].grid(True, alpha=0.3)
    # Panel B: val TER
    for r, c in zip(results, colors):
        axes[0, 1].plot(_epochs(r), r["history"]["val_ter"], color=c, lw=2, label=r["label"])
    axes[0, 1].set_title("Val TER (↓ better)"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend(fontsize=7); axes[0, 1].grid(True, alpha=0.3)
    # Panel C: val frame acc
    for r, c in zip(results, colors):
        axes[1, 0].plot(_epochs(r), r["history"]["val_frame_acc"], color=c, lw=2, label=r["label"])
    axes[1, 0].set_title("Val Frame Accuracy (↑ better)"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].legend(fontsize=7); axes[1, 0].grid(True, alpha=0.3)
    # Panel D: test TER bar
    axes[1, 1].bar(range(len(results)), test_ters, color=colors, edgecolor="black", lw=0.8)
    axes[1, 1].set_xticks(range(len(results)))
    axes[1, 1].set_xticklabels([r["label"] for r in results], rotation=20, ha="right", fontsize=7)
    axes[1, 1].set_title("Test TER (↓ better)")
    axes[1, 1].set_ylabel("TER")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    for xi, v in enumerate(test_ters):
        axes[1, 1].text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("Ablation Study Dashboard", fontsize=14, fontweight="bold")
    _save_plot(fig, output_dir / "00_dashboard.png")

    logger.info("Saved all plots to %s", output_dir)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablation study over 6 lip-reading model variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-dir",    default="s1_lip_crops")
    p.add_argument("--phoneme-csv",  default="phonemes_s1_aligned/phoneme_predictions.csv")
    p.add_argument("--output-dir",   default="ablation_results")
    p.add_argument("--fps",          type=float, default=25.0)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--min-phoneme-count", type=int, default=3)
    p.add_argument("--hidden-dim",   type=int,   default=256)
    p.add_argument("--frame-size",   type=int,   default=96)
    p.add_argument("--frame-stride", type=int,   default=1)
    p.add_argument("--lstm-layers",  type=int,   default=2)
    p.add_argument("--lstm-dropout", type=float, default=0.3)
    p.add_argument("--transformer-heads",  type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--ctc-weight",        type=float, default=0.3)
    p.add_argument("--insertion-penalty", type=float, default=0.5)
    p.add_argument("--warmup-epochs",  type=int,   default=5)
    p.add_argument("--lr-patience",    type=int,   default=6)
    p.add_argument("--lr-factor",      type=float, default=0.5)
    p.add_argument("--min-lr",         type=float, default=1e-6)
    p.add_argument("--early-stop",     type=int,   default=12)
    p.add_argument("--variants",       nargs="+",
                   default=[str(i) for i in range(1, 7)],
                   help="Which variants to run (1-6), e.g. --variants 1 3 5")
    p.add_argument("--dry-run", action="store_true",
                   help="Run one batch per variant and exit (fast sanity check)")
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

    # Load data once
    all_samples = build_all_samples(Path(args.video_dir), Path(args.phoneme_csv))
    train_s, val_s, test_s = split_samples(all_samples, train_ratio=0.7, val_ratio=0.15, seed=args.seed)
    vocab, remap = build_vocab(train_s, min_count=args.min_phoneme_count)
    train_s = apply_remap(train_s, remap)
    val_s   = apply_remap(val_s,   remap)
    test_s  = apply_remap(test_s,  remap)
    id2tok  = {0: BLANK_TOKEN}
    id2tok.update({i+1: tok for i, tok in enumerate(vocab)})

    logger.info("Data — train:%d val:%d test:%d | vocab:%d", len(train_s), len(val_s), len(test_s), len(vocab))

    with (output_dir / "vocab.json").open("w") as fh:
        json.dump({"blank_id": 0, "tokens": vocab, "remap": remap}, fh, indent=2)

    # Select variants
    selected_nums = set(args.variants)
    selected_variants = [
        v for i, v in enumerate(ABLATION_VARIANTS, 1)
        if str(i) in selected_nums
    ]
    logger.info("Running %d variants: %s", len(selected_variants), [v["label"] for v in selected_variants])

    if args.dry_run:
        logger.info("DRY RUN: forward pass only, no training")
        for variant in selected_variants:
            model = LipModel(
                vocab_size=len(vocab), hidden_dim=args.hidden_dim,
                encoder=variant["encoder"],
            ).to(device)
            criterion = AblationLoss(
                vocab=vocab,
                use_soft_targets=variant["use_soft_targets"],
                use_ctc=variant["use_ctc"],
                use_frame_ce=variant["use_frame_ce"],
            ).to(device)
            ds = LipDataset([train_s[0]], vocab, frame_size=args.frame_size)
            batch = collate_fn([ds[0]])
            videos = batch["videos"].to(device)
            vlens  = batch["video_lengths"].to(device)
            logits, lp_ctc, _ = model(videos, vlens)
            loss, bd = criterion(
                logits, lp_ctc,
                batch["frame_targets"].to(device), batch["frame_weights"].to(device),
                vlens, batch["ctc_targets"].to(device), batch["ctc_lengths"].to(device),
            )
            logger.info("[%s] logits=%s loss=%.4f breakdown=%s",
                        variant["label"], tuple(logits.shape), float(loss), bd)
        return

    # Run all variants
    all_results = []
    for variant in selected_variants:
        result = run_variant(variant, args, train_s, val_s, test_s, vocab, id2tok, device)
        all_results.append(result)

    # Summary JSON
    summary = []
    for r in all_results:
        summary.append({
            "name":          r["name"],
            "label":         r["label"],
            "epochs_run":    r["epochs_run"],
            "best_val_ter":  r["best_val_ter"],
            "auc_val_acc":   r["auc_val_acc"],
            "test_ter":      r["test"].get("ter", 1.0),
            "test_loss":     r["test"].get("total_loss", None),
            "test_frame_acc": r["test"].get("frame_acc", None),
        })

    with (output_dir / "ablation_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("\n\n=== ABLATION SUMMARY ===")
    logger.info("%-45s  %8s  %8s  %8s  %8s",
                "Variant", "Epochs", "BestValTER", "TestTER", "AUC")
    for s in summary:
        logger.info("%-45s  %8d  %8.4f  %8.4f  %8.4f",
                    s["label"], s["epochs_run"],
                    s["best_val_ter"], s["test_ter"], s["auc_val_acc"])

    # Generate all plots
    plot_all(all_results, output_dir)

    logger.info("\nAll outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()