#!/usr/bin/env python3
"""Train a noise-robust audio-visual lip-reading model on the GRID corpus.

This extends train_lip_viseme.py's visual-only pipeline (Conv3DFrontEnd +
BiLSTM/Transformer + viseme-aware frame-CE + auxiliary CTC) with a second,
mel-spectrogram-based audio tower, fused via bidirectional cross-attention
and a reliability-gated combiner.

Data contract
-------------
Two input CSVs, joined by clip "stem":

1. `--phoneme-csv` (from extract_phonemes.py): per-frame phoneme *targets*
   used to supervise both towers -- columns `stem`, `sentence`,
   `canonical_phonemes`, `num_frames`, `per_frame_labels`, `spans_json`.

2. `--noise-manifest` (from add_av_noise.py): the *corrupted* video/audio
   file paths plus per-frame reliability scores (0 = corrupted,
   1 = clean) for both modalities -- columns `stem`, `num_frames`,
   `video_path`, `audio_path`, `video_reliability`, `audio_reliability`,
   plus corruption metadata (unused at training time).

The two are joined on `stem`; the phoneme CSV supplies the frame targets,
the noise manifest supplies the (already-corrupted) media and the
reliability signal fed to the fusion gate.

Video / audio are brought onto the same T-length grid: video frames are
loaded at their native rate (25 fps for GRID); audio is converted to a
log-mel spectrogram with hop_length = sample_rate / fps so it produces one
mel frame per video frame with no separate resampling step.

Usage
-----
    python train_av_fusion.py \
        --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
        --noise-manifest av_noise_s1/noise_manifest.csv \
        --output-dir av_fusion_results

    python train_av_fusion.py --dry-run
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
import librosa
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
BLANK_TOKEN = "<blank>"
MISSING_TOKEN = "<missed>"
INSERTION_TOKEN = "<inserted>"

# ---------------------------------------------------------------------------
# Viseme groups / articulatory features -- copied verbatim from
# train_lip_viseme.py so the frame-CE soft-target semantics match exactly.
# ---------------------------------------------------------------------------

VISEME_GROUPS: Dict[str, List[str]] = {
    "bilabial":       ["p", "b", "m"],
    "labiodental":    ["f", "v"],
    "dental":         ["θ", "ð"],
    "alveolar":       ["t", "d", "n", "s", "z", "l"],
    "postalveolar":   ["ʃ", "ʒ", "tʃ", "dʒ"],
    "velar":          ["k", "g", "ŋ"],
    "glottal":        ["h"],
    "approximant":    ["w", "j", "ɹ", "r"],
    "close_front":    ["iː", "i", "ɪ"],
    "close_back":     ["uː", "u", "ʊ"],
    "mid_front":      ["e", "eɪ", "ɛ"],
    "mid_central":    ["ə", "ɐ", "ɜ", "ɚ", "ʌ"],
    "mid_back":       ["ɔ", "ɔɪ", "oʊ", "o"],
    "open_front":     ["æ", "a", "aɪ", "aʊ"],
    "open_back":      ["ɑ", "ɒ"],
}

ARTICULATORY_FEATURES: Dict[str, Tuple[int, int, int]] = {
    "p": (0, 0, 0), "b": (0, 0, 1), "m": (0, 3, 1),
    "f": (1, 1, 0), "v": (1, 1, 1),
    "θ": (2, 1, 0), "ð": (2, 1, 1),
    "t": (3, 0, 0), "d": (3, 0, 1), "n": (3, 3, 1),
    "s": (3, 1, 0), "z": (3, 1, 1), "l": (3, 5, 1),
    "ʃ": (4, 1, 0), "ʒ": (4, 1, 1), "tʃ": (4, 2, 0), "dʒ": (4, 2, 1),
    "k": (6, 0, 0), "g": (6, 0, 1), "ŋ": (6, 3, 1),
    "h": (7, 1, 0),
    "w": (0, 4, 1), "j": (5, 4, 1), "ɹ": (3, 4, 1), "r": (3, 4, 1),
    **{k: (8, 6, 1) for k in [
        "iː", "i", "ɪ", "uː", "u", "ʊ", "e", "eɪ", "ɛ", "ə", "ɐ", "ɜ", "ɚ", "ʌ",
        "ɔ", "ɔɪ", "oʊ", "o", "æ", "a", "aɪ", "aʊ", "ɑ", "ɒ",
    ]},
}


def _build_viseme_map() -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for group_idx, phonemes in enumerate(VISEME_GROUPS.values()):
        for ph in phonemes:
            mapping[ph] = group_idx
    return mapping


PHONEME_TO_VISEME: Dict[str, int] = _build_viseme_map()


def _articulatory_similarity(a: str, b: str) -> float:
    fa = ARTICULATORY_FEATURES.get(a)
    fb = ARTICULATORY_FEATURES.get(b)
    if fa is None or fb is None:
        return 0.0
    diffs = sum(int(x != y) for x, y in zip(fa, fb))
    return 1.0 - diffs / 3.0


def build_similarity_matrix(vocab: Sequence[str]) -> torch.Tensor:
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
            va, vb = PHONEME_TO_VISEME.get(a, -1), PHONEME_TO_VISEME.get(b, -1)
            if va != -1 and va == vb:
                mat[i, j] = 0.95
            else:
                mat[i, j] = _articulatory_similarity(a, b) * 0.4
    return mat


def similarity_matrix_to_soft_targets(sim: torch.Tensor) -> torch.Tensor:
    n = sim.shape[0]
    dist = torch.zeros_like(sim)
    for i in range(n):
        if i == 0:
            dist[i, 0] = 1.0
            continue
        neighbour = sim[i].clone()
        neighbour[i] = 0.0
        neighbour[0] = 0.0
        total = neighbour.sum()
        dist[i, i] = 0.85
        if total > 0:
            dist[i] += 0.15 * (neighbour / total)
    return dist


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhonemeRecord:
    frame_tokens: List[str]
    canonical_tokens: List[str]
    alignment_weights: List[float]   # forced-alignment confidence (existing signal)
    sentence: str
    num_frames: int


@dataclass
class NoiseRecord:
    video_path: Path
    audio_path: Path
    video_reliability: List[float]
    audio_reliability: List[float]
    num_frames: int
    mel_path: Optional[Path] = None


@dataclass
class Sample:
    stem: str
    video_path: Path
    audio_path: Path
    frame_tokens: List[str]
    canonical_tokens: List[str]
    alignment_weights: List[float]
    video_reliability: List[float]
    audio_reliability: List[float]
    sentence: str = ""
    mel_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _parse_alignment_weights(spans_json_str: str, num_frames: int, fps: float = 25.0) -> List[float]:
    """Confidence of the *forced alignment to text* (unrelated to the noise
    manifest's reliability, which reflects *corruption*). Kept as a
    per-frame loss weight exactly as in train_lip_viseme.py."""
    weights = [1.0] * num_frames
    if not spans_json_str.strip():
        return weights
    try:
        spans = json.loads(spans_json_str)
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


def load_phoneme_csv(csv_path: Path, fps: float = 25.0) -> Dict[str, PhonemeRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Phoneme CSV not found: {csv_path}")

    records: Dict[str, PhonemeRecord] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = row.get("stem", "").strip()
            raw = row.get("per_frame_labels", "").strip()
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
            weights = _parse_alignment_weights(row.get("spans_json", ""), len(frame_tokens), fps)

            records[stem] = PhonemeRecord(
                frame_tokens=frame_tokens,
                canonical_tokens=canonical,
                alignment_weights=weights,
                sentence=row.get("sentence", "").strip(),
                num_frames=num_frames,
            )

    if not records:
        raise ValueError(f"No valid phoneme records found in {csv_path}")
    logger.info("Loaded %d phoneme record(s) from %s", len(records), csv_path)
    return records


def load_noise_manifest(csv_path: Path) -> Dict[str, NoiseRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Noise manifest not found: {csv_path}")

    records: Dict[str, NoiseRecord] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = row.get("stem", "").strip()
            if not stem:
                continue
            try:
                video_reliability = json.loads(row["video_reliability"])
                audio_reliability = json.loads(row["audio_reliability"])
            except (KeyError, json.JSONDecodeError):
                continue
            mel_path_str = row.get("mel_path", "").strip()
            records[stem] = NoiseRecord(
                video_path=Path(row["video_path"]),
                audio_path=Path(row["audio_path"]),
                video_reliability=[float(v) for v in video_reliability],
                audio_reliability=[float(v) for v in audio_reliability],
                num_frames=int(row.get("num_frames", len(video_reliability))),
                mel_path=Path(mel_path_str) if mel_path_str else None,
            )

    if not records:
        raise ValueError(f"No valid noise-manifest rows found in {csv_path}")
    logger.info("Loaded %d noise-manifest record(s) from %s", len(records), csv_path)
    has_mel_cache = sum(1 for r in records.values() if r.mel_path is not None)
    if has_mel_cache:
        logger.info("%d/%d record(s) have a precomputed mel_path (from precompute_mel_spectrograms.py).",
                     has_mel_cache, len(records))
    return records


def build_samples(phoneme_csv: Path, noise_manifest: Path, fps: float = 25.0) -> List[Sample]:
    # Join phoneme targets and corrupted media using their shared GRID stem.
    phoneme_records = load_phoneme_csv(phoneme_csv, fps=fps)
    noise_records = load_noise_manifest(noise_manifest)

    samples: List[Sample] = []
    missing = 0
    for stem, phon in phoneme_records.items():
        noise = noise_records.get(stem)
        if noise is None:
            missing += 1
            continue
        samples.append(Sample(
            stem=stem,
            video_path=noise.video_path,
            audio_path=noise.audio_path,
            frame_tokens=phon.frame_tokens,
            canonical_tokens=phon.canonical_tokens,
            alignment_weights=phon.alignment_weights,
            video_reliability=noise.video_reliability,
            audio_reliability=noise.audio_reliability,
            sentence=phon.sentence,
            mel_path=noise.mel_path,
        ))

    if missing:
        logger.warning("%d phoneme record(s) had no matching noise-manifest entry.", missing)
    if not samples:
        raise FileNotFoundError("No samples could be built -- check that stems match between the two CSVs.")
    logger.info("Paired %d sample(s) (phoneme targets + corrupted media)", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Vocabulary (unchanged from train_lip_viseme.py)
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
        my_viseme = PHONEME_TO_VISEME.get(tok, -1)
        best: Optional[str] = None
        best_count = -1
        for candidate in kept:
            if PHONEME_TO_VISEME.get(candidate, -2) == my_viseme and counter[candidate] > best_count:
                best, best_count = candidate, counter[candidate]
        remap[tok] = best if best is not None else counter.most_common(1)[0][0]

    return sorted(kept), remap


def apply_vocab_remap(samples: List[Sample], remap: Dict[str, str]) -> List[Sample]:
    remapped = []
    for s in samples:
        remapped.append(Sample(
            stem=s.stem,
            video_path=s.video_path,
            audio_path=s.audio_path,
            frame_tokens=[remap.get(t, t) if t != "" else "" for t in s.frame_tokens],
            canonical_tokens=[remap.get(t, t) for t in s.canonical_tokens],
            alignment_weights=s.alignment_weights,
            video_reliability=s.video_reliability,
            audio_reliability=s.audio_reliability,
            sentence=s.sentence,
            mel_path=s.mel_path,
        ))
    return remapped


def split_samples(
    samples: List[Sample], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    rng = random.Random(seed)
    s = samples.copy()
    rng.shuffle(s)
    n = len(s)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return s[:n_train], s[n_train:n_train + n_val], s[n_train + n_val:]


# ---------------------------------------------------------------------------
# SpecAugment (extra on-the-fly masking, layered on top of the pre-rendered
# corruption from add_av_noise.py, applied during training only)
# ---------------------------------------------------------------------------

def apply_specaugment(
    mel: torch.Tensor,
    time_mask_param: int = 8,
    freq_mask_param: int = 12,
    num_time_masks: int = 1,
    num_freq_masks: int = 1,
) -> torch.Tensor:
    """mel: (T, n_mels). Masked in-place on a clone; mask value = mel.mean()."""
    mel = mel.clone()
    fill_value = float(mel.mean())
    T, n_mels = mel.shape

    for _ in range(num_time_masks):
        width = random.randint(0, min(time_mask_param, T))
        if width == 0:
            continue
        start = random.randint(0, T - width)
        mel[start:start + width, :] = fill_value

    for _ in range(num_freq_masks):
        width = random.randint(0, min(freq_mask_param, n_mels))
        if width == 0:
            continue
        start = random.randint(0, n_mels - width)
        mel[:, start:start + width] = fill_value

    return mel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AVPhonemeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        vocab: Sequence[str],
        fps: float = 25.0,
        sample_rate: int = 16000,
        n_mels: int = 80,
        frame_size: int = 96,
        frame_stride: int = 1,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.fps = fps
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = int(round(sample_rate / fps))
        self.n_fft = 1024
        self.frame_stride = frame_stride
        self.augment = augment
        self.tok2id = {tok: idx + 1 for idx, tok in enumerate(vocab)}
        self.blank_id = 0

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.video_transform = v2.Compose([
            v2.Resize((frame_size, frame_size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
        self._mean = torch.tensor(mean).view(1, 3, 1, 1)
        self._std = torch.tensor(std).view(1, 3, 1, 1)
        self._cache_warned = False  # avoid spamming the same warning per __getitem__ call

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        frames: List[torch.Tensor] = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        finally:
            cap.release()
        if not frames:
            raise RuntimeError(f"No frames in {path}")
        return torch.stack(frames)

    def _load_mel(self, path: Path) -> torch.Tensor:
        waveform, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        mel_power = librosa.feature.melspectrogram(
            y=waveform, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
        )
        mel_db = librosa.power_to_db(mel_power, ref=np.max)  # (n_mels, T)
        mel_db = mel_db.T  # (T, n_mels)
        mean, std = mel_db.mean(), mel_db.std() + 1e-6
        mel_norm = (mel_db - mean) / std
        return torch.from_numpy(mel_norm.astype(np.float32))

    def _load_mel_from_cache(self, mel_path: Path) -> Optional[torch.Tensor]:
        """Load a mel cached by precompute_mel_spectrograms.py. Returns None
        (rather than raising) on any read/shape problem, so a missing or
        stale cache entry safely falls back to on-the-fly computation
        instead of crashing the whole run."""
        try:
            mel = np.load(mel_path)
        except Exception as exc:
            if not self._cache_warned:
                logger.warning(
                    "Failed to load mel cache %s (%s) -- falling back to on-the-fly computation "
                    "(further cache misses will not be logged individually).", mel_path, exc,
                )
                self._cache_warned = True
            return None

        if mel.ndim != 2 or mel.shape[1] != self.n_mels:
            if not self._cache_warned:
                logger.warning(
                    "Cached mel at %s has shape %s, expected (*, %d). The cache was likely built with "
                    "different --n-mels/--fps/--sample-rate than this run -- falling back to on-the-fly "
                    "computation (further mismatches will not be logged individually). Consider "
                    "re-running precompute_mel_spectrograms.py with matching settings.",
                    mel_path, tuple(mel.shape), self.n_mels,
                )
                self._cache_warned = True
            return None

        return torch.from_numpy(mel.astype(np.float32))

    def _get_mel(self, sample: Sample) -> torch.Tensor:
        if sample.mel_path is not None:
            cached = self._load_mel_from_cache(sample.mel_path)
            if cached is not None:
                return cached
        return self._load_mel(sample.audio_path)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        video = self._load_video(sample.video_path)          # (T_v, C, H, W)
        mel = self._get_mel(sample)                            # (T_a, n_mels)

        T = min(
            video.shape[0], mel.shape[0], len(sample.frame_tokens),
            len(sample.video_reliability), len(sample.audio_reliability),
        )
        video = video[:T]
        mel = mel[:T]
        frame_tokens = sample.frame_tokens[:T]
        alignment_weights = sample.alignment_weights[:T]
        video_reliability = sample.video_reliability[:T]
        audio_reliability = sample.audio_reliability[:T]

        if self.frame_stride > 1:
            video = video[::self.frame_stride]
            mel = mel[::self.frame_stride]
            frame_tokens = frame_tokens[::self.frame_stride]
            alignment_weights = alignment_weights[::self.frame_stride]
            video_reliability = video_reliability[::self.frame_stride]
            audio_reliability = audio_reliability[::self.frame_stride]

        video = self.video_transform(video)

        if self.augment:
            if random.random() < 0.5:
                video = v2.functional.horizontal_flip(video)
            if random.random() < 0.4:
                video = video * self._std + self._mean
                video = v2.functional.adjust_brightness(video, random.uniform(0.8, 1.2))
                video = v2.functional.adjust_contrast(video, random.uniform(0.8, 1.2))
                video = v2.functional.adjust_saturation(video, random.uniform(0.9, 1.1))
                video = (video - self._mean) / self._std
            if random.random() < 0.25:
                _, _, H, W = video.shape
                ch = max(1, int(H * random.uniform(0.08, 0.20)))
                cw = max(1, int(W * random.uniform(0.08, 0.20)))
                y = random.randint(0, max(0, H - ch))
                x = random.randint(0, max(0, W - cw))
                video[:, :, y:y + ch, x:x + cw] = 0.0
            mel = apply_specaugment(mel)

        frame_ids = [self.tok2id.get(t, 0) if t != "" else 0 for t in frame_tokens]
        ctc_ids = [self.tok2id.get(t, 0) for t in sample.canonical_tokens if t in self.tok2id] or [0]

        return {
            "video": video,
            "mel": mel,
            "video_length": torch.tensor(video.shape[0], dtype=torch.long),
            "frame_targets": torch.tensor(frame_ids, dtype=torch.long),
            "alignment_weights": torch.tensor(alignment_weights, dtype=torch.float32),
            "video_reliability": torch.tensor(video_reliability, dtype=torch.float32),
            "audio_reliability": torch.tensor(audio_reliability, dtype=torch.float32),
            "ctc_targets": torch.tensor(ctc_ids, dtype=torch.long),
            "ctc_length": torch.tensor(len(ctc_ids), dtype=torch.long),
            "stem": sample.stem,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    lengths = torch.tensor([b["video_length"] for b in batch], dtype=torch.long)
    T = int(lengths.max())
    C, H, W = batch[0]["video"].shape[1:]
    n_mels = batch[0]["mel"].shape[1]

    videos = torch.zeros(B, T, C, H, W)
    mels = torch.zeros(B, T, n_mels)
    frame_tgts = torch.zeros(B, T, dtype=torch.long)
    align_wts = torch.zeros(B, T)
    video_rel = torch.zeros(B, T)
    audio_rel = torch.zeros(B, T)

    ctc_lens = torch.tensor([b["ctc_length"] for b in batch], dtype=torch.long)
    ctc_tgts = torch.cat([b["ctc_targets"] for b in batch])

    stems: List[str] = []
    for i, b in enumerate(batch):
        t = int(b["video_length"])
        videos[i, :t] = b["video"]
        mels[i, :t] = b["mel"]
        frame_tgts[i, :t] = b["frame_targets"]
        align_wts[i, :t] = b["alignment_weights"]
        video_rel[i, :t] = b["video_reliability"]
        audio_rel[i, :t] = b["audio_reliability"]
        stems.append(b["stem"])

    return {
        "videos": videos, "mels": mels, "video_lengths": lengths,
        "frame_targets": frame_tgts, "alignment_weights": align_wts,
        "video_reliability": video_rel, "audio_reliability": audio_rel,
        "ctc_targets": ctc_tgts, "ctc_lengths": ctc_lens,
        "stems": stems,
    }


# ---------------------------------------------------------------------------
# Model -- video tower kept close to train_lip_viseme.py
# ---------------------------------------------------------------------------

class Conv3DFrontEnd(nn.Module):
    """Unchanged from train_lip_viseme.py. (B, T, C, H, W) -> (B, T, hidden_dim)."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        def _block(in_c: int, out_c: int, spatial_stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(in_c, in_c, kernel_size=(3, 3, 3), stride=(1, spatial_stride, spatial_stride),
                          padding=(1, 1, 1), groups=in_c, bias=False),
                nn.Conv3d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        self.layer1 = _block(3, 32, spatial_stride=2)
        self.layer2 = _block(32, 64, spatial_stride=2)
        self.layer3 = _block(64, 128, spatial_stride=2)
        self.pool = nn.AdaptiveAvgPool3d((None, 3, 3))
        self.projector = nn.Sequential(
            nn.Linear(128 * 9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        B, C, T, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, C * h * w)
        return self.projector(x)


class AudioMelFrontEnd(nn.Module):
    """Anisotropic 2D-conv frontend for log-mel spectrograms.

    Time stride is 1 throughout (frame count is preserved to stay aligned
    with the video tower); only the frequency axis is progressively pooled,
    matching Conv3DFrontEnd's own asymmetric treatment (T preserved, H/W
    pooled). See discussion: frequency carries phoneme identity as a
    whole-band pattern, so it's collapsed via depth rather than kept as a
    spatial axis the way it would be in an image CNN.

    (B, T, n_mels) -> (B, T, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 256, n_mels: int = 80, dropout: float = 0.3):
        super().__init__()

        def _block(in_c: int, out_c: int, freq_stride: int = 2) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, freq_stride),
                          padding=(1, 1), groups=in_c, bias=False),
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.layer1 = _block(1, 32, freq_stride=2)    # n_mels -> n_mels/2
        self.layer2 = _block(32, 64, freq_stride=2)    # -> n_mels/4
        self.layer3 = _block(64, 128, freq_stride=2)   # -> n_mels/8
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.projector = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.unsqueeze(1)          # (B, 1, T, n_mels)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)            # (B, 128, T, n_mels/8)
        x = self.freq_pool(x)         # (B, 128, T, 1)
        x = x.squeeze(-1).transpose(1, 2)  # (B, T, 128)
        return self.projector(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CrossModalAttention(nn.Module):
    """Bidirectional cross-attention: video attends to audio and vice versa,
    each with a residual + LayerNorm, as in Sterpu et al. / MLCA-AVSR."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.video_to_audio = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_to_video = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)

    def forward(
        self, video_feat: torch.Tensor, audio_feat: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_out, _ = self.video_to_audio(video_feat, audio_feat, audio_feat, key_padding_mask=key_padding_mask)
        video_feat = self.norm_v(video_feat + v_out)

        a_out, _ = self.audio_to_video(audio_feat, video_feat, video_feat, key_padding_mask=key_padding_mask)
        audio_feat = self.norm_a(audio_feat + a_out)
        return video_feat, audio_feat


class ReliabilityGatedFusion(nn.Module):
    """Combines the (cross-attended) video/audio streams with a per-frame
    softmax gate conditioned on the known corruption-reliability scores
    (from the noise manifest), so the model learns how much to trust each
    modality's known reliability level rather than treating it as a hard
    on/off mask."""

    def __init__(self, hidden_dim: int, gate_hidden: int = 32):
        super().__init__()
        gate_input_dim = hidden_dim * 3 + 2
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, 2),
        )

    def forward(
        self, video_feat: torch.Tensor, audio_feat: torch.Tensor,
        video_reliability: torch.Tensor, audio_reliability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([
            video_feat,
            audio_feat,
            torch.abs(video_feat - audio_feat),
            video_reliability.unsqueeze(-1),
            audio_reliability.unsqueeze(-1),
        ], dim=-1)  # (B, T, 3 * hidden_dim + 2)
        gate_logits = self.gate_mlp(gate_input)
        gates = F.softmax(gate_logits, dim=-1)  # (B, T, 2)
        g_v = gates[..., 0:1]
        g_a = gates[..., 1:2]
        fused = g_v * video_feat + g_a * audio_feat
        return fused, gates


class AVLipReadingModel(nn.Module):
    """Video tower (Conv3DFrontEnd) + audio tower (AudioMelFrontEnd) ->
    bidirectional cross-attention -> reliability-gated fusion -> joint
    BiLSTM/Transformer encoder -> frame classifier, with optional auxiliary
    per-modality CTC heads applied pre-fusion."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        n_mels: int = 80,
        encoder: str = "bilstm",
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        frontend_dropout: float = 0.3,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        cross_attn_layers: int = 1,
        cross_attn_heads: int = 4,
        use_aux_heads: bool = True,
        modality_dropout: float = 0.15,
        modality: str = "av",
    ):
        super().__init__()
        if modality not in {"av", "audio", "video"}:
            raise ValueError(f"Unknown modality: {modality}")
        self.modality = modality
        self.use_video = modality in {"av", "video"}
        self.use_audio = modality in {"av", "audio"}

        self.video_frontend = Conv3DFrontEnd(hidden_dim=hidden_dim, dropout=frontend_dropout) if self.use_video else None
        self.audio_frontend = AudioMelFrontEnd(hidden_dim=hidden_dim, n_mels=n_mels, dropout=frontend_dropout) if self.use_audio else None
        self.pos_enc_v = SinusoidalPositionalEncoding(hidden_dim, dropout=0.1) if self.use_video else None
        self.pos_enc_a = SinusoidalPositionalEncoding(hidden_dim, dropout=0.1) if self.use_audio else None
        self.modality_dropout = modality_dropout

        if self.modality == "av":
            self.cross_layers = nn.ModuleList([
                CrossModalAttention(hidden_dim, num_heads=cross_attn_heads) for _ in range(cross_attn_layers)
            ])
            self.fusion = ReliabilityGatedFusion(hidden_dim)
        else:
            self.cross_layers = nn.ModuleList()
            self.fusion = None

        self.encoder_type = encoder
        if encoder == "bilstm":
            self.seq_encoder = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim, num_layers=lstm_layers,
                batch_first=True, bidirectional=True,
                dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            )
            enc_dim = hidden_dim * 2
        elif encoder == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=transformer_heads, dim_feedforward=hidden_dim * 4,
                dropout=lstm_dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
            enc_dim = hidden_dim
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        self.norm = nn.LayerNorm(enc_dim)
        self.dropout = nn.Dropout(lstm_dropout)
        self.classifier = nn.Linear(enc_dim, vocab_size + 1)

        self.use_aux_heads = use_aux_heads
        if use_aux_heads:
            if self.use_video:
                self.aux_video_classifier = nn.Linear(hidden_dim, vocab_size + 1)
            if self.use_audio:
                self.aux_audio_classifier = nn.Linear(hidden_dim, vocab_size + 1)

    def _apply_modality_dropout(self, video_feat: torch.Tensor, audio_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0.0:
            return video_feat, audio_feat

        keep_v = (torch.rand(video_feat.size(0), 1, 1, device=video_feat.device) > self.modality_dropout).float()
        keep_a = (torch.rand(audio_feat.size(0), 1, 1, device=audio_feat.device) > self.modality_dropout).float()

        both_dropped = (keep_v + keep_a) == 0
        if both_dropped.any():
            keep_v = keep_v.clone()
            keep_v[both_dropped] = 1.0

        return video_feat * keep_v, audio_feat * keep_a

    def forward(
        self, videos: torch.Tensor, mel: torch.Tensor, video_lengths: torch.Tensor,
        video_reliability: torch.Tensor, audio_reliability: torch.Tensor,
    ):
        video_feat = self.pos_enc_v(self.video_frontend(videos)) if self.use_video else None
        audio_feat = self.pos_enc_a(self.audio_frontend(mel)) if self.use_audio else None

        if self.modality == "av":
            assert video_feat is not None and audio_feat is not None
            video_feat, audio_feat = self._apply_modality_dropout(video_feat, audio_feat)

            T_max = video_feat.size(1)
            pad_mask = torch.arange(T_max, device=video_lengths.device).unsqueeze(0) >= video_lengths.unsqueeze(1)

            for layer in self.cross_layers:
                video_feat, audio_feat = layer(video_feat, audio_feat, key_padding_mask=pad_mask)

            assert self.fusion is not None
            fused, gates = self.fusion(video_feat, audio_feat, video_reliability, audio_reliability)
            aux_video_logits = self.aux_video_classifier(video_feat) if self.use_aux_heads else None
            aux_audio_logits = self.aux_audio_classifier(audio_feat) if self.use_aux_heads else None
        else:
            feature = video_feat if self.use_video else audio_feat
            if feature is None:
                raise RuntimeError("No active modality features were produced.")
            fused = feature
            gates = feature.new_zeros(feature.size(0), feature.size(1), 2)
            if self.modality == "video":
                gates[..., 0] = 1.0
            else:
                gates[..., 1] = 1.0
            aux_video_logits = self.aux_video_classifier(feature) if self.use_aux_heads and self.use_video else None
            aux_audio_logits = self.aux_audio_classifier(feature) if self.use_aux_heads and self.use_audio else None

        T_max = fused.size(1)
        pad_mask = torch.arange(T_max, device=video_lengths.device).unsqueeze(0) >= video_lengths.unsqueeze(1)

        if self.encoder_type == "bilstm":
            packed = nn.utils.rnn.pack_padded_sequence(
                fused, video_lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            enc_out, _ = self.seq_encoder(packed)
            enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)
        else:
            enc_out = self.seq_encoder(fused, src_key_padding_mask=pad_mask)

        enc_out = self.norm(enc_out)
        enc_out = self.dropout(enc_out)
        logits = self.classifier(enc_out)
        log_probs_ctc = F.log_softmax(logits, dim=-1).transpose(0, 1)

        return logits, log_probs_ctc, video_lengths, aux_video_logits, aux_audio_logits, gates


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class AVFusionLoss(nn.Module):
    """Frame-CE with viseme soft targets (weighted by alignment confidence)
    + main CTC on the fused output, plus optional light-weight auxiliary CTC
    terms on the pre-fusion per-modality streams so neither tower is free to
    atrophy behind the gate."""

    def __init__(
        self, vocab: Sequence[str], blank_id: int = 0, ctc_weight: float = 0.3,
        insertion_penalty: float = 0.5, aux_ctc_weight: float = 0.1,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.ctc_weight = ctc_weight
        self.insertion_penalty = insertion_penalty
        self.aux_ctc_weight = aux_ctc_weight
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

        sim = build_similarity_matrix(vocab)
        soft = similarity_matrix_to_soft_targets(sim)
        self.register_buffer("soft_targets", soft)

    def _aux_ctc(self, aux_logits: torch.Tensor, ctc_targets, ctc_lengths, video_lengths) -> torch.Tensor:
        log_probs = F.log_softmax(aux_logits, dim=-1).transpose(0, 1)
        T_max = aux_logits.shape[1]
        in_lens = video_lengths.clamp(max=T_max)
        return self.ctc_loss(log_probs, ctc_targets, in_lens, ctc_lengths)

    def forward(
        self, logits, log_probs_ctc, frame_targets, alignment_weights, video_lengths,
        ctc_targets, ctc_lengths, aux_video_logits=None, aux_audio_logits=None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T_max, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)

        mask = torch.arange(T_max, device=video_lengths.device).unsqueeze(0) < video_lengths.unsqueeze(1)

        flat_lp = log_probs[mask]
        flat_tgt = frame_targets[mask]
        flat_wt = alignment_weights[mask]

        tgt_clamped = flat_tgt.clamp(0, self.soft_targets.shape[0] - 1)
        soft_dist = self.soft_targets[tgt_clamped]
        frame_ce = -(soft_dist * flat_lp).sum(dim=-1)

        is_silence = (flat_tgt == self.blank_id).float()
        blank_prob = flat_lp[:, self.blank_id].exp()
        ins_penalty = self.insertion_penalty * is_silence * (1.0 - blank_prob)

        frame_loss = ((frame_ce + ins_penalty) * flat_wt).mean()

        ctc_in_lens = video_lengths.clamp(max=T_max)
        ctc_l = self.ctc_loss(log_probs_ctc, ctc_targets, ctc_in_lens, ctc_lengths)

        total = frame_loss + self.ctc_weight * ctc_l
        breakdown = {
            "frame_loss": float(frame_loss.detach()),
            "ctc_loss": float(ctc_l.detach()) if ctc_l.isfinite() else 0.0,
        }

        if self.aux_ctc_weight > 0 and aux_video_logits is not None:
            aux_v = self._aux_ctc(aux_video_logits, ctc_targets, ctc_lengths, video_lengths)
            total = total + self.aux_ctc_weight * aux_v
            breakdown["aux_video_ctc"] = float(aux_v.detach()) if aux_v.isfinite() else 0.0

        if self.aux_ctc_weight > 0 and aux_audio_logits is not None:
            aux_a = self._aux_ctc(aux_audio_logits, ctc_targets, ctc_lengths, video_lengths)
            total = total + self.aux_ctc_weight * aux_a
            breakdown["aux_audio_ctc"] = float(aux_a.detach()) if aux_a.isfinite() else 0.0

        breakdown["total_loss"] = float(total.detach())
        return total, breakdown


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def levenshtein(a: Sequence, b: Sequence) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        curr = [i]
        for j, bj in enumerate(b, 1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ai != bj)))
        prev = curr
    return prev[-1]


def greedy_decode(logits: torch.Tensor, length: int, blank_id: int = 0) -> List[int]:
    ids = torch.argmax(logits[:length], dim=-1).tolist()
    tokens, prev = [], None
    for i in ids:
        if i == blank_id:
            prev = None
            continue
        if i == prev:
            continue
        tokens.append(i)
        prev = i
    return tokens


def masked_mean(values: torch.Tensor, lengths: torch.Tensor) -> float:
    T_max = values.shape[1]
    mask = torch.arange(T_max, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    return float((values.detach().squeeze(-1) * mask).sum() / mask.sum().clamp(min=1))


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    gate_v_sum, gate_a_sum, n = 0.0, 0.0, 0

    for batch in tqdm(loader, desc="train", leave=False):
        videos = batch["videos"].to(device)
        mels = batch["mels"].to(device)
        vlens = batch["video_lengths"].to(device)
        ftgts = batch["frame_targets"].to(device)
        align_wts = batch["alignment_weights"].to(device)
        video_rel = batch["video_reliability"].to(device)
        audio_rel = batch["audio_reliability"].to(device)
        ctc_tgts = batch["ctc_targets"].to(device)
        ctc_lens = batch["ctc_lengths"].to(device)

        logits, lp_ctc, out_lens, aux_v, aux_a, gates = model(videos, mels, vlens, video_rel, audio_rel)
        loss, bd = criterion(logits, lp_ctc, ftgts, align_wts, vlens, ctc_tgts, ctc_lens, aux_v, aux_a)

        if not loss.isfinite():
            logger.warning("Non-finite loss -- skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in bd.items():
            totals[k] = totals.get(k, 0.0) + v
        gate_v_sum += masked_mean(gates[..., 0:1], vlens)
        gate_a_sum += masked_mean(gates[..., 1:2], vlens)
        n += 1

    avg = {k: v / max(1, n) for k, v in totals.items()}
    avg["gate_video"] = gate_v_sum / max(1, n)
    avg["gate_audio"] = gate_a_sum / max(1, n)
    return avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, id2tok) -> Tuple[Dict[str, float], List[str], List[str]]:
    model.eval()
    totals: Dict[str, float] = {}
    gate_v_sum, gate_a_sum, n = 0.0, 0.0, 0
    all_preds: List[str] = []
    all_refs: List[str] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        videos = batch["videos"].to(device)
        mels = batch["mels"].to(device)
        vlens = batch["video_lengths"].to(device)
        ftgts = batch["frame_targets"].to(device)
        align_wts = batch["alignment_weights"].to(device)
        video_rel = batch["video_reliability"].to(device)
        audio_rel = batch["audio_reliability"].to(device)
        ctc_tgts = batch["ctc_targets"].to(device)
        ctc_lens = batch["ctc_lengths"].to(device)

        logits, lp_ctc, out_lens, aux_v, aux_a, gates = model(videos, mels, vlens, video_rel, audio_rel)
        _, bd = criterion(logits, lp_ctc, ftgts, align_wts, vlens, ctc_tgts, ctc_lens, aux_v, aux_a)

        for k, v in bd.items():
            totals[k] = totals.get(k, 0.0) + v
        gate_v_sum += masked_mean(gates[..., 0:1], vlens)
        gate_a_sum += masked_mean(gates[..., 1:2], vlens)
        n += 1

        for i in range(logits.shape[0]):
            L = int(vlens[i].item())
            pred = greedy_decode(logits[i].cpu(), L)
            ref = [int(x) for x in ftgts[i, :L].cpu().tolist() if x != 0]
            ref_collapsed: List[int] = []
            prev = None
            for tok in ref:
                if tok != prev:
                    ref_collapsed.append(tok)
                    prev = tok
            all_preds.append(" ".join(id2tok.get(t, "?") for t in pred))
            all_refs.append(" ".join(id2tok.get(t, "?") for t in ref_collapsed))

    avg = {k: v / max(1, n) for k, v in totals.items()}
    avg["gate_video"] = gate_v_sum / max(1, n)
    avg["gate_audio"] = gate_a_sum / max(1, n)

    pred_seqs = [p.split() for p in all_preds]
    ref_seqs = [r.split() for r in all_refs]
    errs = sum(levenshtein(p, r) for p, r in zip(pred_seqs, ref_seqs))
    total = sum(max(1, len(r)) for r in ref_seqs)
    avg["ter"] = errs / total
    return avg, all_preds, all_refs


def phoneme_report(all_preds: List[str], all_refs: List[str], vocab: Sequence[str], output_dir: Path) -> None:
    label_list = list(vocab)
    l2i = {l: i for i, l in enumerate(label_list)}
    n = len(label_list)
    matrix = np.zeros((n + 1, n + 1), dtype=np.int64)
    stats = {l: {"support": 0, "correct": 0, "predicted": 0} for l in label_list}

    def _align(ref: List[str], hyp: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
        R, H = len(ref), len(hyp)
        dp = [[0] * (H + 1) for _ in range(R + 1)]
        bp: List[List[str]] = [[""] * (H + 1) for _ in range(R + 1)]
        for r in range(1, R + 1):
            dp[r][0] = r
            bp[r][0] = "D"
        for h in range(1, H + 1):
            dp[0][h] = h
            bp[0][h] = "I"
        for r in range(1, R + 1):
            for h in range(1, H + 1):
                sc = dp[r - 1][h - 1] + int(ref[r - 1] != hyp[h - 1])
                dc = dp[r - 1][h] + 1
                ic = dp[r][h - 1] + 1
                best, op = min((sc, "S"), (dc, "D"), (ic, "I"), key=lambda x: x[0])
                dp[r][h] = best
                bp[r][h] = op
        pairs: List[Tuple[Optional[str], Optional[str]]] = []
        r, h = R, H
        while r > 0 or h > 0:
            op = bp[r][h]
            if op == "S":
                pairs.append((ref[r - 1], hyp[h - 1])); r -= 1; h -= 1
            elif op == "D":
                pairs.append((ref[r - 1], None)); r -= 1
            else:
                pairs.append((None, hyp[h - 1])); h -= 1
        pairs.reverse()
        return pairs

    for ref_str, pred_str in zip(all_refs, all_preds):
        ref = ref_str.split() if ref_str else []
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
        s, c, pr = stats[l]["support"], stats[l]["correct"], stats[l].get("predicted", 0)
        rows.append({
            "phoneme": l, "support": s, "correct": c,
            "recall": round(c / s, 4) if s else 0.0,
            "precision": round(c / pr, 4) if pr else 0.0,
        })
    rows.sort(key=lambda r: r["recall"])

    csv_path = output_dir / "phoneme_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["phoneme", "support", "correct", "recall", "precision"])
        w.writeheader(); w.writerows(rows)

    size = max(8.0, 0.4 * (n + 1))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = label_list + [MISSING_TOKEN]
    ax.set_xticks(range(n + 1)); ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n + 1)); ax.set_yticklabels(label_list + [INSERTION_TOKEN], fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Phoneme confusion matrix (fused AV model)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Phoneme report -> %s", csv_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a noise-robust audio-visual cross-attention phoneme model on GRID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phoneme-csv", default="phonemes_s1_aligned/phoneme_predictions.csv",
                   help="Per-frame phoneme targets from extract_phonemes.py.")
    p.add_argument("--noise-manifest", default="av_noise_s1/noise_manifest.csv",
                   help="Corrupted media + reliability manifest from add_av_noise.py.")
    p.add_argument("--output-dir", default="av_fusion_results")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=80)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--min-phoneme-count", type=int, default=1)

    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--frame-size", type=int, default=96)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--encoder", choices=["bilstm", "transformer"], default="bilstm")
    p.add_argument("--lstm-layers", type=int, default=2)
    p.add_argument("--lstm-dropout", type=float, default=0.3)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--cross-attn-layers", type=int, default=1)
    p.add_argument("--cross-attn-heads", type=int, default=4)
    p.add_argument("--no-aux-heads", action="store_true", help="Disable auxiliary per-modality CTC heads.")
    p.add_argument("--modality", choices=["av", "audio", "video"], default="av",
                   help="Train the fused AV model, an audio-only model, or a video-only model.")
    p.add_argument("--modality-dropout", type=float, default=0.15,
                   help="Probability of dropping each modality stream during training, independently.")

    p.add_argument("--ctc-weight", type=float, default=0.3)
    p.add_argument("--aux-ctc-weight", type=float, default=0.1)
    p.add_argument("--insertion-penalty", type=float, default=0.5)

    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr-patience", type=int, default=6)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--early-stop", type=int, default=12)

    p.add_argument("--class-aware-sampling", action="store_true")
    p.add_argument("--dry-run", action="store_true")
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

    all_samples = build_samples(Path(args.phoneme_csv), Path(args.noise_manifest), fps=args.fps)
    train_s, val_s, test_s = split_samples(all_samples, train_ratio=0.7, val_ratio=0.15, seed=args.seed)

    vocab, remap = build_vocab(train_s, min_count=args.min_phoneme_count)
    train_s = apply_vocab_remap(train_s, remap)
    val_s = apply_vocab_remap(val_s, remap)
    test_s = apply_vocab_remap(test_s, remap)

    id2tok = {0: BLANK_TOKEN}
    id2tok.update({i + 1: tok for i, tok in enumerate(vocab)})

    with (output_dir / "vocab.json").open("w") as fh:
        json.dump({"blank_id": 0, "tokens": vocab, "remap": remap}, fh, indent=2)

    logger.info("Split -- train:%d val:%d test:%d | vocab:%d", len(train_s), len(val_s), len(test_s), len(vocab))

    train_ds = AVPhonemeDataset(train_s, vocab, fps=args.fps, sample_rate=args.sample_rate, n_mels=args.n_mels,
                                frame_size=args.frame_size, frame_stride=args.frame_stride, augment=True)
    val_ds = AVPhonemeDataset(val_s, vocab, fps=args.fps, sample_rate=args.sample_rate, n_mels=args.n_mels,
                               frame_size=args.frame_size, frame_stride=args.frame_stride, augment=False)
    test_ds = AVPhonemeDataset(test_s, vocab, fps=args.fps, sample_rate=args.sample_rate, n_mels=args.n_mels,
                                frame_size=args.frame_size, frame_stride=args.frame_stride, augment=False)

    if args.class_aware_sampling:
        ph_counts: Counter[str] = Counter()
        for s in train_s:
            ph_counts.update(t for t in s.frame_tokens if t != "")
        inv = {t: 1.0 / max(1, c) for t, c in ph_counts.items()}
        weights = [
            float(np.mean([inv.get(t, 1.0) for t in s.frame_tokens if t != ""]) if any(t != "" for t in s.frame_tokens) else 1.0)
            for s in train_s
        ]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    model = AVLipReadingModel(
        vocab_size=len(vocab), hidden_dim=args.hidden_dim, n_mels=args.n_mels,
        encoder=args.encoder, lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout,
        frontend_dropout=args.lstm_dropout, transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers, cross_attn_layers=args.cross_attn_layers,
        cross_attn_heads=args.cross_attn_heads, use_aux_heads=not args.no_aux_heads,
        modality_dropout=args.modality_dropout, modality=args.modality,
    ).to(device)

    criterion = AVFusionLoss(
        vocab=vocab, blank_id=0, ctc_weight=args.ctc_weight,
        insertion_penalty=args.insertion_penalty, aux_ctc_weight=args.aux_ctc_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    if args.dry_run:
        batch = next(iter(train_loader))
        videos, mels = batch["videos"].to(device), batch["mels"].to(device)
        vlens = batch["video_lengths"].to(device)
        video_rel = batch["video_reliability"].to(device)
        audio_rel = batch["audio_reliability"].to(device)
        logits, lp_ctc, _, aux_v, aux_a, gates = model(videos, mels, vlens, video_rel, audio_rel)
        loss, bd = criterion(
            logits, lp_ctc, batch["frame_targets"].to(device), batch["alignment_weights"].to(device),
            vlens, batch["ctc_targets"].to(device), batch["ctc_lengths"].to(device), aux_v, aux_a,
        )
        logger.info("Dry run -- videos: %s  mels: %s  logits: %s", tuple(videos.shape), tuple(mels.shape), tuple(logits.shape))
        logger.info("Dry run -- loss breakdown: %s", bd)
        logger.info("Dry run -- mean gate video=%.3f audio=%.3f", masked_mean(gates[..., 0:1], vlens), masked_mean(gates[..., 1:2], vlens))
        return

    best_ter = float("inf")
    no_improve = 0
    history: Dict[str, List] = {
        "train_total": [], "train_frame": [], "train_ctc": [],
        "val_total": [], "val_ter": [], "val_gate_video": [], "val_gate_audio": [], "lr": [],
    }

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * epoch / args.warmup_epochs

        train_m = train_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_m, _, _ = evaluate(model, val_loader, criterion, device, id2tok)

        if epoch > args.warmup_epochs:
            scheduler.step(val_m["ter"])

        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_total"].append(train_m.get("total_loss", 0.0))
        history["train_frame"].append(train_m.get("frame_loss", 0.0))
        history["train_ctc"].append(train_m.get("ctc_loss", 0.0))
        history["val_total"].append(val_m.get("total_loss", 0.0))
        history["val_ter"].append(val_m.get("ter", 1.0))
        history["val_gate_video"].append(val_m.get("gate_video", 0.5))
        history["val_gate_audio"].append(val_m.get("gate_audio", 0.5))
        history["lr"].append(cur_lr)

        logger.info(
            "Epoch %03d/%03d | train loss=%.4f (frame=%.4f ctc=%.4f) | val loss=%.4f TER=%.4f "
            "| gate v=%.2f a=%.2f | lr=%.2e",
            epoch, args.epochs, train_m.get("total_loss", 0), train_m.get("frame_loss", 0),
            train_m.get("ctc_loss", 0), val_m.get("total_loss", 0), val_m.get("ter", 1),
            val_m.get("gate_video", 0.5), val_m.get("gate_audio", 0.5), cur_lr,
        )

        if val_m["ter"] < best_ter - 1e-4:
            best_ter = val_m["ter"]
            no_improve = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_ter": best_ter,
                 "vocab": vocab, "config": vars(args)},
                output_dir / "best_model.pt",
            )
            logger.info("  saved best model (val TER %.4f)", best_ter)
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

    torch.save({"model_state_dict": model.state_dict(), "vocab": vocab, "config": vars(args)},
               output_dir / "final_model.pt")
    with (output_dir / "history.json").open("w") as fh:
        json.dump(history, fh, indent=2)

    ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_m, test_preds, test_refs = evaluate(model, test_loader, criterion, device, id2tok)
    logger.info("Test -- loss=%.4f TER=%.4f gate v=%.2f a=%.2f",
                test_m["total_loss"], test_m["ter"], test_m["gate_video"], test_m["gate_audio"])

    phoneme_report(test_preds, test_refs, vocab, output_dir)

    with (output_dir / "test_metrics.json").open("w") as fh:
        json.dump(test_m, fh, indent=2)

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["reference", "prediction"])
        w.writeheader()
        w.writerows({"reference": r, "prediction": p} for r, p in zip(test_refs, test_preds))

    logger.info("Done. Best val TER: %.4f | Test TER: %.4f", best_ter, test_m["ter"])


if __name__ == "__main__":
    main()