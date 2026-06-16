#!/usr/bin/env python3
"""Evaluate lip-reading model: compare predicted vs ground-truth phonemes visually
and acoustically using CTC decoding and Viterbi forced alignment.

Patched to support both the original VisualLSTMFrameCE checkpoint (train_lip_lstm_ctc.py)
and the newer LipReadingModel checkpoint (train_lip_viseme.py).  The script
auto-detects which architecture was saved by inspecting the checkpoint config.

For each sampled test video the script:
  1. Runs the trained CTC model to extract frame-level log-probabilities.
  2. Applies standard CTC greedy decoding to obtain a predicted phoneme sequence.
  3. Uses a Viterbi alignment algorithm to map the predicted token sequence to
     exact video frame boundaries for high-fidelity duration analysis.
  4. Displays a side-by-side terminal table tracking ground-truth vs predicted
     sequences and computes the sequence Levenshtein Token Error Rate (TER).
  5. Synthesises three WAV files using espeak-ng:
       - original_<stem>.wav  : the original audio extracted from the video
       - gt_<stem>.wav        : ground-truth phoneme sequence, frame-accurate durations
       - pred_<stem>.wav      : predicted phoneme sequence, frame-accurate durations
     Each synthesised phoneme segment is time-stretched / compressed to exactly
     fill the duration implied by its frame span (frame_count / fps seconds),
     using librosa's phase-vocoder.

Requirements:
    pip install torch torchvision opencv-python librosa soundfile numpy tqdm
    apt install espeak-ng ffmpeg   # (or brew install on macOS)

Usage (viseme model — train_lip_viseme.py checkpoint):
    python evaluate_lip_lstm.py \
        --checkpoint viseme_results/best_model.pt \
        --vocab-json viseme_results/vocab.json \
        --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
        --video-dir s1_lip_crops \
        --num-videos 5 \
        --output-dir eval_results

Usage (legacy VisualLSTMFrameCE checkpoint):
    python evaluate_lip_lstm.py \
        --checkpoint visual_lstm_ctc_results/best_model.pt \
        --vocab-json visual_lstm_ctc_results/phoneme_vocab.json \
        ...same flags...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BLANK_TOKEN = "<blank>"
SILENCE_TOKEN = ""              # raw per-frame silence label used by viseme CSV
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
SAMPLE_RATE = 22050
DEFAULT_FPS = 25.0
MIN_PHONEME_DURATION = 0.04    # seconds — espeak segments shorter than this are padded
SILENT_TOKENS = {DEFAULT_BLANK_TOKEN, "<missed>", "<inserted>", ""}


# ---------------------------------------------------------------------------
# Architecture A: legacy VisualLSTMFrameCE  (train_lip_lstm_ctc.py)
# ---------------------------------------------------------------------------

class _FrameEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(frames))


class VisualLSTMFrameCE(nn.Module):
    """Original ResNet18 + BiLSTM architecture."""

    def __init__(self, vocab_size: int, hidden_dim: int = 256, lstm_layers: int = 2):
        super().__init__()
        self.frame_encoder = _FrameEncoder(hidden_dim=hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.45 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(
        self,
        videos: torch.Tensor,           # (B, T, C, H, W)
        video_lengths: torch.Tensor,    # (B,)
    ) -> torch.Tensor:                  # (B, T, vocab_size)
        B, T, C, H, W = videos.shape
        flat = videos.reshape(B * T, C, H, W)
        feats = self.frame_encoder(flat).reshape(B, T, -1)
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, video_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Architecture B: LipReadingModel  (train_lip_viseme.py)
# ---------------------------------------------------------------------------

class _Conv3DFrontEnd(nn.Module):
    """Lightweight depthwise-separable 3-D conv frontend."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        def _block(in_c: int, out_c: int, spatial_stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(in_c, in_c,
                          kernel_size=(3, 3, 3),
                          stride=(1, spatial_stride, spatial_stride),
                          padding=(1, 1, 1),
                          groups=in_c, bias=False),
                nn.Conv3d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        self.layer1 = _block(3,   32, spatial_stride=2)
        self.layer2 = _block(32,  64, spatial_stride=2)
        self.layer3 = _block(64, 128, spatial_stride=2)
        self.pool = nn.AdaptiveAvgPool3d((None, 3, 3))
        self.projector = nn.Sequential(
            nn.Linear(128 * 9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # → (B, C, T, H, W)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.pool(x)                              # (B, 128, T, 3, 3)
        B, C, T, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, C * h * w)
        return self.projector(x)                      # (B, T, hidden_dim)


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class LipReadingModel(nn.Module):
    """3D-Conv frontend + positional encoding + BiLSTM/Transformer encoder."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        frontend_dropout: float = 0.3,
        encoder: str = "bilstm",
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()
        self.frontend  = _Conv3DFrontEnd(hidden_dim=hidden_dim, dropout=frontend_dropout)
        self.pos_enc   = _SinusoidalPE(hidden_dim, dropout=0.1)
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
            raise ValueError(f"Unknown encoder: {encoder!r}")

        self.norm       = nn.LayerNorm(enc_dim)
        self.dropout    = nn.Dropout(lstm_dropout)
        self.classifier = nn.Linear(enc_dim, vocab_size + 1)  # +1 for blank at 0

    def forward(
        self,
        videos: torch.Tensor,           # (B, T, C, H, W)
        video_lengths: torch.Tensor,    # (B,)
    ) -> torch.Tensor:                  # (B, T, vocab_size+1)   — logits only
        feat = self.pos_enc(self.frontend(videos))

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

        enc_out = self.dropout(self.norm(enc_out))
        return self.classifier(enc_out)


# ---------------------------------------------------------------------------
# Checkpoint loader — auto-detects architecture
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: Path,
    vocab: List[str],
    device: torch.device,
) -> nn.Module:
    """
    Load whichever architecture was saved in *checkpoint_path*.

    Detection logic:
      - If config contains 'encoder' key  → LipReadingModel  (train_lip_viseme.py)
      - Otherwise                          → VisualLSTMFrameCE (train_lip_lstm_ctc.py)
    """
    ckpt   = torch.load(str(checkpoint_path), map_location=device)
    config = ckpt.get("config", {})

    if "encoder" in config:
        # ---- viseme model ----
        model = LipReadingModel(
            vocab_size          = len(vocab),
            hidden_dim          = config.get("hidden_dim", 256),
            lstm_layers         = config.get("lstm_layers", 2),
            lstm_dropout        = config.get("lstm_dropout", 0.3),
            frontend_dropout    = config.get("lstm_dropout", 0.3),
            encoder             = config.get("encoder", "bilstm"),
            transformer_heads   = config.get("transformer_heads", 4),
            transformer_layers  = config.get("transformer_layers", 2),
        )
        print(f"[INFO] Detected architecture: LipReadingModel "
              f"(encoder={config.get('encoder','bilstm')})")
    else:
        # ---- legacy model ----
        model = VisualLSTMFrameCE(
            vocab_size  = len(vocab) + 1,
            hidden_dim  = config.get("hidden_dim", 256),
            lstm_layers = config.get("lstm_layers", 2),
        )
        print("[INFO] Detected architecture: VisualLSTMFrameCE (legacy)")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Vocab loader — supports both vocab.json formats
# ---------------------------------------------------------------------------

def load_vocab(vocab_json_path: Path) -> Tuple[List[str], Dict[int, str], Dict[str, int]]:
    """
    Returns (vocab, id_to_token, token_to_id).

    Supports two formats:
      • train_lip_viseme.py  : {"tokens": [...], "remap": {...}}
      • train_lip_lstm_ctc.py: {"blank_id": 0, "blank_token": "...", "tokens": [...]}

    In both cases blank is always index 0 and phonemes start at 1.
    """
    with open(str(vocab_json_path), "r", encoding="utf-8") as fh:
        data = json.load(fh)

    vocab = data["tokens"]                          # list of phoneme strings
    id_to_token: Dict[int, str] = {0: DEFAULT_BLANK_TOKEN}
    id_to_token.update({i + 1: tok for i, tok in enumerate(vocab)})
    token_to_id: Dict[str, int] = {tok: i + 1 for i, tok in enumerate(vocab)}
    return vocab, id_to_token, token_to_id


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    video_path: Path
    frame_tokens: List[str]     # per-frame phoneme; blank/silence represented as ""
    sentence: str = ""


def normalize_stem(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_lipcrop")] if stem.endswith("_lipcrop") else stem


def find_lip_videos(data_dir: Path) -> List[Path]:
    return [
        p for p in sorted(data_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]


def load_aligned_phoneme_targets(csv_path: Path) -> Dict[str, List[str]]:
    """
    Reads per_frame_labels from the phoneme CSV.
    Normalises both '' (viseme format) and '<blank>' (lstm_ctc format) to ''
    so downstream code is format-agnostic.
    """
    targets: Dict[str, List[str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = row.get("stem", "").strip()
            raw  = row.get("per_frame_labels", "").strip()
            if not stem or not raw:
                continue
            try:
                raw_labels = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Normalise: None / whitespace / "<blank>" → "" (silence)
            tokens = [
                ""
                if (lbl is None or str(lbl).strip() in ("", DEFAULT_BLANK_TOKEN))
                else str(lbl).strip()
                for lbl in raw_labels
            ]
            targets[stem] = tokens
    return targets


def build_test_samples(
    video_dir: Path,
    phoneme_csv: Path,
    seed: int,
    test_ratio: float = 0.2,
) -> List[Sample]:
    targets     = load_aligned_phoneme_targets(phoneme_csv)
    all_samples: List[Sample] = []

    for vp in find_lip_videos(video_dir):
        key = normalize_stem(vp)
        if key in targets:
            all_samples.append(Sample(video_path=vp, frame_tokens=targets[key]))

    if not all_samples:
        raise FileNotFoundError(
            f"No paired samples found in {video_dir} / {phoneme_csv}"
        )

    rng      = np.random.RandomState(seed)
    indices  = rng.permutation(len(all_samples))
    n_test   = max(1, int(len(all_samples) * test_ratio))
    test_idx = indices[len(all_samples) - n_test :]
    return [all_samples[i] for i in sorted(test_idx)]


# ---------------------------------------------------------------------------
# Run-length encoding
# ---------------------------------------------------------------------------

@dataclass
class PhonemeRun:
    token: str
    start_frame: int
    end_frame: int          # exclusive

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    def duration(self, fps: float) -> float:
        return self.num_frames / fps


def run_length_encode(frame_tokens: Sequence[str]) -> List[PhonemeRun]:
    if not frame_tokens:
        return []
    runs: List[PhonemeRun] = []
    current = frame_tokens[0]
    start   = 0
    for i, tok in enumerate(frame_tokens[1:], start=1):
        if tok != current:
            runs.append(PhonemeRun(current, start, i))
            current = tok
            start   = i
    runs.append(PhonemeRun(current, start, len(frame_tokens)))
    return runs


# ---------------------------------------------------------------------------
# CTC greedy decoding
# ---------------------------------------------------------------------------

def decode_greedy_ctc(
    logits: torch.Tensor,
    id_to_token: Dict[int, str],
    blank_id: int = 0,
) -> List[str]:
    """
    True CTC decoding: collapse repeated IDs first, then drop blanks.
    Returns a list of non-blank phoneme strings.
    """
    pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
    tokens: List[str] = []
    prev_id = -1
    for token_id in pred_ids:
        if token_id != prev_id:
            if token_id != blank_id:
                tok = id_to_token.get(token_id, DEFAULT_BLANK_TOKEN)
                if tok not in SILENT_TOKENS:
                    tokens.append(tok)
            prev_id = token_id
    return tokens


# ---------------------------------------------------------------------------
# Levenshtein TER
# ---------------------------------------------------------------------------

def compute_levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        curr = [i]
        for j, b in enumerate(right, start=1):
            curr.append(min(curr[j-1]+1, prev[j]+1, prev[j-1]+(a != b)))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Viterbi forced alignment
# ---------------------------------------------------------------------------

def viterbi_forced_alignment(
    log_probs: torch.Tensor,        # (T, V+1)  on CPU
    target_tokens: List[str],
    token_to_id: Dict[str, int],
    blank_id: int = 0,
) -> List[int]:
    """
    Standard CTC Viterbi forced alignment.
    Returns one integer token-ID per input frame.
    """
    num_frames = log_probs.size(0)
    target_ids = [token_to_id[tok] for tok in target_tokens if tok in token_to_id]

    if not target_ids:
        return [blank_id] * num_frames

    # Interleave blanks: b t1 b t2 b … tN b
    states: List[int] = [blank_id]
    for tid in target_ids:
        states.append(tid)
        states.append(blank_id)
    num_states = len(states)

    NEG_INF = float("-inf")
    trellis     = [[NEG_INF] * num_states for _ in range(num_frames)]
    backptrs    = [[0]       * num_states for _ in range(num_frames)]

    trellis[0][0] = float(log_probs[0, states[0]])
    if num_states > 1:
        trellis[0][1] = float(log_probs[0, states[1]])

    for t in range(1, num_frames):
        lp = log_probs[t]
        for s in range(num_states):
            best_prev = s
            best_val  = trellis[t-1][s]

            if s > 0 and trellis[t-1][s-1] > best_val:
                best_prev = s - 1
                best_val  = trellis[t-1][s-1]

            # Skip-blank transition: s-2 is allowed when current is not blank
            # and states[s-1] is blank and states[s] ≠ states[s-2]
            if (s > 1
                    and states[s] != blank_id
                    and states[s-1] == blank_id
                    and states[s] != states[s-2]
                    and trellis[t-1][s-2] > best_val):
                best_prev = s - 2
                best_val  = trellis[t-1][s-2]

            trellis[t][s] = best_val + float(lp[states[s]])
            backptrs[t][s] = best_prev

    # Backtrack from best final state (last blank or last phoneme)
    last = num_states - 1
    best_final = last if trellis[-1][last] >= trellis[-1][last-1] else last - 1

    path: List[int] = []
    s = best_final
    for t in range(num_frames - 1, -1, -1):
        path.append(states[s])
        s = backptrs[t][s]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video_frames(
    video_path: Path,
    frame_size: int = 96,
) -> Tuple[torch.Tensor, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
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
        raise RuntimeError(f"No frames extracted from {video_path}")

    transform = v2.Compose([
        v2.Resize((frame_size, frame_size),
                  interpolation=v2.InterpolationMode.BILINEAR,
                  antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(torch.stack(frames, dim=0))  # (T, C, H, W)
    return tensor, fps


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------

def check_espeak() -> str:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        sys.exit("espeak-ng not found on PATH. Install with: apt install espeak-ng")
    return exe


def synth_phoneme_segment(
    token: str,
    duration_sec: float,
    espeak_exe: str,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    n_samples = max(1, int(round(duration_sec * sr)))
    if token in SILENT_TOKENS:
        return np.zeros(n_samples, dtype=np.float32)

    tmp_path: Optional[str] = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        # Try IPA mode first, then SAMPA bracket fallback
        result = subprocess.run(
            [espeak_exe, "--ipa=1", "-q", "--stdout", "-s", "150", f"/{token}/"],
            capture_output=True,
        )
        if result.returncode != 0 or not result.stdout:
            result = subprocess.run(
                [espeak_exe, "-q", "--stdout", f"[[{token}]]"],
                capture_output=True,
            )
        if not result.stdout:
            return np.zeros(n_samples, dtype=np.float32)

        with open(tmp_path, "wb") as fh:
            fh.write(result.stdout)
        raw, _ = librosa.load(tmp_path, sr=sr, mono=True)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if raw is None or len(raw) == 0:
        return np.zeros(n_samples, dtype=np.float32)

    target_dur = max(duration_sec, MIN_PHONEME_DURATION)
    raw_dur    = len(raw) / sr
    if abs(raw_dur - target_dur) > 0.005:
        rate = float(np.clip(raw_dur / target_dur, 0.2, 5.0))
        try:
            raw = librosa.effects.time_stretch(raw, rate=rate)
        except Exception:
            pass

    if len(raw) >= n_samples:
        return raw[:n_samples].astype(np.float32)
    return np.concatenate([raw, np.zeros(n_samples - len(raw))]).astype(np.float32)


def synthesise_runs(
    runs: List[PhonemeRun],
    fps: float,
    espeak_exe: str,
    sr: int = SAMPLE_RATE,
    total_frames: Optional[int] = None,
) -> np.ndarray:
    if total_frames is not None:
        total_samples = int(round(total_frames / fps * sr))
    else:
        total_samples = int(round(sum(r.duration(fps) for r in runs) * sr))

    output    = np.zeros(total_samples, dtype=np.float32)
    write_pos = 0

    for run in runs:
        dur     = run.duration(fps)
        n       = int(round(dur * sr))
        segment = synth_phoneme_segment(run.token, dur, espeak_exe, sr=sr)
        end     = min(write_pos + len(segment), total_samples)
        output[write_pos:end] = segment[: end - write_pos]
        write_pos += n

    return output


def extract_original_audio(
    video_path: Path,
    sr: int = SAMPLE_RATE,
) -> Optional[np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    tmp_path: Optional[str] = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        cmd = [
            ffmpeg, "-y", "-i", str(video_path),
            "-ar", str(sr), "-ac", "1", "-vn", "-loglevel", "error",
            tmp_path,
        ]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return None
        audio, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return audio
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def print_comparison(
    stem: str,
    gt_seq: List[str],
    pred_seq: List[str],
    ter: float,
) -> None:
    width = shutil.get_terminal_size((120, 40)).columns
    sep   = "─" * width
    print(f"\n{sep}")
    print(f"\033[1;36m  Clip: {stem}\033[0m")
    print(sep)
    print(f"\033[1;32m  Ground Truth:\033[0m  {' '.join(gt_seq) if gt_seq else '(empty)'}")
    print(f"\033[1;33m  Prediction:  \033[0m  {' '.join(pred_seq) if pred_seq else '(empty)'}")
    print(f"  TER: {ter:.2%}")
    print(sep)


# ---------------------------------------------------------------------------
# Collapse ground-truth frame tokens → phoneme sequence
# ---------------------------------------------------------------------------

def collapse_gt_frames(frame_tokens: Sequence[str]) -> List[str]:
    """
    Remove silence/blank tokens and collapse consecutive duplicates,
    matching how CTC greedy decoding collapses the prediction side.
    """
    result: List[str] = []
    prev: Optional[str] = None
    for tok in frame_tokens:
        if tok in SILENT_TOKENS:
            prev = None
            continue
        if tok != prev:
            result.append(tok)
            prev = tok
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lip-reading evaluator with Viterbi alignment — supports "
                    "both VisualLSTMFrameCE and LipReadingModel checkpoints"
    )
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--vocab-json",   required=True,
                        help="Path to vocab.json (viseme) or phoneme_vocab.json (legacy)")
    parser.add_argument("--phoneme-csv",  default="phonemes_s1_aligned/phoneme_predictions.csv")
    parser.add_argument("--video-dir",    default="s1_lip_crops")
    parser.add_argument("--output-dir",   default="eval_results")
    parser.add_argument("--num-videos",   type=int,   default=5)
    parser.add_argument("--frame-size",   type=int,   default=96,
                        help="Must match the frame_size used during training "
                             "(96 for viseme model, 224 for legacy)")
    parser.add_argument("--sample-rate",  type=int,   default=SAMPLE_RATE)
    parser.add_argument("--test-ratio",   type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no-audio",     action="store_true",
                        help="Skip espeak synthesis and WAV export")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # -- Vocab --
    vocab, id_to_token, token_to_id = load_vocab(Path(args.vocab_json))
    print(f"[INFO] Vocabulary: {len(vocab)} phonemes")

    # -- Model --
    model = load_model_from_checkpoint(Path(args.checkpoint), vocab, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {n_params:,}")

    # -- Data --
    test_samples = build_test_samples(
        Path(args.video_dir),
        Path(args.phoneme_csv),
        seed=args.seed,
        test_ratio=args.test_ratio,
    )
    print(f"[INFO] Test samples: {len(test_samples)} (evaluating first {args.num_videos})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    espeak_exe = check_espeak() if not args.no_audio else None

    all_gt:   List[List[str]] = []
    all_pred: List[List[str]] = []

    # -- Per-video evaluation --
    for sample in tqdm(test_samples[: args.num_videos], desc="Evaluating"):
        stem = normalize_stem(sample.video_path)
        try:
            video, fps = load_video_frames(sample.video_path, frame_size=args.frame_size)
        except Exception as exc:
            print(f"[WARN] Skipping {stem}: {exc}")
            continue

        # Forward pass — model returns logits (B, T, V+1)
        with torch.no_grad():
            v_in  = video.unsqueeze(0).to(device)           # (1, T, C, H, W)
            v_len = torch.tensor([video.size(0)], dtype=torch.long, device=device)
            logits = model(v_in, v_len).squeeze(0)          # (T, V+1)
            log_probs = F.log_softmax(logits, dim=-1).cpu() # (T, V+1)

        # Sequences
        gt_collapsed   = collapse_gt_frames(sample.frame_tokens)
        pred_collapsed = decode_greedy_ctc(logits.cpu(), id_to_token, blank_id=0)

        dist = compute_levenshtein_distance(gt_collapsed, pred_collapsed)
        ter  = dist / max(1, len(gt_collapsed))

        all_gt.append(gt_collapsed)
        all_pred.append(pred_collapsed)

        print_comparison(stem, gt_collapsed, pred_collapsed, ter)

        if args.no_audio:
            continue

        # Original audio
        orig = extract_original_audio(sample.video_path, sr=args.sample_rate)
        if orig is not None:
            sf.write(
                str(output_dir / f"original_{stem}.wav"),
                orig, args.sample_rate, subtype="PCM_16",
            )

        # Viterbi alignment → run-length encode → synthesise
        gt_path   = viterbi_forced_alignment(log_probs, gt_collapsed,   token_to_id, blank_id=0)
        pred_path = viterbi_forced_alignment(log_probs, pred_collapsed, token_to_id, blank_id=0)

        gt_runs   = run_length_encode([id_to_token[i] for i in gt_path])
        pred_runs = run_length_encode([id_to_token[i] for i in pred_path])

        gt_audio   = synthesise_runs(gt_runs,   fps, espeak_exe,
                                     sr=args.sample_rate, total_frames=len(gt_path))
        pred_audio = synthesise_runs(pred_runs, fps, espeak_exe,
                                     sr=args.sample_rate, total_frames=len(pred_path))

        sf.write(str(output_dir / f"gt_{stem}.wav"),
                 gt_audio,   args.sample_rate, subtype="PCM_16")
        sf.write(str(output_dir / f"pred_{stem}.wav"),
                 pred_audio, args.sample_rate, subtype="PCM_16")

    # -- Aggregate TER --
    if all_gt:
        total_errs  = sum(
            compute_levenshtein_distance(p, r)
            for p, r in zip(all_pred, all_gt)
        )
        total_toks  = sum(max(1, len(r)) for r in all_gt)
        overall_ter = total_errs / total_toks
        print(f"\n[RESULT] Overall TER across {len(all_gt)} clip(s): {overall_ter:.2%}")

        # Save summary CSV
        summary_path = output_dir / "evaluation_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["clip", "gt", "prediction", "ter"])
            writer.writeheader()
            for i, (gt, pred) in enumerate(zip(all_gt, all_pred)):
                d   = compute_levenshtein_distance(pred, gt)
                ter = d / max(1, len(gt))
                writer.writerow({
                    "clip":       i,
                    "gt":         " ".join(gt),
                    "prediction": " ".join(pred),
                    "ter":        f"{ter:.4f}",
                })
        print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()