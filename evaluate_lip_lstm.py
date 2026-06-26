#!/usr/bin/env python3
"""Evaluate lip-reading LSTM: compare predicted vs ground-truth phonemes visually
and acoustically using CTC decoding and Viterbi forced alignment.

For each sampled test video the script:
  1. Runs the trained CTC model to extract frame-level log-probabilities.
  2. Applies standard CTC greedy decoding to obtain a predicted phoneme sequence.
  3. Uses a Viterbi alignment algorithm to map the predicted token sequence to 
     exact video frame boundaries for high-fidelity duration analysis.
  4. Displays a side-by-side terminal table tracking ground-truth vs predicted sequences
     and computes the sequence Levenshtein Token Error Rate (TER).
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

Usage:
    python evaluate_lip_lstm.py --checkpoint visual_lstm_ctc_results/best_model.pt \
                                --vocab-json visual_lstm_ctc_results/phoneme_vocab.json \
                                --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
                                --video-dir s1_lip_crops \
                                --num-videos 5 \
                                --output-dir eval_results
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
from torchvision.transforms import v2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BLANK_TOKEN = "<blank>"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
SAMPLE_RATE = 22050          # output WAV sample rate
DEFAULT_FPS = 29.97          # NTSC standard for exact audio-visual synchronization
MIN_PHONEME_DURATION = 0.04  # seconds – espeak segments shorter than this are padded

# IPA tokens that espeak-ng cannot synthesise in isolation get a short silence.
SILENT_TOKENS = {DEFAULT_BLANK_TOKEN, "<missed>", "<inserted>"}

# Characters that would break out of the SSML ph="..." attribute if left
# unescaped. espeak-ng's IPA phoneme strings should never legitimately
# contain these, but we escape defensively since tokens come from model
# vocab / CSV data rather than a hard-coded list.
_SSML_ESCAPES = {
    "&": "&amp;",
    '"': "&quot;",
    "<": "&lt;",
    ">": "&gt;",
}


def _escape_ssml_attr(token: str) -> str:
    out = token
    for char, escaped in _SSML_ESCAPES.items():
        out = out.replace(char, escaped)
    return out


# ---------------------------------------------------------------------------
# Model Architecture
#
# FIXED: this used to define a ResNet18 + LSTM model ("VisualLSTMFrameCE")
# that does NOT match what train_lip_viseme.py actually trains. The real
# checkpoint architecture is Conv3DFrontEnd -> sinusoidal positional encoding
# -> BiLSTM/Transformer ("LipReadingModel"), which is why loading best_model.pt
# raised "Missing key(s)" / "Unexpected key(s)" — the two architectures share
# no parameter names. These classes are copied verbatim (structurally) from
# train_lip_viseme.py so the state_dict keys match exactly.
# ---------------------------------------------------------------------------

class Conv3DFrontEnd(nn.Module):
    """Lightweight 3D convolutional frontend. (B, T, C, H, W) -> (B, T, hidden_dim)."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        def _block(in_c: int, out_c: int, spatial_stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(in_c, in_c,
                          kernel_size=(3, 3, 3),
                          stride=(1, spatial_stride, spatial_stride),
                          padding=(1, 1, 1),
                          groups=in_c,
                          bias=False),
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
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        B, C, T, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, C * h * w)
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


class LipReadingModel(nn.Module):
    """3D-Conv frontend -> positional encoding -> BiLSTM/Transformer -> frame classifier.

    forward() returns (logits, log_probs_ctc, out_lengths) to match the
    training script's signature.
    """

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
        self.frontend = Conv3DFrontEnd(hidden_dim=hidden_dim, dropout=frontend_dropout)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, dropout=0.1)

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

        self.norm = nn.LayerNorm(enc_dim)
        self.dropout = nn.Dropout(lstm_dropout)
        self.classifier = nn.Linear(enc_dim, vocab_size + 1)  # +1 for blank at index 0

    def forward(
        self,
        videos: torch.Tensor,
        video_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.frontend(videos)
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
        logits = self.classifier(enc_out)
        log_probs_ctc = F.log_softmax(logits, dim=-1).transpose(0, 1)

        return logits, log_probs_ctc, video_lengths


# ---------------------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    video_path: Path
    frame_tokens: List[str]
    sentence: str = ""
    canonical_tokens: List[str] = field(default_factory=list)


def normalize_stem(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_lipcrop")] if stem.endswith("_lipcrop") else stem


def find_lip_videos(data_dir: Path) -> List[Path]:
    return [p for p in sorted(data_dir.rglob("*")) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def load_aligned_phoneme_targets(csv_path: Path) -> Dict[str, List[str]]:
    targets: Dict[str, List[str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = row.get("stem", "").strip()
            raw = row.get("per_frame_labels", "").strip()
            if not stem or not raw:
                continue
            try:
                tokens = [str(t).strip() or DEFAULT_BLANK_TOKEN for t in json.loads(raw)]
            except json.JSONDecodeError:
                continue
            targets[stem] = tokens
    return targets


def build_test_samples(video_dir: Path, phoneme_csv: Path, seed: int, test_ratio: float = 0.2) -> List[Sample]:
    targets = load_aligned_phoneme_targets(phoneme_csv)
    all_samples: List[Sample] = []
    for vp in find_lip_videos(video_dir):
        key = normalize_stem(vp)
        if key in targets:
            all_samples.append(Sample(video_path=vp, frame_tokens=targets[key]))
    if not all_samples:
        raise FileNotFoundError(f"No paired samples in {video_dir} / {phoneme_csv}")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_samples))
    n_test = max(1, int(len(all_samples) * test_ratio))
    test_indices = indices[len(all_samples) - n_test :]
    return [all_samples[i] for i in sorted(test_indices)]


# ---------------------------------------------------------------------------
# Run-Length Encodings & Levenshtein Metrics
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
    """Collapse consecutive identical tokens into structured segments."""
    if not frame_tokens:
        return []
    runs: List[PhonemeRun] = []
    current = frame_tokens[0]
    start = 0
    for i, tok in enumerate(frame_tokens[1:], start=1):
        if tok != current:
            runs.append(PhonemeRun(current, start, i))
            current = tok
            start = i
    runs.append(PhonemeRun(current, start, len(frame_tokens)))
    return runs


def decode_greedy_ctc(logits: torch.Tensor, id_to_token: Dict[int, str], blank_id: int = 0) -> List[str]:
    """CHANGED: True CTC decoding: collapse identical adjacent frames *before* dropping blanks."""
    pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
    collapsed_tokens: List[str] = []
    previous_id = -1
    for token_id in pred_ids:
        if token_id != previous_id:
            token_str = id_to_token.get(token_id, DEFAULT_BLANK_TOKEN)
            if token_str != DEFAULT_BLANK_TOKEN:
                collapsed_tokens.append(token_str)
        previous_id = token_id
    return collapsed_tokens


def compute_levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if not left: return len(right)
    if not right: return len(left)
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


# ---------------------------------------------------------------------------
# CHANGED: Monotonic Viterbi Forced Alignment Algorithm
# ---------------------------------------------------------------------------

def viterbi_forced_alignment(
    log_probs: torch.Tensor, 
    target_tokens: List[str], 
    token_to_id: Dict[str, int], 
    blank_id: int = 0
) -> List[int]:
    """Forces emission logits to map structurally against a target sequence.
    
    Returns:
        A list of integer IDs mapping each individual frame sequentially to a token.
    """
    num_frames = log_probs.size(0)
    target_ids = [token_to_id[tok] for tok in target_tokens if tok in token_to_id]
    
    # Interleave blanks into states
    states = [blank_id]
    for tid in target_ids:
        states.append(tid)
        states.append(blank_id)
    num_states = len(states)

    trellis = torch.full((num_frames, num_states), float("-inf"))
    backpointers = torch.zeros((num_frames, num_states), dtype=torch.long)

    # Init
    trellis[0, 0] = log_probs[0, states[0]]
    if num_states > 1:
        trellis[0, 1] = log_probs[0, states[1]]

    # Forward Recursion
    for t in range(1, num_frames):
        for s in range(num_states):
            log_p = log_probs[t, states[s]]
            best_prev_s = s
            best_prob = trellis[t - 1, s]

            if s > 0 and trellis[t - 1, s - 1] > best_prob:
                best_prev_s = s - 1
                best_prob = trellis[t - 1, s - 1]

            if s > 1 and states[s] != blank_id and states[s - 1] == blank_id and states[s] != states[s - 2]:
                if trellis[t - 1, s - 2] > best_prob:
                    best_prev_s = s - 2
                    best_prob = trellis[t - 1, s - 2]

            trellis[t, s] = best_prob + log_p
            backpointers[t, s] = best_prev_s

    # Backtracking Pass
    best_final_s = num_states - 1 if trellis[-1, -1] > trellis[-1, -2] else num_states - 2
    
    path = []
    current_s = best_final_s
    for t in range(num_frames - 1, -1, -1):
        path.append(states[current_s])
        current_s = int(backpointers[t, current_s])
    
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Video Loading & Video Inference Processing
# ---------------------------------------------------------------------------

def load_video_frames(video_path: Path, frame_size: int = 224) -> Tuple[torch.Tensor, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    transform = v2.Compose([
        v2.Resize((frame_size, frame_size), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(torch.stack(frames, dim=0))
    return tensor, fps


# ---------------------------------------------------------------------------
# Synthesis & Audio Generation Track Exporters
# ---------------------------------------------------------------------------

def check_espeak() -> str:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        sys.exit("espeak-ng utility binary not located on PATH variable hierarchy.")
    return exe


def synth_phoneme_segment(token: str, duration_sec: float, espeak_exe: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    n_samples = max(1, int(round(duration_sec * sr)))
    if token in SILENT_TOKENS:
        return np.zeros(n_samples, dtype=np.float32)

    # Two temp files: one for the SSML input, one for the WAV output.
    #
    # Why not `--stdout`?  On Windows, espeak-ng's stdout pipe can produce
    # corrupt or truncated WAV data due to text-mode buffering in cmd.exe /
    # CreateProcess, causing librosa to silently return empty audio.
    #
    # Why not pass SSML as a CLI argument?  Python's subprocess serialises the
    # argument list through Windows CreateProcess quoting rules, which breaks
    # strings that contain both single and double quotes (our SSML has `"`
    # inside the `ph="..."` attribute and `<`/`>` angle brackets).  Writing
    # the SSML to a file and passing `-f <file>` sidesteps this entirely.
    #
    # Why SSML / `<phoneme alphabet="ipa">`?  espeak-ng's `[[...]]` bracket
    # mode uses its own ASCII mnemonic alphabet, not Unicode IPA.  The only
    # supported way to feed raw IPA glyphs (ɪ, ʃ, θ, …) for synthesis is
    # SSML with `alphabet="ipa"`, enabled by the `-m` flag.
    ssml_path: Optional[str] = None
    wav_path: Optional[str] = None
    raw: Optional[np.ndarray] = None

    try:
        escaped_token = _escape_ssml_attr(token)
        ssml = f'<phoneme alphabet="ipa" ph="{escaped_token}">x</phoneme>'

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as ssml_f:
            ssml_path = ssml_f.name
            ssml_f.write(ssml)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
            wav_path = wav_f.name

        cmd = [espeak_exe, "-m", "-s", "150", "-f", ssml_path, "-w", wav_path]
        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            return np.zeros(n_samples, dtype=np.float32)

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            return np.zeros(n_samples, dtype=np.float32)

        raw, _ = librosa.load(wav_path, sr=sr, mono=True)

    finally:
        for p in (ssml_path, wav_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    if raw is None or len(raw) == 0:
        return np.zeros(n_samples, dtype=np.float32)

    raw_dur = len(raw) / sr
    target_dur = max(duration_sec, MIN_PHONEME_DURATION)

    if abs(raw_dur - target_dur) < 0.005:
        stretched = raw
    else:
        rate = float(np.clip(raw_dur / target_dur, 0.2, 5.0))
        try:
            stretched = librosa.effects.time_stretch(raw, rate=rate)
        except Exception:
            stretched = raw

    if len(stretched) >= n_samples:
        return stretched[:n_samples].astype(np.float32)
    return np.concatenate([stretched, np.zeros(n_samples - len(stretched))]).astype(np.float32)


def synthesise_runs(runs: List[PhonemeRun], fps: float, espeak_exe: str, sr: int = SAMPLE_RATE, total_frames: Optional[int] = None) -> np.ndarray:
    if total_frames is not None:
        total_samples = int(round(total_frames / fps * sr))
    else:
        total_samples = int(round(sum(r.duration(fps) for r in runs) * sr))

    output = np.zeros(total_samples, dtype=np.float32)
    write_pos = 0
    for run in runs:
        dur = run.duration(fps)
        n = int(round(dur * sr))
        segment = synth_phoneme_segment(run.token, dur, espeak_exe, sr=sr)
        end = min(write_pos + len(segment), total_samples)
        output[write_pos:end] = segment[:(end - write_pos)]
        write_pos += n
    return output


def extract_original_audio(video_path: Path, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None: return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [ffmpeg, "-y", "-i", str(video_path), "-ar", str(sr), "-ac", "1", "-vn", "-loglevel", "error", tmp_path]
        if subprocess.run(cmd, capture_output=True).returncode != 0: return None
        audio, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return audio
    except: return None
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass


# ---------------------------------------------------------------------------
# Visual Formatting Diagnostics
# ---------------------------------------------------------------------------

def print_comparison(stem: str, gt_seq: List[str], pred_seq: List[str], ter: float) -> None:
    sep = "─" * shutil.get_terminal_size((120, 40)).columns
    print(f"\n{sep}")
    print(f"\033[1;36m  Clip Identifier: {stem}\033[0m")
    print(sep)
    print(f"\033[1;32m  Ground Truth Sequence:\033[0m  {' '.join(gt_seq) if gt_seq else '(empty)'}")
    print(f"\033[1;33m  Greedy CTC Prediction:\033[0m  {' '.join(pred_seq) if pred_seq else '(empty)'}")
    print(f"  Sequence Token Error Rate (TER): {ter:.2%}")
    print(sep)


# ---------------------------------------------------------------------------
# Execution Block Initialization
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="CTC Lip-reading Evaluator with Viterbi Durations")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    p.add_argument("--vocab-json", type=str, required=True, help="Path to phoneme_vocab.json")
    p.add_argument("--phoneme-csv", type=str, default="phonemes_s1_aligned/phoneme_predictions.csv")
    p.add_argument("--video-dir", type=str, default="s1_lip_crops")
    p.add_argument("--output-dir", type=str, default="eval_results")
    p.add_argument("--num-videos", type=int, default=5)
    p.add_argument("--frame-size", type=int, default=None,
                    help="Spatial resolution fed to the model. Defaults to the value stored "
                         "in the checkpoint's training config if not given.")
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-audio", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.vocab_json, "r", encoding="utf-8") as fh:
        vocab_data = json.load(fh)
    vocab = vocab_data["tokens"]

    token_to_id = {token: idx + 1 for idx, token in enumerate(vocab)}
    id_to_token = {idx + 1: token for idx, token in enumerate(vocab)}
    id_to_token[0] = DEFAULT_BLANK_TOKEN

    # Instantiate and restore architecture
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})

    frame_size = args.frame_size if args.frame_size is not None else cfg.get("frame_size", 96)

    model = LipReadingModel(
        vocab_size=len(vocab),
        hidden_dim=cfg.get("hidden_dim", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
        lstm_dropout=cfg.get("lstm_dropout", 0.3),
        frontend_dropout=cfg.get("lstm_dropout", 0.3),
        encoder=cfg.get("encoder", "bilstm"),
        transformer_heads=cfg.get("transformer_heads", 4),
        transformer_layers=cfg.get("transformer_layers", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    test_samples = build_test_samples(Path(args.video_dir), Path(args.phoneme_csv), seed=args.seed, test_ratio=args.test_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    espeak_exe = check_espeak() if not args.no_audio else None

    for sample in tqdm(test_samples[:args.num_videos], desc="Evaluating Sequences"):
        stem = normalize_stem(sample.video_path)
        try: video, fps = load_video_frames(sample.video_path, frame_size=frame_size)
        except Exception as e: continue

        # Forward Pass
        with torch.no_grad():
            v_input = video.unsqueeze(0).to(device)
            v_len = torch.tensor([video.size(0)], dtype=torch.long, device=device)
            logits, _, _ = model(v_input, v_len)
            logits = logits.squeeze(0)  # [T, Classes]
            log_probs = F.log_softmax(logits, dim=-1)

        # Standard clean targets (remove ground-truth sequential frame duplicate stretches)
        gt_collapsed = [t for i, t in enumerate(sample.frame_tokens) if t != DEFAULT_BLANK_TOKEN and (i == 0 or t != sample.frame_tokens[i-1])]
        pred_collapsed = decode_greedy_ctc(logits, id_to_token, blank_id=0)

        dist = compute_levenshtein_distance(gt_collapsed, pred_collapsed)
        ter = dist / max(1, len(gt_collapsed))

        print_comparison(stem, gt_collapsed, pred_collapsed, ter)

        if args.no_audio: continue

        # Save baseline audio track
        orig_audio = extract_original_audio(sample.video_path, sr=args.sample_rate)
        if orig_audio is not None:
            sf.write(str(output_dir / f"original_{stem}.wav"), orig_audio, args.sample_rate, subtype="PCM_16")

        # CHANGED: Derive forced timelines tracking paths using explicit Viterbi logic
        gt_frame_path = viterbi_forced_alignment(log_probs, gt_collapsed, token_to_id, blank_id=0)
        pred_frame_path = viterbi_forced_alignment(log_probs, pred_collapsed, token_to_id, blank_id=0)

        # Convert back to clean run representations
        gt_string_frames = [id_to_token[i] for i in gt_frame_path]
        pred_string_frames = [id_to_token[i] for i in pred_frame_path]

        gt_runs = run_length_encode(gt_string_frames)
        pred_runs = run_length_encode(pred_string_frames)

        # Synthesize target acoustic WAV configurations
        gt_audio = synthesise_runs(gt_runs, fps, espeak_exe, sr=args.sample_rate, total_frames=len(gt_frame_path))
        sf.write(str(output_dir / f"gt_{stem}.wav"), gt_audio, args.sample_rate, subtype="PCM_16")

        pred_audio = synthesise_runs(pred_runs, fps, espeak_exe, sr=args.sample_rate, total_frames=len(pred_frame_path))
        sf.write(str(output_dir / f"pred_{stem}.wav"), pred_audio, args.sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    main()