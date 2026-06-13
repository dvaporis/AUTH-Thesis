#!/usr/bin/env python3
"""Evaluate lip-reading LSTM: compare predicted vs ground-truth phonemes visually
and acoustically.

For each sampled test video the script:
  1. Runs the trained model to get per-frame predicted phoneme IDs.
  2. Collapses repeated IDs and blanks to a phoneme sequence (greedy decode).
  3. Displays a side-by-side terminal table: ground-truth vs predicted,
     frame-level and collapsed.
  4. Synthesises three WAV files using espeak-ng:
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
    python evaluate_lip_lstm.py --checkpoint visual_lstm_ce_results/best_model.pt \\
                                --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \\
                                --video-dir s1_lip_crops \\
                                --num-videos 5 \\
                                --output-dir eval_results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BLANK_TOKEN = "<blank>"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
SAMPLE_RATE = 22050          # output WAV sample rate
DEFAULT_FPS = 25.0           # fallback if video metadata is unavailable
MIN_PHONEME_DURATION = 0.04  # seconds – espeak segments shorter than this are padded

# IPA tokens that espeak-ng cannot synthesise in isolation get a short silence.
SILENT_TOKENS = {DEFAULT_BLANK_TOKEN, "<missed>", "<inserted>"}


# ---------------------------------------------------------------------------
# Model (mirrors train_lip_lstm.py exactly so the checkpoint loads cleanly)
# ---------------------------------------------------------------------------

class FrameEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, pretrained: bool = False, finetune: bool = True):
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
    def __init__(self, vocab_size: int, hidden_dim: int = 256, lstm_layers: int = 2):
        super().__init__()
        self.frame_encoder = FrameEncoder(hidden_dim=hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.45 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, videos: torch.Tensor, video_lengths: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = videos.shape
        flat = videos.reshape(B * T, C, H, W)
        feats = self.frame_encoder(flat).reshape(B, T, -1)
        packed = nn.utils.rnn.pack_padded_sequence(feats, video_lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Data helpers (subset of train_lip_lstm.py)
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
    # Use the same tail slice that train_lip_lstm.py uses for the test split.
    test_indices = indices[len(all_samples) - n_test :]
    return [all_samples[i] for i in sorted(test_indices)]


# ---------------------------------------------------------------------------
# Video loading & inference
# ---------------------------------------------------------------------------

def load_video_frames(video_path: Path, frame_size: int = 224) -> Tuple[torch.Tensor, float]:
    """Return (frames [T,C,H,W] float32 normalised, fps)."""
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
        raise RuntimeError(f"No frames in {video_path}")

    transform = v2.Compose([
        v2.Resize((frame_size, frame_size), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(torch.stack(frames, dim=0))  # [T,C,H,W]
    return tensor, fps


@torch.no_grad()
def predict_frame_tokens(
    model: VisualLSTMFrameCE,
    video: torch.Tensor,
    id_to_token: Dict[int, str],
    device: torch.device,
    blank_id: int = 0,
) -> List[str]:
    """Return a per-frame token list (same length as video)."""
    video = video.unsqueeze(0).to(device)           # [1,T,C,H,W]
    lengths = torch.tensor([video.shape[1]], dtype=torch.long)
    logits = model(video, lengths)                  # [1,T,vocab]
    pred_ids = torch.argmax(logits[0], dim=-1).cpu().tolist()
    return [id_to_token.get(int(i), DEFAULT_BLANK_TOKEN) for i in pred_ids]


# ---------------------------------------------------------------------------
# Phoneme run-length encoding
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
    """Collapse consecutive identical tokens into runs."""
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


def collapsed_sequence(frame_tokens: Sequence[str]) -> List[str]:
    """CTC-style collapse: remove blanks, then deduplicate adjacent identical tokens."""
    no_blank = [t for t in frame_tokens if t != DEFAULT_BLANK_TOKEN]
    out: List[str] = []
    for tok in no_blank:
        if not out or tok != out[-1]:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# espeak-ng synthesis helpers
# ---------------------------------------------------------------------------

def check_espeak() -> str:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        sys.exit(
            "espeak-ng not found. Install with:\n"
            "  Linux : sudo apt install espeak-ng\n"
            "  macOS : brew install espeak-ng\n"
            "  Windows: https://github.com/espeak-ng/espeak-ng/releases"
        )
    return exe


def synth_phoneme_segment(
    token: str,
    duration_sec: float,
    espeak_exe: str,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesise a single IPA phoneme with espeak-ng, then time-stretch it
    to exactly `duration_sec` seconds using librosa's phase vocoder.

    Returns a float32 mono array at sample rate `sr`.
    """
    n_samples = max(1, int(round(duration_sec * sr)))

    # Silent tokens → silence
    if token in SILENT_TOKENS:
        return np.zeros(n_samples, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # --ipa=1 tells espeak-ng the input is IPA text.
        # We wrap the token in /…/ which is espeak's IPA notation.
        cmd = [
            espeak_exe,
            "--ipa=1",
            "-q",               # quiet: no text output
            "--stdout",
            "-s", "150",        # speaking rate (words/min); doesn't affect phoneme
            f"/{token}/",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not result.stdout:
            # Fallback: try passing as plain X-SAMPA / phoneme name
            cmd_fb = [espeak_exe, "-q", "--stdout", f"[[{token}]]"]
            result = subprocess.run(cmd_fb, capture_output=True)

        if not result.stdout:
            # Give up and return silence for this token
            return np.zeros(n_samples, dtype=np.float32)

        with open(tmp_path, "wb") as fh:
            fh.write(result.stdout)

        raw, raw_sr = librosa.load(tmp_path, sr=sr, mono=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if raw is None or len(raw) == 0:
        return np.zeros(n_samples, dtype=np.float32)

    raw_dur = len(raw) / sr
    target_dur = max(duration_sec, MIN_PHONEME_DURATION)

    if abs(raw_dur - target_dur) < 0.005:
        # Already close enough — just trim / pad
        stretched = raw
    else:
        rate = raw_dur / target_dur          # >1 means speed up, <1 slow down
        rate = float(np.clip(rate, 0.2, 5.0))
        try:
            stretched = librosa.effects.time_stretch(raw, rate=rate)
        except Exception:
            stretched = raw

    # Trim or zero-pad to exactly n_samples
    if len(stretched) >= n_samples:
        return stretched[:n_samples].astype(np.float32)
    else:
        return np.concatenate([stretched, np.zeros(n_samples - len(stretched))]).astype(np.float32)


def synthesise_runs(
    runs: List[PhonemeRun],
    fps: float,
    espeak_exe: str,
    sr: int = SAMPLE_RATE,
    total_frames: Optional[int] = None,
) -> np.ndarray:
    """Synthesise all phoneme runs and concatenate into one audio array."""
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
        copy_len = end - write_pos
        output[write_pos:end] = segment[:copy_len]
        write_pos += n  # advance by the *intended* frame count, not len(segment)

    return output


# ---------------------------------------------------------------------------
# Original audio extraction
# ---------------------------------------------------------------------------

def extract_original_audio(video_path: Path, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Extract original audio track from the lip-crop video.
    Returns None if no audio stream is present."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("  [warn] ffmpeg not found; skipping original audio extraction.")
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            ffmpeg, "-y", "-i", str(video_path),
            "-ar", str(sr), "-ac", "1", "-vn",
            "-loglevel", "error",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            return None
        audio, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return audio
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def _colour(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def print_comparison(
    stem: str,
    gt_frame_tokens: List[str],
    pred_frame_tokens: List[str],
    gt_collapsed: List[str],
    pred_collapsed: List[str],
    fps: float,
    num_display_frames: int = 60,
) -> None:
    width = shutil.get_terminal_size((120, 40)).columns
    sep = "─" * width

    print(f"\n{sep}")
    print(_colour(f"  Clip: {stem}  ({len(gt_frame_tokens)} frames @ {fps:.1f} fps)", "1;36"))
    print(sep)

    # --- Collapsed sequences ---
    print(_colour("  Collapsed ground truth :", "1;32"), " ".join(gt_collapsed) or "(empty)")
    print(_colour("  Collapsed prediction   :", "1;33"), " ".join(pred_collapsed) or "(empty)")

    # --- Quick accuracy numbers ---
    matched_len = min(len(gt_frame_tokens), len(pred_frame_tokens))
    frame_correct = sum(g == p for g, p in zip(gt_frame_tokens[:matched_len], pred_frame_tokens[:matched_len]))
    frame_acc = frame_correct / max(1, matched_len)
    print(f"  Frame-level accuracy   : {frame_acc:.1%}  ({frame_correct}/{matched_len} frames)")
    print(sep)

    # --- Per-frame table (first num_display_frames frames) ---
    n = min(num_display_frames, len(gt_frame_tokens), len(pred_frame_tokens))
    col_w = 8
    header_frames = "".join(f"{i:<{col_w}}" for i in range(n))
    print(f"  {'Frame':<16}{header_frames}")

    gt_row = "".join(f"{t[:col_w-1]:<{col_w}}" for t in gt_frame_tokens[:n])
    print(f"  {'GT':<16}", end="")
    print(_colour(gt_row, "32"))

    pred_row_parts: List[str] = []
    for g, p in zip(gt_frame_tokens[:n], pred_frame_tokens[:n]):
        tok = f"{p[:col_w-1]:<{col_w}}"
        pred_row_parts.append(_colour(tok, "33") if g == p else _colour(tok, "31"))
    print(f"  {'Pred':<16}", end="")
    print("".join(pred_row_parts))

    if len(gt_frame_tokens) > num_display_frames:
        remaining = len(gt_frame_tokens) - num_display_frames
        print(f"  ... {remaining} more frames not shown")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[VisualLSTMFrameCE, Dict[int, str], Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    vocab: List[str] = ckpt["vocab"]
    config: Dict = ckpt.get("config", {})

    hidden_dim = config.get("hidden_dim", 256)
    lstm_layers = config.get("lstm_layers", 2)
    vocab_size = len(vocab) + 1

    model = VisualLSTMFrameCE(vocab_size=vocab_size, hidden_dim=hidden_dim, lstm_layers=lstm_layers)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    id_to_token: Dict[int, str] = {idx + 1: tok for idx, tok in enumerate(vocab)}
    id_to_token[0] = DEFAULT_BLANK_TOKEN

    return model, id_to_token, config


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lip-reading LSTM evaluator with phoneme-accurate TTS comparison")
    p.add_argument("--checkpoint", type=str, default="visual_lstm_ce_results/best_model.pt")
    p.add_argument("--phoneme-csv", type=str, default="phonemes_s1_aligned/phoneme_predictions.csv")
    p.add_argument("--video-dir", type=str, default="s1_lip_crops")
    p.add_argument("--output-dir", type=str, default="eval_results")
    p.add_argument("--num-videos", type=int, default=5, help="Number of test-set videos to evaluate")
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    p.add_argument("--test-ratio", type=float, default=0.2, help="Must match value used during training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--display-frames", type=int, default=60, help="Max frames to show in terminal table")
    p.add_argument("--no-audio", action="store_true", help="Skip TTS synthesis (useful for quick checks)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    espeak_exe = None
    if not args.no_audio:
        espeak_exe = check_espeak()
        print(f"espeak-ng: {espeak_exe}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model, id_to_token, train_config = load_checkpoint(checkpoint_path, device)
    print(f"  vocab size : {len(id_to_token) - 1} phonemes + blank")
    print(f"  hidden_dim : {train_config.get('hidden_dim', 256)}")
    print(f"  lstm_layers: {train_config.get('lstm_layers', 2)}")

    # ── Build test split (same seed/ratio as training) ──────────────────────
    test_samples = build_test_samples(
        Path(args.video_dir),
        Path(args.phoneme_csv),
        seed=args.seed,
        test_ratio=args.test_ratio,
    )
    print(f"\nTest set size : {len(test_samples)} clips")

    n = min(args.num_videos, len(test_samples))
    selected = test_samples[:n]
    print(f"Evaluating    : {n} clips\n")

    # ── Per-clip loop ────────────────────────────────────────────────────────
    for sample in tqdm(selected, desc="Clips", unit="clip"):
        stem = normalize_stem(sample.video_path)

        # 1. Load video & run model
        try:
            video, fps = load_video_frames(sample.video_path, frame_size=args.frame_size)
        except RuntimeError as exc:
            print(f"  [skip] {stem}: {exc}")
            continue

        pred_frame_tokens = predict_frame_tokens(model, video, id_to_token, device)
        gt_frame_tokens = sample.frame_tokens

        # Align lengths (model may output one frame fewer due to packing)
        matched = min(len(gt_frame_tokens), len(pred_frame_tokens), video.shape[0])
        gt_frame_tokens = gt_frame_tokens[:matched]
        pred_frame_tokens = pred_frame_tokens[:matched]

        gt_collapsed = collapsed_sequence(gt_frame_tokens)
        pred_collapsed = collapsed_sequence(pred_frame_tokens)

        # 2. Terminal display
        print_comparison(
            stem=stem,
            gt_frame_tokens=gt_frame_tokens,
            pred_frame_tokens=pred_frame_tokens,
            gt_collapsed=gt_collapsed,
            pred_collapsed=pred_collapsed,
            fps=fps,
            num_display_frames=args.display_frames,
        )

        if args.no_audio:
            continue

        # 3. Build run-length encodings for TTS
        gt_runs = run_length_encode(gt_frame_tokens)
        pred_runs = run_length_encode(pred_frame_tokens)

        # 4. Original audio
        orig_path = output_dir / f"original_{stem}.wav"
        orig_audio = extract_original_audio(sample.video_path, sr=args.sample_rate)
        if orig_audio is not None:
            sf.write(str(orig_path), orig_audio, args.sample_rate, subtype="PCM_16")
            print(f"  Saved original audio : {orig_path}")
        else:
            print(f"  [warn] No audio stream in {sample.video_path.name}; original WAV skipped.")

        # 5. Ground-truth TTS
        print(f"  Synthesising GT  phonemes ({len(gt_runs)} runs) …", end=" ", flush=True)
        gt_audio = synthesise_runs(gt_runs, fps, espeak_exe, sr=args.sample_rate, total_frames=matched)
        gt_path = output_dir / f"gt_{stem}.wav"
        sf.write(str(gt_path), gt_audio, args.sample_rate, subtype="PCM_16")
        print(f"→ {gt_path}")

        # 6. Predicted TTS
        print(f"  Synthesising pred phonemes ({len(pred_runs)} runs) …", end=" ", flush=True)
        pred_audio = synthesise_runs(pred_runs, fps, espeak_exe, sr=args.sample_rate, total_frames=matched)
        pred_path = output_dir / f"pred_{stem}.wav"
        sf.write(str(pred_path), pred_audio, args.sample_rate, subtype="PCM_16")
        print(f"→ {pred_path}")

    print(f"\nDone. Outputs written to: {output_dir}/")


if __name__ == "__main__":
    main()