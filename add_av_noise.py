#!/usr/bin/env python3
"""Generate noise-corrupted audio/video pairs for robust AV lip-reading training.

This script takes clean GRID lip-crop videos (silent) and their matching
source audio (e.g. the original .mpg files), and produces a corrupted copy
of each modality plus a manifest recording exactly which frames were
corrupted and how badly. The manifest is consumed by train_av_fusion.py to
build per-frame "reliability" scores (0 = fully corrupted, 1 = clean) that
feed the cross-modal fusion gate.

Two independent corruption families are applied, matching the scenario
described for the fusion model. Every clip is corrupted in both modalities
-- there is no probability gate that can leave a clip clean:
  - Video: a single *contiguous* run of frames is obstructed (never
    scattered), with length drawn around a target of 30 frames (uniformly
    from `video_obstruct_frames +/- video_obstruct_jitter`, so it varies a
    bit clip-to-clip while always staying one unbroken block) (modes:
    "zero" blackout, "freeze" repeat-last-frame, "blur" heavy Gaussian
    blur).
  - Audio: additive noise (white or a crude broadband "babble"
    approximation) is always mixed in at a randomly sampled SNR; an
    *additional*, optional silence dropout over an exact-frame-count span
    (or several) can also be layered on top (`--audio-dropout-prob`).

Only the *severity* (SNR, run length/position, which frames) is
randomised per clip -- the presence of corruption itself is guaranteed,
which is what lets the fusion gate learn to always partially discount each
modality rather than learning a shortcut of "assume clean unless told
otherwise."

Usage:
    python add_av_noise.py \
        --video-dir s1_lip_crops \
        --audio-source-dir s1 \
        --output-dir av_noise_s1 \
        --fps 25.0

    # Quick smoke-test
    python add_av_noise.py --max-files 3 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This script requires librosa. Install with: pip install librosa") from exc

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This script requires soundfile. Install with: pip install soundfile") from exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_OR_AV_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wav", ".flac"}
TARGET_SR = 16_000
VIDEO_CORRUPTION_MODES = ("zero", "freeze", "blur")
AUDIO_NOISE_TYPES = ("white", "babble")

# Reliability floor/ceiling used when mapping a clip-level SNR to a
# per-frame audio reliability score. Below snr_floor_db the frame is
# treated as (almost) unusable; above snr_ceiling_db it's treated as clean.
SNR_FLOOR_DB = -5.0
SNR_CEILING_DB = 20.0
DROPOUT_RELIABILITY = 0.05  # residual reliability for fully-dropped-out spans


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VideoCorruptionMeta:
    mode: str
    spans: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class AudioCorruptionMeta:
    snr_db: Optional[float]
    noise_type: Optional[str]
    dropout_spans_sec: List[Tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# File discovery / matching (mirrors the stem convention used elsewhere in
# this codebase: lip-crop files are "<stem>_lipcrop.<ext>")
# ---------------------------------------------------------------------------

def normalize_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_lipcrop"):
        return stem[: -len("_lipcrop")]
    return stem


def find_lip_videos(video_dir: Path) -> List[Path]:
    return [
        p for p in sorted(video_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]


def build_audio_source_index(audio_source_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in audio_source_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_OR_AV_EXTS:
            index[p.stem.lower()] = p
    log.info("Indexed %d source audio/AV file(s) in %s", len(index), audio_source_dir)
    return index


# ---------------------------------------------------------------------------
# Audio loading (waveform, mono, 16 kHz) -- tries a small backend chain
# ---------------------------------------------------------------------------

def resolve_ffmpeg_executable() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


SOUNDFILE_NATIVE_EXTS = {".wav", ".flac", ".ogg"}


def _ffmpeg_extract(path: Path, sr: int, ffmpeg: str) -> np.ndarray:
    tmp_path = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp_file.name
        tmp_file.close()
        cmd = [ffmpeg, "-y", "-i", str(path), "-ar", str(sr), "-ac", "1", "-vn", tmp_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        y, _ = sf.read(tmp_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32)
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def load_audio_16k(path: Path, sr: int = TARGET_SR) -> np.ndarray:
    """Load audio as mono float32 at `sr` Hz.

    For containers librosa/soundfile can't decode natively (.mpg, .mp4,
    .avi, .mov, .mkv), go straight to an ffmpeg subprocess extraction
    rather than calling librosa.load first: librosa silently falls back to
    its deprecated `audioread` backend for these formats, which is orders
    of magnitude slower than a direct ffmpeg extraction and prints a
    FutureWarning per file.
    """
    suffix = path.suffix.lower()

    if suffix in SOUNDFILE_NATIVE_EXTS:
        try:
            y, _ = librosa.load(str(path), sr=sr, mono=True)
            return y
        except Exception:
            pass

    ffmpeg = resolve_ffmpeg_executable()
    if ffmpeg is not None:
        try:
            return _ffmpeg_extract(path, sr, ffmpeg)
        except Exception:
            pass

    # Last resort: let librosa try (may hit the slow audioread path, but
    # only reached if ffmpeg is unavailable or failed).
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y
    except Exception as exc:
        raise RuntimeError(f"Could not decode audio from {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Video corruption
# ---------------------------------------------------------------------------

def read_video_frames(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)  # (T, H, W, 3) BGR uint8


def write_video_frames(frames: np.ndarray, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def sample_single_contiguous_span(
    num_frames: int, target_length: int, jitter: int, rng: random.Random,
) -> List[Tuple[int, int]]:
    """Return exactly one contiguous (start, end) span, with length drawn
    uniformly from [target_length - jitter, target_length + jitter]
    (clamped to [1, num_frames]) and placed at a random start position.

    Unlike sample_fixed_count_spans, this never splits the obstruction into
    multiple bursts -- the corrupted frames are always consecutive ("in a
    row"), only the span's length and position vary per clip.
    """
    low = max(1, target_length - jitter)
    high = min(num_frames, target_length + jitter)
    if low > high:
        low = high = min(max(1, target_length), num_frames)
    length = rng.randint(low, high)
    start = rng.randint(0, max(0, num_frames - length))
    return [(start, start + length)]


def sample_fixed_count_spans(
    num_frames: int, target_count: int, max_spans: int, rng: random.Random,
) -> List[Tuple[int, int]]:
    """Return non-overlapping (start, end) spans whose lengths sum to
    exactly `target_count` frames (clamped to num_frames), split across
    1..max_spans contiguous bursts at random positions.

    This guarantees every clip receives exactly the same *amount* of
    corruption while still varying *how* it's distributed -- e.g. one
    15-frame occlusion vs. three shorter ones scattered through the clip
    (closer to how a hand or head turn actually occludes lips in bursts,
    rather than one probabilistic on/off coin-flip per clip).
    """
    target_count = max(0, min(target_count, num_frames))
    if target_count == 0:
        return []

    max_spans = max(1, min(max_spans, target_count))
    n_spans = rng.randint(1, max_spans)

    # Random composition of target_count into n_spans positive integers
    # (stars-and-bars via random cut points).
    if n_spans == 1:
        span_lengths = [target_count]
    else:
        cut_points = sorted(rng.sample(range(1, target_count), n_spans - 1))
        span_lengths = []
        prev = 0
        for cp in cut_points:
            span_lengths.append(cp - prev)
            prev = cp
        span_lengths.append(target_count - prev)

    # Scatter the untouched frames into n_spans + 1 gaps (before / between /
    # after the corrupted spans) so the spans never overlap and the full
    # clip length is covered exactly.
    remaining = num_frames - target_count
    gaps = [0] * (n_spans + 1)
    for _ in range(remaining):
        gaps[rng.randint(0, n_spans)] += 1

    spans: List[Tuple[int, int]] = []
    cursor = gaps[0]
    for i, length in enumerate(span_lengths):
        start = cursor
        end = start + length
        spans.append((start, end))
        cursor = end + gaps[i + 1]
    return spans


def apply_video_corruption(
    frames: np.ndarray,
    mode: str,
    spans: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (corrupted_frames, per_frame_reliability) for the given spans."""
    corrupted = frames.copy()
    num_frames = frames.shape[0]
    reliability = np.ones(num_frames, dtype=np.float32)

    for start, end in spans:
        start = max(0, start)
        end = min(num_frames, end)
        if end <= start:
            continue

        if mode == "zero":
            corrupted[start:end] = 0
            reliability[start:end] = 0.0
        elif mode == "freeze":
            hold_idx = start - 1 if start > 0 else end  # nearest still-valid frame
            hold_idx = min(max(hold_idx, 0), num_frames - 1)
            corrupted[start:end] = corrupted[hold_idx]
            reliability[start:end] = 0.0
        elif mode == "blur":
            for i in range(start, end):
                corrupted[i] = cv2.GaussianBlur(corrupted[i], (25, 25), sigmaX=12)
            reliability[start:end] = 0.3
        else:
            raise ValueError(f"Unknown video corruption mode: {mode}")

    return corrupted, reliability


# ---------------------------------------------------------------------------
# Audio corruption
# ---------------------------------------------------------------------------

def make_noise(length: int, noise_type: str, rng: np.random.RandomState) -> np.ndarray:
    if noise_type == "white":
        return rng.normal(0.0, 1.0, size=length).astype(np.float32)
    if noise_type == "babble":
        # Crude broadband "babble-like" noise: sum of independently
        # band-limited white-noise streams (simple cumulative-sum /
        # differencing gives a rough 1/f-ish coloring without extra
        # dependencies), then renormalized. Not a substitute for a real
        # babble corpus, but gives spectrally different noise than pure
        # white for augmentation diversity.
        base = rng.normal(0.0, 1.0, size=length + 1).astype(np.float32)
        colored = np.cumsum(base)[1:] - np.cumsum(base)[:-1]
        colored = colored - colored.mean()
        return colored
    raise ValueError(f"Unknown noise type: {noise_type}")


def snr_to_reliability(snr_db: float) -> float:
    clamped = min(max(snr_db, SNR_FLOOR_DB), SNR_CEILING_DB)
    return float((clamped - SNR_FLOOR_DB) / (SNR_CEILING_DB - SNR_FLOOR_DB))


def add_noise_at_snr(waveform: np.ndarray, snr_db: float, noise_type: str, rng: np.random.RandomState) -> np.ndarray:
    signal_power = float(np.mean(waveform ** 2)) + 1e-12
    noise = make_noise(len(waveform), noise_type, rng)
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    mixed = waveform + noise
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def apply_audio_dropout(
    waveform: np.ndarray,
    spans_sec: Sequence[Tuple[float, float]],
    sr: int,
) -> np.ndarray:
    corrupted = waveform.copy()
    for start_sec, end_sec in spans_sec:
        s = max(0, int(start_sec * sr))
        e = min(len(corrupted), int(end_sec * sr))
        if e > s:
            corrupted[s:e] = 0.0
    return corrupted


def build_audio_frame_reliability(
    num_video_frames: int,
    fps: float,
    base_reliability: float,
    dropout_spans_sec: Sequence[Tuple[float, float]],
) -> np.ndarray:
    reliability = np.full(num_video_frames, base_reliability, dtype=np.float32)
    for start_sec, end_sec in dropout_spans_sec:
        # round() rather than int() truncation: dropout_spans_sec was
        # derived from integer frame indices via start_sec = start / fps,
        # so round-tripping back through * fps should recover the exact
        # integer even when float representation lands a hair under it
        # (e.g. 2.28 * 25 == 56.999999999999993, not 57.0).
        first_frame = int(round(start_sec * fps))
        last_frame = int(round(end_sec * fps))
        for f in range(max(0, first_frame), min(num_video_frames, last_frame)):
            reliability[f] = DROPOUT_RELIABILITY
    return reliability


# ---------------------------------------------------------------------------
# Per-clip pipeline
# ---------------------------------------------------------------------------

def process_clip(
    video_path: Path,
    audio_path: Path,
    stem: str,
    output_video_dir: Path,
    output_audio_dir: Path,
    fps: float,
    sample_rate: int,
    args: argparse.Namespace,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> Dict:
    frames = read_video_frames(video_path)
    num_frames = frames.shape[0]

    # ---- video corruption (always applied -- every clip loses one
    # contiguous run of frames, length randomised around
    # args.video_obstruct_frames, no probability gate) ----
    mode = rng.choice(args.video_corruption_modes)
    video_spans = sample_single_contiguous_span(
        num_frames, args.video_obstruct_frames, args.video_obstruct_jitter, rng,
    )
    frames, video_reliability = apply_video_corruption(frames, mode, video_spans)
    video_meta = VideoCorruptionMeta(mode=mode, spans=video_spans)

    corrupted_video_path = output_video_dir / f"{stem}_lipcrop.mp4"
    write_video_frames(frames, corrupted_video_path, fps=fps)

    # ---- audio corruption ----
    waveform = load_audio_16k(audio_path, sr=sample_rate)
    clip_duration_sec = num_frames / fps
    # Trim/pad waveform to match the video clip's duration so the two
    # modalities describe the same span of time.
    target_len = int(round(clip_duration_sec * sample_rate))
    if len(waveform) >= target_len:
        waveform = waveform[:target_len]
    else:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))

    # Additive noise is always applied (no probability gate) -- only the
    # SNR and noise type are randomised, so no audio file ever stays clean.
    snr_db = rng.uniform(*args.audio_snr_db_range)
    noise_type = rng.choice(args.audio_noise_types)
    waveform = add_noise_at_snr(waveform, snr_db, noise_type, np_rng)
    base_audio_reliability = snr_to_reliability(snr_db)

    # Silence dropout is an *additional*, optional corruption layered on
    # top of the always-present noise (the "never unchanged" guarantee is
    # already met by the additive noise above).
    dropout_spans_sec: List[Tuple[float, float]] = []
    if rng.random() < args.audio_dropout_prob:
        frame_spans = sample_fixed_count_spans(num_frames, args.audio_dropout_frames, args.audio_dropout_max_spans, rng)
        dropout_spans_sec = [(s / fps, e / fps) for s, e in frame_spans]
        waveform = apply_audio_dropout(waveform, dropout_spans_sec, sample_rate)

    audio_reliability = build_audio_frame_reliability(num_frames, fps, base_audio_reliability, dropout_spans_sec)

    corrupted_audio_path = output_audio_dir / f"{stem}.wav"
    corrupted_audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(corrupted_audio_path), waveform, sample_rate, subtype="PCM_16")

    audio_meta = AudioCorruptionMeta(snr_db=snr_db, noise_type=noise_type, dropout_spans_sec=dropout_spans_sec)

    return {
        "stem": stem,
        "num_frames": num_frames,
        "video_path": str(corrupted_video_path),
        "audio_path": str(corrupted_audio_path),
        "video_reliability": json.dumps([round(float(v), 4) for v in video_reliability]),
        "audio_reliability": json.dumps([round(float(v), 4) for v in audio_reliability]),
        "video_corruption_mode": video_meta.mode,
        "video_corruption_spans": json.dumps(video_meta.spans),
        "audio_snr_db": round(snr_db, 2),
        "audio_noise_type": noise_type,
        "audio_dropout_spans_sec": json.dumps([[round(s, 4), round(e, 4)] for s, e in dropout_spans_sec]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate corrupted audio/video pairs + reliability manifest for robust AV lip-reading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-dir", default="s1_lip_crops", help="Directory of clean, silent lip-crop videos.")
    p.add_argument("--audio-source-dir", default="s1",
                   help="Directory containing the original audio/AV source files (matched to lip crops by stem).")
    p.add_argument("--output-dir", default="av_noise_s1")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--sample-rate", type=int, default=TARGET_SR)

    p.add_argument("--video-obstruct-frames", type=int, default=30,
                   help="Target length (in frames) of the single contiguous obstructed run. "
                        "Every clip is corrupted -- this is not gated by a probability.")
    p.add_argument("--video-obstruct-jitter", type=int, default=8,
                   help="Actual obstruction length is drawn uniformly from "
                        "[video_obstruct_frames - jitter, video_obstruct_frames + jitter], "
                        "clamped to the clip length, so the run is 'around' the target rather than exact.")
    p.add_argument("--video-corruption-modes", nargs="+", default=list(VIDEO_CORRUPTION_MODES),
                   choices=list(VIDEO_CORRUPTION_MODES))

    p.add_argument("--audio-snr-db-range", type=float, nargs=2, default=(-5.0, 15.0),
                   help="Additive noise is applied to every clip; SNR is drawn uniformly from this range.")
    p.add_argument("--audio-noise-types", nargs="+", default=list(AUDIO_NOISE_TYPES),
                   choices=list(AUDIO_NOISE_TYPES))
    p.add_argument("--audio-dropout-prob", type=float, default=0.3,
                   help="Probability of an *additional* silence dropout layered on top of the "
                        "always-applied additive noise (not required for the no-clean-clip guarantee).")
    p.add_argument("--audio-dropout-frames", type=int, default=15,
                   help="Exact number of frames' worth of audio to silence when dropout triggers.")
    p.add_argument("--audio-dropout-max-spans", type=int, default=1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files", type=int, default=0, help="0 = process all files.")
    p.add_argument("--dry-run", action="store_true", help="Process only the first file, print the manifest row, exit.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    video_dir = Path(args.video_dir)
    audio_source_dir = Path(args.audio_source_dir)
    output_dir = Path(args.output_dir)
    output_video_dir = output_dir / "video"
    output_audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    lip_videos = find_lip_videos(video_dir)
    if not lip_videos:
        raise FileNotFoundError(f"No lip-crop videos found in {video_dir}")

    audio_index = build_audio_source_index(audio_source_dir)

    pairs: List[Tuple[Path, Path, str]] = []
    missing = 0
    for video_path in lip_videos:
        stem = normalize_stem(video_path)
        audio_path = audio_index.get(stem.lower())
        if audio_path is None:
            missing += 1
            continue
        pairs.append((video_path, audio_path, stem))

    if missing:
        log.warning("%d lip-crop video(s) had no matching source audio and were skipped.", missing)
    if not pairs:
        raise FileNotFoundError("No (video, audio) pairs could be matched by stem.")

    if args.dry_run:
        pairs = pairs[:1]
    elif args.max_files > 0:
        pairs = pairs[: args.max_files]

    log.info("Processing %d clip(s)...", len(pairs))

    rows: List[Dict] = []
    for i, (video_path, audio_path, stem) in enumerate(pairs, start=1):
        try:
            row = process_clip(
                video_path=video_path,
                audio_path=audio_path,
                stem=stem,
                output_video_dir=output_video_dir,
                output_audio_dir=output_audio_dir,
                fps=args.fps,
                sample_rate=args.sample_rate,
                args=args,
                rng=rng,
                np_rng=np_rng,
            )
            rows.append(row)
            log.info(
                "[%d/%d] %s  video_mode=%s  audio_snr=%s  audio_noise=%s",
                i, len(pairs), stem, row["video_corruption_mode"], row["audio_snr_db"], row["audio_noise_type"],
            )
        except Exception as exc:
            log.error("Failed on %s: %s", stem, exc)

    if args.dry_run:
        print(json.dumps(rows[0], indent=2)[:2000])
        log.info("Dry run complete -- no manifest written.")
        return

    if not rows:
        raise RuntimeError("No clips were processed successfully.")

    manifest_path = output_dir / "noise_manifest.csv"
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d row(s) to %s", len(rows), manifest_path)
    log.info("Corrupted video -> %s", output_video_dir)
    log.info("Corrupted audio -> %s", output_audio_dir)


if __name__ == "__main__":
    main()