#!/usr/bin/env python3
"""Precompute log-mel spectrograms for the audio referenced by a manifest
(typically the corrupted audio in a noise manifest from add_av_noise.py),
so train_av_fusion.py doesn't recompute the same deterministic STFT +
mel-filterbank + dB conversion on every single epoch of training.

Why this is safe to cache
--------------------------
train_av_fusion.py's mel extraction is per-utterance normalised (its own
mean/std, not a corpus statistic) and depends only on fixed hyperparameters
(sample_rate, hop_length, n_fft, n_mels) -- nothing epoch- or
batch-dependent. Given a fixed corrupted .wav file and fixed
hyperparameters, the exact same array comes out every time, so recomputing
it once per epoch (up to --epochs times) is pure redundant CPU work that
can bottleneck the DataLoader while the GPU sits idle.

SpecAugment (time/frequency masking) is intentionally *not* baked into the
cache -- it needs fresh random masks every epoch, so train_av_fusion.py
still applies it after loading the cached array, exactly as it does for a
freshly-computed one.

IMPORTANT: the cache is only valid for the exact (--fps, --sample-rate,
--n-mels, --n-fft) combination it was built with. If you retune any of
those for an experiment, regenerate the cache. train_av_fusion.py detects a
mismatched cached shape and falls back to on-the-fly computation with a
warning rather than silently feeding the wrong-shaped features.

If you pass --audio-source-dir, the script resolves each stem against the
clean source audio files in that directory, writes those clean paths back
into the output manifest, and sets audio_reliability to 1.0 for every frame.
That is the recommended mode for a clean-audio-only run.

Audio loading and the core mel computation reuse the same backend chain
(PyAV -> librosa -> ffmpeg subprocess) and compute_mel() pattern as
preprocess_mels_full.py / preprocess_ravdess_audio.py, since the corrupted
media here can in principle be any container add_av_noise.py's audio_path
points to, not just .wav.

Usage
-----
    python precompute_mel_spectrograms.py \
        --noise-manifest av_noise_s1/noise_manifest.csv \
        --audio-source-dir s1 \
        --output-dir av_clean_audio_s1

    python precompute_mel_spectrograms.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


librosa = try_import("librosa")
if librosa is None:
    raise RuntimeError("This script requires librosa. Install with: pip install librosa")


VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".mpg", ".mpeg")
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")
SOURCE_AUDIO_EXTS = VIDEO_EXTS + AUDIO_EXTS


# ---------------------------------------------------------------------------
# Audio loading -- same backend chain as preprocess_mels_full.py /
# preprocess_ravdess_audio.py (PyAV first, then librosa directly for native
# audio formats, then an ffmpeg subprocess extraction as a last resort for
# containers neither of those can decode).
# ---------------------------------------------------------------------------

def resolve_ffmpeg_executable() -> Optional[str]:
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    imageio_ffmpeg = try_import("imageio_ffmpeg")
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def build_audio_source_index(audio_source_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in audio_source_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SOURCE_AUDIO_EXTS:
            index[p.stem.lower()] = p
    log.info("Indexed %d clean source audio/AV file(s) in %s", len(index), audio_source_dir)
    return index


def load_audio(path: Path, sr: int) -> np.ndarray:
    ext = path.suffix.lower()

    av_mod = try_import("av")
    if av_mod is not None:
        try:
            container = av_mod.open(str(path))
            audio_stream = container.streams.audio[0] if container.streams.audio else None
            if audio_stream:
                audio_frames = []
                for frame in container.decode(audio=0):
                    audio_data = frame.to_ndarray()
                    if audio_data.ndim == 1:
                        audio_data = audio_data.reshape(-1, 1)
                    elif audio_data.shape[0] < audio_data.shape[1]:
                        audio_data = audio_data.T
                    audio_frames.append(audio_data)

                container.close()
                if audio_frames:
                    audio_full = np.concatenate(audio_frames, axis=0)
                    if np.issubdtype(audio_full.dtype, np.integer):
                        max_val = float(2 ** (8 * audio_full.dtype.itemsize - 1))
                        audio_full = audio_full.astype(np.float32) / max_val
                    else:
                        audio_full = audio_full.astype(np.float32)

                    if audio_full.ndim > 1:
                        audio_full = audio_full.mean(axis=1)

                    stream_rate = getattr(audio_stream, "rate", None)
                    if stream_rate and stream_rate != sr:
                        audio_full = librosa.resample(audio_full, orig_sr=stream_rate, target_sr=sr)

                    return audio_full
        except Exception:
            pass

    if ext in AUDIO_EXTS:
        try:
            y, _ = librosa.load(str(path), sr=sr, mono=True)
            return y
        except Exception:
            pass

    ffmpeg = resolve_ffmpeg_executable()
    if ext in VIDEO_EXTS or ffmpeg is not None:
        if ffmpeg is None:
            raise RuntimeError(
                f"No decode backend available for {path}. Install ffmpeg on PATH, install imageio-ffmpeg, "
                "or install PyAV."
            )
        tmp_path = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_path = tmp_file.name
            tmp_file.close()
            cmd = [ffmpeg, "-y", "-i", str(path), "-ar", str(sr), "-ac", "1", "-vn", str(tmp_path)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            y, _ = librosa.load(tmp_path, sr=sr, mono=True)
            return y
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


# ---------------------------------------------------------------------------
# Mel computation -- same shape as preprocess_mels_full.py's compute_mel(),
# plus the per-utterance normalisation train_av_fusion.py's _load_mel
# applies, so a cached array is byte-for-byte what the training script
# would have computed on the fly.
# ---------------------------------------------------------------------------

def compute_mel(y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int, power: float = 2.0) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db  # (n_mels, T)


def compute_normalized_mel(y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    mel_db = compute_mel(y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = mel_db.T  # (T, n_mels), matching train_av_fusion.py's _load_mel
    mean, std = mel_db.mean(), mel_db.std() + 1e-6
    mel_norm = (mel_db - mean) / std
    return mel_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def read_manifest_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def write_manifest_rows(rows: List[Dict[str, str]], csv_path: Path) -> None:
    fieldnames = list(rows[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Precompute + cache log-mel spectrograms for a noise manifest's corrupted audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--noise-manifest", default="av_noise_s1/noise_manifest.csv",
                   help="Manifest produced by add_av_noise.py (must have 'stem' and 'audio_path' columns).")
    p.add_argument("--audio-source-dir", default="",
                   help="Optional directory containing the clean source audio/AV files matched by stem. "
                        "When set, each row's audio_path is rewritten to the clean source file and the "
                        "output manifest marks audio_reliability as 1.0 for all frames.")
    p.add_argument("--output-dir", default="",
                   help="Where to write mel_cache/ and the augmented manifest. Defaults to the "
                        "noise manifest's own directory.")
    p.add_argument("--output-manifest", default="",
                   help="Path for the augmented manifest (adds a 'mel_path' column). Defaults to "
                        "<output-dir>/noise_manifest_with_mel.csv.")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=0,
                   help="0 = auto (round(sample_rate / fps)), matching train_av_fusion.py's frame-rate "
                        "alignment so one mel frame lines up with one video frame.")
    p.add_argument("--max-files", type=int, default=0, help="0 = process all rows.")
    p.add_argument("--dry-run", action="store_true", help="Process only the first row, print stats, exit.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    noise_manifest_path = Path(args.noise_manifest)
    output_dir = Path(args.output_dir) if args.output_dir else noise_manifest_path.parent
    output_manifest_path = (
        Path(args.output_manifest) if args.output_manifest else output_dir / "noise_manifest_with_mel.csv"
    )
    mel_cache_dir = output_dir / "mel_cache"
    mel_cache_dir.mkdir(parents=True, exist_ok=True)
    audio_source_index: Optional[Dict[str, Path]] = None
    if args.audio_source_dir:
        audio_source_index = build_audio_source_index(Path(args.audio_source_dir))

    hop_length = args.hop_length if args.hop_length > 0 else int(round(args.sample_rate / args.fps))
    log.info(
        "sample_rate=%d  fps=%.2f  n_mels=%d  n_fft=%d  hop_length=%d (%.1fms/frame)",
        args.sample_rate, args.fps, args.n_mels, args.n_fft, hop_length, 1000.0 * hop_length / args.sample_rate,
    )

    rows = read_manifest_rows(noise_manifest_path)
    log.info("Loaded %d row(s) from %s", len(rows), noise_manifest_path)

    # A dry run exercises one row before a full cache is generated.
    if args.dry_run:
        rows = rows[:1]
    elif args.max_files > 0:
        rows = rows[: args.max_files]

    processed_rows: List[Dict[str, str]] = []
    mismatch_count = 0

    for i, row in enumerate(rows, start=1):
        stem = row.get("stem", "").strip()
        audio_path = Path(row.get("audio_path", ""))
        if not stem or not audio_path:
            log.warning("Skipping row %d: missing stem/audio_path", i)
            continue

        if audio_source_index is not None:
            clean_audio_path = audio_source_index.get(stem.lower())
            if clean_audio_path is None:
                log.warning("Skipping %s: no clean source audio found in %s", stem, args.audio_source_dir)
                continue
            audio_path = clean_audio_path

        try:
            waveform = load_audio(audio_path, sr=args.sample_rate)
            mel = compute_normalized_mel(waveform, args.sample_rate, args.n_mels, args.n_fft, hop_length)
        except Exception as exc:
            log.error("Failed on %s (%s): %s", stem, audio_path, exc)
            continue

        expected_frames_str = row.get("num_frames", "").strip()
        if expected_frames_str.isdigit():
            expected_frames = int(expected_frames_str)
            if abs(mel.shape[0] - expected_frames) > 1:
                mismatch_count += 1
                log.warning(
                    "%s: mel has %d frames but manifest num_frames=%d -- check --fps/--sample-rate "
                    "match what add_av_noise.py used.",
                    stem, mel.shape[0], expected_frames,
                )

        mel_path = mel_cache_dir / f"{stem}.npy"
        np.save(mel_path, mel)

        row = dict(row)
        if audio_source_index is not None:
            row["audio_path"] = str(audio_path)
            row["audio_reliability"] = json.dumps([1.0] * int(row.get("num_frames", mel.shape[0]) or mel.shape[0]))
        row["mel_path"] = str(mel_path)
        processed_rows.append(row)

        if args.dry_run:
            log.info(
                "%s: waveform=%d samples (%.2fs)  mel=%s  mean=%.4f std=%.4f  -> %s",
                stem, len(waveform), len(waveform) / args.sample_rate, tuple(mel.shape),
                float(mel.mean()), float(mel.std()), mel_path,
            )
        elif i % 50 == 0 or i == len(rows):
            log.info("[%d/%d] processed", i, len(rows))

    if not processed_rows:
        raise RuntimeError("No rows were processed successfully.")

    if args.dry_run:
        log.info("Dry run complete -- no manifest written. mel_cache file(s) above were still saved to disk.")
        return

    write_manifest_rows(processed_rows, output_manifest_path)
    log.info("Wrote %d row(s) to %s", len(processed_rows), output_manifest_path)
    log.info("Mel cache -> %s", mel_cache_dir)
    if mismatch_count:
        log.warning(
            "%d/%d clip(s) had a mel frame count that didn't match the manifest's num_frames by more than "
            "1 frame -- double check --fps/--sample-rate against add_av_noise.py's settings.",
            mismatch_count, len(processed_rows),
        )
    log.info("Point train_av_fusion.py at this manifest with --noise-manifest %s", output_manifest_path)



if __name__ == "__main__":
    main()