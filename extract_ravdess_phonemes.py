#!/usr/bin/env python3
"""Extract phoneme predictions from RAVDESS audio using a pretrained wav2vec2 CTC model.

This is a first-pass audio-only probe for the lip-reading pipeline. It loads each
RAVDESS audio/video clip, resamples it to 16 kHz, runs a phoneme-recognition
wav2vec2 model, and writes the decoded predictions to a CSV.

Usage:
    python extract_ravdess_phonemes.py --input-dir ravdess_videos_only --output-dir phoneme_results
    python extract_ravdess_phonemes.py --max-files 3 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download


VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi")
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


librosa = try_import("librosa")
if librosa is None:
    raise RuntimeError("This script requires librosa. Install with: pip install librosa")


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


def load_audio(path: Path, sr: int):
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
    if ext in VIDEO_EXTS:
        if ffmpeg is None:
            raise RuntimeError(
                f"No video decode backend available for {path}. Install ffmpeg on PATH, install imageio-ffmpeg, or install PyAV."
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
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def find_ravdess_files(input_dir: Path):
    files = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS + VIDEO_EXTS:
            files.append(path)
    return files


def load_vocab(model_name: str) -> dict[int, str]:
    vocab_path = hf_hub_download(repo_id=model_name, filename="vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    return {int(index): token for token, index in vocab.items()}


def decode_ctc_ids(predicted_ids: list[int], id_to_token: dict[int, str], blank_id: Optional[int]) -> str:
    decoded_tokens: list[str] = []
    previous_id: Optional[int] = None
    special_tokens = {"<pad>", "<s>", "</s>", "<unk>"}

    for token_id in predicted_ids:
        if blank_id is not None and token_id == blank_id:
            previous_id = token_id
            continue
        if previous_id == token_id:
            continue

        token = id_to_token.get(int(token_id), "")
        if not token or token in special_tokens:
            previous_id = token_id
            continue

        decoded_tokens.append(token)

        previous_id = token_id

    return " ".join(piece for piece in decoded_tokens if piece).replace(" | ", " ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract wav2vec2 phoneme predictions from RAVDESS clips")
    parser.add_argument("--input-dir", type=str, default="ravdess_videos_only")
    parser.add_argument("--output-dir", type=str, default="phoneme_results")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-files", type=int, default=0, help="Process at most this many files; 0 means all files")
    parser.add_argument("--dry-run", action="store_true", help="Process only the first file and print the decoded phoneme string")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_ravdess_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No audio/video files found in {input_dir}")

    if args.dry_run:
        files = files[:1]
    elif args.max_files > 0:
        files = files[: args.max_files]

    if args.model_name == "facebook/wav2vec2-lv60-espeak-cv-ft":
        args.model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    model.eval()

    id_to_token = load_vocab(args.model_name)
    blank_id = getattr(model.config, "pad_token_id", None)

    rows = []
    for path in files:
        audio = load_audio(path, sr=args.sample_rate)
        if audio is None or len(audio) == 0:
            continue

        input_values = feature_extractor(audio, return_tensors="pt", sampling_rate=args.sample_rate).input_values
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        phoneme_text = decode_ctc_ids(predicted_ids[0].tolist(), id_to_token, blank_id)

        rows.append(
            {
                "file": str(path),
                "num_samples": int(len(audio)),
                "num_frames": int(logits.shape[1]),
                "phonemes": phoneme_text,
            }
        )
        print(f"{path.name}: {phoneme_text}")

    if not rows:
        raise RuntimeError("No files were processed successfully")

    csv_path = output_dir / "phoneme_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "num_samples", "num_frames", "phonemes"])
        writer.writeheader()
        writer.writerows(rows)

    if args.dry_run:
        print(f"Dry run complete. Wrote {csv_path}")
    else:
        print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()