#!/usr/bin/env python3
"""Preprocess RAVDESS audio: split into 0.5s chunks and save 128x50 Mel tensors.

Saves per-chunk files as numpy .npz with keys: 'mel' (float32, shape (128,50)),
and writes a manifest CSV with metadata.

Usage:
    python preprocess_ravdess_audio.py --input-dir ravdess_videos_only --output-dir data/ravdess_mels_0.5s --sr 16000
"""
from pathlib import Path
import argparse
import importlib
import traceback
import numpy as np
import csv
import shutil
import tempfile
import subprocess
import os


def try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


librosa = try_import('librosa')
if librosa is None:
    raise RuntimeError('This script requires librosa. Install with: pip install librosa')

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kw):
        return x


def load_audio(path: Path, sr: int):
    ext = path.suffix.lower()
    # Prefer PyAV (same approach as test_mel_reconstruction.py)
    av_mod = try_import('av')
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
                    # convert to float32 in [-1,1] if integer type
                    if np.issubdtype(audio_full.dtype, np.integer):
                        max_val = float(2 ** (8 * audio_full.dtype.itemsize - 1))
                        audio_full = audio_full.astype(np.float32) / max_val
                    else:
                        audio_full = audio_full.astype(np.float32)

                    if audio_full.ndim > 1:
                        audio_full = audio_full.mean(axis=1)

                    # Resample if needed
                    stream_rate = getattr(audio_stream, 'rate', None)
                    if stream_rate and stream_rate != sr:
                        audio_full = librosa.resample(audio_full, orig_sr=stream_rate, target_sr=sr)

                    return audio_full
        except Exception:
            # fall back to other loaders below
            pass

    # Next, try librosa directly for common audio files; if it fails use ffmpeg extraction
    if ext in ('.wav', '.flac', '.mp3', '.ogg'):
        try:
            y, _ = librosa.load(str(path), sr=sr, mono=True)
            return y
        except Exception:
            pass

    # If file is a video container or librosa failed, try using ffmpeg to extract a temporary WAV and load that.
    if ext in ('.mp4', '.mkv', '.mov', '.avi') or True:
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            # last resort: try librosa.load (may fail)
            y, _ = librosa.load(str(path), sr=sr, mono=True)
            return y

        tmp = None
        try:
            tmp_f = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp = tmp_f.name
            tmp_f.close()
            cmd = [ffmpeg, '-y', '-i', str(path), '-ar', str(sr), '-ac', '1', '-vn', str(tmp)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            y, _ = librosa.load(tmp, sr=sr, mono=True)
            return y
        finally:
            if tmp is not None and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


def compute_mel(chunk, sr, n_mels, n_fft, hop_length, power=2.0):
    S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    # Convert to dB
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def process_file(path: Path, out_dir: Path, sr: int, chunk_secs: float, frames_per_chunk: int, n_mels: int, n_fft: int):
    y = load_audio(path, sr)
    total_samples = y.shape[0]
    chunk_samples = int(round(chunk_secs * sr))
    hop_length = int(round(chunk_samples / frames_per_chunk))

    num_full = total_samples // chunk_samples
    results = []
    for idx in range(num_full):
        start = idx * chunk_samples
        chunk = y[start:start + chunk_samples]
        mel = compute_mel(chunk, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        # Ensure shape (n_mels, frames_per_chunk)
        if mel.shape[1] < frames_per_chunk:
            pad_width = frames_per_chunk - mel.shape[1]
            mel = np.pad(mel, ((0,0),(0,pad_width)), mode='constant', constant_values=mel.min())
        elif mel.shape[1] > frames_per_chunk:
            mel = mel[:, :frames_per_chunk]

        mel = mel.astype(np.float32)
        out_name = f"{path.stem}__chunk{idx:04d}.npz"
        out_path = out_dir / out_name
        np.savez_compressed(str(out_path), mel=mel)
        results.append({'file': str(out_path.relative_to(out_dir.parent)), 'source': str(path), 'chunk_idx': idx, 'start_s': start / sr, 'sr': sr})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='ravdess_videos_only')
    parser.add_argument('--output-dir', type=str, default='data/ravdess_mels_0.5s')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--chunk-secs', type=float, default=0.5)
    parser.add_argument('--frames-per-chunk', type=int, default=50)
    parser.add_argument('--n-mels', type=int, default=128)
    parser.add_argument('--n-fft', type=int, default=1024)
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / 'manifest.csv'
    fields = ['file', 'source', 'chunk_idx', 'start_s', 'sr']

    audio_files = [p for p in inp.rglob('*') if p.suffix.lower() in ('.wav', '.flac', '.mp3', '.ogg', '.mp4', '.mkv', '.mov', '.avi')]

    all_rows = []
    for p in tqdm(audio_files, desc='Files'):
        try:
            rows = process_file(p, out, sr=args.sr, chunk_secs=args.chunk_secs, frames_per_chunk=args.frames_per_chunk, n_mels=args.n_mels, n_fft=args.n_fft)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Failed to process {p}: {e}")
            traceback.print_exc()

    # write manifest
    with manifest_path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Wrote {len(all_rows)} chunks to {out}")


if __name__ == '__main__':
    main()
