#!/usr/bin/env python3
"""Preprocess audio to full-length mel-spectrograms (default 80 bins, 22050 Hz).

Saves per-file .npz with key 'mel' (float32, shape (n_mels, frames)).
Writes a manifest CSV with metadata: file, source, duration_s, sr, frames

Usage:
    python preprocess_mels_full.py --input-dir ravdess_videos_only --output-dir data/ravdess_mels_full
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


VIDEO_EXTS = ('.mp4', '.mkv', '.mov', '.avi', '.mpg', '.mpeg')
AUDIO_EXTS = ('.wav', '.flac', '.mp3', '.ogg')


def resolve_ffmpeg_executable():
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg

    imageio_ffmpeg = try_import('imageio_ffmpeg')
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    return None


def load_audio(path: Path, sr: int):
    ext = path.suffix.lower()
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
                    if np.issubdtype(audio_full.dtype, np.integer):
                        max_val = float(2 ** (8 * audio_full.dtype.itemsize - 1))
                        audio_full = audio_full.astype(np.float32) / max_val
                    else:
                        audio_full = audio_full.astype(np.float32)

                    if audio_full.ndim > 1:
                        audio_full = audio_full.mean(axis=1)

                    stream_rate = getattr(audio_stream, 'rate', None)
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
                f"No video decode backend available for {path}. Install ffmpeg or PyAV."
            )

        tmp = None
        try:
            tmp_f = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp = tmp_f.name
            tmp_f.close()
            cmd = [ffmpeg, '-y', '-i', str(path), '-ar', str(sr), '-ac', '1', '-vn', str(tmp)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            y, _ = librosa.load(tmp, sr=sr, mono=True)
            return y
        except subprocess.CalledProcessError as e:
            stderr_tail = (e.stderr or '').strip().splitlines()[-10:]
            stderr_msg = '\n'.join(stderr_tail) if stderr_tail else '<no ffmpeg stderr>'
            raise RuntimeError(f"ffmpeg failed for {path}:\n{stderr_msg}") from e
        finally:
            if tmp is not None and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def compute_mel(y, sr, n_mels, n_fft, hop_length, power=2.0):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def process_file(path: Path, out_dir: Path, sr: int, n_mels: int, n_fft: int, hop_length: int):
    y = load_audio(path, sr)
    if y is None or y.size == 0:
        return None

    mel = compute_mel(y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel = mel.astype(np.float32)
    out_name = f"{path.stem}.npz"
    out_path = out_dir / out_name
    np.savez_compressed(str(out_path), mel=mel)
    duration = float(y.shape[0]) / float(sr)
    frames = int(mel.shape[1])
    return {'file': str(out_path.relative_to(out_dir.parent)), 'source': str(path), 'duration_s': duration, 'sr': sr, 'frames': frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='ravdess_videos_only')
    parser.add_argument('--output-dir', type=str, default='data/ravdess_mels_full')
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--n-mels', type=int, default=80)
    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--hop-length', type=int, default=512)
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / 'manifest.csv'
    fields = ['file', 'source', 'duration_s', 'sr', 'frames']

    audio_files = [p for p in inp.rglob('*') if p.suffix.lower() in (AUDIO_EXTS + VIDEO_EXTS)]

    has_video_inputs = any(p.suffix.lower() in VIDEO_EXTS for p in audio_files)
    if has_video_inputs and try_import('av') is None and resolve_ffmpeg_executable() is None:
        raise RuntimeError(
            'Detected video inputs but neither PyAV nor ffmpeg is available in this environment.'
        )

    all_rows = []
    for p in tqdm(audio_files, desc='Files'):
        try:
            row = process_file(p, out, sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)
            if row is not None:
                all_rows.append(row)
        except Exception as e:
            print(f"Failed to process {p}: {type(e).__name__}: {e!r}")
            traceback.print_exc()

    with manifest_path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Wrote {len(all_rows)} files to {out}")


if __name__ == '__main__':
    main()
