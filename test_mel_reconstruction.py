"""
Mel Spectrogram Reconstruction Test Script

Process:
1. Extract audio from RAVDESS dataset.
2. Select a 0.5s high-energy speech segment.
3. Compute mel spectrograms with increasing mel resolution.
4. Reconstruct audio from each mel spectrogram.
5. Save all reconstructed clips and quality summaries.
"""

import csv
import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import av
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MelConfig:
    """Mel reconstruction configuration for short speech clips."""

    sample_rate: int = 16000
    chunk_duration: float = 0.5
    n_fft: int = 400
    hop_length: int = 160
    f_min: float = 80.0
    f_max: float = 7600.0

    # Sweep from coarse to fine mel resolution.
    mel_sweep: tuple[int, ...] = (16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256)


def extract_audio_from_video(video_path: Path, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int]:
    """Extract and resample mono audio from a video file."""
    try:
        container = av.open(str(video_path))
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        if not audio_stream:
            container.close()
            return None, target_sr

        audio_frames = []
        for frame in container.decode(audio=0):
            audio_data = frame.to_ndarray()
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)
            elif audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T
            audio_frames.append(audio_data)

        container.close()
        if not audio_frames:
            return None, target_sr

        audio_full = np.concatenate(audio_frames, axis=0)
        audio_full = audio_full.astype(np.float32) / (2**15 if audio_full.dtype == np.int16 else 1.0)
        if audio_full.ndim > 1:
            audio_full = audio_full.mean(axis=1)

        if audio_stream.rate != target_sr:
            audio_full = librosa.resample(audio_full, orig_sr=audio_stream.rate, target_sr=target_sr)

        return audio_full, target_sr
    except Exception as exc:
        logger.warning(f"Audio extraction failed: {exc}")
        return None, target_sr


def find_ravdess_video_files() -> list[Path]:
    """Find all RAVDESS videos under the expected local folder."""
    ravdess_path = Path("ravdess_videos_only")
    video_files: list[Path] = []
    if ravdess_path.exists():
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            video_files.extend(ravdess_path.rglob(f"*{ext}"))
    return sorted(video_files)


def find_speech_segment(audio: np.ndarray, sr: int, duration: float = 0.5) -> np.ndarray:
    """Return the highest-energy window of the requested duration."""
    num_samples = int(duration * sr)
    if len(audio) < num_samples:
        return np.pad(audio, (0, num_samples - len(audio)))

    hop = max(1, num_samples // 4)
    frames = librosa.util.frame(audio, frame_length=num_samples, hop_length=hop)
    rms = librosa.feature.rms(y=audio, frame_length=num_samples, hop_length=hop)
    best_idx = int(np.argmax(rms))
    return frames[:, best_idx]


def reconstruct_audio_from_mel(mel_spec: np.ndarray, config: MelConfig, num_iters: int = 64) -> np.ndarray:
    """Invert mel spectrogram back to waveform via Griffin-Lim."""
    audio_recon = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_iter=num_iters,
        fmin=config.f_min,
        fmax=config.f_max,
        window="hann",
        power=2.0,
    )
    return librosa.util.normalize(audio_recon)


def align_audio_lengths(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Pad/trim candidate to match reference length."""
    if len(candidate) < len(reference):
        return np.pad(candidate, (0, len(reference) - len(candidate)))
    return candidate[: len(reference)]


def evaluate_reconstruction(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    """Compute basic objective quality metrics."""
    eps = 1e-12
    diff = reference - reconstruction
    mse = float(np.mean(diff**2))
    ref_power = float(np.mean(reference**2)) + eps
    snr_db = 10.0 * np.log10(ref_power / (mse + eps))

    if np.std(reference) <= eps or np.std(reconstruction) <= eps:
        corr = 0.0
    else:
        corr = float(np.corrcoef(reference, reconstruction)[0, 1])

    return {"mse": mse, "snr_db": float(snr_db), "corr": corr}


def main() -> None:
    config = MelConfig()
    video_files = find_ravdess_video_files()

    if not video_files:
        logger.error("No RAVDESS files found.")
        return

    selected_video = random.choice(video_files)
    audio, sr = extract_audio_from_video(selected_video, target_sr=config.sample_rate)
    if audio is None or len(audio) == 0:
        logger.error(f"Could not extract audio from selected file: {selected_video}")
        return

    audio_chunk = find_speech_segment(audio, sr, duration=config.chunk_duration)

    out_dir = Path("mel_results")
    out_dir.mkdir(exist_ok=True)
    sf.write(out_dir / "original.wav", audio_chunk, sr)

    valid_sweep = sorted({n for n in config.mel_sweep if n > 0})
    results: list[dict[str, float | int]] = []
    waveform_examples: list[tuple[int, np.ndarray]] = []

    logger.info(f"Running mel sweep for bins: {valid_sweep}")
    for n_mels in valid_sweep:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_chunk,
            sr=sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=n_mels,
            fmin=config.f_min,
            fmax=config.f_max,
            power=2.0,
        )

        audio_reconstructed = reconstruct_audio_from_mel(mel_spec, config)
        audio_reconstructed = align_audio_lengths(audio_chunk, audio_reconstructed)

        metrics = evaluate_reconstruction(audio_chunk, audio_reconstructed)
        metrics["n_mels"] = n_mels
        results.append(metrics)

        sf.write(out_dir / f"reconstructed_mel_{n_mels:03d}.wav", audio_reconstructed, sr)
        logger.info(
            f"Saved reconstructed_mel_{n_mels:03d}.wav | "
            f"SNR={metrics['snr_db']:.2f} dB, Corr={metrics['corr']:.4f}, MSE={metrics['mse']:.6f}"
        )

        if n_mels in (valid_sweep[0], valid_sweep[len(valid_sweep) // 2], valid_sweep[-1]):
            waveform_examples.append((n_mels, audio_reconstructed.copy()))

    results.sort(key=lambda x: float(x["snr_db"]), reverse=True)

    metrics_csv = out_dir / "reconstruction_mel_sweep_metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["n_mels", "snr_db", "corr", "mse"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary_txt = out_dir / "reconstruction_mel_sweep_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Selected file: {selected_video}\n")
        file_obj.write(f"Clip duration: {config.chunk_duration:.2f}s | Sample rate: {sr} Hz\n")
        file_obj.write("\nRanked by SNR (higher is better):\n")
        for row in results:
            file_obj.write(
                f"MEL={int(row['n_mels']):>3d} | "
                f"SNR={float(row['snr_db']):.2f} dB | "
                f"Corr={float(row['corr']):.4f} | "
                f"MSE={float(row['mse']):.6f}\n"
            )

    if waveform_examples:
        plt.figure(figsize=(10, 8))
        for idx, (n_mels, waveform) in enumerate(waveform_examples, start=1):
            plt.subplot(len(waveform_examples), 1, idx)
            librosa.display.waveshow(audio_chunk, sr=sr, alpha=0.5, label="Original")
            librosa.display.waveshow(waveform, sr=sr, alpha=0.5, label=f"Recon MEL={n_mels}", color="r")
            plt.legend(loc="upper right")
            plt.title(f"Waveform Comparison (MEL={n_mels})")
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_mel_sweep_waveforms.png")

    logger.info(f"Done! Files saved to {out_dir.absolute()}")
    logger.info(f"Metrics CSV: {metrics_csv}")
    logger.info(f"Summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
