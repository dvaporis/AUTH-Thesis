"""
MFCC Reconstruction Test Script (Fixed Version)

Process:
1. Extract audio from RAVDESS dataset.
2. Compute MFCC features.
3. Use librosa.feature.inverse to transform MFCC -> Mel -> Magnitude -> Time.
4. Save and compare results.
"""

import librosa
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
import csv
import random
import av
from scipy import signal as scipy_signal
import soundfile as sf
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MFCCConfig:
    """MFCC configuration matching standard DSP pipelines."""
    sample_rate: int = 16000
    chunk_duration: float = 0.5
    num_mfcc: int = 13
    mfcc_sweep: tuple[int, ...] = (8, 10, 13, 16, 20, 24, 30, 40, 60, 80, 100, 128)
    n_mels: int = 128  # Essential for reconstruction mapping
    n_fft: int = 400
    hop_length: int = 160
    f_min: float = 80.0
    f_max: float = 7600.0
    
    # Derived
    num_samples: int = int(sample_rate * chunk_duration)
    num_frames: int = 1 + (num_samples) // hop_length 

def extract_audio_from_video(video_path: Path, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int]:
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
        if not audio_frames: return None, target_sr
        
        audio_full = np.concatenate(audio_frames, axis=0)
        audio_full = audio_full.astype(np.float32) / (2**15 if audio_full.dtype == np.int16 else 1.0)
        if audio_full.ndim > 1: audio_full = audio_full.mean(axis=1)
        
        if audio_stream.rate != target_sr:
            audio_full = librosa.resample(audio_full, orig_sr=audio_stream.rate, target_sr=target_sr)
        
        return audio_full, target_sr
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return None, target_sr

def find_ravdess_video_files() -> list[Path]:
    ravdess_path = Path("ravdess_videos_only")
    video_files = []
    if ravdess_path.exists():
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(ravdess_path.rglob(f"*{ext}"))
    return sorted(video_files)

def find_speech_segment(audio: np.ndarray, sr: int, duration: float = 0.5) -> np.ndarray:
    """Simple energy-based segment extractor."""
    num_samples = int(duration * sr)
    if len(audio) < num_samples:
        return np.pad(audio, (0, num_samples - len(audio)))
    
    # Simple trick: find the 0.5s window with the highest RMS energy
    frames = librosa.util.frame(audio, frame_length=num_samples, hop_length=num_samples // 4)
    rms = librosa.feature.rms(y=audio, frame_length=num_samples, hop_length=num_samples // 4)
    best_idx = np.argmax(rms)
    return frames[:, best_idx]

def reconstruct_audio_from_mfcc(mfcc: np.ndarray, config: MFCCConfig, num_iters: int = 64) -> np.ndarray:
    """
    Reconstructs audio using librosa's built-in inversion pipeline.
    This handles the IDCT, log-to-power conversion, and Griffin-Lim.
    """
    logger.info("Starting inversion: MFCC -> Mel -> Magnitude -> Audio")
    
    # 1. Map MFCC back to Mel Spectrogram (Inverts the DCT)
    # Ensure dct_type and norm match what was used in extraction
    mel_spec = librosa.feature.inverse.mfcc_to_mel(
        mfcc, 
        n_mels=config.n_mels, 
        dct_type=2, 
        norm='ortho'
    )
    
    # 2. Map Mel back to Audio (Inverts Mel-scale and estimates phase)
    audio_recon = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_iter=num_iters,
        window='hann'
    )
    
    return librosa.util.normalize(audio_recon)


def align_audio_lengths(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Pad or trim reconstruction so metric comparison is length-consistent."""
    if len(candidate) < len(reference):
        return np.pad(candidate, (0, len(reference) - len(candidate)))
    return candidate[:len(reference)]


def evaluate_reconstruction(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    """Compute simple objective metrics to compare MFCC reconstruction quality."""
    eps = 1e-12
    diff = reference - reconstruction
    mse = float(np.mean(diff ** 2))
    ref_power = float(np.mean(reference ** 2)) + eps
    snr_db = 10.0 * np.log10(ref_power / (mse + eps))
    corr = float(np.corrcoef(reference, reconstruction)[0, 1]) if np.std(reconstruction) > eps else 0.0
    return {
        'mse': mse,
        'snr_db': float(snr_db),
        'corr': corr,
    }

def main():
    config = MFCCConfig()
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

    # Save results
    out_dir = Path('mfcc_results')
    out_dir.mkdir(exist_ok=True)
    sf.write(out_dir / 'original.wav', audio_chunk, sr)

    valid_sweep = sorted({n for n in config.mfcc_sweep if 1 <= n <= config.n_mels})
    results = []
    waveform_examples = []

    logger.info(f"Running MFCC sweep for coefficients: {valid_sweep}")
    for n_mfcc in valid_sweep:
        mfcc = librosa.feature.mfcc(
            y=audio_chunk,
            sr=sr,
            n_mfcc=n_mfcc,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            fmin=config.f_min,
            fmax=config.f_max,
            norm='ortho'  # Using ortho norm makes inversion more stable
        )

        audio_reconstructed = reconstruct_audio_from_mfcc(mfcc, config)
        audio_reconstructed = align_audio_lengths(audio_chunk, audio_reconstructed)

        metrics = evaluate_reconstruction(audio_chunk, audio_reconstructed)
        metrics['n_mfcc'] = n_mfcc
        results.append(metrics)

        sf.write(out_dir / f'reconstructed_mfcc_{n_mfcc:03d}.wav', audio_reconstructed, sr)
        logger.info(
            f"Saved reconstructed_mfcc_{n_mfcc:03d}.wav | "
            f"SNR={metrics['snr_db']:.2f} dB, Corr={metrics['corr']:.4f}, MSE={metrics['mse']:.6f}"
        )

        if n_mfcc in (valid_sweep[0], valid_sweep[len(valid_sweep) // 2], valid_sweep[-1]):
            waveform_examples.append((n_mfcc, audio_reconstructed.copy()))

    # Sort by reconstruction quality (higher SNR is better)
    results.sort(key=lambda x: x['snr_db'], reverse=True)

    metrics_csv = out_dir / 'reconstruction_sweep_metrics.csv'
    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['n_mfcc', 'snr_db', 'corr', 'mse'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary_txt = out_dir / 'reconstruction_sweep_summary.txt'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(f"Selected file: {selected_video}\n")
        f.write(f"Clip duration: {config.chunk_duration:.2f}s | Sample rate: {sr} Hz\n")
        f.write("\nRanked by SNR (higher is better):\n")
        for row in results:
            f.write(
                f"MFCC={row['n_mfcc']:>3d} | "
                f"SNR={row['snr_db']:.2f} dB | "
                f"Corr={row['corr']:.4f} | "
                f"MSE={row['mse']:.6f}\n"
            )

    # Plotting: waveform overlays for a low/mid/high MFCC case
    if waveform_examples:
        plt.figure(figsize=(10, 8))
        for idx, (n_mfcc, waveform) in enumerate(waveform_examples, start=1):
            plt.subplot(len(waveform_examples), 1, idx)
            librosa.display.waveshow(audio_chunk, sr=sr, alpha=0.5, label='Original')
            librosa.display.waveshow(waveform, sr=sr, alpha=0.5, label=f'Recon MFCC={n_mfcc}', color='r')
            plt.legend(loc='upper right')
            plt.title(f"Waveform Comparison (MFCC={n_mfcc})")
        plt.tight_layout()
        plt.savefig(out_dir / 'comparison_sweep_waveforms.png')

    logger.info(f"Done! Files saved to {out_dir.absolute()}")
    logger.info(f"Metrics CSV: {metrics_csv}")
    logger.info(f"Summary TXT: {summary_txt}")

if __name__ == '__main__':
    main()