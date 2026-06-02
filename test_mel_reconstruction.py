"""
Mel spectrogram reconstruction test script.

Process:
1. Extract the full audio clip from a RAVDESS video.
2. Build a mel spectrogram for the entire clip.
3. Save the waveform, mel features, and a visual comparison.
4. Reconstruct audio with a neural vocoder when available.
5. Report comparison metrics such as MSE, SNR, and correlation.
"""

import argparse
import csv
import importlib.util
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import av
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass

HIFI_GAN_REPO = Path("hifi-gan")
if HIFI_GAN_REPO.exists():
    sys.path.insert(0, str(HIFI_GAN_REPO.resolve()))


def load_hifigan_module(module_name: str, file_name: str):
    module_path = HIFI_GAN_REPO / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


env_module = load_hifigan_module("hifigan_env", "env.py")
meldataset = load_hifigan_module("hifigan_meldataset", "meldataset.py")
models = load_hifigan_module("hifigan_models", "models.py")

AttrDict = env_module.AttrDict
MAX_WAV_VALUE = meldataset.MAX_WAV_VALUE
Generator = models.Generator

_mel_basis_cache: dict[str, torch.Tensor] = {}
_hann_window_cache: dict[str, torch.Tensor] = {}


def _spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(magnitudes, min=1e-5))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MelConfig:
    """Mel reconstruction configuration for the provided HiFi-GAN model."""

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_size: int = 1024
    f_min: float = 0.0
    f_max: float = 8000.0
    n_mels: int = 80
    max_duration: Optional[float] = None
    config_file: Path = Path("UNIVERSAL_V1/config.json")
    checkpoint_file: Path = Path("UNIVERSAL_V1/g_02500000")


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
        if np.issubdtype(audio_full.dtype, np.integer):
            max_val = float(2 ** (8 * audio_full.dtype.itemsize - 1))
            audio_full = audio_full.astype(np.float32) / max_val
        else:
            audio_full = audio_full.astype(np.float32)
        if audio_full.ndim > 1:
            audio_full = audio_full.mean(axis=1)

        if audio_stream.rate != target_sr:
            audio_full = librosa.resample(audio_full, orig_sr=audio_stream.rate, target_sr=target_sr)

        return audio_full, target_sr
    except Exception as exc:
        logger.warning(f"Audio extraction failed: {exc}")
        return None, target_sr


def get_full_audio_clip(audio: np.ndarray, sr: int, max_duration: Optional[float] = None) -> np.ndarray:
    """Return the full clip, optionally truncating to a maximum duration."""
    if max_duration is None:
        return audio

    max_samples = int(max_duration * sr)
    if max_samples <= 0:
        return audio
    return audio[:max_samples]


def load_hifigan_model(config_file: Path, checkpoint_file: Path, device: torch.device) -> tuple[Generator, AttrDict]:
    if not config_file.exists():
        raise FileNotFoundError(f"HiFi-GAN config not found: {config_file}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"HiFi-GAN checkpoint not found: {checkpoint_file}")

    h = AttrDict(json.loads(config_file.read_text(encoding="utf-8")))
    generator = Generator(h).to(device)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
        raise RuntimeError(f"Unexpected checkpoint format: {checkpoint_file}")

    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator, h


def compute_hifigan_mel(audio: np.ndarray, h: AttrDict, device: torch.device) -> torch.Tensor:
    """Compute the mel tensor expected by the provided HiFi-GAN model."""
    audio_tensor = torch.FloatTensor(audio).to(device)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    cache_key = f"{h.sampling_rate}_{h.n_fft}_{h.num_mels}_{h.fmin}_{h.fmax}_{device}"
    if cache_key not in _mel_basis_cache:
        mel = librosa.filters.mel(
            sr=h.sampling_rate,
            n_fft=h.n_fft,
            n_mels=h.num_mels,
            fmin=h.fmin,
            fmax=h.fmax,
        )
        _mel_basis_cache[cache_key] = torch.from_numpy(mel).float().to(device)
        _hann_window_cache[str(device)] = torch.hann_window(h.win_size).to(device)

    mel_basis = _mel_basis_cache[cache_key]
    hann_window = _hann_window_cache[str(device)]

    audio_tensor = torch.nn.functional.pad(
        audio_tensor.unsqueeze(1),
        (int((h.n_fft - h.hop_size) / 2), int((h.n_fft - h.hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)

    spec = torch.stft(
        audio_tensor,
        h.n_fft,
        hop_length=h.hop_size,
        win_length=h.win_size,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(spec.abs().pow(2) + 1e-9)
    mel = torch.matmul(mel_basis, spec)
    mel = _spectral_normalize_torch(mel)
    return mel


def reconstruct_audio_with_hifigan(mel_tensor: torch.Tensor, generator: Generator) -> np.ndarray:
    """Run the provided HiFi-GAN generator on a full-clip mel tensor."""
    with torch.no_grad():
        audio = generator(mel_tensor).squeeze().detach().cpu().numpy()
    return audio.astype(np.float32)


def find_ravdess_video_files() -> list[Path]:
    """Find all RAVDESS videos under the expected local folder."""
    ravdess_path = Path("ravdess_videos_only")
    video_files: list[Path] = []
    if ravdess_path.exists():
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            video_files.extend(ravdess_path.rglob(f"*{ext}"))
    return sorted(video_files)


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
    parser = argparse.ArgumentParser(description="Reconstruct full-clip audio with the provided HiFi-GAN model")
    parser.add_argument("--max-duration", type=float, default=None, help="Optional maximum clip duration in seconds")
    parser.add_argument("--config-file", type=str, default=str(MelConfig.config_file), help="Path to the HiFi-GAN config.json")
    parser.add_argument("--checkpoint-file", type=str, default=str(MelConfig.checkpoint_file), help="Path to the HiFi-GAN generator checkpoint")
    parser.add_argument("--output-dir", type=str, default="mel_results", help="Directory for saved outputs")
    args = parser.parse_args()

    config = MelConfig(
        max_duration=args.max_duration,
        config_file=Path(args.config_file),
        checkpoint_file=Path(args.checkpoint_file),
    )
    video_files = find_ravdess_video_files()

    if not video_files:
        logger.error("No RAVDESS files found.")
        return

    selected_video = random.choice(video_files)
    audio, sr = extract_audio_from_video(selected_video, target_sr=config.sample_rate)
    if audio is None or len(audio) == 0:
        logger.error(f"Could not extract audio from selected file: {selected_video}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, h = load_hifigan_model(config.config_file, config.checkpoint_file, device)

    audio_clip = get_full_audio_clip(audio, sr, config.max_duration)
    audio_clip = librosa.util.normalize(audio_clip) * 0.95

    if sr != h.sampling_rate:
        audio_clip = librosa.resample(audio_clip, orig_sr=sr, target_sr=h.sampling_rate)
        sr = h.sampling_rate

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    sf.write(out_dir / "original.wav", audio_clip, sr)

    mel_tensor = compute_hifigan_mel(audio_clip, h, device)
    mel_np = mel_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    mel_npy = out_dir / "full_clip_hifigan_mel.npy"
    np.save(mel_npy, mel_np)

    audio_reconstructed = reconstruct_audio_with_hifigan(mel_tensor, generator)
    audio_reconstructed = align_audio_lengths(audio_clip, audio_reconstructed)
    sf.write(out_dir / "reconstructed_hifigan.wav", audio_reconstructed, sr)

    metrics = evaluate_reconstruction(audio_clip, audio_reconstructed)
    metrics["n_mels"] = h.num_mels
    metrics["samples"] = len(audio_clip)
    metrics["sample_rate"] = sr

    metrics_csv = out_dir / "reconstruction_metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["n_mels", "samples", "sample_rate", "snr_db", "corr", "mse"])
        writer.writeheader()
        writer.writerow(metrics)

    summary_txt = out_dir / "reconstruction_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Selected file: {selected_video}\n")
        file_obj.write(f"Clip duration: {len(audio_clip) / sr:.2f}s | Sample rate: {sr} Hz\n")
        file_obj.write(f"Mel bins: {h.num_mels}\n")
        file_obj.write(f"HiFi-GAN config: {config.config_file}\n")
        file_obj.write(f"HiFi-GAN checkpoint: {config.checkpoint_file}\n")
        file_obj.write("\nMetrics:\n")
        file_obj.write(f"MSE: {metrics['mse']:.6f}\n")
        file_obj.write(f"SNR: {metrics['snr_db']:.2f} dB\n")
        file_obj.write(f"Corr: {metrics['corr']:.4f}\n")
        file_obj.write(f"Samples: {metrics['samples']}\n")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_clip, sr=sr, alpha=0.8, label="Original")
    librosa.display.waveshow(audio_reconstructed, sr=sr, alpha=0.6, label="Reconstructed (HiFi-GAN)", color="r")
    plt.legend(loc="upper right")
    plt.title("Waveform Comparison")

    plt.subplot(2, 1, 2)
    img = librosa.display.specshow(mel_np, sr=sr, hop_length=h.hop_size, x_axis="time", y_axis="mel", fmin=h.fmin, fmax=h.fmax, cmap="magma")
    plt.colorbar(img, format="%+2.0f")
    plt.title("Full-Clip HiFi-GAN Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_full_clip.png", dpi=180, bbox_inches="tight")

    logger.info(f"Done! Files saved to {out_dir.absolute()}")
    logger.info(f"Mel NPY: {mel_npy}")
    logger.info(f"Metrics CSV: {metrics_csv}")
    logger.info(f"Summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
