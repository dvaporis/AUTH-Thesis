"""
Enhanced EnCodec test using Kaggle audio-visual dataset.

This script extends the basic test to work with real audio files from the 
Audio-Visual Database of Emotional Speech and Song dataset.
Extracts fixed sample counts (no padding) and displays actual duration based on sample rate.

Usage:
    # First download the dataset
    python download_kaggle_dataset.py
    
    # Then test with dataset audio files
    python test_encodec_with_kaggle.py --use-kaggle
    
    # Or test specific file
    python test_encodec_with_kaggle.py --audio path/to/audio.wav
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_kaggle_audio_files(kaggle_dir: str = "kaggle_datasets") -> list:
    """
    Find all audio files in the Kaggle dataset directory.
    
    Args:
        kaggle_dir: Root directory of Kaggle dataset
        
    Returns:
        List of audio file paths
    """
    import os
    
    # Check multiple possible locations
    possible_paths = [
        Path(kaggle_dir),
        Path(os.path.expanduser("~/.cache/kagglehub/datasets")),
        Path(os.environ.get("USERPROFILE", "~")) / ".cache/kagglehub/datasets",
    ]
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []
    
    for kaggle_path in possible_paths:
        if kaggle_path.exists():
            for file in kaggle_path.rglob("*"):
                if file.suffix.lower() in audio_extensions:
                    audio_files.append(file)
            
            if audio_files:
                logger.info(f"Found {len(audio_files)} audio files in {kaggle_path}")
                break
    
    return audio_files


def load_or_create_audio(audio_path: Optional[str] = None, duration: float = 5.0, 
                         sample_rate: int = 48000, num_samples: int = 25600) -> Tuple[torch.Tensor, int, float, str]:
    """
    Load audio file or create a test signal.
    Extracts exactly num_samples without padding (matching video encoder approach).
    
    Args:
        audio_path: Path to audio file (optional)
        duration: Duration of test signal in seconds
        sample_rate: Sample rate in Hz
        num_samples: Number of samples to extract (default 25600 ≈ 16 frames at 30fps at 48kHz)
    
    Returns:
        Tuple of (waveform, sample_rate, duration_seconds, source_description)
        where duration_seconds is the actual duration of the extracted samples
    """
    if audio_path and Path(audio_path).exists():
        logger.info(f"Loading audio from: {audio_path}")
        try:
            # Try scipy first (more reliable for WAV files)
            from scipy.io import wavfile
            sr, audio_data = wavfile.read(audio_path)
            
            # Convert to float and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # Convert to torch tensor
            if audio_data.ndim == 1:
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_data.T)
            
            # Resample if necessary
            if sr != sample_rate:
                logger.info(f"Resampling from {sr}Hz to {sample_rate}Hz...")
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # Preserve stereo for 48kHz model, convert to mono for 24kHz
            if sample_rate == 48000:
                # Preserve stereo (pad mono to stereo, truncate >2 channels to 2)
                if waveform.shape[0] == 1:
                    logger.info("Padding mono to stereo")
                    waveform = waveform.repeat(2, 1)
                elif waveform.shape[0] > 2:
                    logger.info(f"Truncating {waveform.shape[0]} channels to stereo")
                    waveform = waveform[:2, :]
            else:
                # Convert to mono for 24kHz
                if waveform.shape[0] > 1:
                    logger.info(f"Converting stereo to mono ({waveform.shape[0]} channels)")
                    waveform = waveform.mean(dim=0, keepdim=True)
            
            # Limit duration if too long
            max_samples = int(sample_rate * 30)  # 30 second max
            if waveform.shape[1] > max_samples:
                logger.info(f"Trimming audio to 30 seconds...")
                waveform = waveform[:, :max_samples]
            
            # Extract exactly num_samples (no padding, matching video encoder)
            if waveform.shape[1] > num_samples:
                logger.info(f"Extracting {num_samples} samples ({num_samples}/{sample_rate}={num_samples/sample_rate:.3f}s)...")
                # Take a random segment from the audio
                max_start = waveform.shape[1] - num_samples
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                waveform = waveform[:, start_idx:start_idx + num_samples]
            else:
                logger.warning(f"Audio shorter than {num_samples} samples ({waveform.shape[1]} available)")
            
            # Calculate actual duration based on extracted samples
            actual_duration = waveform.shape[1] / sample_rate
            source_desc = f"Real audio: {Path(audio_path).name} ({waveform.shape[1]} samples, {actual_duration:.3f}s @ {sample_rate}Hz)"
            logger.info(f"✓ Loaded: {waveform.shape}, {source_desc}")
            return waveform, sample_rate, actual_duration, source_desc
            
        except Exception as e:
            logger.warning(f"Could not load {audio_path}: {e}")
            logger.info("Falling back to generated test signal...")
    
    # Generate test signal
    logger.info(f"Creating test signal: {duration}s at {sample_rate}Hz")
    num_full_samples = int(duration * sample_rate)
    t = np.arange(num_full_samples) / sample_rate
    
    # Determine channels based on sample rate
    channels = 2 if sample_rate == 48000 else 1
    
    if channels == 2:
        # Create stereo signal with phase-shifted channels
        left = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
            0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 note
            0.15 * np.sin(2 * np.pi * 220 * t) +  # A3 note
            0.1 * np.random.randn(num_full_samples)  # Noise
        )
        right = (
            0.3 * np.sin(2 * np.pi * 440.1 * t) +  # Slightly detuned
            0.2 * np.sin(2 * np.pi * 880.05 * t) +
            0.15 * np.sin(2 * np.pi * 220.15 * t) +
            0.1 * np.random.randn(num_full_samples)
        )
        # Normalize
        left = left / np.max(np.abs(left)) * 0.95
        right = right / np.max(np.abs(right)) * 0.95
        waveform = torch.FloatTensor(np.stack([left, right], axis=0))
    else:
        # Mono signal
        signal = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
            0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 note
            0.15 * np.sin(2 * np.pi * 220 * t) +  # A3 note
            0.1 * np.random.randn(num_full_samples)  # Noise
        )
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.95
        waveform = torch.FloatTensor(signal).unsqueeze(0)
    
    # Extract only num_samples if available
    if waveform.shape[1] > num_samples:
        max_start = waveform.shape[1] - num_samples
        start_idx = random.randint(0, max_start)
        waveform = waveform[:, start_idx:start_idx + num_samples]
    
    # Calculate actual duration
    actual_duration = waveform.shape[1] / sample_rate
    source_desc = f"Generated test signal: {waveform.shape[1]} samples, {actual_duration:.3f}s @ {sample_rate}Hz"
    
    return waveform, sample_rate, actual_duration, source_desc


def compute_error_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """
    Compute reconstruction error metrics.
    
    Args:
        original: Original audio tensor [channels, samples]
        reconstructed: Reconstructed audio tensor [channels, samples]
    
    Returns:
        Dictionary with error metrics
    """
    # Ensure same shape
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]
    
    metrics = {}
    
    # Mean Squared Error
    mse = torch.mean((original - reconstructed) ** 2).item()
    metrics['MSE'] = mse
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(original - reconstructed)).item()
    metrics['MAE'] = mae
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    metrics['RMSE'] = rmse
    
    # Signal-to-Noise Ratio
    signal_power = torch.mean(original ** 2).item()
    noise_power = torch.mean((original - reconstructed) ** 2).item()
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    metrics['SNR_dB'] = snr
    
    # Peak SNR
    peak_value = torch.max(torch.abs(original)).item()
    psnr = 20 * np.log10(peak_value / (rmse + 1e-10))
    metrics['PSNR_dB'] = psnr
    
    # Cosine similarity
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()
    cos_sim = torch.cosine_similarity(
        original_flat.unsqueeze(0), 
        reconstructed_flat.unsqueeze(0)
    ).item()
    metrics['Cosine_Similarity'] = cos_sim
    
    return metrics


def test_encodec_with_audio(audio_path: Optional[str] = None, use_kaggle: bool = False):
    """
    Test EnCodec encoder-decoder with real or generated audio.
    
    Args:
        audio_path: Path to audio file (optional)
        use_kaggle: Whether to use a random file from Kaggle dataset
    """
    logger.info("="*70)
    logger.info("Testing EnCodec Audio Codec (Fixed Sample Count)")
    logger.info("="*70)
    
    # Determine audio to test
    if use_kaggle:
        kaggle_files = find_kaggle_audio_files()
        if not kaggle_files:
            logger.error("No audio files found in Kaggle dataset!")
            logger.info("Run: python download_kaggle_dataset.py")
            raise FileNotFoundError("Kaggle dataset not found or empty")
        
        audio_path = str(random.choice(kaggle_files))
        logger.info(f"Using random Kaggle audio: {audio_path}")
    
    # Load audio (extracts fixed num_samples, no padding)
    waveform, sample_rate, duration_seconds, source_desc = load_or_create_audio(audio_path)
    logger.info(f"Audio shape: {waveform.shape}, Sample rate: {sample_rate}Hz")
    logger.info(f"Duration: {duration_seconds:.3f} seconds")
    logger.info(f"Source: {source_desc}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    waveform = waveform.to(device)
    
    # Add batch dimension
    if waveform.dim() == 2:
        waveform_batch = waveform.unsqueeze(0)
    else:
        waveform_batch = waveform
    
    logger.info(f"Batch shape: {waveform_batch.shape}")
    
    # Load EnCodec model
    try:
        from transformers import AutoModel
        logger.info("\nLoading EnCodec model from HuggingFace...")
        model_name = "facebook/encodec_48khz" if sample_rate == 48000 else "facebook/encodec_24khz"
        logger.info(f"Using model: {model_name}")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device)
        model.eval()
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Encode and reconstruct
    logger.info("\nProcessing audio...")
    with torch.no_grad():
        # Encode
        logger.info("  Encoding...")
        encoded = model.encode(waveform_batch)
        if hasattr(encoded, 'audio_codes'):
            logger.info(f"  ✓ Encoded to codes: {encoded.audio_codes.shape}")
        else:
            logger.info(f"  ✓ Encoded (type: {type(encoded).__name__})")
        
        # Decode
        logger.info("  Decoding...")
        if hasattr(encoded, 'audio_codes'):
            # HuggingFace model format
            decoded_output = model.decode(encoded.audio_codes, encoded.audio_scales)
        else:
            decoded_output = model.decode(encoded)
        
        # Extract audio from decoder output
        if hasattr(decoded_output, 'audio_values'):
            reconstructed = decoded_output.audio_values
        elif isinstance(decoded_output, tuple):
            reconstructed = decoded_output[0]
        else:
            reconstructed = decoded_output
        
        logger.info(f"  ✓ Reconstructed: {reconstructed.shape}")
    
    # Prepare for metrics
    original = waveform_batch.squeeze(0)
    reconstructed = reconstructed.squeeze(0)
    
    # Compute metrics
    logger.info("\nComputing error metrics...")
    metrics = compute_error_metrics(original, reconstructed)
    
    # Log results
    logger.info("\nReconstruction Quality Metrics:")
    logger.info("-" * 50)
    logger.info(f"  Source                 : {source_desc}")
    logger.info(f"  Audio duration         : {duration_seconds:.3f} seconds")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name:20s}: {metric_value:10.6f}")
    logger.info("-" * 50)
    
    # Save results
    output_dir = Path("audio_encodec_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create descriptive filename
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem if audio_path else "generated"
    
    # Save audio
    original_path = output_dir / f"{timestamp}_{base_name}_original.wav"
    reconstructed_path = output_dir / f"{timestamp}_{base_name}_reconstructed.wav"
    
    try:
        from scipy.io import wavfile
        # Convert to PCM format (int16)
        original_pcm = (original.cpu().numpy() * 32767).astype(np.int16)
        reconstructed_pcm = (reconstructed.cpu().numpy() * 32767).astype(np.int16)
        
        wavfile.write(original_path, sample_rate, 
                     original_pcm.T if original_pcm.ndim > 1 else original_pcm)
        wavfile.write(reconstructed_path, sample_rate, 
                     reconstructed_pcm.T if reconstructed_pcm.ndim > 1 else reconstructed_pcm)
    except Exception as e:
        logger.warning(f"Could not save audio: {e}")
    
    logger.info(f"\n✓ Saved original to: {original_path}")
    logger.info(f"✓ Saved reconstructed to: {reconstructed_path}")
    
    # Visualize
    visualize_results(original.cpu().numpy(), reconstructed.cpu().numpy(), 
                     sample_rate, metrics, source_desc, output_dir, timestamp, base_name)
    
    return metrics


def visualize_results(original: np.ndarray, reconstructed: np.ndarray, 
                     sample_rate: int, metrics: Dict[str, float], 
                     source_desc: str, output_dir: Path, timestamp: str, base_name: str):
    """
    Create and save visualizations.
    """
    logger.info("\nCreating visualizations...")
    
    # Flatten if needed
    if original.ndim > 1:
        original = original.flatten()
    if reconstructed.ndim > 1:
        reconstructed = reconstructed.flatten()
    
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time = np.arange(len(original)) / sample_rate
    
    # Waveform comparison
    axes[0].plot(time, original, label='Original', alpha=0.7, linewidth=0.8)
    axes[0].plot(time, reconstructed, label='Reconstructed', alpha=0.7, linewidth=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Waveform Comparison - {source_desc}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error plot
    error = original - reconstructed
    axes[1].plot(time, error, label='Error', color='red', alpha=0.7, linewidth=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Reconstruction Error (RMSE: {metrics["RMSE"]:.6f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Spectrum
    max_samples = min(len(original), 3 * sample_rate)
    original_spec = np.abs(np.fft.rfft(original[:max_samples]))
    reconstructed_spec = np.abs(np.fft.rfft(reconstructed[:max_samples]))
    freqs = np.fft.rfftfreq(max_samples, 1/sample_rate)
    
    axes[2].semilogy(freqs, original_spec, label='Original', alpha=0.7, linewidth=0.8)
    axes[2].semilogy(freqs, reconstructed_spec, label='Reconstructed', alpha=0.7, linewidth=0.8)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude')
    axes[2].set_title('Frequency Spectrum')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plot_path = output_dir / f"{timestamp}_{base_name}_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved comparison plot to: {plot_path}")
    plt.close()
    
    # Metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    colors = ['green' if v > 0 or k.endswith('_dB') or k == 'Cosine_Similarity' else 'blue' 
              for k, v in metrics.items()]
    
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.7)
    ax.set_xlabel('Value')
    ax.set_title(f'Metrics - {source_desc}')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax.text(value, i, f' {value:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    metrics_path = output_dir / f"{timestamp}_{base_name}_metrics.png"
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved metrics plot to: {metrics_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EnCodec with real audio files")
    parser.add_argument("--audio", type=str, default=None, 
                       help="Path to audio file to test")
    parser.add_argument("--use-kaggle", action="store_true",
                       help="Use a random audio file from Kaggle dataset")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Duration for generated test signal")
    
    args = parser.parse_args()
    
    try:
        metrics = test_encodec_with_audio(
            audio_path=args.audio,
            use_kaggle=args.use_kaggle
        )
        logger.info("\n" + "="*70)
        logger.info("✓ EnCodec test completed successfully!")
        logger.info("="*70)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        exit(1)
