"""
Test EnCodec audio encoder-decoder from Meta.

This script:
1. Loads audio or generates a test signal
2. Encodes audio using EnCodec
3. Decodes the encoded representation
4. Calculates reconstruction error metrics
5. Saves and visualizes results
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
from scipy.io import wavfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_error_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """
    Compute various error metrics between original and reconstructed audio.
    
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
    
    # Signal-to-Noise Ratio (SNR) in dB
    signal_power = torch.mean(original ** 2).item()
    noise_power = torch.mean((original - reconstructed) ** 2).item()
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    metrics['SNR_dB'] = snr
    
    # Peak Signal-to-Noise Ratio (PSNR)
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


def load_or_create_audio(audio_path: str = None, duration: float = 5.0, sample_rate: int = 48000, channels: int = 2) -> tuple:
    """
    Load audio file or create a test signal if no file is provided.
    
    Args:
        audio_path: Path to audio file (optional)
        duration: Duration of test signal in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels (2 for stereo, 1 for mono)
    
    Returns:
        Tuple of (waveform, sample_rate)
    """
    if audio_path and Path(audio_path).exists():
        logger.info(f"Loading audio from {audio_path}")
        waveform, sr = torchaudio.load(audio_path)
        
        # Adjust channels if needed
        if waveform.shape[0] < channels:
            # Duplicate channels if too few
            waveform = waveform.repeat(channels, 1)[:channels]
        elif waveform.shape[0] > channels:
            # Average down if too many
            waveform = waveform[:channels]
        
        # Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        return waveform, sample_rate
    else:
        logger.info(f"Creating test signal: {duration}s at {sample_rate}Hz, {channels} channel(s)")
        # Create a test signal with multiple frequency components
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        
        # Combination of sine waves to create interesting audio
        signal = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
            0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 note
            0.15 * np.sin(2 * np.pi * 220 * t) +  # A3 note
            0.1 * np.random.randn(num_samples)  # Add some noise
        )
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.95
        waveform = torch.FloatTensor(signal).unsqueeze(0)
        
        # Create stereo if needed
        if channels == 2:
            # Create slightly different signal for right channel
            signal_r = (
                0.3 * np.sin(2 * np.pi * 440 * t + 0.1) +  # Slight phase shift
                0.2 * np.sin(2 * np.pi * 880 * t + 0.05) +
                0.15 * np.sin(2 * np.pi * 220 * t + 0.15) +
                0.1 * np.random.randn(num_samples)
            )
            signal_r = signal_r / np.max(np.abs(signal_r)) * 0.95
            waveform_r = torch.FloatTensor(signal_r).unsqueeze(0)
            waveform = torch.cat([waveform, waveform_r], dim=0)
        
        return waveform, sample_rate


def load_encodec_model(bandwidth: str = "24kbps", sample_rate: int = 48000, device: torch.device = None):
    """
    Load EnCodec model from encodec library.
    
    Args:
        bandwidth: Bandwidth setting ("6kbps", "24kbps", etc.)
        sample_rate: Sample rate (24000 or 48000)
        device: Device to load model on
        
    Returns:
        Loaded model and model type string
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load using encodec library (correct API)
    try:
        from encodec import EncodecModel
        logger.info(f"Loading EnCodec from encodec library...")
        model = EncodecModel.encodec_model_48khz()
        model.set_target_bandwidth(24.0)  # Set bandwidth to 24kbps for optimal quality
        model = model.to(device)
        model.eval()
        logger.info(f"✓ EnCodec model loaded: 48kHz, 24kbps bandwidth")
        return model, "encodec"
    except Exception as e:
        logger.error(f"Failed to load EnCodec model: {e}")
        raise RuntimeError(f"Could not load EnCodec model: {e}")


def test_encodec(audio_path: str = None, bandwidth: str = "24kbps", sample_rate: int = 48000):
    """
    Test EnCodec encoder-decoder.
    
    Args:
        audio_path: Path to audio file (optional, will generate test signal if not provided)
        bandwidth: Bandwidth setting for EnCodec ("1.5kbps", "3kbps", "6kbps", "12kbps", "24kbps")
        sample_rate: Target sample rate (24000 or 48000)
    """
    logger.info("="*60)
    logger.info(f"Testing EnCodec with {bandwidth} bandwidth at {sample_rate}Hz")
    logger.info("="*60)
    
    # Determine number of channels based on sample rate
    channels = 2 if sample_rate == 48000 else 1
    
    # Load audio
    waveform, original_sr = load_or_create_audio(audio_path, channels=channels)
    
    # Resample if needed to match target sample rate
    if original_sr != sample_rate:
        logger.info(f"Resampling from {original_sr}Hz to {sample_rate}Hz")
        waveform = torchaudio.functional.resample(waveform, original_sr, sample_rate)
    
    logger.info(f"Audio shape: {waveform.shape}, Sample rate: {sample_rate}Hz")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    waveform = waveform.to(device)
    
    # Add batch dimension if needed
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [batch, channels, samples]
    
    logger.info(f"Waveform shape for model: {waveform.shape}")
    
    # Load model
    model, model_type = load_encodec_model(bandwidth, sample_rate, device)
    
    # Encode and Decode with encodec library
    logger.info("Encoding audio...")
    with torch.no_grad():
        # encodec library returns list of frames, each containing (codes, scales)
        encoded_frames = model.encode(waveform)
        logger.info(f"Encoded frames: {len(encoded_frames)}")
        for i, frame in enumerate(encoded_frames):
            codes, scales = frame
            logger.info(f"  Frame {i+1}: codes shape {codes.shape}")
        
        # Decode
        logger.info("Decoding audio...")
        reconstructed = model.decode(encoded_frames)
    
    logger.info(f"Reconstructed shape: {reconstructed.shape}")
    
    # Remove batch dimension for comparison
    original = waveform.squeeze(0)
    reconstructed = reconstructed.squeeze(0)
    
    # Compute error metrics
    logger.info("\nComputing error metrics...")
    metrics = compute_error_metrics(original, reconstructed)
    
    # Log metrics
    logger.info("\nReconstruction Error Metrics:")
    logger.info("-" * 40)
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name:20s}: {metric_value:.6f}")
    logger.info("-" * 40)
    
    # Save results
    output_dir = Path("audio_encodec_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save audio files
    original_path = output_dir / "original_audio.wav"
    reconstructed_path = output_dir / "reconstructed_audio.wav"
    
    # Use scipy.io.wavfile to avoid torchcodec dependency
    original_np = original.cpu().numpy().T  # scipy expects [samples, channels]
    reconstructed_np = reconstructed.cpu().numpy().T
    wavfile.write(str(original_path), sample_rate, original_np)
    wavfile.write(str(reconstructed_path), sample_rate, reconstructed_np)
    logger.info(f"\nSaved original audio to: {original_path}")
    logger.info(f"Saved reconstructed audio to: {reconstructed_path}")
    
    # Visualize
    visualize_comparison(original.cpu().numpy(), reconstructed.cpu().numpy(), sample_rate, metrics, output_dir)
    
    return metrics, waveform.shape


def visualize_comparison(original: np.ndarray, reconstructed: np.ndarray, 
                         sample_rate: int, metrics: dict, output_dir: Path):
    """
    Create visualizations comparing original and reconstructed audio.
    
    Args:
        original: Original audio array [channels, samples]
        reconstructed: Reconstructed audio array [channels, samples]
        sample_rate: Sample rate in Hz
        metrics: Dictionary of error metrics
        output_dir: Directory to save plots
    """
    logger.info("Creating visualizations...")
    
    # Work with first channel if stereo
    if original.shape[0] > 1:
        original = original[0]
    if reconstructed.shape[0] > 1:
        reconstructed = reconstructed[0]
    
    # Trim reconstructed to match original length (EnCodec may pad)
    min_length = min(original.shape[-1], reconstructed.shape[-1])
    original = original[:min_length]
    reconstructed = reconstructed[:min_length]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Set time axis
    num_samples = original.shape[-1]
    time = np.arange(num_samples) / sample_rate
    
    # Plot 1: Original vs Reconstructed waveform
    axes[0].plot(time, original, label='Original', alpha=0.7, linewidth=0.8)
    axes[0].plot(time, reconstructed, label='Reconstructed', alpha=0.7, linewidth=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original vs Reconstructed Audio Waveform')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction error
    error = original - reconstructed
    axes[1].plot(time, error, label='Error', color='red', alpha=0.7, linewidth=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Reconstruction Error (RMSE: {metrics["RMSE"]:.6f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Spectrogram comparison
    # Only use first 3 seconds for spectrogram for clarity
    max_samples = min(num_samples, 3 * sample_rate)
    original_spec = np.abs(np.fft.rfft(original[:max_samples]))
    reconstructed_spec = np.abs(np.fft.rfft(reconstructed[:max_samples]))
    freqs = np.fft.rfftfreq(max_samples, 1/sample_rate)
    
    axes[2].semilogy(freqs, original_spec, label='Original', alpha=0.7, linewidth=0.8)
    axes[2].semilogy(freqs, reconstructed_spec, label='Reconstructed', alpha=0.7, linewidth=0.8)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude')
    axes[2].set_title('Frequency Spectrum (First 3s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plot_path = output_dir / "encodec_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to: {plot_path}")
    plt.close()
    
    # Create metrics summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Normalize values for better visualization (show absolute values and scale if needed)
    colors = ['green' if v > 0 or k.endswith('_dB') or k == 'Cosine_Similarity' else 'red' 
              for k, v in metrics.items()]
    
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.7)
    ax.set_xlabel('Value')
    ax.set_title('EnCodec Reconstruction Error Metrics')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax.text(value, i, f' {value:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    metrics_path = output_dir / "metrics_summary.png"
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved metrics summary to: {metrics_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EnCodec audio encoder-decoder")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (optional)")
    parser.add_argument("--bandwidth", type=str, default="6kbps", 
                       choices=["1.5kbps", "3kbps", "6kbps", "12kbps", "24kbps"],
                       help="EnCodec bandwidth")
    parser.add_argument("--sample-rate", type=int, default=48000,
                       choices=[24000, 48000],
                       help="Sample rate (default: 48000 stereo; 24000 for mono)")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Duration of test signal in seconds (if no audio file provided)")
    
    args = parser.parse_args()
    
    try:
        metrics, shape = test_encodec(audio_path=args.audio, bandwidth=args.bandwidth, sample_rate=args.sample_rate)
        logger.info("\n" + "="*60)
        logger.info("EnCodec test completed successfully!")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"\nError during EnCodec test: {e}", exc_info=True)
        exit(1)
