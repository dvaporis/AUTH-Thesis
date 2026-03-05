"""
Enhanced EnCodec test using Kaggle audio-visual dataset.

This script extends the basic test to work with real audio/video files from the 
Audio-Visual Database of Emotional Speech and Song dataset.
Supports extracting audio from MP4 videos or loading from WAV files.
Extracts fixed sample counts (no padding) and displays actual duration based on sample rate.

Usage:
    # First download the dataset
    python download_kaggle_dataset.py
    
    # Then test with dataset files (prefers MP4 videos)
    python test_encodec_with_kaggle.py --use-kaggle
    
    # Or test specific file
    python test_encodec_with_kaggle.py --audio path/to/audio.wav
    python test_encodec_with_kaggle.py --audio path/to/video.mp4
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
import random
import av

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_kaggle_audio_files(kaggle_dir: str = "kaggle_datasets") -> list:
    """Find video files (01-*.mp4) from Kaggle dataset - extracts audio from videos only."""
    import os
    
    possible_paths = [
        Path(kaggle_dir),
        Path(os.path.expanduser("~/.cache/kagglehub/datasets")),
        Path(os.environ.get("USERPROFILE", "~")) / ".cache/kagglehub/datasets",
    ]
    
    video_files = []
    
    for kaggle_path in possible_paths:
        if kaggle_path.exists():
            # Find video files starting with "01-" (speech videos with audio)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(kaggle_path.rglob(f"*{ext}"))
            
            # Filter to only "01-" prefix (speech videos with embedded audio)
            video_files = [v for v in all_videos if v.name.startswith('01-')]
            
            if video_files:
                logger.info(f"Found {len(video_files)} speech video files (01-*.mp4, will extract audio) in {kaggle_path}")
                break
    
    if not video_files:
        logger.warning("No video files with '01-' prefix found! Ensure Kaggle dataset is downloaded.")
    
    return video_files


def load_or_create_audio(audio_path: Optional[str] = None, duration: float = 5.0,
                         sample_rate: int = 48000, num_samples: Optional[int] = None,
                         num_video_frames: int = 16, full_audio: bool = False) -> Tuple[torch.Tensor, int, float, str]:
    """
    Load audio from audio/video file or create a test signal.
    Supports extracting audio from MP4 videos using PyAV.
    Extracts a frame-aligned sample count for video files using actual FPS:
    samples = round(num_video_frames / fps * sample_rate).
    
    Args:
        audio_path: Path to audio/video file (optional)
        duration: Duration of test signal in seconds
        sample_rate: Sample rate in Hz
        num_samples: Optional explicit sample count override
        num_video_frames: Number of video frames represented by each audio bite
        full_audio: If True, load entire audio file without frame-alignment limitation
    
    Returns:
        Tuple of (waveform, sample_rate, duration_seconds, source_description)
        where duration_seconds is the actual duration of the extracted samples
    """
    default_samples = int(round(sample_rate * num_video_frames / 30.0))
    target_num_samples = num_samples if num_samples is not None else default_samples

    if audio_path and Path(audio_path).exists():
        logger.info(f"Loading audio from: {audio_path}")
        file_path = Path(audio_path)
        is_video = file_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}
        
        try:
            if is_video:
                # Extract audio from video using PyAV
                logger.info(f"Extracting audio from video file...")
                container = av.open(str(audio_path))

                # Derive sample count from actual video fps for exact frame alignment
                video_fps = None
                video_stream = container.streams.video[0] if container.streams.video else None
                if video_stream:
                    if video_stream.average_rate is not None:
                        video_fps = float(video_stream.average_rate)
                    elif video_stream.base_rate is not None:
                        video_fps = float(video_stream.base_rate)

                if video_fps and video_fps > 0:
                    target_num_samples = int(round(sample_rate * num_video_frames / video_fps))
                    logger.info(
                        f"Frame-aligned extraction: {num_video_frames} frames @ {video_fps:.3f} fps "
                        f"=> {target_num_samples} samples"
                    )
                else:
                    logger.warning(
                        f"Could not determine video FPS, using fallback {target_num_samples} samples "
                        f"(~{num_video_frames} frames @ 30fps)"
                    )
                
                audio_stream = container.streams.audio[0] if container.streams.audio else None
                if not audio_stream:
                    container.close()
                    raise ValueError("No audio stream in video file")
                
                sr = audio_stream.rate
                
                # Decode all audio frames
                audio_frames = []
                for frame in container.decode(audio=0):
                    audio_data = frame.to_ndarray()
                    
                    # Ensure [samples, channels] format
                    if audio_data.ndim == 1:
                        audio_data = audio_data.reshape(-1, 1)
                    elif audio_data.shape[0] < audio_data.shape[1]:
                        audio_data = audio_data.T
                    
                    audio_frames.append(audio_data)
                
                container.close()
                
                if not audio_frames:
                    raise ValueError("No audio frames decoded")
                
                # Concatenate and convert
                audio_data = np.concatenate(audio_frames, axis=0)
                
                # Normalize
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    audio_data = audio_data.astype(np.float32)
                
                # Convert to torch [channels, samples]
                if audio_data.ndim == 1:
                    waveform = torch.from_numpy(audio_data).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(audio_data.T)
                
            else:
                # Load WAV file with scipy
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
            
            # Extract frame-aligned target samples (unless full_audio is requested)
            if not full_audio:
                if waveform.shape[1] > target_num_samples:
                    logger.info(
                        f"Extracting {target_num_samples} samples "
                        f"({target_num_samples/sample_rate:.3f}s target duration)..."
                    )
                    # Take a random segment from the audio
                    max_start = waveform.shape[1] - target_num_samples
                    start_idx = random.randint(0, max_start) if max_start > 0 else 0
                    waveform = waveform[:, start_idx:start_idx + target_num_samples]
                else:
                    logger.warning(
                        f"Audio shorter than target bite ({target_num_samples} samples, "
                        f"{waveform.shape[1]} available)"
                    )
            else:
                logger.info(
                    f"Loading full audio file: {waveform.shape[1]} samples "
                    f"({waveform.shape[1]/sample_rate:.3f}s)"
                )
            
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
    
    # Extract only target_num_samples if available
    if waveform.shape[1] > target_num_samples:
        max_start = waveform.shape[1] - target_num_samples
        start_idx = random.randint(0, max_start)
        waveform = waveform[:, start_idx:start_idx + target_num_samples]
    
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


def test_encodec_with_audio(audio_path: Optional[str] = None, use_kaggle: bool = False,
                            num_video_frames: int = 16, full_audio: bool = False):
    """
    Test EnCodec encoder-decoder with real or generated audio.
    
    Args:
        audio_path: Path to audio file (optional)
        use_kaggle: Whether to use a random file from Kaggle dataset
        num_video_frames: Number of video frames represented by each audio bite
        full_audio: If True, use entire audio file instead of frame-aligned extraction
    """
    logger.info("="*70)
    if full_audio:
        logger.info("Testing EnCodec Audio Codec (Full Audio)")
    else:
        logger.info(f"Testing EnCodec Audio Codec (Frame-Aligned: {num_video_frames} video frames)")
    logger.info("="*70)
    
    # Determine audio to test
    if use_kaggle:
        kaggle_files = find_kaggle_audio_files()
        if not kaggle_files:
            logger.info("Run: python download_kaggle_dataset.py")
            raise FileNotFoundError("Kaggle dataset not found or empty")
        
        audio_path = str(random.choice(kaggle_files))
        logger.info(f"Using random Kaggle audio: {audio_path}")
    
    # Load audio
    waveform, sample_rate, duration_seconds, source_desc = load_or_create_audio(
        audio_path,
        num_video_frames=num_video_frames,
        full_audio=full_audio
    )
    
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
    
    # Load EnCodec model directly using encodec library
    try:
        from encodec import EncodecModel
        logger.info("\nLoading EnCodec model...")
        
        # Create the EnCodec model with maximum bandwidth for best reconstruction
        model = EncodecModel.encodec_model_48khz()
        # Set bandwidth to 24kbps (maximum quality, supports 1.5/3/6/12/24)
        model.set_target_bandwidth(24.0)
        model.to(device)
        model.eval()
        logger.info("✓ EnCodec 48kHz model loaded successfully (bandwidth: 24kbps)")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Make sure encodec library is properly installed: pip install encodec")
        raise
    
    # Encode and reconstruct
    logger.info("\nProcessing audio...")
    with torch.no_grad():
        # Encode
        logger.info("  Encoding...")
        encoded_frames = model.encode(waveform_batch)
        logger.info(f"  ✓ Encoded to {len(encoded_frames)} frame(s)")
        
        # Log embedding dimensions
        logger.info(f"\n  Embedding Details:")
        for i, frame in enumerate(encoded_frames):
            logger.info(f"    Frame {i+1}: codes shape = {frame[0].shape}, scale shape = {frame[1].shape if len(frame) > 1 else 'N/A'}")
        
        # Decode
        logger.info("  Decoding...")
        reconstructed = model.decode(encoded_frames)
        logger.info(f"  ✓ Reconstructed: {reconstructed.shape}")
    
    # Prepare for metrics
    original = waveform_batch.squeeze(0)
    reconstructed = reconstructed.squeeze(0)
    
    # Trim reconstructed to match original length (EnCodec may pad to frame boundaries)
    if reconstructed.shape[-1] > original.shape[-1]:
        logger.info(f"  Trimming reconstructed from {reconstructed.shape[-1]} to {original.shape[-1]} samples to match original length")
        reconstructed = reconstructed[..., :original.shape[-1]]
    
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
    
    # Take first channel if stereo, don't flatten all channels together
    if original.ndim > 1:
        original = original[0]  # Take first channel
    if reconstructed.ndim > 1:
        reconstructed = reconstructed[0]  # Take first channel
    
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
    
    # Spectrum - use actual audio length, no padding
    original_spec = np.abs(np.fft.rfft(original))
    reconstructed_spec = np.abs(np.fft.rfft(reconstructed))
    freqs = np.fft.rfftfreq(len(original), 1/sample_rate)
    
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
    parser.add_argument("--num-video-frames", type=int, default=16,
                       help="Number of video frames represented by each audio bite")
    parser.add_argument("--full-audio", action="store_true",
                       help="Load and process entire audio file (ignores --num-video-frames)")
    
    args = parser.parse_args()
    
    try:
        metrics = test_encodec_with_audio(
            audio_path=args.audio,
            use_kaggle=args.use_kaggle,
            num_video_frames=args.num_video_frames,
            full_audio=args.full_audio
        )
        logger.info("\n" + "="*70)
        logger.info("✓ EnCodec test completed successfully!")
        logger.info("="*70)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        exit(1)
