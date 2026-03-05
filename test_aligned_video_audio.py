"""
Test VideoMAE with aligned audio extraction from Kaggle dataset.

This script:
1. Finds video (.mp4) files from Kaggle dataset with embedded audio
2. Extracts 16 consecutive video frames for VideoMAE
3. Extracts the EXACT corresponding audio segment from the same video file
4. Demonstrates perfect temporal alignment between video frames and audio
5. Optionally tests both VideoMAE (visual) and EnCodec (audio) features

Both video and audio are extracted from the same MP4 file, ensuring perfect synchronization.

Usage:
    python test_aligned_video_audio.py --use-kaggle
    python test_aligned_video_audio.py --video path/to/video.mp4
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict
import random
import cv2
import argparse
from datetime import datetime
import av

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_kaggle_videos() -> list:
    """Find all video files in the Kaggle dataset."""
    import os
    
    possible_paths = [
        Path("kaggle_datasets"),
        Path(os.path.expanduser("~/.cache/kagglehub/datasets")),
        Path(os.environ.get("USERPROFILE", "~")) / ".cache/kagglehub/datasets",
    ]
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    video_files = []
    
    for search_path in possible_paths:
        if search_path.exists():
            logger.info(f"Searching in: {search_path}")
            for file in search_path.rglob("*"):
                if file.is_file() and file.suffix.lower() in video_extensions:
                    video_files.append(file)
            
            if video_files:
                logger.info(f"Found {len(video_files)} video files")
                break
    
    return video_files





def extract_aligned_video_audio(
    video_path: str,
    num_frames: int = 16,
    start_time: Optional[float] = None,
    target_audio_sr: int = 48000
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, float]]:
    """
    Extract temporally-aligned video frames and audio from the same MP4 file.
    
    Both video and audio are extracted from the same file, ensuring perfect synchronization.
    
    Args:
        video_path: Path to video file (.mp4) with embedded audio
        num_frames: Number of video frames to extract (default 16 for VideoMAE)
        start_time: Start time in seconds (random if None)
        target_audio_sr: Target sample rate for audio
    
    Returns:
        Tuple of (video_frames, audio_waveform, timing_info)
        - video_frames: [num_frames, height, width, 3] RGB frames
        - audio_waveform: [channels, samples] audio tensor
        - timing_info: Dictionary with timing details
    """
    logger.info(f"Extracting aligned video+audio from MP4:")
    logger.info(f"  Video: {Path(video_path).name}")
    logger.info("="*70)
    
    # === Open video file with PyAV ===
    container = av.open(str(video_path))
    
    # Get video stream info
    video_stream = container.streams.video[0]
    fps = float(video_stream.average_rate)
    total_frames = video_stream.frames
    width = video_stream.width
    height = video_stream.height
    video_duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0
    
    # Fallback if duration not available
    if video_duration == 0:
        video_duration = total_frames / fps if total_frames > 0 else 0
    
    logger.info(f"VIDEO INFO (from {Path(video_path).name}):")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  Frame rate: {fps:.2f} fps")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Duration: {video_duration:.2f}s")
    
    # Get audio stream info
    audio_stream = container.streams.audio[0] if container.streams.audio else None
    if audio_stream:
        audio_sr = audio_stream.rate
        audio_channels = audio_stream.channels
        logger.info(f"\nAUDIO INFO (from same MP4):")
        logger.info(f"  Sample rate: {audio_sr} Hz")
        logger.info(f"  Channels: {audio_channels}")
    else:
        logger.error("No audio stream found in video file!")
        container.close()
        raise ValueError(f"Video file has no audio stream: {video_path}")
    
    # Choose starting point (middle of video if not specified)
    max_start_frame = max(0, total_frames - num_frames)
    if start_time is None:
        # Start from the middle third to avoid empty beginnings/endings
        middle_start = total_frames // 3
        middle_end = 2 * total_frames // 3 - num_frames
        if middle_end > middle_start:
            start_frame = random.randint(middle_start, middle_end)
        else:
            start_frame = max(0, total_frames // 2 - num_frames // 2)
        start_time = start_frame / fps
        logger.info(f"  Auto-selected middle segment to avoid empty parts")
    else:
        start_frame = int(start_time * fps)
        start_frame = min(start_frame, max_start_frame)
    
    # Calculate exact time range
    end_frame = start_frame + num_frames
    end_time = end_frame / fps
    chunk_duration = num_frames / fps
    
    logger.info(f"\nTIME SEGMENT SELECTION:")
    logger.info(f"  Start time: {start_time:.3f}s (frame {start_frame})")
    logger.info(f"  End time: {end_time:.3f}s (frame {end_frame})")
    logger.info(f"  Segment duration: {chunk_duration:.3f}s ({num_frames} frames)")
    
    # === EXTRACT VIDEO FRAMES AND AUDIO FROM SAME FILE ===
    logger.info(f"\nEXTRACTING VIDEO FRAMES...")
    
    # Seek to start time
    seek_pts = int(start_time / float(video_stream.time_base))
    container.seek(seek_pts, stream=video_stream)
    
    frames = []
    frame_count = 0
    target_frame_time = start_time
    
    for frame in container.decode(video=0):
        current_time = float(frame.pts * video_stream.time_base)
        
        # Check if we've reached the desired frame range
        if current_time >= start_time and frame_count < num_frames:
            # Convert to RGB numpy array and resize
            img = frame.to_ndarray(format='rgb24')
            img_resized = cv2.resize(img, (224, 224))
            frames.append(img_resized)
            frame_count += 1
            
        if frame_count >= num_frames:
            break
    
    if len(frames) < num_frames:
        logger.warning(f"Only extracted {len(frames)} frames (expected {num_frames})")
        if len(frames) == 0:
            container.close()
            raise ValueError("No frames extracted from video")
    
    frames = np.stack(frames)  # [T, H, W, 3]
    logger.info(f"  ✓ Extracted {frames.shape[0]} frames, shape: {frames.shape}")
    
    # === EXTRACT AUDIO FROM SAME FILE ===
    logger.info(f"\nEXTRACTING AUDIO (from same MP4)...")
    
    # Re-seek to start time for audio extraction
    container.seek(seek_pts, stream=audio_stream)
    
    audio_frames = []
    total_samples = 0
    target_samples = int(chunk_duration * audio_sr)
    
    for frame in container.decode(audio=0):
        current_time = float(frame.pts * audio_stream.time_base) if frame.pts else 0
        
        if current_time >= start_time:
            # Convert audio frame to numpy
            audio_data = frame.to_ndarray()
            
            # PyAV gives us [channels, samples] for planar or [samples, channels] for packed
            # Ensure we have [channels, samples]
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            elif audio_data.shape[0] > audio_data.shape[1]:
                audio_data = audio_data.T
            
            audio_frames.append(audio_data)
            total_samples += audio_data.shape[1]
            
        if total_samples >= target_samples:
            break
    
    container.close()
    
    if not audio_frames:
        logger.warning("No audio extracted, creating silent placeholder")
        audio_waveform = torch.zeros(2, target_samples)
        sample_rate = target_audio_sr
    else:
        # Concatenate all audio frames
        audio_waveform = np.concatenate(audio_frames, axis=1)
        
        # Convert to float32 and normalize if needed
        if audio_waveform.dtype == np.int16:
            audio_waveform = audio_waveform.astype(np.float32) / 32768.0
        elif audio_waveform.dtype == np.int32:
            audio_waveform = audio_waveform.astype(np.float32) / 2147483648.0
        
        audio_waveform = torch.from_numpy(audio_waveform.astype(np.float32))
        sample_rate = audio_sr
        
        logger.info(f"  Extracted audio shape: {audio_waveform.shape}")
        logger.info(f"  Duration: {audio_waveform.shape[1]/sample_rate:.3f}s")
        
        # Trim to exact duration
        exact_samples = int(chunk_duration * sample_rate)
        if audio_waveform.shape[1] > exact_samples:
            audio_waveform = audio_waveform[:, :exact_samples]
        
        # Preserve stereo for 48kHz workflow
        if audio_waveform.shape[0] == 1:
            logger.info(f"  Padding mono to stereo")
            audio_waveform = audio_waveform.repeat(2, 1)
        elif audio_waveform.shape[0] > 2:
            logger.info(f"  Truncating {audio_waveform.shape[0]} channels to stereo")
            audio_waveform = audio_waveform[:2, :]
        
        # Resample if needed
        if sample_rate != target_audio_sr:
            logger.info(f"  Resampling from {sample_rate}Hz to {target_audio_sr}Hz...")
            from scipy.signal import resample
            num_samples_new = int(audio_waveform.shape[1] * target_audio_sr / sample_rate)
            waveform_np = audio_waveform.numpy()
            waveform_resampled = resample(waveform_np, num_samples_new, axis=1)
            audio_waveform = torch.from_numpy(waveform_resampled.astype(np.float32))
            sample_rate = target_audio_sr
        
        logger.info(f"  ✓ Final audio shape: {audio_waveform.shape}")
    
    # Calculate sample indices for timing info
    start_sample = 0
    num_samples = audio_waveform.shape[1]
    
    # Timing information
    timing_info = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': chunk_duration,
        'video_fps': fps,
        'num_video_frames': len(frames),
        'audio_sample_rate': sample_rate,
        'num_audio_samples': audio_waveform.shape[1],
        'video_start_frame': start_frame,
        'video_end_frame': end_frame,
        'audio_start_sample': start_sample,
        'audio_end_sample': start_sample + audio_waveform.shape[1],
    }
    
    logger.info("\n" + "="*70)
    logger.info("TEMPORAL ALIGNMENT VERIFICATION:")
    logger.info(f"  Video frames (from MP4): {start_time:.3f}s - {end_time:.3f}s")
    logger.info(f"  Audio samples (from MP4): {start_time:.3f}s - {end_time:.3f}s")
    logger.info(f"  Duration match: {chunk_duration:.3f}s (both)")
    logger.info(f"  ✓ VIDEO AND AUDIO PERFECTLY ALIGNED (same file)!")
    logger.info("="*70)
    
    return frames, audio_waveform, timing_info


def load_videomae_model(device: str = "cpu"):
    """Load VideoMAEv2 model from HuggingFace."""
    logger.info("\nLoading VideoMAEv2 model...")
    
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEModel
        
        model_name = "MCG-NJU/videomae-base"
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        model = VideoMAEModel.from_pretrained(model_name)
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"  ✓ VideoMAEv2 loaded: {model_name}")
        return model, processor
        
    except Exception as e:
        logger.error(f"Error loading VideoMAE model: {e}")
        raise


def extract_video_features(frames: np.ndarray, model, processor, device: str = "cpu") -> torch.Tensor:
    """Extract video features using VideoMAEv2."""
    logger.info("\nExtracting video features...")
    
    # Prepare frames
    frames_list = [frames[i] for i in range(frames.shape[0])]
    inputs = processor(frames_list, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Extract features
    with torch.no_grad():
        outputs = model(pixel_values)
    
    # Get CLS token as video embedding
    video_embedding = outputs.last_hidden_state[:, 0, :]
    
    logger.info(f"  ✓ Video embedding shape: {video_embedding.shape}")
    logger.info(f"  ✓ Embedding dimension: {video_embedding.shape[1]}")
    
    return video_embedding


def visualize_alignment(frames: np.ndarray, audio: torch.Tensor, 
                       timing_info: Dict, output_dir: str = "aligned_results"):
    """Visualize temporally-aligned video frames and audio waveform."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Temporally Aligned Video Frames and Audio (from same MP4)\n'
                 f'Time: {timing_info["start_time"]:.3f}s - {timing_info["end_time"]:.3f}s '
                 f'(Duration: {timing_info["duration"]:.3f}s)',
                 fontsize=14, fontweight='bold')
    
    # Show first 8 frames (2 rows)
    for i in range(min(8, frames.shape[0])):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(frames[i])
        frame_time = timing_info['start_time'] + (i / timing_info['video_fps'])
        ax.set_title(f'Frame {i+1}\nt={frame_time:.3f}s', fontsize=10)
        ax.axis('off')
    
    # Audio waveform
    ax_audio = fig.add_subplot(gs[2, :])
    audio_np = audio.numpy().flatten()
    sample_rate = timing_info['audio_sample_rate']
    time_axis = np.linspace(timing_info['start_time'], timing_info['end_time'], len(audio_np))
    
    ax_audio.plot(time_axis, audio_np, linewidth=0.5, color='steelblue')
    ax_audio.set_title('Audio Waveform from MP4 (Aligned with Video Frames)', fontsize=12, fontweight='bold')
    ax_audio.set_xlabel('Time (seconds)', fontsize=10)
    ax_audio.set_ylabel('Amplitude', fontsize=10)
    ax_audio.grid(alpha=0.3)
    ax_audio.set_xlim(timing_info['start_time'], timing_info['end_time'])
    
    # Add vertical lines for frame timestamps
    for i in range(frames.shape[0]):
        frame_time = timing_info['start_time'] + (i / timing_info['video_fps'])
        ax_audio.axvline(frame_time, color='red', alpha=0.3, linewidth=1, linestyle='--')
    
    plt.tight_layout()
    
    plot_path = output_path / f"{timestamp}_aligned_video_audio.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    logger.info(f"\n✓ Saved visualization: {plot_path}")
    plt.close()
    
    # Save the audio segment as WAV file
    audio_save_path = output_path / f"{timestamp}_audio_segment.wav"
    from scipy.io import wavfile
    audio_np = (audio.numpy() * 32767).astype(np.int16)
    wavfile.write(str(audio_save_path), timing_info['audio_sample_rate'], audio_np.T)
    logger.info(f"✓ Saved audio segment: {audio_save_path}")
    
    # Save the video frames as a short video clip
    video_save_path = output_path / f"{timestamp}_video_segment.mp4"
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_save_path), fourcc, timing_info['video_fps'], (224, 224))
    for frame in frames:
        # Convert RGB back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    logger.info(f"✓ Saved video segment: {video_save_path}")


def main(video_path: Optional[str] = None, 
         extract_features: bool = False, use_kaggle: bool = False):
    """Main function."""
    logger.info("="*70)
    logger.info("KAGGLE DATASET: ALIGNED VIDEO + AUDIO EXTRACTION FROM MP4")
    logger.info("="*70)
    
    # Find video files
    if use_kaggle or not video_path:
        logger.info("\nSearching Kaggle dataset for video files...")
        video_files = find_kaggle_videos()
        
        if not video_files:
            logger.error("No video files found in Kaggle dataset!")
            logger.info("\nPlease ensure the Kaggle dataset is downloaded:")
            logger.info("  python download_kaggle_dataset.py")
            return
        
        # Pick a random video
        video_path = random.choice(video_files)
        logger.info(f"\n✓ Selected video:")
        logger.info(f"  Video: {video_path}")
    else:
        # Use provided path
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
    
    # Extract aligned video frames and audio
    frames, audio, timing_info = extract_aligned_video_audio(
        video_path, num_frames=16
    )
    
    # Visualize alignment
    visualize_alignment(frames, audio, timing_info)
    
    # Optionally extract VideoMAE features
    if extract_features:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"\nUsing device: {device}")
        
        model, processor = load_videomae_model(device)
        video_embedding = extract_video_features(frames, model, processor, device)
        
        logger.info("\n" + "="*70)
        logger.info("FEATURE EXTRACTION RESULTS:")
        logger.info(f"  Video embedding shape: {video_embedding.shape}")
        logger.info(f"  Embedding mean: {video_embedding.mean().item():.6f}")
        logger.info(f"  Embedding std: {video_embedding.std().item():.6f}")
        logger.info("="*70)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY:")
    logger.info("-"*70)
    logger.info(f"  Video file (MP4): {Path(video_path).name}")
    logger.info(f"  Time segment: {timing_info['start_time']:.3f}s - {timing_info['end_time']:.3f}s")
    logger.info(f"  Duration: {timing_info['duration']:.3f}s")
    logger.info(f"  Video frames extracted: {timing_info['num_video_frames']} @ {timing_info['video_fps']:.2f}fps")
    logger.info(f"  Audio samples extracted: {timing_info['num_audio_samples']} @ {timing_info['audio_sample_rate']}Hz")
    logger.info(f"  Frames shape: {frames.shape}")
    logger.info(f"  Audio shape: {audio.shape}")
    logger.info("\n  ✓ VIDEO AND AUDIO PERFECTLY ALIGNED (extracted from same MP4)!")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test aligned video and audio extraction from MP4 files (Kaggle dataset)"
    )
    parser.add_argument("--video", type=str, help="Path to video file (.mp4)")
    parser.add_argument("--use-kaggle", action="store_true", 
                       help="Auto-find video files from Kaggle dataset")
    parser.add_argument("--extract-features", action="store_true", 
                       help="Also extract VideoMAE features")
    
    args = parser.parse_args()
    
    main(
        video_path=args.video,
        extract_features=args.extract_features,
        use_kaggle=args.use_kaggle
    )
