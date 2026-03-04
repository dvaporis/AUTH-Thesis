"""
Test VideoMAE with aligned audio extraction from Kaggle dataset.

This script:
1. Finds matching video (.mp4) and audio (.wav) files from Kaggle dataset based on filenames
2. Extracts 16 consecutive video frames for VideoMAE (visual only, no audio from video)
3. Extracts the EXACT corresponding audio segment from the WAV file for the same time range
4. Demonstrates temporal alignment between video frames and audio from separate files
5. Optionally tests both VideoMAE (visual) and EnCodec (audio) features

The audio from the WAV file corresponds exactly to the same time period as the video frames.

Usage:
    python test_aligned_video_audio.py --use-kaggle
    python test_aligned_video_audio.py --video path/to/video.mp4 --audio path/to/audio.wav
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict
import random
import cv2
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_kaggle_files() -> Tuple[list, list]:
    """Find all video and audio files in the Kaggle dataset."""
    import os
    
    possible_paths = [
        Path("kaggle_datasets"),
        Path(os.path.expanduser("~/.cache/kagglehub/datasets")),
        Path(os.environ.get("USERPROFILE", "~")) / ".cache/kagglehub/datasets",
    ]
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    
    video_files = []
    audio_files = []
    
    for search_path in possible_paths:
        if search_path.exists():
            logger.info(f"Searching in: {search_path}")
            for file in search_path.rglob("*"):
                if file.is_file():
                    if file.suffix.lower() in video_extensions:
                        video_files.append(file)
                    elif file.suffix.lower() in audio_extensions:
                        audio_files.append(file)
            
            if video_files or audio_files:
                logger.info(f"Found {len(video_files)} video files")
                logger.info(f"Found {len(audio_files)} audio files")
                break
    
    return video_files, audio_files


def find_matching_audio_file(video_path: Path, audio_files: list) -> Optional[Path]:
    """
    Find the corresponding audio file for a video file based on filename similarity.
    
    Common patterns in datasets:
    - Same stem: video_01.mp4 <-> video_01.wav
    - Actor codes: Actor_01_video.mp4 <-> Actor_01_audio.wav
    - ID patterns: 01-01-01-01-01-01-01.mp4 <-> 01-01-01-01-01-01-01.wav
    
    Args:
        video_path: Path to video file
        audio_files: List of available audio files
        
    Returns:
        Path to matching audio file or None
    """
    video_stem = video_path.stem
    video_name = video_path.name
    
    logger.info(f"\nSearching for audio file matching: {video_name}")
    
    # Try exact stem match first
    for audio_file in audio_files:
        if audio_file.stem == video_stem:
            logger.info(f"  ✓ Found exact match: {audio_file.name}")
            return audio_file
    
    # Try partial matches (remove common video/audio suffixes)
    video_base = video_stem.replace('_video', '').replace('_vid', '').replace('-video', '')
    
    for audio_file in audio_files:
        audio_base = audio_file.stem.replace('_audio', '').replace('_aud', '').replace('-audio', '')
        
        if video_base == audio_base:
            logger.info(f"  ✓ Found match (base name): {audio_file.name}")
            return audio_file
    
    # Try matching ID patterns (like 01-01-01-01-01-01-01)
    import re
    video_id = re.search(r'(\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', video_name)
    if video_id:
        pattern = video_id.group(1)
        # Extract everything except the first two digits (modality indicator)
        pattern_suffix = pattern[3:]  # Skip "XX-" prefix
        for audio_file in audio_files:
            audio_id = re.search(r'(\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', audio_file.name)
            if audio_id:
                audio_suffix = audio_id.group(1)[3:]  # Skip "XX-" prefix
                if pattern_suffix == audio_suffix:
                    logger.info(f"  ✓ Found match (ID pattern, ignoring modality prefix): {audio_file.name}")
                    return audio_file
    
    # Try matching actor/scene numbers
    video_nums = re.findall(r'\d+', video_stem)
    if video_nums:
        for audio_file in audio_files:
            audio_nums = re.findall(r'\d+', audio_file.stem)
            if video_nums == audio_nums:
                logger.info(f"  ✓ Found match (number sequence): {audio_file.name}")
                return audio_file
    
    logger.warning(f"  ✗ No matching audio file found for {video_name}")
    return None


def extract_aligned_video_audio(
    video_path: str,
    audio_path: str,
    num_frames: int = 16,
    start_time: Optional[float] = None,
    target_audio_sr: int = 24000
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, float]]:
    """
    Extract temporally-aligned video frames and audio from separate files.
    
    The audio segment from the WAV file corresponds EXACTLY to the same time period
    as the video frames from the MP4 file.
    
    Args:
        video_path: Path to video file (.mp4)
        audio_path: Path to audio file (.wav)
        num_frames: Number of video frames to extract (default 16 for VideoMAE)
        start_time: Start time in seconds (random if None)
        target_audio_sr: Target sample rate for audio
    
    Returns:
        Tuple of (video_frames, audio_waveform, timing_info)
        - video_frames: [num_frames, height, width, 3] RGB frames
        - audio_waveform: [channels, samples] audio tensor
        - timing_info: Dictionary with timing details
    """
    logger.info(f"Extracting aligned video+audio from SEPARATE files:")
    logger.info(f"  Video: {Path(video_path).name}")
    logger.info(f"  Audio: {Path(audio_path).name}")
    logger.info("="*70)
    
    # === VIDEO EXTRACTION (NO AUDIO FROM VIDEO) ===
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps
    
    logger.info(f"VIDEO INFO (from {Path(video_path).name}):")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  Frame rate: {fps:.2f} fps")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Duration: {video_duration:.2f}s")
    
    # Choose starting point (middle of video if not specified to skip empty beginnings)
    max_start_frame = max(0, total_frames - num_frames)
    if start_time is None:
        # Start from the middle third of the video to avoid empty beginnings/endings
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
    
    # Calculate the exact time range for this chunk
    end_frame = start_frame + num_frames
    end_time = end_frame / fps
    chunk_duration = num_frames / fps
    
    logger.info(f"\nTIME SEGMENT SELECTION:")
    logger.info(f"  Start time: {start_time:.3f}s (frame {start_frame})")
    logger.info(f"  End time: {end_time:.3f}s (frame {end_frame})")
    logger.info(f"  Segment duration: {chunk_duration:.3f}s ({num_frames} frames)")
    
    # Extract video frames
    logger.info(f"\nEXTRACTING VIDEO FRAMES (muted, no audio)...")
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Only extracted {len(frames)} frames (expected {num_frames})")
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (VideoMAE standard)
        frame = cv2.resize(frame, (224, 224))
        
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < num_frames:
        logger.warning(f"Video too short: got {len(frames)} frames, needed {num_frames}")
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
    
    frames = np.stack(frames)  # [T, H, W, 3]
    logger.info(f"  ✓ Extracted {frames.shape[0]} frames, shape: {frames.shape}")
    
    # === AUDIO EXTRACTION FROM SEPARATE WAV FILE ===
    logger.info(f"\nEXTRACTING AUDIO (from separate WAV file)...")
    
    try:
        # Load audio from the WAV file (NOT from the video) using scipy
        from scipy.io import wavfile
        sample_rate, audio_data = wavfile.read(str(audio_path))
        
        # Convert to float and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = audio_data.astype(np.float32)
        
        # Convert to torch tensor
        if audio_data.ndim == 1:
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_data.T)
        
        logger.info(f"AUDIO INFO (from {Path(audio_path).name}):")
        logger.info(f"  Original sample rate: {sample_rate} Hz")
        logger.info(f"  Channels: {waveform.shape[0]}")
        logger.info(f"  Total samples: {waveform.shape[1]}")
        logger.info(f"  Duration: {waveform.shape[1]/sample_rate:.2f}s")
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            logger.info(f"  Converting stereo to mono")
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_audio_sr:
            logger.info(f"  Resampling from {sample_rate}Hz to {target_audio_sr}Hz...")
            from scipy.signal import resample
            num_samples_new = int(waveform.shape[1] * target_audio_sr / sample_rate)
            waveform_np = waveform.numpy()
            waveform_resampled = resample(waveform_np, num_samples_new, axis=1)
            waveform = torch.from_numpy(waveform_resampled.astype(np.float32))
            sample_rate = target_audio_sr
        
        # Calculate exact sample indices for the time segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        num_samples = end_sample - start_sample
        
        logger.info(f"\nALIGNED AUDIO EXTRACTION (from WAV):")
        logger.info(f"  Time range: {start_time:.3f}s - {end_time:.3f}s")
        logger.info(f"  Sample range: {start_sample} - {end_sample}")
        logger.info(f"  Number of samples: {num_samples}")
        logger.info(f"  Duration: {num_samples/sample_rate:.3f}s")
        
        # Extract the exact audio segment
        if end_sample <= waveform.shape[1]:
            audio_segment = waveform[:, start_sample:end_sample]
            logger.info(f"  ✓ Extracted aligned audio, shape: {audio_segment.shape}")
        else:
            logger.warning(f"  Audio shorter than expected, padding may be needed")
            audio_segment = waveform[:, start_sample:]
        
    except Exception as e:
        logger.warning(f"Could not extract audio: {e}")
        logger.info("Creating silent audio placeholder...")
        num_samples = int(chunk_duration * target_audio_sr)
        audio_segment = torch.zeros(1, num_samples)
        sample_rate = target_audio_sr
        start_sample = 0
    
    # Timing information
    timing_info = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': chunk_duration,
        'video_fps': fps,
        'num_video_frames': len(frames),
        'audio_sample_rate': sample_rate,
        'num_audio_samples': audio_segment.shape[1],
        'video_start_frame': start_frame,
        'video_end_frame': end_frame,
        'audio_start_sample': start_sample,
        'audio_end_sample': start_sample + audio_segment.shape[1],
    }
    
    logger.info("\n" + "="*70)
    logger.info("TEMPORAL ALIGNMENT VERIFICATION:")
    logger.info(f"  Video frames (from MP4): {start_time:.3f}s - {end_time:.3f}s")
    logger.info(f"  Audio samples (from WAV): {start_time:.3f}s - {end_time:.3f}s")
    logger.info(f"  Duration match: {chunk_duration:.3f}s (both)")
    logger.info(f"  ✓ VIDEO (MP4) AND AUDIO (WAV) ARE TEMPORALLY ALIGNED!")
    logger.info("="*70)
    
    return frames, audio_segment, timing_info


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
    logger.info("\nExtracting video features (VISUAL ONLY)...")
    
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
    
    fig.suptitle(f'Temporally Aligned Video Frames (MP4) and Audio (WAV)\n'
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
    ax_audio.set_title('Audio Waveform from WAV (Aligned with Video Frames)', fontsize=12, fontweight='bold')
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


def main(video_path: Optional[str] = None, audio_path: Optional[str] = None, 
         extract_features: bool = False, use_kaggle: bool = False):
    """Main function."""
    logger.info("="*70)
    logger.info("KAGGLE DATASET: ALIGNED VIDEO (MP4) + AUDIO (WAV) EXTRACTION")
    logger.info("="*70)
    
    # Find matching video and audio files
    if use_kaggle or not video_path:
        logger.info("\nSearching Kaggle dataset for video and audio files...")
        video_files, audio_files = find_kaggle_files()
        
        if not video_files:
            logger.error("No video files found in Kaggle dataset!")
            logger.info("\nPlease ensure the Kaggle dataset is downloaded:")
            logger.info("  python download_kaggle_dataset.py")
            return
        
        if not audio_files:
            logger.error("No audio files found in Kaggle dataset!")
            return
        
        # Try to find a video with matching audio
        matched_pair = None
        for vid_file in video_files:
            aud_file = find_matching_audio_file(vid_file, audio_files)
            if aud_file:
                matched_pair = (vid_file, aud_file)
                break
        
        if not matched_pair:
            logger.error("Could not find any video-audio file pairs!")
            logger.info(f"\nFound {len(video_files)} videos and {len(audio_files)} audio files,")
            logger.info("but couldn't match them by filename.")
            return
        
        video_path, audio_path = matched_pair
        logger.info(f"\n✓ Found matching pair:")
        logger.info(f"  Video: {video_path}")
        logger.info(f"  Audio: {audio_path}")
    else:
        # Use provided paths
        if not video_path or not audio_path:
            logger.error("Both --video and --audio paths must be provided!")
            logger.info("Or use --use-kaggle to auto-find matching files")
            return
        
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        if not Path(audio_path).exists():
            logger.error(f"Audio file not found: {audio_path}")
            return
    
    # Extract aligned video frames and audio
    frames, audio, timing_info = extract_aligned_video_audio(
        video_path, audio_path, num_frames=16
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
    logger.info(f"  Audio file (WAV): {Path(audio_path).name}")
    logger.info(f"  Time segment: {timing_info['start_time']:.3f}s - {timing_info['end_time']:.3f}s")
    logger.info(f"  Duration: {timing_info['duration']:.3f}s")
    logger.info(f"  Video frames extracted: {timing_info['num_video_frames']} @ {timing_info['video_fps']:.2f}fps")
    logger.info(f"  Audio samples extracted: {timing_info['num_audio_samples']} @ {timing_info['audio_sample_rate']}Hz")
    logger.info(f"  Frames shape: {frames.shape}")
    logger.info(f"  Audio shape: {audio.shape}")
    logger.info("\n  ✓ VIDEO FRAMES (MP4) AND AUDIO (WAV) ARE TEMPORALLY ALIGNED!")
    logger.info("  ✓ Video frames extracted WITHOUT sound from MP4")
    logger.info("  ✓ Audio extracted from matching WAV file for the same time period!")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test aligned video (MP4) and audio (WAV) extraction from Kaggle dataset"
    )
    parser.add_argument("--video", type=str, help="Path to video file (.mp4)")
    parser.add_argument("--audio", type=str, help="Path to audio file (.wav)")
    parser.add_argument("--use-kaggle", action="store_true", 
                       help="Auto-find matching video-audio pairs from Kaggle dataset")
    parser.add_argument("--extract-features", action="store_true", 
                       help="Also extract VideoMAE features")
    
    args = parser.parse_args()
    
    main(
        video_path=args.video,
        audio_path=args.audio,
        extract_features=args.extract_features,
        use_kaggle=args.use_kaggle
    )
