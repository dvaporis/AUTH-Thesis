"""
Create a simple test video with audio for testing alignment.

This creates a short video with:
- Visual frames showing frame numbers
- Audio with a tone that changes over time

Usage:
    python create_test_video.py
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_video_with_audio(
    output_path: str = "test_video.mp4",
    duration: float = 3.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    audio_freq_start: int = 440,
    audio_freq_end: int = 880
):
    """
    Create a test video with synchronized audio.
    
    Args:
        output_path: Output video file path
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
        audio_freq_start: Starting frequency for audio tone (Hz)
        audio_freq_end: Ending frequency for audio tone (Hz)
    """
    logger.info(f"Creating test video: {output_path}")
    logger.info(f"  Duration: {duration}s")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Resolution: {width}x{height}")
    
    total_frames = int(duration * fps)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate frames
    logger.info(f"Generating {total_frames} frames...")
    for frame_idx in range(total_frames):
        # Create a frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gradient based on frame number
        progress = frame_idx / total_frames
        color_value = int(progress * 255)
        frame[:, :] = [color_value, 100, 255 - color_value]
        
        # Add frame number and time
        time_sec = frame_idx / fps
        text = f"Frame {frame_idx+1}/{total_frames}"
        time_text = f"Time: {time_sec:.3f}s"
        
        cv2.putText(frame, text, (50, height//2 - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, time_text, (50, height//2 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Add a moving circle
        circle_x = int((width - 100) * progress + 50)
        circle_y = height // 2 + 100
        cv2.circle(frame, (circle_x, circle_y), 30, (255, 255, 0), -1)
        
        video_writer.write(frame)
    
    video_writer.release()
    logger.info(f"✓ Video frames written")
    
    # Generate audio
    logger.info(f"Generating audio...")
    sample_rate = 24000
    total_samples = int(duration * sample_rate)
    
    # Create a swept frequency tone
    t = np.linspace(0, duration, total_samples)
    freq = np.linspace(audio_freq_start, audio_freq_end, total_samples)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    audio = 0.3 * np.sin(phase)
    
    # Add some harmonics for richness
    audio += 0.1 * np.sin(2 * phase)
    audio += 0.05 * np.sin(3 * phase)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    audio = (audio * 32767).astype(np.int16)
    
    # Save audio temporarily
    import tempfile
    import subprocess
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
        
        from scipy.io import wavfile
        wavfile.write(temp_audio_path, sample_rate, audio)
        logger.info(f"✓ Audio generated: {len(audio)} samples @ {sample_rate}Hz")
    
    # Combine video and audio using ffmpeg
    logger.info("Combining video and audio with ffmpeg...")
    output_with_audio = output_path.replace('.mp4', '_with_audio.mp4')
    
    try:
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', output_path,
            '-i', temp_audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_with_audio
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✓ Video with audio created: {output_with_audio}")
            # Remove the video-only file
            Path(output_path).unlink()
            Path(temp_audio_path).unlink()
            logger.info(f"\n✓ Test video ready: {output_with_audio}")
            logger.info(f"  Duration: {duration}s")
            logger.info(f"  Frames: {total_frames} @ {fps}fps")
            logger.info(f"  Audio: {len(audio)} samples @ {sample_rate}Hz")
            return output_with_audio
        else:
            logger.warning("ffmpeg failed, audio not added to video")
            logger.warning(result.stderr)
            Path(temp_audio_path).unlink()
            return output_path
            
    except FileNotFoundError:
        logger.warning("ffmpeg not found. Video created without audio.")
        logger.info("Install ffmpeg to add audio: https://ffmpeg.org/download.html")
        Path(temp_audio_path).unlink()
        return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create test video with audio")
    parser.add_argument("--output", default="test_video_with_audio.mp4", help="Output file")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    
    args = parser.parse_args()
    
    create_test_video_with_audio(
        output_path=args.output,
        duration=args.duration,
        fps=args.fps
    )
