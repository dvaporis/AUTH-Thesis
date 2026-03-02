"""
Download and manage the Audio-Visual Database of Emotional Speech and Song from Kaggle.

Usage:
    python download_kaggle_dataset.py
    
This will download the dataset to ./kaggle_datasets/
"""

import os
import logging
from pathlib import Path
import kagglehub

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_audio_visual_dataset(cache_dir: str = "kaggle_datasets") -> Path:
    """
    Download the Audio-Visual Database of Emotional Speech and Song from Kaggle.
    
    Args:
        cache_dir: Directory to cache the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    dataset_name = "thbdh5765/audio-visual-database-of-emotional-speech-and-song"
    
    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Cache directory: {cache_dir}")
    
    try:
        # Create cache directory if it doesn't exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Set environment variable for kagglehub cache
        os.environ["KAGGLEHUB_HOME"] = os.path.abspath(cache_dir)
        
        # Download dataset
        logger.info("This may take a few minutes on first download...")
        path = kagglehub.dataset_download(dataset_name)
        
        logger.info(f"✓ Dataset downloaded successfully to: {path}")
        return Path(path)
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("\nTo use this dataset, you need:")
        logger.info("1. A Kaggle account (free): https://www.kaggle.com/signup")
        logger.info("2. API credentials: https://www.kaggle.com/settings/account")
        logger.info("3. Create ~/.kaggle/kaggle.json with your credentials")
        logger.info("4. Then run this script again")
        raise


def list_audio_files(dataset_path: Path, extensions: list = ['.wav', '.mp3', '.flac', '.ogg']) -> list:
    """
    List all audio files in the downloaded dataset.
    
    Args:
        dataset_path: Path to the dataset
        extensions: Audio file extensions to look for
        
    Returns:
        List of audio file paths
    """
    audio_files = []
    
    logger.info(f"\nSearching for audio files in: {dataset_path}")
    
    for ext in extensions:
        audio_files.extend(dataset_path.rglob(f"*{ext}"))
        audio_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    if audio_files:
        logger.info("\nFirst 10 audio files:")
        for i, file in enumerate(audio_files[:10], 1):
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"  {i}. {file.relative_to(dataset_path)} ({size_mb:.1f} MB)")
    
    return audio_files


def list_video_files(dataset_path: Path, extensions: list = ['.mp4', '.avi', '.mov', '.mkv']) -> list:
    """
    List all video files in the downloaded dataset.
    
    Args:
        dataset_path: Path to the dataset
        extensions: Video file extensions to look for
        
    Returns:
        List of video file paths
    """
    video_files = []
    
    logger.info(f"\nSearching for video files in: {dataset_path}")
    
    for ext in extensions:
        video_files.extend(dataset_path.rglob(f"*{ext}"))
        video_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(video_files)} video files")
    
    if video_files:
        logger.info("\nFirst 10 video files:")
        for i, file in enumerate(video_files[:10], 1):
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"  {i}. {file.relative_to(dataset_path)} ({size_mb:.1f} MB)")
    
    return video_files


def get_dataset_stats(dataset_path: Path) -> dict:
    """
    Get statistics about the downloaded dataset.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    audio_files = list_audio_files(dataset_path)
    video_files = list_video_files(dataset_path)
    
    # Calculate total size
    total_size_bytes = 0
    for file in dataset_path.rglob("*"):
        if file.is_file():
            total_size_bytes += file.stat().st_size
    
    stats = {
        'num_audio_files': len(audio_files),
        'num_video_files': len(video_files),
        'total_size_mb': total_size_bytes / (1024 * 1024),
        'audio_files': audio_files,
        'video_files': video_files,
    }
    
    logger.info("\n" + "="*60)
    logger.info("Dataset Statistics:")
    logger.info("="*60)
    logger.info(f"Audio files:     {stats['num_audio_files']}")
    logger.info(f"Video files:     {stats['num_video_files']}")
    logger.info(f"Total size:      {stats['total_size_mb']:.1f} MB")
    logger.info("="*60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kaggle audio-visual dataset")
    parser.add_argument("--cache-dir", type=str, default="kaggle_datasets",
                       help="Directory to cache the dataset")
    parser.add_argument("--list-only", action="store_true",
                       help="Only list files, don't download")
    
    args = parser.parse_args()
    
    try:
        # Download dataset
        dataset_path = download_audio_visual_dataset(args.cache_dir)
        
        # Get statistics
        stats = get_dataset_stats(dataset_path)
        
        logger.info("\n✓ Dataset ready for testing!")
        logger.info(f"Use in test: python test_encodec_simple.py --audio {stats['audio_files'][0]}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)
