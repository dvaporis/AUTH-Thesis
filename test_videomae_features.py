"""
Extract visual features from videos using VideoMAEv2.

This script uses Meta's VideoMAEv2 model from HuggingFace to extract video embeddings
from 16-frame chunks (no frame cloning), compatible with contrastive learning.

Usage:
    python test_videomae_features.py --use-kaggle
    python test_videomae_features.py --video path/to/video.mp4
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, Optional
import random
import cv2
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_kaggle_video_files() -> list:
    """Find all video files in the Kaggle dataset directory."""
    import os
    
    possible_paths = [
        Path("kaggle_datasets"),
        Path(os.path.expanduser("~/.cache/kagglehub/datasets")),
        Path(os.environ.get("USERPROFILE", "~")) / ".cache/kagglehub/datasets",
    ]
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = []
    
    for kaggle_path in possible_paths:
        if kaggle_path.exists():
            for file in kaggle_path.rglob("*"):
                if file.suffix.lower() in video_extensions:
                    video_files.append(file)
            
            if video_files:
                logger.info(f"Found {len(video_files)} video files in {kaggle_path}")
                break
    
    return video_files


def load_video_frames(video_path: str, num_frames: int = 16, start_frame: Optional[int] = None) -> Tuple[np.ndarray, float, str]:
    """
    Load exactly 16 consecutive frames from video for VideoMAE processing.
    No frame cloning - extracts 16 consecutive frames from the original video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default 16 for VideoMAE)
        start_frame: Starting frame index (random if None)
    
    Returns:
        Tuple of (frames, duration_seconds, source_description)
        frames shape: [num_frames, height, width, channels]
    """
    logger.info(f"Loading video from: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {width}x{height}, {fps:.2f}fps, {total_frames} frames")
    
    # Choose random starting point if not specified
    max_start = max(0, total_frames - num_frames)
    if start_frame is None:
        start_frame = random.randint(0, max_start) if max_start > 0 else 0
    else:
        start_frame = min(start_frame, max_start)
    
    # Calculate actual duration in seconds for this chunk
    duration_seconds = num_frames / fps
    
    logger.info(f"Extracting {num_frames} consecutive frames starting at frame {start_frame}")
    logger.info(f"Chunk duration: {duration_seconds:.3f} seconds ({num_frames} frames @ {fps:.2f}fps)")
    
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
    
    source_desc = f"{Path(video_path).name} ({frames.shape[0]} frames, {duration_seconds:.3f}s @ {fps:.2f}fps)"
    logger.info(f"✓ Loaded: {frames.shape}, {source_desc}")
    
    return frames, duration_seconds, source_desc


def load_videomae_model(device: str = "cpu"):
    """
    Load VideoMAEv2 model from HuggingFace.
    
    Args:
        device: Device to use (cpu or cuda)
    
    Returns:
        Model and image processor
    """
    logger.info("Loading VideoMAEv2 model from HuggingFace...")
    
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEModel
        
        # Load VideoMAE v2 base model
        model_name = "MCG-NJU/videomae-base"
        
        logger.info(f"Loading model: {model_name}")
        
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        model = VideoMAEModel.from_pretrained(model_name)
        
        model = model.to(device)
        model.eval()
        
        logger.info("✓ VideoMAEv2 model loaded successfully")
        logger.info(f"  Input size: {processor.size}")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


def extract_video_features(frames: np.ndarray, model, processor, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract video features using VideoMAEv2.
    
    Args:
        frames: Video frames [T, H, W, 3]
        model: VideoMAE model
        processor: VideoMAE image processor
        device: Device to use
    
    Returns:
        Tuple of (video_embedding, sequence_features)
    """
    logger.info("Extracting video features with VideoMAEv2...")
    
    try:
        # VideoMAE expects list of frames in [0, 255] range (uint8)
        # The processor will handle normalization
        frames_list = [frames[i] for i in range(frames.shape[0])]
        
        logger.info(f"Processing {len(frames_list)} frames")
        
        # Preprocess frames - processor expects uint8 [0, 255] images
        inputs = processor(frames_list, return_tensors="pt")
        
        # Move to device
        pixel_values = inputs['pixel_values'].to(device)
        
        logger.info(f"Input tensor shape: {pixel_values.shape}")
        
        # Extract features
        with torch.no_grad():
            outputs = model(pixel_values)
        
        # Get the last hidden state (sequence of patch embeddings)
        sequence_output = outputs.last_hidden_state  # [batch, num_patches, hidden_size]
        
        logger.info(f"Sequence output shape: {sequence_output.shape}")
        
        # Extract CLS token (first token) as video embedding
        video_embedding = sequence_output[:, 0, :]  # [batch, hidden_size]
        
        # Also get mean pooled features as alternative
        pooled_features = sequence_output.mean(dim=1)  # [batch, hidden_size]
        
        logger.info(f"✓ Video embedding (CLS token): {video_embedding.shape}")
        logger.info(f"✓ Pooled features: {pooled_features.shape}")
        
        return video_embedding, sequence_output
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        raise


def visualize_results(video_embedding: torch.Tensor, sequence_features: torch.Tensor,
                     video_name: str, output_dir: str = "videomae_results") -> None:
    """Create visualizations for extracted features."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert to numpy
    embed_np = video_embedding.cpu().numpy()
    seq_feat_np = sequence_features.cpu().numpy().squeeze()  # Remove batch dimension
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'VideoMAEv2 Features: {video_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: Video embedding distribution
    axes[0, 0].hist(embed_np.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].set_title(f'Video Embedding Distribution (dim={embed_np.shape[1]})')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Video embedding heatmap
    im1 = axes[0, 1].imshow(embed_np, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('Video Embedding (CLS Token)')
    axes[0, 1].set_xlabel('Feature Dimension')
    axes[0, 1].set_ylabel('Video')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot 3: Sequence features heatmap
    im2 = axes[1, 0].imshow(seq_feat_np, aspect='auto', cmap='viridis')
    axes[1, 0].set_title(f'Sequence Features ({seq_feat_np.shape[0]} patches)')
    axes[1, 0].set_xlabel('Feature Dimension')
    axes[1, 0].set_ylabel('Patch Index')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 4: Feature magnitudes across patches
    patch_magnitudes = np.linalg.norm(seq_feat_np, axis=1)
    axes[1, 1].plot(range(len(patch_magnitudes)), patch_magnitudes,
                   marker='o', linewidth=2, markersize=4, color='steelblue')
    axes[1, 1].set_title('Feature Magnitude Across Patches')
    axes[1, 1].set_xlabel('Patch Index')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_path / f"{timestamp}_{Path(video_name).stem}_videomae_features.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {plot_path}")
    plt.close()
    
    # Save embeddings
    embed_path = output_path / f"{timestamp}_{Path(video_name).stem}_video_embedding.npy"
    np.save(embed_path, embed_np)
    logger.info(f"✓ Saved embedding: {embed_path}")


def main(video_path: Optional[str] = None):
    """Main function."""
    logger.info("="*70)
    logger.info("VideoMAEv2 Visual Feature Extraction (16-frame chunks)")
    logger.info("="*70)
    
    # Load video
    if not video_path:
        video_files = find_kaggle_video_files()
        if not video_files:
            logger.error("No video files found")
            return
        video_path = random.choice(video_files)
        logger.info(f"Using random Kaggle video: {video_path}")
    
    # Load exactly 16 consecutive frames (no cloning)
    frames, duration_seconds, source_desc = load_video_frames(video_path, num_frames=16)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load VideoMAE model
    model, processor = load_videomae_model(device)
    
    # Extract features
    video_embedding, sequence_features = extract_video_features(frames, model, processor, device)
    
    logger.info("\n" + "="*70)
    logger.info("Results Summary:")
    logger.info("-"*70)
    logger.info(f"  Video                     : {source_desc}")
    logger.info(f"  Chunk duration            : {duration_seconds:.3f} seconds")
    logger.info(f"  Video embedding shape     : {video_embedding.shape}")
    logger.info(f"  Sequence features shape   : {sequence_features.shape}")
    logger.info(f"  Embedding dimension       : {video_embedding.shape[1]}")
    logger.info(f"  Number of patches         : {sequence_features.shape[1]}")
    logger.info(f"  Embedding mean            : {video_embedding.mean().item():.6f}")
    logger.info(f"  Embedding std             : {video_embedding.std().item():.6f}")
    logger.info(f"  Embedding L2 norm         : {torch.norm(video_embedding).item():.6f}")
    logger.info("="*70)
    
    # Visualize
    visualize_results(video_embedding, sequence_features, Path(video_path).name)
    
    logger.info("\n✓ VideoMAEv2 visual feature extraction completed!")
    logger.info(f"  Video embedding dimension: {video_embedding.shape[1]}")
    logger.info("  Ready for contrastive learning with audio embeddings!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoMAEv2 visual feature extraction")
    parser.add_argument("--use-kaggle", action="store_true", help="Use random Kaggle video")
    parser.add_argument("--video", type=str, help="Path to specific video file")
    
    args = parser.parse_args()
    
    video_path = args.video if args.video else (None if args.use_kaggle else None)
    
    main(video_path=video_path)
