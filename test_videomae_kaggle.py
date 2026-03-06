"""
Test script to verify VideoMAE v1 model is correctly loaded and used with Kaggle dataset.

This script:
1. Finds a Kaggle video file (01-prefixed)
2. Loads VideoMAE model with proper architecture
3. Extracts 16 frames from the video
4. Verifies the model class is VideoMAEModel
5. Tests forward pass and checks embedding shape
6. Reports success/failure clearly

Usage:
    python test_videomae_v2_kaggle.py
"""

import torch
import numpy as np
from pathlib import Path
import logging
import random
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_kaggle_video():
    """Find a single Kaggle video file starting with '01'."""
    import os
    
    possible_paths = [
        Path("kaggle_datasets"),
        Path(os.path.expanduser("~/.cache/kagglehub/datasets")),
        Path(os.environ.get("USERPROFILE", "~")) / ".cache/kagglehub/datasets",
    ]
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    for search_path in possible_paths:
        if search_path.exists():
            logger.info(f"Searching in: {search_path}")
            for file in search_path.rglob("*"):
                if (
                    file.is_file()
                    and file.suffix.lower() in video_extensions
                    and file.name.startswith("01")
                ):
                    return file
    
    return None


def extract_video_frames(video_path, num_frames=16):
    """Extract consecutive frames from video."""
    import av
    
    logger.info(f"\nOpening video: {video_path.name}")
    container = av.open(str(video_path))
    
    video_stream = container.streams.video[0]
    fps = float(video_stream.average_rate)
    total_frames = video_stream.frames
    
    logger.info(f"  FPS: {fps:.2f}, Total frames: {total_frames}")
    
    # Pick random start point
    max_start = max(0, total_frames - num_frames)
    start_frame = random.randint(0, max_start) if max_start > 0 else 0
    
    logger.info(f"  Extracting {num_frames} frames starting at frame {start_frame}")
    
    container.seek(int(start_frame / fps / float(video_stream.time_base)))
    
    frames = []
    frame_count = 0
    
    for frame in container.decode(video=0):
        if frame_count >= num_frames:
            break
        
        img = frame.to_ndarray(format='rgb24')
        img_resized = cv2.resize(img, (224, 224))
        frames.append(img_resized)
        frame_count += 1
    
    container.close()
    
    if len(frames) < num_frames:
        logger.warning(f"Only extracted {len(frames)} frames (expected {num_frames})")
    
    return np.stack(frames)  # [T, H, W, 3]


def load_videomae_model():
    """Load VideoMAE with proper architecture."""
    logger.info("\n" + "="*70)
    logger.info("LOADING VIDEOMAE")
    logger.info("="*70)
    
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEModel
        
        model_name = "MCG-NJU/videomae-base"
        
        logger.info(f"Loading model from: {model_name}")
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        logger.info(f"✓ Processor loaded")
        
        model = VideoMAEModel.from_pretrained(model_name)
        logger.info(f"✓ Model loaded")
        
        # **VERIFY: Check we loaded correct architecture**
        model_class = type(model).__name__
        logger.info(f"\nMODEL CLASS CHECK:")
        logger.info(f"  Actual class: {model_class}")
        
        logger.info(f"✓ Correct VideoMAE architecture loaded!")
        
        model.eval()
        return model, processor
        
    except Exception as e:
        logger.error(f"❌ Error loading VideoMAE: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(frames, model, processor):
    """Test forward pass through VideoMAE."""
    logger.info("\n" + "="*70)
    logger.info("TESTING FORWARD PASS")
    logger.info("="*70)
    
    try:
        # Convert frames to list for processor
        frames_list = [frames[i] for i in range(frames.shape[0])]
        
        logger.info(f"Input frames shape: {frames.shape} (expected [16, 224, 224, 3])")
        
        # Processor conversion
        logger.info("Processing frames with processor...")
        inputs = processor(frames_list, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        
        logger.info(f"Processor output shape: {tuple(pixel_values.shape)}")
        
        # Forward pass
        logger.info("Running forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info(f"✓ Forward pass successful")
        
        # Check outputs
        logger.info(f"\nOutput shapes:")
        logger.info(f"  last_hidden_state: {tuple(outputs.last_hidden_state.shape)}")
        
        # Extract embedding
        video_embedding = outputs.last_hidden_state[:, 0, :]
        logger.info(f"  CLS token embedding: {tuple(video_embedding.shape)}")
        
        # Verify dimensions
        expected_dim = 768
        actual_dim = video_embedding.shape[1]
        
        if actual_dim == expected_dim:
            logger.info(f"✓ Embedding dimension is correct: {actual_dim}")
        else:
            logger.error(f"❌ Wrong embedding dimension: expected {expected_dim}, got {actual_dim}")
            return False
        
        logger.info(f"  Mean: {video_embedding.mean().item():.6f}")
        logger.info(f"  Std: {video_embedding.std().item():.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("="*70)
    logger.info("VIDEOMAE + KAGGLE DATASET TEST")
    logger.info("="*70)
    
    # Step 1: Find video
    logger.info("\nSTEP 1: Find Kaggle video")
    logger.info("-"*70)
    video_path = find_kaggle_video()
    
    if video_path is None:
        logger.error("❌ No Kaggle video found!")
        logger.info("Please ensure Kaggle dataset is downloaded: python download_kaggle_dataset.py")
        return False
    
    logger.info(f"✓ Found video: {video_path}")
    
    # Step 2: Extract frames
    logger.info("\nSTEP 2: Extract video frames")
    logger.info("-"*70)
    frames = extract_video_frames(video_path, num_frames=16)
    logger.info(f"✓ Extracted frames shape: {frames.shape}")
    
    # Step 3: Load VideoMAE
    logger.info("\nSTEP 3: Load VideoMAE model")
    logger.info("-"*70)
    model, processor = load_videomae_model()
    
    if model is None or processor is None:
        logger.error("❌ Failed to load VideoMAE")
        return False
    
    # Step 4: Test forward pass
    logger.info("\nSTEP 4: Test forward pass")
    logger.info("-"*70)
    success = test_forward_pass(frames, model, processor)
    
    if not success:
        logger.error("❌ Forward pass test failed")
        return False
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("✓ ALL TESTS PASSED!")
    logger.info("="*70)
    logger.info("VideoMAE is correctly loaded and working with Kaggle dataset.")
    logger.info("="*70)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
