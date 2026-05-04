"""
Save all RAVDESS videos as reduced lip-only videos.

This script reuses the MediaPipe lip-cropping pipeline from crop_ravdess_lips.py
and runs it over the full input folder by default.

Usage:
    python save_all_lip_videos.py
    python save_all_lip_videos.py --input-dir ravdess_videos_only --output-dir lip_crop_results_full
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from crop_ravdess_lips import (
    DEFAULT_MODEL_PATH,
    build_face_landmarker,
    ensure_face_landmarker_model,
    process_ravdess_folder,
    require_mediapipe,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line arguments for full lip-only video export."""

    parser = argparse.ArgumentParser(description="Save all videos as reduced lip-only versions")
    parser.add_argument("--input-dir", type=str, default="ravdess_videos_only", help="Folder containing source videos")
    parser.add_argument("--output-dir", type=str, default="lip_crop_results_full", help="Folder where lip-only videos will be saved")
    parser.add_argument("--crop-margin", type=float, default=0.30, help="Padding around detected lip landmarks")
    parser.add_argument("--crop-size", type=int, default=224, help="Output lip crop size in pixels; use 0 for native crop size")
    return parser


def main() -> None:
    """Run full-folder lip-only conversion."""

    parser = build_arg_parser()
    args = parser.parse_args()

    require_mediapipe()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    crop_size = None if args.crop_size == 0 else args.crop_size

    # Keep an example directory for API compatibility; example frames are disabled.
    example_dir = output_dir / "_unused_examples"

    model_path = ensure_face_landmarker_model(DEFAULT_MODEL_PATH)
    landmarker = build_face_landmarker(model_path)

    try:
        results = process_ravdess_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            example_dir=example_dir,
            landmarker=landmarker,
            max_videos=None,
            save_cropped_video=True,
            example_frame_count=0,
            crop_margin=args.crop_margin,
            crop_output_size=crop_size,
        )
    finally:
        landmarker.close()

    if not results:
        logger.info("No videos were processed.")
        return

    total_videos = len(results)
    total_frames = sum(result.frames_processed for result in results)
    total_face_frames = sum(result.frames_with_face for result in results)
    total_fallback_frames = sum(result.frames_with_fallback_crop for result in results)

    logger.info("=" * 80)
    logger.info("Saved lip-only videos for %d input videos", total_videos)
    logger.info("Total frames processed: %d", total_frames)
    logger.info("Frames with detected lips: %d", total_face_frames)
    logger.info("Frames using fallback crop: %d", total_fallback_frames)
    logger.info("Lip-only output folder: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
