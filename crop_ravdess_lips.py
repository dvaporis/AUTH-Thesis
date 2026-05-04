"""
Crop RAVDESS videos around the lip region using MediaPipe Face Mesh.

This script:
1. Scans the ravdess_videos_only folder for video files
2. Uses MediaPipe Face Mesh to locate the lip region in each frame
3. Crops a padded mouth/lip bounding box to reduce frame size
4. Saves cropped videos for later training use
5. Saves a few example comparison frames so you can inspect the crops

Usage:
    python crop_ravdess_lips.py
    python crop_ravdess_lips.py --input-dir ravdess_videos_only --max-videos 3
    python crop_ravdess_lips.py --save-cropped-video
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python import vision
except ImportError as exc:  # pragma: no cover - handled at runtime
    mp = None
    base_options = None
    vision = None
    _MEDIAPIPE_IMPORT_ERROR = exc
else:
    _MEDIAPIPE_IMPORT_ERROR = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("lip_crop_results") / "models" / "face_landmarker.task"


@dataclass
class LipCropResult:
    """Summary information for one processed video."""

    video_path: Path
    output_video_path: Optional[Path]
    example_frame_paths: List[Path]
    frames_processed: int
    frames_with_face: int
    frames_with_fallback_crop: int
    last_timestamp_ms: int


def require_mediapipe() -> None:
    """Raise a clear error if MediaPipe is not installed."""

    if mp is None or base_options is None or vision is None:
        raise ImportError(
            "MediaPipe Tasks is required for lip cropping. Install it with: pip install mediapipe"
        ) from _MEDIAPIPE_IMPORT_ERROR


def ensure_face_landmarker_model(model_path: Path) -> Path:
    """Download the official face landmarker model if it is not already cached."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    logger.info("Downloading MediaPipe Face Landmarker model to %s", model_path)
    with urllib.request.urlopen(FACE_LANDMARKER_MODEL_URL) as response, open(model_path, "wb") as output_file:
        output_file.write(response.read())
    return model_path


def build_face_landmarker(model_path: Path):
    """Create a FaceLandmarker configured for decoded video frames."""

    require_mediapipe()
    options = vision.FaceLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def find_ravdess_video_files(input_dir: Path) -> List[Path]:
    """Find video files under the RAVDESS video folder."""

    if not input_dir.exists():
        logger.warning("Input directory does not exist: %s", input_dir)
        return []

    video_files = [
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    video_files.sort()
    logger.info("Found %d video files in %s", len(video_files), input_dir)
    return video_files


@lru_cache(maxsize=1)
def get_lip_landmark_indices() -> Sequence[int]:
    """Return the unique Face Mesh landmark indices used for the lips."""

    require_mediapipe()
    lip_indices = set()
    for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
        lip_indices.add(connection.start)
        lip_indices.add(connection.end)
    return sorted(lip_indices)


def landmarks_to_bbox(
    landmarks: Sequence,
    image_width: int,
    image_height: int,
    margin: float = 0.30,
) -> Tuple[int, int, int, int]:
    """Convert lip landmarks into a padded pixel bounding box."""

    lip_indices = get_lip_landmark_indices()
    xs = [landmarks[index].x * image_width for index in lip_indices]
    ys = [landmarks[index].y * image_height for index in lip_indices]

    min_x = max(0, int(min(xs)))
    max_x = min(image_width - 1, int(max(xs)))
    min_y = max(0, int(min(ys)))
    max_y = min(image_height - 1, int(max(ys)))

    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    pad_x = int(width * margin)
    pad_y = int(height * margin)

    left = max(0, min_x - pad_x)
    right = min(image_width, max_x + pad_x)
    top = max(0, min_y - pad_y)
    bottom = min(image_height, max_y + pad_y)

    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)

    return left, top, right, bottom


def fallback_mouth_bbox(image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    """Return a conservative lower-face crop when landmarks are unavailable."""

    crop_width = int(image_width * 0.50)
    crop_height = int(image_height * 0.35)
    left = max(0, (image_width - crop_width) // 2)
    top = int(image_height * 0.42)
    right = min(image_width, left + crop_width)
    bottom = min(image_height, top + crop_height)
    return left, top, right, bottom


def crop_with_padding(frame: np.ndarray, bbox: Tuple[int, int, int, int], output_size: Optional[int] = None) -> np.ndarray:
    """Crop a frame to bbox and optionally resize it to a square output."""

    left, top, right, bottom = bbox
    cropped = frame[top:bottom, left:right]

    if cropped.size == 0:
        raise ValueError("Empty crop generated from bbox")

    if output_size is not None:
        cropped = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)

    return cropped


def make_comparison_frame(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    crop: np.ndarray,
    frame_index: int,
    output_size: int = 256,
) -> np.ndarray:
    """Create a side-by-side original/crop inspection frame."""

    left, top, right, bottom = bbox
    annotated = frame.copy()
    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.putText(
        annotated,
        f"frame {frame_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    annotated_height = output_size
    annotated_width = int(annotated.shape[1] * annotated_height / annotated.shape[0])
    annotated_resized = cv2.resize(annotated, (annotated_width, annotated_height), interpolation=cv2.INTER_AREA)

    crop_preview = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    spacer = np.full((output_size, 12, 3), 25, dtype=np.uint8)
    combined = np.concatenate([annotated_resized, spacer, crop_preview], axis=1)

    return combined


def process_video_lip_crops(
    video_path: Path,
    output_dir: Path,
    example_dir: Path,
    landmarker,
    timestamp_offset_ms: int = 0,
    save_cropped_video: bool = True,
    example_frame_count: int = 4,
    crop_margin: float = 0.30,
    crop_output_size: Optional[int] = 224,
) -> LipCropResult:
    """Process one video and save cropped output plus inspection frames."""

    require_mediapipe()

    output_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    logger.info(
        "Processing %s (%dx%d, %.2f fps, %d frames)",
        video_path.name,
        frame_width,
        frame_height,
        fps,
        total_frames,
    )

    output_video_path: Optional[Path] = None
    writer = None
    if save_cropped_video:
        output_video_path = output_dir / f"{video_path.stem}_lipcrop.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        target_width = crop_output_size or frame_width
        target_height = crop_output_size or frame_height
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (target_width, target_height))

    sample_indices: set[int]
    if total_frames > 0 and example_frame_count > 0:
        raw_samples = np.linspace(0, max(total_frames - 1, 0), num=min(example_frame_count, total_frames), dtype=int)
        sample_indices = set(int(index) for index in raw_samples.tolist())
    else:
        sample_indices = set()

    example_paths: List[Path] = []
    frames_processed = 0
    frames_with_face = 0
    frames_with_fallback_crop = 0
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_timestamp_ms = timestamp_offset_ms

    while True:
        success, frame_bgr = capture.read()
        if not success:
            break

        frame_index = frames_processed
        frames_processed += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        timestamp_ms = timestamp_offset_ms + int(round(frame_index * 1000.0 / fps))
        last_timestamp_ms = timestamp_ms
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            bbox = landmarks_to_bbox(landmarks, frame_width, frame_height, margin=crop_margin)
            last_bbox = bbox
            frames_with_face += 1
        elif last_bbox is not None:
            bbox = last_bbox
            frames_with_fallback_crop += 1
        else:
            bbox = fallback_mouth_bbox(frame_width, frame_height)
            frames_with_fallback_crop += 1

        crop = crop_with_padding(frame_bgr, bbox, output_size=crop_output_size)

        if writer is not None:
            writer.write(crop)

        if frame_index in sample_indices:
            comparison = make_comparison_frame(frame_bgr, bbox, crop, frame_index)
            example_path = example_dir / f"{video_path.stem}_frame{frame_index:04d}_comparison.jpg"
            cv2.imwrite(str(example_path), comparison)
            example_paths.append(example_path)

    capture.release()
    if writer is not None:
        writer.release()

    return LipCropResult(
        video_path=video_path,
        output_video_path=output_video_path,
        example_frame_paths=example_paths,
        frames_processed=frames_processed,
        frames_with_face=frames_with_face,
        frames_with_fallback_crop=frames_with_fallback_crop,
        last_timestamp_ms=last_timestamp_ms,
    )


def process_ravdess_folder(
    input_dir: Path,
    output_dir: Path,
    example_dir: Path,
    landmarker,
    max_videos: Optional[int] = None,
    save_cropped_video: bool = True,
    example_frame_count: int = 4,
    crop_margin: float = 0.30,
    crop_output_size: Optional[int] = 224,
) -> List[LipCropResult]:
    """Process all videos in the RAVDESS folder."""

    video_files = find_ravdess_video_files(input_dir)
    if max_videos is not None:
        video_files = video_files[:max_videos]

    if not video_files:
        logger.warning("No video files found to process.")
        return []

    results: List[LipCropResult] = []
    timestamp_offset_ms = 0
    for index, video_path in enumerate(video_files, start=1):
        logger.info("[%d/%d] Cropping lips for %s", index, len(video_files), video_path.name)
        result = process_video_lip_crops(
            video_path=video_path,
            output_dir=output_dir,
            example_dir=example_dir,
            landmarker=landmarker,
            timestamp_offset_ms=timestamp_offset_ms,
            save_cropped_video=save_cropped_video,
            example_frame_count=example_frame_count,
            crop_margin=crop_margin,
            crop_output_size=crop_output_size,
        )
        results.append(result)
        # MediaPipe VIDEO mode requires strictly increasing timestamps for one landmarker instance.
        timestamp_offset_ms = result.last_timestamp_ms + 1
        logger.info(
            "  processed=%d face_frames=%d fallback_frames=%d examples=%d",
            result.frames_processed,
            result.frames_with_face,
            result.frames_with_fallback_crop,
            len(result.example_frame_paths),
        )

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Crop RAVDESS videos around the lip area with MediaPipe Face Mesh")
    parser.add_argument("--input-dir", type=str, default="ravdess_videos_only", help="Folder containing source videos")
    parser.add_argument("--output-dir", type=str, default="lip_crop_results", help="Folder for cropped videos")
    parser.add_argument("--example-dir", type=str, default="lip_crop_results/examples", help="Folder for example comparison frames")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit how many videos to process")
    parser.add_argument("--example-frames", type=int, default=4, help="How many comparison frames to save per video")
    parser.add_argument("--crop-margin", type=float, default=0.30, help="Extra padding around the lip bounding box")
    parser.add_argument("--crop-size", type=int, default=224, help="Output crop size in pixels; use 0 to keep native crop size")
    parser.add_argument("--no-cropped-video", action="store_true", help="Only save example frames, not cropped videos")
    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args()

    require_mediapipe()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    example_dir = Path(args.example_dir)
    crop_size = None if args.crop_size == 0 else args.crop_size
    model_path = ensure_face_landmarker_model(DEFAULT_MODEL_PATH)
    landmarker = build_face_landmarker(model_path)

    try:
        results = process_ravdess_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            example_dir=example_dir,
            landmarker=landmarker,
            max_videos=args.max_videos,
            save_cropped_video=not args.no_cropped_video,
            example_frame_count=args.example_frames,
            crop_margin=args.crop_margin,
            crop_output_size=crop_size,
        )
    finally:
        landmarker.close()

    if not results:
        logger.info("Nothing was processed.")
        return

    total_videos = len(results)
    total_frames = sum(result.frames_processed for result in results)
    total_face_frames = sum(result.frames_with_face for result in results)
    total_fallback_frames = sum(result.frames_with_fallback_crop for result in results)

    logger.info("=" * 80)
    logger.info("Completed lip cropping for %d videos", total_videos)
    logger.info("Total frames processed: %d", total_frames)
    logger.info("Frames with detected lips: %d", total_face_frames)
    logger.info("Frames using fallback crop: %d", total_fallback_frames)
    logger.info("Cropped videos saved to: %s", output_dir.resolve())
    logger.info("Example frames saved to: %s", example_dir.resolve())


if __name__ == "__main__":
    main()