"""
Visualize the video augmentation methods used in the contrastive-learning setup.

This script creates a PNG showing a reference lip-video clip and the result of
each augmentation method from the video contrastive training script:
1. Rotation
2. Horizontal flip
3. Color jitter
4. Grayscale
5. Gaussian noise
6. Time stretch

Usage:
    python visualize_video_augmentations.py --output-dir video_contrastive_results
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_video_contrastive import VideoAugmentation, VideoConfig, find_lip_video_files


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_video_clip(video_path: Path, config: VideoConfig) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    start_frame = max(0, (total_frames - config.clip_length) // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(config.clip_length):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (config.frame_size, config.frame_size))
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        frames.append(frame_tensor)

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {video_path.name}")

    while len(frames) < config.clip_length:
        frames.append(frames[-1].clone())

    return torch.stack(frames[: config.clip_length], dim=1)


def _build_synthetic_clip(config: VideoConfig) -> torch.Tensor:
    logger.warning("No usable video found; generating a synthetic reference clip.")
    frames = []
    for frame_idx in range(config.clip_length):
        canvas = np.zeros((config.frame_size, config.frame_size, 3), dtype=np.uint8)

        # Soft gradient background so color jitter and grayscale are visible.
        gradient = np.linspace(30, 90, config.frame_size, dtype=np.uint8)
        canvas[:, :, 0] = gradient[:, None]
        canvas[:, :, 1] = gradient[None, :]
        canvas[:, :, 2] = 60

        center_x = int(config.frame_size * (0.25 + 0.5 * (frame_idx / max(1, config.clip_length - 1))))
        center_y = int(config.frame_size * (0.45 + 0.12 * np.sin(frame_idx / 2.0)))
        cv2.circle(canvas, (center_x, center_y), config.frame_size // 8, (220, 90, 70), thickness=-1)
        cv2.rectangle(
            canvas,
            (config.frame_size // 5, config.frame_size // 3),
            (config.frame_size - config.frame_size // 5, config.frame_size - config.frame_size // 3),
            (15, 15, 15),
            thickness=2,
        )

        frame_tensor = torch.from_numpy(canvas).float().permute(2, 0, 1) / 255.0
        frames.append(frame_tensor)

    return torch.stack(frames, dim=1)


def _build_reference_clip(config: VideoConfig) -> Tuple[torch.Tensor, str]:
    video_files = find_lip_video_files()
    if video_files:
        for candidate in video_files:
            try:
                clip = _load_video_clip(candidate, config)
                return clip, f"video: {candidate.name}"
            except Exception as exc:
                logger.warning(f"Skipping {candidate.name}: {exc}")

    return _build_synthetic_clip(config), "synthetic reference clip"


def _select_frame_indices(num_frames: int, num_tiles: int = 4) -> List[int]:
    if num_frames <= 1:
        return [0] * num_tiles
    return np.linspace(0, num_frames - 1, num=num_tiles, dtype=int).tolist()


def _make_contact_sheet(clip: torch.Tensor, num_tiles: int = 4) -> np.ndarray:
    indices = _select_frame_indices(clip.shape[1], num_tiles=num_tiles)
    frames = []

    for frame_idx in indices:
        frame = clip[:, frame_idx].detach().cpu().clamp(0.0, 1.0)
        frame_np = (frame.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        frames.append(frame_np)

    gap = np.full((frames[0].shape[0], 8, 3), 245, dtype=np.uint8)
    sheet = frames[0]
    for frame in frames[1:]:
        sheet = np.concatenate([sheet, gap, frame], axis=1)

    return sheet


def _plot_sheet(ax, sheet: np.ndarray, title: str, subtitle: str) -> None:
    ax.imshow(sheet)
    ax.set_title(title, fontsize=11, pad=8)
    ax.text(
        0.01,
        0.04,
        subtitle,
        transform=ax.transAxes,
        fontsize=8,
        color="white",
        va="bottom",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
    )
    ax.axis("off")


def build_showcase(output_dir: Path, seed: int = 42) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = VideoConfig()
    augmentation = VideoAugmentation(config)
    reference_clip, source = _build_reference_clip(config)

    augmentation_methods = [
        ("Reference clip", lambda clip: clip.clone()),
        ("Rotation", augmentation._augment_rotation),
        ("Horizontal flip", augmentation._augment_horizontal_flip),
        ("Color jitter", augmentation._augment_color_jitter),
        ("Grayscale", augmentation._augment_grayscale),
        ("Gaussian noise", augmentation._augment_gaussian_noise),
        ("Time stretch", augmentation._augment_time_stretch),
    ]

    rows = []
    for title, func in augmentation_methods:
        try:
            augmented_clip = func(reference_clip.clone())
        except Exception as exc:
            logger.warning(f"{title} failed, using reference clip instead: {exc}")
            augmented_clip = reference_clip.clone()

        rows.append(
            (
                title,
                _make_contact_sheet(reference_clip),
                _make_contact_sheet(augmented_clip),
            )
        )

    fig, axes = plt.subplots(len(rows), 2, figsize=(16, 3.4 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.array([axes])

    fig.patch.set_facecolor("white")

    for row_idx, (title, original_sheet, augmented_sheet) in enumerate(rows):
        _plot_sheet(
            axes[row_idx, 0],
            original_sheet,
            "Original frames",
            source if row_idx == 0 else "reference clip",
        )
        _plot_sheet(
            axes[row_idx, 1],
            augmented_sheet,
            title,
            "augmentation result",
        )

    fig.suptitle("Video Augmentation Showcase", fontsize=16, fontweight="bold", y=1.01)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "video_augmentation_showcase.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved augmentation showcase to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PNG showcase of video augmentations.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="video_contrastive_results",
        help="Directory for the PNG output",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the showcase")
    args = parser.parse_args()

    output_path = build_showcase(Path(args.output_dir), seed=args.seed)
    logger.info(f"Showcase ready: {output_path}")


if __name__ == "__main__":
    main()