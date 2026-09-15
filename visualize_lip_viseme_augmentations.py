#!/usr/bin/env python3
"""Visualize the data augmentations used by train_lip_viseme.py.

This script samples one lip video clip, applies each augmentation that the
training dataset actually uses, and saves a side-by-side comparison PNG for
each augmentation:

1. Horizontal flip
2. Brightness/contrast/saturation jitter
3. Random cutout

Each PNG shows a small contact sheet of example frames from the original clip
on the left and the augmented clip on the right.

Usage:
    python visualize_lip_viseme_augmentations.py \
        --video-dir s1_lip_crops \
        --output-dir viseme_results

    python visualize_lip_viseme_augmentations.py \
        --video-path s1_lip_crops/bbaf2n_lipcrop.mp4
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import torch
from torchvision.transforms import v2

matplotlib.use("Agg")
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _find_videos(video_dir: Path) -> List[Path]:
    return [
        p for p in sorted(video_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]


def _load_frames(video_path: Path, frame_size: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frames: List[torch.Tensor] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No readable frames in {video_path}")

    return torch.stack(frames, dim=0)  # (T, C, H, W) uint8


def _build_synthetic_clip(frame_size: int, num_frames: int = 16) -> torch.Tensor:
    logger.warning("No usable video found; generating a synthetic reference clip.")
    frames = []
    for frame_idx in range(num_frames):
        canvas = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

        gradient = np.linspace(30, 90, frame_size, dtype=np.uint8)
        canvas[:, :, 0] = gradient[:, None]
        canvas[:, :, 1] = gradient[None, :]
        canvas[:, :, 2] = 60

        center_x = int(frame_size * (0.25 + 0.5 * (frame_idx / max(1, num_frames - 1))))
        center_y = int(frame_size * (0.45 + 0.12 * np.sin(frame_idx / 2.0)))
        cv2.circle(canvas, (center_x, center_y), frame_size // 8, (220, 90, 70), thickness=-1)
        cv2.rectangle(
            canvas,
            (frame_size // 5, frame_size // 3),
            (frame_size - frame_size // 5, frame_size - frame_size // 3),
            (15, 15, 15),
            thickness=2,
        )

        frames.append(torch.from_numpy(canvas).permute(2, 0, 1))

    return torch.stack(frames, dim=0)


def _select_frame_indices(num_frames: int, num_tiles: int) -> List[int]:
    if num_frames <= 1:
        return [0] * num_tiles
    return np.linspace(0, num_frames - 1, num=num_tiles, dtype=int).tolist()


def _make_contact_sheet(clip: torch.Tensor, num_tiles: int = 4) -> np.ndarray:
    indices = _select_frame_indices(clip.shape[0], num_tiles)
    frames = []

    for frame_idx in indices:
        frame = clip[frame_idx].detach().cpu().clamp(0.0, 1.0)
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


def _normalize_clip(clip: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (clip - mean) / std


def _unnormalize_clip(clip: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return clip * std + mean


def _prepare_display_clip(clip: torch.Tensor) -> torch.Tensor:
    return _unnormalize_clip(clip).clamp(0.0, 1.0)


def _apply_horizontal_flip(clip: torch.Tensor, rng: random.Random) -> torch.Tensor:
    del rng
    return v2.functional.horizontal_flip(clip)


def _apply_color_jitter(clip: torch.Tensor, rng: random.Random) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    denorm = clip * std + mean
    denorm = v2.functional.adjust_brightness(denorm, rng.uniform(0.8, 1.2))
    denorm = v2.functional.adjust_contrast(denorm, rng.uniform(0.8, 1.2))
    denorm = v2.functional.adjust_saturation(denorm, rng.uniform(0.9, 1.1))
    return (denorm - mean) / std


def _apply_random_cutout(clip: torch.Tensor, rng: random.Random) -> torch.Tensor:
    augmented = clip.clone()
    _, _, height, width = augmented.shape
    cutout_h = max(1, int(height * rng.uniform(0.08, 0.20)))
    cutout_w = max(1, int(width * rng.uniform(0.08, 0.20)))
    y = rng.randint(0, max(0, height - cutout_h))
    x = rng.randint(0, max(0, width - cutout_w))
    augmented[:, :, y:y + cutout_h, x:x + cutout_w] = 0.0
    return augmented


def _select_reference_video(video_dir: Optional[Path], video_path: Optional[Path], frame_size: int) -> Tuple[torch.Tensor, str]:
    if video_path is not None:
        return _load_frames(video_path, frame_size=frame_size), f"video: {video_path.name}"

    if video_dir is not None:
        videos = _find_videos(video_dir)
        if videos:
            return _load_frames(videos[0], frame_size=frame_size), f"video: {videos[0].name}"

    return _build_synthetic_clip(frame_size=frame_size), "synthetic reference clip"


def build_showcase(
    output_dir: Path,
    video_dir: Optional[Path] = None,
    video_path: Optional[Path] = None,
    seed: int = 42,
    frame_size: int = 96,
    num_tiles: int = 4,
) -> List[Path]:
    # Show the same three augmentations used by LipDataset after normalisation.
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    reference_raw, source = _select_reference_video(video_dir, video_path, frame_size=frame_size)
    reference = _normalize_clip(reference_raw.float() / 255.0)

    augmentations = [
        ("horizontal_flip", "Horizontal flip", _apply_horizontal_flip),
        ("color_jitter", "Brightness / contrast / saturation", _apply_color_jitter),
        ("random_cutout", "Random cutout", _apply_random_cutout),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for short_name, title, func in augmentations:
        augmented = func(reference.clone(), rng)

        original_sheet = _make_contact_sheet(_prepare_display_clip(reference), num_tiles=num_tiles)
        augmented_sheet = _make_contact_sheet(_prepare_display_clip(augmented), num_tiles=num_tiles)

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=False)
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(top=0.83, hspace=0.34)
        _plot_sheet(axes[0], original_sheet, "Original frames", source)
        _plot_sheet(axes[1], augmented_sheet, title, "training-time augmentation")
        fig.suptitle(f"Data Augmentation: {title}", fontsize=15, fontweight="bold", y=0.97)

        output_path = output_dir / f"{short_name}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(output_path)
        logger.info("Saved %s", output_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PNG comparisons for train_lip_viseme.py augmentations.")
    parser.add_argument("--video-dir", type=str, default="s1_lip_crops", help="Directory containing lip videos")
    parser.add_argument("--video-path", type=str, default="", help="Optional explicit video to visualize")
    parser.add_argument("--output-dir", type=str, default="viseme_results", help="Directory for PNG outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic examples")
    parser.add_argument("--frame-size", type=int, default=96, help="Resize frames to this square size")
    parser.add_argument("--num-tiles", type=int, default=4, help="Number of example frames in each contact sheet")
    args = parser.parse_args()

    video_dir = Path(args.video_dir) if args.video_dir else None
    video_path = Path(args.video_path) if args.video_path else None
    outputs = build_showcase(
        output_dir=Path(args.output_dir),
        video_dir=video_dir,
        video_path=video_path,
        seed=args.seed,
        frame_size=args.frame_size,
        num_tiles=args.num_tiles,
    )
    logger.info("Wrote %d PNG files", len(outputs))


if __name__ == "__main__":
    main()