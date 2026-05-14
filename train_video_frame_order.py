"""Train a video encoder with frame-order verification (real vs. shuffled).

Dataset:
 - Splits each clip into sliding windows of 8 frames (stride=1), covering every
     frame in the clip. For each window we produce two examples: a "natural"
     ordered sequence (label=1) and an "artificial" shuffled sequence (label=0).

Model:
 - `torchvision.models.video.r2plus1d_18` as encoder (fc replaced with Identity,
     producing a 512-D feature vector).
 - A linear head (512 -> 1) and we use `BCEWithLogitsLoss`.

Usage examples:
        python train_video_frame_order.py --data-dir lip_crop_results_full --dry-run
        python train_video_frame_order.py --data-dir lip_crop_results_full --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Use 8-frame windows as requested
FRAMES_PER_CLIP = 8


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def find_clip_dirs(root: Path, min_frames: int = FRAMES_PER_CLIP) -> List[Path]:
    """Return directories or video files that contain at least `min_frames` frames.

    This will return either a directory Path (containing image frames) or a
    video file Path (e.g., .mp4) if the file has >= min_frames frames.
    """
    clips: List[Path] = []
    root = Path(root)
    if not root.exists():
        return []

    # First, collect directories that contain image frames
    for p in root.rglob("*"):
        if p.is_dir():
            imgs = [f for f in sorted(p.iterdir()) if f.is_file() and is_image_file(f)]
            if len(imgs) >= min_frames:
                clips.append(p)

    # Next, check for video files directly under root (non-recursive)
    for f in sorted(root.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            if not HAS_CV2:
                # can't inspect frame counts without OpenCV; include file and
                # let dataset raise a helpful error later
                clips.append(f)
                continue
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                cap.release()
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if frame_count >= min_frames:
                clips.append(f)

    # Also support frames directly in root as a single clip
    imgs_root = [f for f in sorted(root.iterdir()) if f.is_file() and is_image_file(f)]
    if len(imgs_root) >= min_frames:
        clips.append(root)

    return sorted(set(clips))


class VideoFrameOrderDataset(Dataset):
    """Dataset that returns ordered and shuffled 8-frame clips using sliding windows.

    The dataset enumerates all length-`FRAMES_PER_CLIP` windows with stride=1 for
    each clip directory, so every frame is included in at least one window.
    For each window we return two samples (natural and shuffled) so classes are
    balanced overall.
    """

    def __init__(self, root: Path, frames_per_clip: int = FRAMES_PER_CLIP, transforms_new=None, clip_dirs: Optional[List[Path]] = None):
        self.root = Path(root)
        self.frames_per_clip = int(frames_per_clip)
        # either use provided clip_dirs (filtered) or discover under root
        if clip_dirs is not None:
            self.clip_dirs = [Path(d) for d in clip_dirs]
        else:
            self.clip_dirs = find_clip_dirs(self.root, min_frames=self.frames_per_clip)

        # prepare samples as (clip_dir, start_index) covering all sliding windows
        self.samples: List[Tuple[Path, int]] = []
        for cd in self.clip_dirs:
            cd = Path(cd)
            # If cd is a video file, probe frame count via OpenCV
            if cd.is_file() and cd.suffix.lower() in VIDEO_EXTS:
                if not HAS_CV2:
                    raise RuntimeError("OpenCV is required to read video files. Install with: pip install opencv-python")
                cap = cv2.VideoCapture(str(cd))
                if not cap.isOpened():
                    cap.release()
                    continue
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                if frame_count < self.frames_per_clip:
                    continue
                n_windows = frame_count - self.frames_per_clip + 1
                for s in range(n_windows):
                    self.samples.append((cd, s))
            elif cd.is_dir():
                frames = [f for f in sorted(cd.iterdir()) if f.is_file() and is_image_file(f)]
                if len(frames) < self.frames_per_clip:
                    continue
                n_windows = len(frames) - self.frames_per_clip + 1
                for s in range(n_windows):
                    self.samples.append((cd, s))
            else:
                # unknown entry (neither dir nor supported video file) — skip
                continue

        if not self.samples:
            raise FileNotFoundError(f"No windows (length={self.frames_per_clip}) found in {self.root}")

        self.transforms = transforms_new or transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        # two samples per window (natural/shuffled)
        return len(self.samples) * 2

    def __getitem__(self, index: int):
        window_idx = index // 2
        is_natural = (index % 2 == 0)
        clip_dir, start = self.samples[window_idx]
        # clip_dir may be a directory of images or a video file
        if clip_dir.is_file() and clip_dir.suffix.lower() in VIDEO_EXTS:
            if not HAS_CV2:
                raise RuntimeError("OpenCV is required to read video files. Install with: pip install opencv-python")
            # read frames from video using OpenCV
            cap = cv2.VideoCapture(str(clip_dir))
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Failed to open video file {clip_dir}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
            imgs = []
            for i in range(self.frames_per_clip):
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgs.append(self.transforms(Image.fromarray(frame)))
            cap.release()
            if len(imgs) < self.frames_per_clip:
                raise RuntimeError(f"Video {clip_dir} did not yield enough frames at start={start}")
        else:
            frames = [f for f in sorted(clip_dir.iterdir()) if f.is_file() and is_image_file(f)]
            selected = frames[start:start + self.frames_per_clip]
            imgs = [self.transforms(Image.open(p).convert("RGB")) for p in selected]

        if not is_natural:
            order = list(range(len(imgs)))
            random.shuffle(order)
            imgs = [imgs[i] for i in order]

        video = torch.stack(imgs, dim=0)  # [T, C, H, W]
        video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
        label = float(is_natural)
        return video, torch.tensor(label, dtype=torch.float32)


class VideoOrderNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Try modern weights API, fallback to pretrained flag if needed
        try:
            weights = models.video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            backbone = models.video.r2plus1d_18(weights=weights)
        except Exception:
            backbone = models.video.r2plus1d_18(pretrained=pretrained)

        # Replace final classifier with identity so encoder returns 512-d vector
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()
        self.encoder = backbone
        self.classifier = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        feats = self.encoder(x)
        # feats: [B, 512]
        logits = self.classifier(feats).squeeze(1)
        return logits, feats


def split_list(items: List, train_ratio=0.8, val_ratio=0.1, seed=42) -> Tuple[List, List, List]:
    random.seed(seed)
    items = items.copy()
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for vids, labels in pbar:
        vids = vids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(vids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item()) * vids.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += int((preds == labels).sum().item())
        total += vids.size(0)
        pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for vids, labels in tqdm(loader, desc="Val", leave=False):
        vids = vids.to(device)
        labels = labels.to(device)
        logits, _ = model(vids)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * vids.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += int((preds == labels).sum().item())
        total += vids.size(0)

    if total == 0:
        return {"loss": float("inf"), "acc": 0.0}
    return {"loss": total_loss / total, "acc": correct / total}


def main():
    parser = argparse.ArgumentParser(description="Train video encoder on frame-order verification")
    parser.add_argument("--data-dir", type=str, default="lip_crop_results_full")
    parser.add_argument("--frames", type=int, default=FRAMES_PER_CLIP, help="Number of frames per input window (default 8)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="video_frame_order_results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover clips and split using requested window length
    frames_per_clip = int(args.frames)
    clip_dirs = find_clip_dirs(data_dir, min_frames=frames_per_clip)
    if not clip_dirs:
        # helpful debug: list top-level entries to help diagnose path issues
        entries = sorted([p.name for p in data_dir.iterdir()]) if data_dir.exists() else []
        logger.error("No clips found matching requirements.")
        logger.info(f"Searched data-dir: {data_dir.resolve()}")
        logger.info(f"Required minimum frames per clip: {frames_per_clip}")
        logger.info(f"Top-level entries: {entries[:50]}")
        raise FileNotFoundError(f"No clips with >={frames_per_clip} frames found in {data_dir}")

    train_dirs, val_dirs, test_dirs = split_list(clip_dirs, train_ratio=0.8, val_ratio=0.1, seed=args.seed)

    # create datasets that point only at the selected clip dirs
    def dataset_from_dirs(dirs):
        return VideoFrameOrderDataset(data_dir, frames_per_clip=frames_per_clip, clip_dirs=dirs)

    train_ds = dataset_from_dirs(train_dirs)
    val_ds = dataset_from_dirs(val_dirs)
    test_ds = dataset_from_dirs(test_dirs)

    logger.info(f"Train clips: {len(train_ds.clip_dirs)}, Val clips: {len(val_ds.clip_dirs)}, Test clips: {len(test_ds.clip_dirs)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = VideoOrderNet(pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.dry_run:
        # fetch a single batch and run a forward pass
        vids, labels = next(iter(train_loader))
        logger.info(f"Dry run batch vids shape: {vids.shape}, labels shape: {labels.shape}")
        vids = vids.to(device)
        logits, feats = model(vids)
        logger.info(f"Logits shape: {logits.shape}, feats shape: {feats.shape}")
        return

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch}/{args.epochs} | Train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.3f} | Val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f}")

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        # checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
            }, out_dir / "best_model.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, out_dir / "last_model.pt")

    # final test
    test_metrics = validate(model, test_loader, criterion, device)
    logger.info(f"Test loss: {test_metrics['loss']:.4f}, Test acc: {test_metrics['acc']:.3f}")

    # save metrics summary
    with (out_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(history["train_loss"])):
            writer.writerow([i + 1, history["train_loss"][i], history["train_acc"][i], history["val_loss"][i], history["val_acc"][i]])

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
