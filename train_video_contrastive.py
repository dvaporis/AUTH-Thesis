"""
Video contrastive learning for lip videos with a pre-trained 3D ResNet and NT-Xent loss.

This script implements contrastive learning for lip videos using:
1. Fixed-length clip extraction from videos in lip_crop_results_full
2. Pre-trained 3D ResNet backbone for visual feature extraction
3. 2-layer MLP projection head (input -> 512 -> 128)
4. NT-Xent loss with positive pairs formed by original clips and their augmented versions
5. Exactly one augmentation per clip: rotation, horizontal flip, color jitter, grayscale,
   gaussian noise, or time stretching
6. Training progress bars, early stopping, and loss plotting

Usage:
    python train_video_contrastive.py --epochs 50 --batch-size 8 --lr 0.0003
"""

import argparse
import logging
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    clip_length: int = 16
    frame_size: int = 112

    rotation_degrees: float = 12.0
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: Tuple[float, float] = (-0.08, 0.08)
    noise_std: float = 0.03
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 0.0003
    weight_decay: float = 1e-4
    temperature: float = 0.07

    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    finetune_backbone: bool = True
    projection_hidden_dim: int = 512
    projection_output_dim: int = 128

    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3


class VideoAugmentation:
    """Exactly one augmentation per video clip."""

    def __init__(self, config: VideoConfig):
        self.config = config

    def augment(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply exactly one augmentation to a clip [C, T, H, W]."""
        augmentation_methods = [
            ("rotation", self._augment_rotation),
            ("horizontal_flip", self._augment_horizontal_flip),
            ("color_jitter", self._augment_color_jitter),
            ("grayscale", self._augment_grayscale),
            ("gaussian_noise", self._augment_gaussian_noise),
            ("time_stretch", self._augment_time_stretch),
        ]

        _, aug_func = random.choice(augmentation_methods)
        return aug_func(clip.clone())

    def _apply_per_frame(self, clip: torch.Tensor, fn) -> torch.Tensor:
        frames = [fn(clip[:, frame_idx]) for frame_idx in range(clip.shape[1])]
        return torch.stack(frames, dim=1)

    def _augment_rotation(self, clip: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.config.rotation_degrees, self.config.rotation_degrees)

        def rotate_frame(frame: torch.Tensor) -> torch.Tensor:
            return TF.rotate(frame, angle=angle, interpolation=InterpolationMode.BILINEAR)

        return self._apply_per_frame(clip, rotate_frame)

    def _augment_horizontal_flip(self, clip: torch.Tensor) -> torch.Tensor:
        return torch.flip(clip, dims=[3])

    def _augment_color_jitter(self, clip: torch.Tensor) -> torch.Tensor:
        brightness = random.uniform(*self.config.brightness_range)
        contrast = random.uniform(*self.config.contrast_range)
        saturation = random.uniform(*self.config.saturation_range)
        hue = random.uniform(*self.config.hue_range)

        def jitter_frame(frame: torch.Tensor) -> torch.Tensor:
            frame = TF.adjust_brightness(frame, brightness)
            frame = TF.adjust_contrast(frame, contrast)
            frame = TF.adjust_saturation(frame, saturation)
            frame = TF.adjust_hue(frame, hue)
            return frame

        return self._apply_per_frame(clip, jitter_frame)

    def _augment_grayscale(self, clip: torch.Tensor) -> torch.Tensor:
        def grayscale_frame(frame: torch.Tensor) -> torch.Tensor:
            return TF.rgb_to_grayscale(frame, num_output_channels=3)

        return self._apply_per_frame(clip, grayscale_frame)

    def _augment_gaussian_noise(self, clip: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(clip) * self.config.noise_std
        return torch.clamp(clip + noise, 0.0, 1.0)

    def _augment_time_stretch(self, clip: torch.Tensor) -> torch.Tensor:
        stretch = random.uniform(*self.config.time_stretch_range)
        original_length = clip.shape[1]
        stretched_length = max(2, int(round(original_length * stretch)))

        stretched = F.interpolate(
            clip.unsqueeze(0),
            size=(stretched_length, clip.shape[2], clip.shape[3]),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        if stretched.shape[1] != original_length:
            stretched = F.interpolate(
                stretched.unsqueeze(0),
                size=(original_length, clip.shape[2], clip.shape[3]),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        return torch.clamp(stretched, 0.0, 1.0)


class VideoClipDataset(torch.utils.data.Dataset):
    """Dataset for fixed-length clips from lip videos."""

    def __init__(
        self,
        video_files: List[Path],
        config: VideoConfig,
        augmentation: Optional[VideoAugmentation] = None,
    ):
        self.video_files = video_files
        self.config = config
        self.augmentation = augmentation
        self.clips: List[Dict[str, object]] = []
        self._build_clip_index()

    def _build_clip_index(self) -> None:
        logger.info("Building video clip index from lip_crop_results_full...")

        for file_idx, video_file in enumerate(self.video_files):
            try:
                total_frames = self._get_total_frames(video_file)
                if total_frames < self.config.clip_length:
                    logger.warning(f"Video file {video_file.name} is too short for one clip")
                    continue

                num_clips = total_frames // self.config.clip_length
                if num_clips == 0:
                    continue

                for clip_idx in range(num_clips):
                    start_frame = clip_idx * self.config.clip_length
                    end_frame = start_frame + self.config.clip_length

                    self.clips.append(
                        {
                            "file_idx": file_idx,
                            "file_path": video_file,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "clip_idx": clip_idx,
                            "total_frames": total_frames,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to process {video_file}: {e}")

        logger.info(f"Built index with {len(self.clips)} clips from {len(self.video_files)} video files")

    def _get_total_frames(self, video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def __len__(self) -> int:
        return len(self.clips)

    def _read_clip(self, video_path: Path, start_frame: int) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.config.clip_length):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.config.frame_size, self.config.frame_size))
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            frames.append(frame_tensor)

        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from {video_path.name}")

        while len(frames) < self.config.clip_length:
            frames.append(frames[-1].clone())

        clip = torch.stack(frames[: self.config.clip_length], dim=1)
        return clip

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_info = self.clips[idx]

        try:
            video_path = clip_info["file_path"]
            start_frame = int(clip_info["start_frame"])
            clip_original = self._read_clip(video_path, start_frame)

            result = {
                "clip_original": clip_original,
                "file_idx": clip_info["file_idx"],
                "clip_idx": idx,
            }

            if self.augmentation is not None:
                result["clip_augmented"] = self.augmentation.augment(clip_original)

            return result

        except Exception as e:
            logger.error(f"Error loading clip {idx}: {e}")
            fallback = torch.zeros(3, self.config.clip_length, self.config.frame_size, self.config.frame_size)
            return {
                "clip_original": fallback,
                "clip_augmented": fallback.clone(),
                "file_idx": -1,
                "clip_idx": idx,
            }


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=1)


class VideoResNetContrastiveModel(nn.Module):
    """Contrastive model with a pre-trained 3D ResNet backbone."""

    def __init__(self, config: TrainingConfig):
        super().__init__()

        weights = R3D_18_Weights.DEFAULT
        self.backbone = r3d_18(weights=weights)

        if not config.finetune_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("3D ResNet backbone frozen (no fine-tuning)")
        else:
            logger.info("3D ResNet backbone fine-tuning enabled")

        backbone_output_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projection_head = ProjectionHead(
            input_dim=backbone_output_dim,
            hidden_dim=config.projection_hidden_dim,
            output_dim=config.projection_output_dim,
        )

        meta = getattr(weights, "meta", {}) if hasattr(weights, "meta") else {}
        mean = torch.tensor(meta.get("mean", [0.43216, 0.394666, 0.37645]), dtype=torch.float32)
        std = torch.tensor(meta.get("std", [0.22803, 0.22145, 0.216989]), dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1, 1))

        logger.info(
            f"Model initialized: 3D ResNet -> {backbone_output_dim}D -> "
            f"ProjectionHead(hidden={config.projection_hidden_dim}, output={config.projection_output_dim})"
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        clip = (clip - self.mean) / self.std
        features = self.backbone(clip)
        return self.projection_head(features)


class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        device = z_i.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        eye = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(eye, float("-inf"))

        pos_indices = torch.arange(batch_size, device=device)
        targets = torch.cat([pos_indices + batch_size, pos_indices], dim=0)

        loss = F.cross_entropy(sim_matrix, targets)
        return loss


def find_lip_video_files() -> List[Path]:
    """Find lip videos from lip_crop_results_full."""
    lip_path = Path("lip_crop_results_full")

    video_files: List[Path] = []
    if lip_path.exists():
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        for ext in video_extensions:
            video_files.extend(lip_path.rglob(f"*{ext}"))

        if video_files:
            logger.info(f"Found {len(video_files)} video files in {lip_path}")
            return video_files

    logger.warning(f"No video files found in {lip_path}!")
    return []


def split_dataset(
    video_files: List[Path],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split videos into train/val/test sets."""
    random.seed(seed)
    video_files_shuffled = video_files.copy()
    random.shuffle(video_files_shuffled)

    n = len(video_files_shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = video_files_shuffled[:n_train]
    val_files = video_files_shuffled[n_train : n_train + n_val]
    test_files = video_files_shuffled[n_train + n_val :]

    logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    return train_files, val_files, test_files


def train_epoch(
    model: VideoResNetContrastiveModel,
    dataloader: torch.utils.data.DataLoader,
    criterion: NTXentLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch", leave=True)

    for batch in pbar:
        clip_original = batch["clip_original"].to(device)
        clip_augmented = batch["clip_augmented"].to(device)
        file_idx = batch["file_idx"].to(device)

        valid_mask = file_idx >= 0
        if not valid_mask.any():
            continue

        clip_original = clip_original[valid_mask]
        clip_augmented = clip_augmented[valid_mask]

        if clip_original.shape[0] < 2:
            continue

        projections_original = model(clip_original)
        projections_augmented = model(clip_augmented)
        loss = criterion(projections_original, projections_augmented)

        if not torch.isfinite(loss):
            logger.warning("Skipping batch with non-finite loss")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": total_loss / num_batches})

    if num_batches == 0:
        return {"loss": float("inf")}

    return {"loss": total_loss / num_batches}


@torch.no_grad()
def validate(
    model: VideoResNetContrastiveModel,
    dataloader: torch.utils.data.DataLoader,
    criterion: NTXentLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        clip_original = batch["clip_original"].to(device)
        clip_augmented = batch["clip_augmented"].to(device)
        file_idx = batch["file_idx"].to(device)

        valid_mask = file_idx >= 0
        if not valid_mask.any():
            continue

        clip_original = clip_original[valid_mask]
        clip_augmented = clip_augmented[valid_mask]

        if clip_original.shape[0] < 2:
            continue

        projections_original = model(clip_original)
        projections_augmented = model(clip_augmented)
        loss = criterion(projections_original, projections_augmented)

        if not torch.isfinite(loss):
            logger.warning("Skipping batch with non-finite validation loss")
            continue

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return {"loss": float("inf")}

    return {"loss": total_loss / num_batches}


def plot_training_history(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot and save training history."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("NT-Xent Loss", fontsize=12)
    ax.set_title("Lip Video Contrastive Learning - Training Progress", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Training history saved to {save_path}")


def main() -> None:
    logger.info("=" * 80)
    logger.info("Video Contrastive Learning with 3D ResNet")
    logger.info("=" * 80)

    parser = argparse.ArgumentParser(description="Video contrastive learning with 3D ResNet")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.07, help="NT-Xent temperature")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze 3D ResNet backbone")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3, help="Minimum improvement for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="video_contrastive_results", help="Output directory")
    parser.add_argument("--clip-length", type=int, default=16, help="Frames per clip")
    parser.add_argument("--frame-size", type=int, default=112, help="Frame size for model input")

    args = parser.parse_args()
    logger.info("[1/9] Configuration loaded")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.info(f"[2/9] Random seeds set (seed={args.seed})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[3/9] Device initialized: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"[4/9] Output directory created: {output_dir.absolute()}")

    video_config = VideoConfig(clip_length=args.clip_length, frame_size=args.frame_size)
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        finetune_backbone=not args.freeze_backbone,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    logger.info(
        f"Video config: clip_length={video_config.clip_length}, frame_size={video_config.frame_size}"
    )
    logger.info(
        f"Training config: epochs={training_config.num_epochs}, batch_size={training_config.batch_size}, "
        f"lr={training_config.learning_rate}, temperature={training_config.temperature}"
    )
    logger.info(
        f"Early stopping: patience={training_config.early_stopping_patience}, "
        f"min_delta={training_config.early_stopping_min_delta}"
    )

    logger.info("[5a/9] Searching for lip videos in lip_crop_results_full...")
    video_files = find_lip_video_files()
    if len(video_files) == 0:
        logger.error("No video files found! Please ensure lip_crop_results_full contains video files.")
        return

    logger.info(f"[5b/9] Found {len(video_files)} total video files")

    logger.info("[5c/9] Splitting dataset into train/val/test...")
    train_files, val_files, test_files = split_dataset(
        video_files,
        train_ratio=training_config.train_ratio,
        val_ratio=training_config.val_ratio,
        test_ratio=training_config.test_ratio,
        seed=args.seed,
    )

    logger.info("[5d/9] Initializing augmentation pipeline...")
    augmentation = VideoAugmentation(video_config)

    logger.info("[5e/9] Creating video datasets and clip index...")
    train_dataset = VideoClipDataset(train_files, video_config, augmentation=augmentation)
    val_dataset = VideoClipDataset(val_files, video_config, augmentation=augmentation)
    test_dataset = VideoClipDataset(test_files, video_config, augmentation=augmentation)

    logger.info(
        f"[5f/9] Dataset sizes - Train: {len(train_dataset)} clips, Val: {len(val_dataset)} clips, Test: {len(test_dataset)} clips"
    )

    logger.info("[5g/9] Creating data loaders...")
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    logger.info(
        f"Data loaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches, {len(test_loader)} test batches"
    )

    logger.info("[6/9] Creating pre-trained 3D ResNet model with projection head...")
    model = VideoResNetContrastiveModel(training_config).to(device)

    logger.info("[7/9] Setting up loss function and optimizer...")
    criterion = NTXentLoss(temperature=training_config.temperature)
    logger.info(f"Loss function: NT-Xent (temperature={training_config.temperature})")

    if not training_config.finetune_backbone:
        optimizer = torch.optim.Adam(
            model.projection_head.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        logger.info("Optimizer: Adam (projection head only)")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        logger.info("Optimizer: Adam (all parameters including 3D ResNet fine-tuning)")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    logger.info("Learning rate scheduler: ReduceLROnPlateau")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    logger.info("=" * 80)
    logger.info("[8/9] STARTING TRAINING LOOP")
    logger.info("=" * 80)

    for epoch in range(training_config.num_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch + 1,
            total_epochs=training_config.num_epochs,
        )

        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        logger.info(
            f"  Epoch {epoch + 1:3d}/{training_config.num_epochs} │ "
            f"Train: {train_metrics['loss']:.4f} │ Val: {val_metrics['loss']:.4f} │ "
            f"No-improve: {epochs_without_improvement}/{training_config.early_stopping_patience}"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        if val_metrics["loss"] < (best_val_loss - training_config.early_stopping_min_delta):
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "config": training_config.__dict__,
                },
                output_dir / "best_model.pt",
            )
            logger.info(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs (best val_loss: {best_val_loss:.4f})"
                )
                break

    logger.info("=" * 80)
    logger.info("[9/9] EVALUATING ON TEST SET")
    logger.info("=" * 80)

    best_model_path = output_dir / "best_model.pt"
    if not best_model_path.exists():
        logger.warning("Best model checkpoint was not saved during training; skipping test evaluation.")
        return

    logger.info(f"Loading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Evaluating on {len(test_loader)} test batches...")
    test_metrics = validate(model, test_loader, criterion, device)
    logger.info(f"✓ Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Best Val Loss: {best_val_loss:.4f}")

    logger.info("Saving test metrics...")
    with open(output_dir / "test_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
    logger.info(f"✓ Test metrics saved to: {output_dir / 'test_metrics.txt'}")

    logger.info("Generating training history plot...")
    plot_training_history(history, output_dir / "training_history.png")
    logger.info(f"✓ Training plot saved to: {output_dir / 'training_history.png'}")

    logger.info("Saving final model...")
    torch.save(
        {
            "epoch": training_config.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": training_config.__dict__,
        },
        output_dir / "final_model.pt",
    )
    logger.info(f"✓ Final model saved to: {output_dir / 'final_model.pt'}")

    logger.info("=" * 80)
    logger.info("✓✓✓ TRAINING COMPLETE ✓✓✓")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir.absolute()}")
    logger.info("Files created:")
    logger.info("  - best_model.pt (best validation checkpoint)")
    logger.info("  - final_model.pt (final checkpoint)")
    logger.info("  - test_metrics.txt (test performance)")
    logger.info("  - training_history.png (loss curves)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()