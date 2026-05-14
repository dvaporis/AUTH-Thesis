"""Mel-spectrogram CPC training for RAVDESS.

This script follows the MFCC contrastive reference style, but uses the precomputed
128x50 mel tensors in `data/ravdess_mels_0.5s`.
Architecture:
    mel chunk [128, 50]
        -> split into 25 overlapping slices of [128, 6] with stride 2
        -> ResNet18 encoder per slice -> 25 x 512 latent vectors
        -> GRU predictor consumes z_1..z_10
        -> predict z_11
        -> CPC / InfoNCE loss against batch negatives

Default training objective matches the user request:
    context steps = 10
    future horizons = +3..+7 steps
    latent dimension = 512

Usage:
    python train_ravdess_mel_cpc.py --data-dir data/ravdess_mels_0.5s --epochs 50

Dry run / shape check:
    python train_ravdess_mel_cpc.py --data-dir data/ravdess_mels_0.5s --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MelConfig:
    sample_rate: int = 16000
    mel_bins: int = 128
    frames_per_chunk: int = 50
    time_steps: int = 25
    frames_per_step: int = 2
    frames_per_slice: int = 6
    min_prediction_horizon: int = 3


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.2
    context_steps: int = 10
    prediction_horizons: Tuple[int, ...] = (3, 4, 5, 6, 7)
    predictor_hidden_dim: int = 512
    finetune_resnet: bool = True
    pretrained_resnet: bool = True
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3


class MelChunkDataset(torch.utils.data.Dataset):
    """Dataset for saved RAVDESS mel chunks in `*.npz` format."""

    def __init__(self, data_dir: Path, manifest_path: Optional[Path] = None, files: Optional[List[Path]] = None):
        self.data_dir = Path(data_dir)
        self.files: List[Path] = files[:] if files is not None else []

        if not self.files and manifest_path is not None and manifest_path.exists():
            with manifest_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rel = row["file"].strip()
                    candidate = self.data_dir.parent / rel
                    if candidate.exists():
                        self.files.append(candidate)

        if not self.files:
            self.files = sorted(self.data_dir.glob("*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No .npz mel files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        mel_path = self.files[index]
        with np.load(mel_path) as data:
            mel = data["mel"].astype(np.float32)

        if mel.shape != (128, 50):
            raise ValueError(f"Expected mel shape (128, 50), got {mel.shape} for {mel_path}")

        # Normalize each chunk independently so the encoder focuses on structure,
        # not on absolute scale differences between recordings.
        mean = float(mel.mean())
        std = float(mel.std())
        mel = (mel - mean) / (std + 1e-6)

        # Increase temporal receptive field: pad 50 frames to 54 so we can extract
        # 25 slices of [128, 6] with stride=2. Each slice is 60ms (6 frames × 10ms hop).
        # Stride-2 slicing: positions 0, 2, 4, ..., 48 (25 positions total)
        mel = np.pad(mel, ((0, 0), (0, 4)), mode="edge")  # [128, 54]

        # Create 25 overlapping slices of [128, 6] using stride-2 windows.
        mel_tensor = torch.from_numpy(mel)
        mel_slices = []
        for i in range(0, 50, 2):  # Positions: 0, 2, 4, ..., 48
            mel_slices.append(mel_tensor[:, i:i+6])
        mel_slices = torch.stack(mel_slices)  # [25, 128, 6]

        return {
            "mel_slices": mel_slices,
            "file_idx": torch.tensor(index, dtype=torch.long),
        }


class ResNet18MelEncoder(nn.Module):
    """Pretrained ResNet18 adapted for 1-channel mel slices."""

    def __init__(self, finetune: bool = True, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Replace the RGB stem with a single-channel stem for mel input.
        # Use stride=(1, 1) instead of (2, 2) to preserve temporal information
        # in the small mel slices [128, 6].
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=original_conv.kernel_size,
            stride=(1, 1),  # Reduce temporal stride for better lip-sync alignment
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize the new stem from averaged ImageNet weights.
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

        self.backbone.fc = nn.Identity()

        if not finetune:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, mel_slice: torch.Tensor) -> torch.Tensor:
        # mel_slice: [batch, 1, 128, 2]
        return self.backbone(mel_slice)


class MultiStepPredictor(nn.Module):
    """GRU context encoder with one linear head per future horizon."""

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 512, horizons: Tuple[int, ...] = (1, 2, 3, 4, 5)):
        super().__init__()
        # Encode the observed context sequence z_1..z_t into a single hidden state.
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.horizons = tuple(int(h) for h in horizons)
        # Use a separate projection head per horizon because each future offset
        # has a different relationship to the context representation.
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, latent_dim),
            )
            for h in self.horizons
        })

    def forward(self, sequence: torch.Tensor) -> Dict[int, torch.Tensor]:
        # sequence: [batch, context_steps, latent_dim]
        output, hidden = self.gru(sequence)
        context = hidden[-1]
        predictions: Dict[int, torch.Tensor] = {}
        for horizon in self.horizons:
            prediction = self.heads[str(horizon)](context)
            predictions[horizon] = F.normalize(prediction, dim=1)
        return predictions


class CPCInfoNCELoss(nn.Module):
    """InfoNCE loss using batch negatives for CPC prediction."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape[0] < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        predictions = F.normalize(predictions, dim=1)
        targets = F.normalize(targets, dim=1)

        logits = torch.matmul(predictions, targets.T) / self.temperature
        labels = torch.arange(predictions.shape[0], device=predictions.device)
        return F.cross_entropy(logits, labels)


class MultiStepCPCInfoNCELoss(nn.Module):
    """Average InfoNCE loss across multiple future horizons."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.base_loss = CPCInfoNCELoss(temperature=temperature)

    def forward(self, predictions: Dict[int, torch.Tensor], targets: Dict[int, torch.Tensor]) -> torch.Tensor:
        losses = []
        for horizon, prediction in predictions.items():
            target = targets[horizon]
            losses.append(self.base_loss(prediction, target))

        if not losses:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device, requires_grad=True)

        return torch.stack(losses).mean()


class MelCPCAudioModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        if not 1 <= config.context_steps < MelConfig.time_steps:
            raise ValueError(
                f"Expected 1 <= context_steps < {MelConfig.time_steps}, "
                f"got context_steps={config.context_steps}"
            )
        if not config.prediction_horizons:
            raise ValueError("prediction_horizons cannot be empty")
        if min(config.prediction_horizons) < MelConfig.min_prediction_horizon:
            raise ValueError(
                f"prediction_horizons must start at >= {MelConfig.min_prediction_horizon} "
                f"to keep targets outside the overlapping 6-frame context window"
            )
        if max(config.prediction_horizons) + config.context_steps > MelConfig.time_steps:
            raise ValueError(
                f"Context steps {config.context_steps} plus max horizon {max(config.prediction_horizons)} "
                f"exceeds available time steps {MelConfig.time_steps}"
            )
        # Encoder learns one 512-D vector per 2-frame mel slice.
        self.encoder = ResNet18MelEncoder(finetune=config.finetune_resnet, pretrained=config.pretrained_resnet)
        self.predictor = MultiStepPredictor(latent_dim=512, hidden_dim=config.predictor_hidden_dim, horizons=config.prediction_horizons)

    def encode_slices(self, mel_slices: torch.Tensor) -> torch.Tensor:
        # mel_slices: [batch, 25, 128, 2]
        batch_size, time_steps, height, width = mel_slices.shape
        reshaped = mel_slices.reshape(batch_size * time_steps, 1, height, width)
        z = self.encoder(reshaped)
        return z.reshape(batch_size, time_steps, -1)

    def forward(self, mel_slices: torch.Tensor, context_steps: int = 10) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        z_seq = self.encode_slices(mel_slices)
        predictions = self.predictor(z_seq[:, :context_steps, :])
        return z_seq, predictions


def split_dataset(files: List[Path], train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    random.seed(seed)
    shuffled = files.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]
    return train_files, val_files, test_files


def train_epoch(model: MelCPCAudioModel, dataloader: torch.utils.data.DataLoader, criterion: MultiStepCPCInfoNCELoss, optimizer: torch.optim.Optimizer, device: torch.device, context_steps: int, prediction_horizons: Tuple[int, ...], epoch: int = 1, total_epochs: int = 1) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Batch-level progress bar shows the current epoch and running loss.
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch", leave=False)

    for batch in pbar:
        mel_slices = batch["mel_slices"].to(device)

        if mel_slices.shape[0] < 2:
            continue

        # Predict several future horizons from the observed context.
        z_seq, predictions = model(mel_slices, context_steps=context_steps)
        targets = {horizon: z_seq[:, context_steps + horizon - 1, :] for horizon in prediction_horizons}

        loss = criterion(predictions, targets)
        if not torch.isfinite(loss):
            logger.warning("Skipping batch with non-finite loss")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        pbar.set_postfix({"loss": total_loss / num_batches})

    if num_batches == 0:
        return {"loss": float("inf")}
    return {"loss": total_loss / num_batches}


@torch.no_grad()
def validate(model: MelCPCAudioModel, dataloader: torch.utils.data.DataLoader, criterion: MultiStepCPCInfoNCELoss, device: torch.device, context_steps: int, prediction_horizons: Tuple[int, ...]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation", unit="batch", leave=False):
        mel_slices = batch["mel_slices"].to(device)
        if mel_slices.shape[0] < 2:
            continue

        z_seq, predictions = model(mel_slices, context_steps=context_steps)
        targets = {horizon: z_seq[:, context_steps + horizon - 1, :] for horizon in prediction_horizons}
        loss = criterion(predictions, targets)

        if not torch.isfinite(loss):
            logger.warning("Skipping batch with non-finite validation loss")
            continue

        total_loss += float(loss.item())
        num_batches += 1

    if num_batches == 0:
        return {"loss": float("inf")}
    return {"loss": total_loss / num_batches}


def dry_run(model: MelCPCAudioModel, dataset: torch.utils.data.Dataset, device: torch.device, context_steps: int, prediction_horizons: Tuple[int, ...]) -> None:
    # Quick tensor-shape sanity check before a full training run.
    sample = dataset[0]
    mel_slices = sample["mel_slices"].unsqueeze(0).to(device)
    with torch.no_grad():
        z_seq, predictions = model(mel_slices, context_steps=context_steps)
    logger.info(f"Dry run mel_slices: {mel_slices.shape}")
    logger.info(f"Dry run z_seq: {z_seq.shape}")
    logger.info(f"Dry run prediction horizons: {prediction_horizons}")
    for horizon, prediction in predictions.items():
        logger.info(f"Dry run prediction[{horizon}]: {prediction.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mel-based CPC audio encoder on RAVDESS chunks")
    parser.add_argument("--data-dir", type=str, default="data/ravdess_mels_0.5s")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--context-steps", type=int, default=10)
    parser.add_argument(
        "--prediction-horizons",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7],
        help="Future offsets in slice steps; start at +3 so targets do not overlap the context window",
    )
    parser.add_argument("--predictor-hidden-dim", type=int, default=512)
    parser.add_argument("--freeze-resnet", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Initialize ResNet18 without ImageNet weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="ravdess_mel_cpc_results")
    parser.add_argument("--dry-run", action="store_true", help="Load one batch and print tensor shapes")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(data_dir.glob("*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    train_files, val_files, test_files = split_dataset(all_files, seed=args.seed)
    train_dataset = MelChunkDataset(data_dir, files=train_files)
    val_dataset = MelChunkDataset(data_dir, files=val_files)
    test_dataset = MelChunkDataset(data_dir, files=test_files)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        context_steps=args.context_steps,
        prediction_horizons=tuple(args.prediction_horizons),
        predictor_hidden_dim=args.predictor_hidden_dim,
        finetune_resnet=not args.freeze_resnet,
        pretrained_resnet=not args.no_pretrained,
    )

    model = MelCPCAudioModel(training_config).to(device)
    criterion = MultiStepCPCInfoNCELoss(temperature=training_config.temperature)
    optimizer = torch.optim.Adam(
        model.parameters() if training_config.finetune_resnet else model.predictor.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    if args.dry_run:
        dry_run(model, train_dataset, device, training_config.context_steps, training_config.prediction_horizons)
        return

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Top-level progress bar for epochs.
    epoch_bar = tqdm(range(training_config.num_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            context_steps=training_config.context_steps,
            prediction_horizons=training_config.prediction_horizons,
            epoch=epoch + 1,
            total_epochs=training_config.num_epochs,
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            context_steps=training_config.context_steps,
            prediction_horizons=training_config.prediction_horizons,
        )

        logger.info(
            f"Epoch {epoch + 1:03d}/{training_config.num_epochs} | Train {train_metrics['loss']:.4f} | Val {val_metrics['loss']:.4f}"
        )
        epoch_bar.set_postfix(train_loss=f"{train_metrics['loss']:.4f}", val_loss=f"{val_metrics['loss']:.4f}")

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
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config.early_stopping_patience:
            logger.info("Early stopping triggered")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": training_config.__dict__,
        },
        output_dir / "final_model.pt",
    )

    test_metrics = validate(
        model,
        test_loader,
        criterion,
        device,
        context_steps=training_config.context_steps,
        prediction_horizons=training_config.prediction_horizons,
    )

    with (output_dir / "test_metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"test_loss={test_metrics['loss']:.6f}\n")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Test loss: {test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
