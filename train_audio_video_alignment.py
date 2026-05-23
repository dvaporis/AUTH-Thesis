"""Train audio-video alignment from pretrained CPC and frame-order encoders.

Pipeline per 0.5s paired chunk:
    video chunk -> 11 x 512 tokens from pretrained video encoder
    -> 1D transposed convolution (11 -> 25 tokens)
    -> 3-layer Transformer encoder
    -> aligned video token sequence [25, 512]

Targets:
    audio chunk -> pretrained mel CPC encoder -> 25 x 512 tokens

Training schedule:
1) Joint phase: train audio encoder + video encoder + upsampler + transformer
   until validation loss converges (early stopping).
2) Transformer-only phase: freeze both encoders and train only transformer
   (and its positional embedding) until convergence.

Usage:
    python train_audio_video_alignment.py --dry-run
    python train_audio_video_alignment.py --joint-epochs 40 --transformer-only-epochs 30
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    import cv2
except Exception as exc:  # pragma: no cover - explicit runtime dependency guard
    raise RuntimeError("OpenCV is required for video loading. Install with: pip install opencv-python") from exc

from train_ravdess_mel_cpc import MelCPCAudioModel, TrainingConfig
from train_video_frame_order import VideoOrderNet


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class PairSample:
    audio_path: Path
    video_path: Path
    clip_stem: str
    chunk_idx: int


class PairedRavdessChunkDataset(Dataset):
    """Return temporally aligned audio mel slices and video frame chunks."""

    def __init__(
        self,
        pairs: Sequence[PairSample],
        chunk_seconds: float = 0.5,
        video_tokens_per_chunk: int = 15,
    ):
        if not pairs:
            raise ValueError("Paired dataset is empty")
        self.pairs = list(pairs)
        self.chunk_seconds = float(chunk_seconds)
        self.video_tokens_per_chunk = int(video_tokens_per_chunk)

        self.video_transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_audio_slices(self, audio_path: Path) -> torch.Tensor:
        with np.load(audio_path) as data:
            mel = data["mel"].astype(np.float32)

        if mel.shape != (128, 50):
            raise ValueError(f"Expected mel shape (128, 50), got {mel.shape} for {audio_path}")

        mean = float(mel.mean())
        std = float(mel.std())
        mel = (mel - mean) / (std + 1e-6)

        mel = np.pad(mel, ((0, 0), (0, 4)), mode="edge")  # [128, 54]
        mel_tensor = torch.from_numpy(mel)
        mel_slices = [mel_tensor[:, i : i + 6] for i in range(0, 50, 2)]
        return torch.stack(mel_slices)  # [25, 128, 6]

    def _read_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        if not ok:
            return None
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _load_video_tokens(self, video_path: Path, chunk_idx: int) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        start_frame = int(round(chunk_idx * self.chunk_seconds * fps))
        chunk_frames = max(1, int(round(self.chunk_seconds * fps)))
        end_frame = min(total_frames - 1, start_frame + chunk_frames - 1)
        start_frame = min(start_frame, total_frames - 1)

        sampled_idx = np.linspace(
            start_frame,
            end_frame,
            num=self.video_tokens_per_chunk,
            dtype=np.int64,
        )

        frames: List[torch.Tensor] = []
        for idx in sampled_idx.tolist():
            frame_rgb = self._read_frame(cap, int(idx))
            if frame_rgb is None:
                break
            frame_tensor = self.video_transform(Image.fromarray(frame_rgb))
            frames.append(frame_tensor)

        cap.release()

        if not frames:
            raise RuntimeError(f"Failed to decode frames for {video_path}, chunk={chunk_idx}")

        while len(frames) < self.video_tokens_per_chunk:
            frames.append(frames[-1].clone())

        frames = frames[: self.video_tokens_per_chunk]
        return torch.stack(frames, dim=0)  # [15, 3, 112, 112]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.pairs[index]
        mel_slices = self._load_audio_slices(sample.audio_path)
        video_frames = self._load_video_tokens(sample.video_path, sample.chunk_idx)

        return {
            "mel_slices": mel_slices,
            "video_frames": video_frames,
        }


def discover_video_files(video_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in sorted(video_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        stem = path.stem
        lookup.setdefault(stem, path)
    return lookup


def pair_audio_video_chunks(audio_dir: Path, video_dir: Path) -> List[PairSample]:
    audio_files = sorted(audio_dir.glob("*.npz"))
    if not audio_files:
        raise FileNotFoundError(f"No mel chunks found in {audio_dir}")

    video_by_stem = discover_video_files(video_dir)
    if not video_by_stem:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    pairs: List[PairSample] = []
    skipped = 0
    pattern = re.compile(r"^(?P<stem>.+)__chunk(?P<idx>\d+)$")

    for audio_path in audio_files:
        match = pattern.match(audio_path.stem)
        if not match:
            skipped += 1
            continue

        stem = match.group("stem")
        chunk_idx = int(match.group("idx"))

        candidates = [stem, f"{stem}_lipcrop"]
        video_path = None
        for key in candidates:
            if key in video_by_stem:
                video_path = video_by_stem[key]
                break

        if video_path is None:
            skipped += 1
            continue

        pairs.append(PairSample(audio_path=audio_path, video_path=video_path, clip_stem=stem, chunk_idx=chunk_idx))

    if not pairs:
        raise RuntimeError(
            "No paired chunks found. Ensure audio names look like '<stem>__chunk0000.npz' "
            "and video names match '<stem>' or '<stem>_lipcrop'."
        )

    logger.info("Paired chunks: %d | skipped: %d", len(pairs), skipped)
    return pairs


def split_pairs(
    pairs: Sequence[PairSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[PairSample], List[PairSample], List[PairSample]]:
    # Split by clip stem to avoid identity leakage across train/val/test.
    stems = sorted({p.clip_stem for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_stems = set(stems[:n_train])
    val_stems = set(stems[n_train : n_train + n_val])
    test_stems = set(stems[n_train + n_val :])

    train = [p for p in pairs if p.clip_stem in train_stems]
    val = [p for p in pairs if p.clip_stem in val_stems]
    test = [p for p in pairs if p.clip_stem in test_stems]

    return train, val, test


class VideoTokenEncoder(nn.Module):
    """Convert a 0.5s frame chunk into a sequence of 11 latent vectors."""

    def __init__(self, video_order_net: VideoOrderNet, window_size: int = 5):
        super().__init__()
        self.encoder = video_order_net.encoder
        self.window_size = int(window_size)

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        # video_frames: [B, T=15, C, H, W]
        batch, time_steps, channels, height, width = video_frames.shape

        x = video_frames.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        windows = []
        for t in range(time_steps - self.window_size + 1):
            clip = x[:, :, t : t + self.window_size, :, :]
            windows.append(clip)

        clip_batch = torch.stack(windows, dim=1)  # [B, 11, C, 5, H, W]
        clip_batch = clip_batch.reshape(batch * len(windows), channels, self.window_size, height, width)

        features = self.encoder(clip_batch)  # [B*T, 512]
        return features.reshape(batch, len(windows), 512)


class TemporalUpsampler(nn.Module):
    """Use 1D transposed convolution to map 11 video tokens to 25 tokens."""

    def __init__(self, dim: int = 512, in_tokens: int = 11, out_tokens: int = 25):
        super().__init__()
        if out_tokens <= in_tokens:
            raise ValueError("out_tokens must be larger than in_tokens")

        kernel_size = out_tokens - in_tokens + 1
        self.deconv = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, video_tokens: torch.Tensor) -> torch.Tensor:
        # [B, 11, 512] -> [B, 25, 512]
        x = video_tokens.transpose(1, 2)
        x = self.deconv(x)
        x = x.transpose(1, 2)
        return self.norm(x)


class VideoAlignmentTransformer(nn.Module):
    """3-layer Transformer encoder over upsampled video tokens."""

    def __init__(self, dim: int = 512, num_layers: int = 3, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 25, dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.out_norm(x)


class AudioVideoAlignmentModel(nn.Module):
    def __init__(
        self,
        audio_model: MelCPCAudioModel,
        video_token_encoder: VideoTokenEncoder,
        upsampler: TemporalUpsampler,
        transformer: VideoAlignmentTransformer,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_token_encoder = video_token_encoder
        self.upsampler = upsampler
        self.transformer = transformer

    def forward(self, mel_slices: torch.Tensor, video_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_tokens = self.audio_model.encode_slices(mel_slices)  # [B, 25, 512]
        video_tokens_15 = self.video_token_encoder(video_frames)  # [B, 15, 512]
        video_tokens_25 = self.upsampler(video_tokens_15)  # [B, 25, 512]
        video_aligned = self.transformer(video_tokens_25)  # [B, 25, 512]
        return audio_tokens, video_aligned


class TokenContrastiveAlignmentLoss(nn.Module):
    """Symmetric soft InfoNCE over flattened token positions with temporal smoothing."""

    def __init__(self, temperature: float = 0.07, temporal_smoothing_sigma: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.temporal_smoothing_sigma = float(temporal_smoothing_sigma)

    def _build_temporal_targets(
        self,
        batch_size: int,
        time_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.temporal_smoothing_sigma <= 0.0:
            return torch.eye(batch_size * time_steps, device=device, dtype=dtype)

        time_indices = torch.arange(time_steps, device=device, dtype=dtype)
        time_diff = time_indices[:, None] - time_indices[None, :]
        kernel = torch.exp(-0.5 * (time_diff / self.temporal_smoothing_sigma) ** 2)
        kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)

        batch_eye = torch.eye(batch_size, device=device, dtype=dtype)
        targets = batch_eye[:, None, :, None] * kernel[None, :, None, :]
        return targets.reshape(batch_size * time_steps, batch_size * time_steps)

    @staticmethod
    def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()

    def forward(self, audio_tokens: torch.Tensor, video_tokens: torch.Tensor) -> torch.Tensor:
        bsz, t, dim = audio_tokens.shape
        if bsz < 2:
            return torch.tensor(0.0, device=audio_tokens.device, requires_grad=True)

        # Normalize before similarity so the objective is driven by angular alignment.
        audio = F.normalize(audio_tokens.reshape(bsz * t, dim), dim=1)
        video = F.normalize(video_tokens.reshape(bsz * t, dim), dim=1)

        logits = torch.matmul(video, audio.T) / self.temperature
        targets = self._build_temporal_targets(
            batch_size=bsz,
            time_steps=t,
            device=video.device,
            dtype=video.dtype,
        )

        loss_v2a = self._soft_cross_entropy(logits, targets)
        loss_a2v = self._soft_cross_entropy(logits.T, targets)
        return 0.5 * (loss_v2a + loss_a2v)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def set_training_mode(model: nn.Module, trainable_modules: Optional[Sequence[nn.Module]] = None) -> None:
    if trainable_modules is None:
        model.train(True)
        return

    model.eval()
    for module in trainable_modules:
        module.train(True)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_epoch(
    model: AudioVideoAlignmentModel,
    loader: DataLoader,
    criterion: TokenContrastiveAlignmentLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    desc: str,
    trainable_modules: Optional[Sequence[nn.Module]] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    set_training_mode(model, trainable_modules=trainable_modules if is_train else None)

    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        mel_slices = batch["mel_slices"].to(device)
        video_frames = batch["video_frames"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            audio_tokens, video_tokens = model(mel_slices, video_frames)
            loss = criterion(audio_tokens, video_tokens)

            if not torch.isfinite(loss):
                logger.warning("Skipping non-finite loss batch")
                continue

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1
        pbar.set_postfix(loss=f"{total_loss / max(total_steps, 1):.4f}")

    if total_steps == 0:
        return {"loss": float("inf")}
    return {"loss": total_loss / total_steps}


def train_until_converged(
    model: AudioVideoAlignmentModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: TokenContrastiveAlignmentLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
    phase_name: str,
    checkpoint_path: Path,
    trainable_modules: Optional[Sequence[nn.Module]] = None,
) -> Tuple[List[Dict[str, float]], float]:
    history: List[Dict[str, float]] = []
    best_val = float("inf")
    epochs_without_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            desc=f"{phase_name} Train {epoch}/{max_epochs}",
            trainable_modules=trainable_modules,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            desc=f"{phase_name} Val {epoch}/{max_epochs}",
        )

        row = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }
        history.append(row)

        logger.info(
            "%s | epoch %03d/%03d | train %.4f | val %.4f",
            phase_name,
            epoch,
            max_epochs,
            train_metrics["loss"],
            val_metrics["loss"],
        )

        if val_metrics["loss"] < best_val - min_delta:
            best_val = val_metrics["loss"]
            epochs_without_improve = 0
            torch.save(
                {
                    "phase": phase_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            logger.info("%s converged (early stopping after %d stagnant epochs)", phase_name, patience)
            break

    return history, best_val


def load_audio_model(checkpoint_path: Path) -> MelCPCAudioModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = TrainingConfig()
    model = MelCPCAudioModel(cfg)

    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Audio checkpoint missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Audio checkpoint unexpected keys: %d", len(unexpected))

    return model


def load_video_model(checkpoint_path: Path) -> VideoOrderNet:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = VideoOrderNet(pretrained=False)

    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Video checkpoint missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Video checkpoint unexpected keys: %d", len(unexpected))

    return model


def save_history(rows: List[Dict[str, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train aligned audio-video token spaces with a 3-layer transformer")
    parser.add_argument("--audio-data-dir", type=str, default="data/ravdess_mels_0.5s")
    parser.add_argument("--video-data-dir", type=str, default="lip_crop_results_full")
    parser.add_argument("--audio-checkpoint", type=str, default="ravdess_mel_cpc_results/best_model.pt")
    parser.add_argument("--video-checkpoint", type=str, default="video_frame_order_results/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="aligned_results")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-joint", type=float, default=1e-4)
    parser.add_argument("--lr-transformer", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--temporal-smoothing-sigma", type=float, default=0.5)

    parser.add_argument("--joint-epochs", type=int, default=30)
    parser.add_argument("--transformer-only-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    audio_data_dir = Path(args.audio_data_dir)
    video_data_dir = Path(args.video_data_dir)
    audio_ckpt = Path(args.audio_checkpoint)
    video_ckpt = Path(args.video_checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_audio_video_chunks(audio_data_dir, video_data_dir)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, seed=args.seed)

    if not train_pairs or not val_pairs or not test_pairs:
        raise RuntimeError(
            f"Invalid split sizes: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}"
        )

    train_ds = PairedRavdessChunkDataset(train_pairs, chunk_seconds=args.chunk_seconds)
    val_ds = PairedRavdessChunkDataset(val_pairs, chunk_seconds=args.chunk_seconds)
    test_ds = PairedRavdessChunkDataset(test_pairs, chunk_seconds=args.chunk_seconds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        "Split sizes | train=%d val=%d test=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    audio_model = load_audio_model(audio_ckpt)
    video_model = load_video_model(video_ckpt)

    alignment_model = AudioVideoAlignmentModel(
        audio_model=audio_model,
        video_token_encoder=VideoTokenEncoder(video_model),
        upsampler=TemporalUpsampler(dim=512, in_tokens=11, out_tokens=25),
        transformer=VideoAlignmentTransformer(dim=512, num_layers=3, num_heads=8, dropout=0.1),
    ).to(device)

    criterion = TokenContrastiveAlignmentLoss(
        temperature=args.temperature,
        temporal_smoothing_sigma=args.temporal_smoothing_sigma,
    )

    if args.dry_run:
        sample = next(iter(train_loader))
        mel_slices = sample["mel_slices"].to(device)
        video_frames = sample["video_frames"].to(device)
        with torch.no_grad():
            audio_tokens, video_tokens = alignment_model(mel_slices, video_frames)
        logger.info("Dry run mel_slices: %s", tuple(mel_slices.shape))
        logger.info("Dry run video_frames: %s", tuple(video_frames.shape))
        logger.info("Dry run audio_tokens: %s", tuple(audio_tokens.shape))
        logger.info("Dry run video_tokens: %s", tuple(video_tokens.shape))
        return

    history_rows: List[Dict[str, float]] = []

    # Phase 1: jointly train both encoders + upsampler + transformer.
    set_requires_grad(alignment_model.audio_model, True)
    set_requires_grad(alignment_model.video_token_encoder, True)
    set_requires_grad(alignment_model.upsampler, True)
    set_requires_grad(alignment_model.transformer, True)

    optimizer_joint = build_optimizer(alignment_model, lr=args.lr_joint, weight_decay=args.weight_decay)

    phase1_ckpt = out_dir / "phase1_best.pt"
    phase1_hist, phase1_best = train_until_converged(
        model=alignment_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_joint,
        device=device,
        max_epochs=args.joint_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        phase_name="joint",
        checkpoint_path=phase1_ckpt,
        trainable_modules=None,
    )
    history_rows.extend(phase1_hist)

    if phase1_ckpt.exists():
        ckpt = torch.load(phase1_ckpt, map_location=device)
        alignment_model.load_state_dict(ckpt["model_state_dict"])

    # Phase 2: freeze encoders and upsampler, train transformer only.
    set_requires_grad(alignment_model.audio_model, False)
    set_requires_grad(alignment_model.video_token_encoder, False)
    set_requires_grad(alignment_model.upsampler, False)
    set_requires_grad(alignment_model.transformer, True)

    optimizer_transformer = build_optimizer(alignment_model, lr=args.lr_transformer, weight_decay=args.weight_decay)

    phase2_ckpt = out_dir / "phase2_best.pt"
    phase2_hist, phase2_best = train_until_converged(
        model=alignment_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_transformer,
        device=device,
        max_epochs=args.transformer_only_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        phase_name="transformer_only",
        checkpoint_path=phase2_ckpt,
        trainable_modules=[alignment_model.transformer],
    )
    history_rows.extend(phase2_hist)

    if phase2_ckpt.exists():
        ckpt = torch.load(phase2_ckpt, map_location=device)
        alignment_model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = run_epoch(
        model=alignment_model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        desc="Test",
    )

    save_history(history_rows, out_dir / "alignment_history.csv")

    torch.save(
        {
            "model_state_dict": alignment_model.state_dict(),
            "phase1_best_val_loss": phase1_best,
            "phase2_best_val_loss": phase2_best,
            "test_loss": test_metrics["loss"],
            "args": vars(args),
        },
        out_dir / "alignment_model_final.pt",
    )

    with (out_dir / "test_metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"test_loss={test_metrics['loss']:.6f}\n")

    logger.info("Done.")
    logger.info("Best val loss (joint): %.4f", phase1_best)
    logger.info("Best val loss (transformer_only): %.4f", phase2_best)
    logger.info("Test loss: %.4f", test_metrics["loss"])


if __name__ == "__main__":
    main()
