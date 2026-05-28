"""Train a video-conditioned diffusion model to generate mel spectrogram chunks.

Goal:
    Learn p(mel | video_tokens) where:
      - mel target is a 0.5s chunk shaped [128, 50]
      - conditioning tokens come from lip-centered video clips

Conditioning pipeline:
    video clip chunk (0.5s, 15 sampled frames)
        -> pretrained frame-order encoder (11 temporal windows)
        -> temporal upsampler (11 -> 25 tokens)
        -> 3-layer transformer
        -> video tokens [25, 512]

Diffusion model:
    DDPM-style epsilon prediction on mel "images" [1, 128, 50],
    conditioned via cross-attention over the full video token sequence.

Usage:
    python train_video_conditioned_mel_diffusion.py --dry-run

    python train_video_conditioned_mel_diffusion.py \
        --audio-data-dir data/ravdess_mels_0.5s \
        --video-data-dir lip_crop_results_full \
        --video-checkpoint aligned_results/stage2_best.pt \
        --alignment-checkpoint aligned_results/stage2_best.pt \
        --epochs 40 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
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

from train_audio_video_alignment import TemporalUpsampler, VideoAlignmentTransformer, VideoTokenEncoder
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


class PairedMelVideoChunkDataset(Dataset):
    """Return normalized mel targets and video frame chunks for paired samples."""

    def __init__(
        self,
        pairs: Sequence[PairSample],
        chunk_seconds: float = 0.5,
        video_frames_per_chunk: int = 15,
    ):
        if not pairs:
            raise ValueError("Paired dataset is empty")

        self.pairs = list(pairs)
        self.chunk_seconds = float(chunk_seconds)
        self.video_frames_per_chunk = int(video_frames_per_chunk)

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

    def _read_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        if not ok:
            return None
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _load_video_chunk(self, video_path: Path, chunk_idx: int) -> torch.Tensor:
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
            num=self.video_frames_per_chunk,
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

        while len(frames) < self.video_frames_per_chunk:
            frames.append(frames[-1].clone())

        frames = frames[: self.video_frames_per_chunk]
        return torch.stack(frames, dim=0)  # [15, 3, 112, 112]

    @staticmethod
    def _normalize_mel(mel: np.ndarray) -> np.ndarray:
        # Per-chunk z-score then scale to roughly [-1, 1] for diffusion stability.
        mean = float(mel.mean())
        std = float(mel.std())
        mel = (mel - mean) / (std + 1e-6)
        mel = np.clip(mel, -4.0, 4.0) / 4.0
        return mel.astype(np.float32)

    def _load_mel(self, audio_path: Path) -> torch.Tensor:
        with np.load(audio_path) as data:
            mel = data["mel"].astype(np.float32)

        if mel.shape != (128, 50):
            raise ValueError(f"Expected mel shape (128, 50), got {mel.shape} for {audio_path}")

        mel = self._normalize_mel(mel)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # [1, 128, 50]
        return mel_tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.pairs[index]
        mel = self._load_mel(sample.audio_path)
        video_frames = self._load_video_chunk(sample.video_path, sample.chunk_idx)

        return {
            "mel": mel,
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
    # Split by identity stem to prevent leakage across splits.
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


class VideoConditionEncoder(nn.Module):
    """Extract [B, 25, 512] conditioning tokens from video frames."""

    def __init__(self, video_order_net: VideoOrderNet):
        super().__init__()
        self.video_token_encoder = VideoTokenEncoder(video_order_net=video_order_net, window_size=5)
        self.upsampler = TemporalUpsampler(dim=512, in_tokens=11, out_tokens=25)
        self.transformer = VideoAlignmentTransformer(dim=512, num_layers=3, num_heads=8, dropout=0.1)

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        tokens_11 = self.video_token_encoder(video_frames)  # [B, 11, 512]
        tokens_25 = self.upsampler(tokens_11)  # [B, 25, 512]
        tokens_25 = self.transformer(tokens_25)  # [B, 25, 512]
        return tokens_25


def load_video_order_model(checkpoint_path: Path) -> VideoOrderNet:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = VideoOrderNet(pretrained=False)

    state = ckpt.get("model_state_dict", ckpt)

    # Support both formats:
    # 1) Frame-order checkpoint keys like "encoder.*" / "classifier.*"
    # 2) Alignment checkpoint keys like "video_token_encoder.encoder.*"
    if any(k.startswith("video_token_encoder.encoder.") for k in state.keys()):
        remapped_state: Dict[str, torch.Tensor] = {}
        prefix = "video_token_encoder.encoder."
        for key, value in state.items():
            if key.startswith(prefix):
                remapped_state["encoder." + key[len(prefix) :]] = value
        state = remapped_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Video checkpoint missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Video checkpoint unexpected keys: %d", len(unexpected))

    return model


def maybe_load_alignment_weights(condition_encoder: VideoConditionEncoder, checkpoint_path: Optional[Path]) -> None:
    if checkpoint_path is None:
        return
    if not checkpoint_path.exists():
        logger.warning("Alignment checkpoint not found, skipping load: %s", checkpoint_path)
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    allowed_prefixes = (
        "video_token_encoder.",
        "upsampler.",
        "transformer.",
    )
    filtered = {k: v for k, v in state.items() if k.startswith(allowed_prefixes)}
    if not filtered:
        logger.warning("No video/alignment keys found in checkpoint: %s", checkpoint_path)
        return

    missing, unexpected = condition_encoder.load_state_dict(filtered, strict=False)
    if missing:
        logger.info("Condition encoder missing keys after partial load: %d", len(missing))
    if unexpected:
        logger.info("Condition encoder unexpected keys after partial load: %d", len(unexpected))

    logger.info("Loaded conditioning weights from: %s (keys=%d)", checkpoint_path, len(filtered))


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half_dim = dim // 2
    freq_const = math.log(10000.0) / max(half_dim - 1, 1)
    freqs = torch.exp(torch.arange(half_dim, device=device) * -freq_const)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    group = min(preferred, channels)
    while group > 1 and channels % group != 0:
        group -= 1
    return max(group, 1)


class FiLMResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(_valid_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_channels * 2)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale_shift = self.emb_proj(F.silu(emb))
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(h))

        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Cross-attention from mel feature queries to video token keys/values."""

    def __init__(self, dim: int, context_dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        inner_dim = self.heads * self.dim_head

        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C], context: [B, T, D]
        bsz, n_tokens, _ = x.shape

        q = self.to_q(self.norm_q(x))
        k = self.to_k(self.norm_ctx(context))
        v = self.to_v(self.norm_ctx(context))

        q = q.view(bsz, n_tokens, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(bsz, context.shape[1], self.heads, self.dim_head).transpose(1, 2)
        v = v.view(bsz, context.shape[1], self.heads, self.dim_head).transpose(1, 2)

        scale = self.dim_head ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, self.heads * self.dim_head)
        return self.to_out(out)


class MelConditionedUNet(nn.Module):
    """Small conditioned U-Net for noise prediction epsilon_theta(x_t, t, cond)."""

    def __init__(self, cond_dim: int = 512, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_dim = int(time_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.in_conv = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        self.down1 = FiLMResBlock(base_channels, base_channels, self.time_dim)
        self.down2 = FiLMResBlock(base_channels, base_channels * 2, self.time_dim)

        self.mid = FiLMResBlock(base_channels * 2, base_channels * 2, self.time_dim)
        self.mid_cross_attn = CrossAttention(dim=base_channels * 2, context_dim=cond_dim, heads=4, dim_head=32)

        self.up2 = FiLMResBlock(base_channels * 4, base_channels, self.time_dim)
        self.up1 = FiLMResBlock(base_channels * 2, base_channels, self.time_dim)

        self.out_norm = nn.GroupNorm(_valid_group_count(base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        emb = t_emb

        x0 = self.in_conv(x_t)
        d1 = self.down1(x0, emb)

        x = F.avg_pool2d(d1, kernel_size=2, stride=2, ceil_mode=True)
        d2 = self.down2(x, emb)

        x = F.avg_pool2d(d2, kernel_size=2, stride=2, ceil_mode=True)
        x = self.mid(x, emb)

        # Inject full temporal conditioning sequence at the bottleneck.
        bsz, channels, height, width = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_seq = x_seq + self.mid_cross_attn(x_seq, cond_tokens)
        x = x_seq.transpose(1, 2).reshape(bsz, channels, height, width)

        x = F.interpolate(x, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, d2], dim=1)
        x = self.up2(x, emb)

        x = F.interpolate(x, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, d1], dim=1)
        x = self.up1(x, emb)

        return self.out_conv(F.silu(self.out_norm(x)))


class GaussianDiffusion(nn.Module):
    """DDPM scheduler with fixed linear betas and epsilon prediction objective."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.timesteps = int(timesteps)

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)

        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))

    def _extract(self, values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bar, t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bar, t, x_start.shape)
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    def p_losses(self, model: MelConditionedUNet, x_start: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = model(x_t, t, cond_tokens)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model: MelConditionedUNet, cond_tokens: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        device = cond_tokens.device
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)

            beta_t = self._extract(self.betas, t, x.shape)
            sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bar, t, x.shape)
            sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

            pred_noise = model(x, t, cond_tokens)
            model_mean = sqrt_recip_alphas_t * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * pred_noise)

            if i == 0:
                x = model_mean
            else:
                posterior_var_t = self._extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_var_t) * noise

        return x.clamp(-1.0, 1.0)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def build_optimizer(
    diffusion_model: MelConditionedUNet,
    condition_encoder: VideoConditionEncoder,
    lr: float,
    weight_decay: float,
    finetune_conditioner: bool,
) -> torch.optim.Optimizer:
    params: List[torch.nn.Parameter] = []
    params.extend([p for p in diffusion_model.parameters() if p.requires_grad])
    if finetune_conditioner:
        params.extend([p for p in condition_encoder.parameters() if p.requires_grad])

    if not params:
        raise RuntimeError("No trainable parameters found")

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_epoch(
    diffusion_model: MelConditionedUNet,
    condition_encoder: VideoConditionEncoder,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    desc: str,
    amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    finetune_conditioner: bool,
) -> Dict[str, float]:
    is_train = optimizer is not None
    diffusion_model.train(is_train)

    if finetune_conditioner and is_train:
        condition_encoder.train(True)
    else:
        condition_encoder.eval()

    total_loss = 0.0
    total_steps = 0

    autocast_enabled = amp and device.type == "cuda"
    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        mel = batch["mel"].to(device)  # [B, 1, 128, 50]
        video_frames = batch["video_frames"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if finetune_conditioner and is_train:
                cond_tokens = condition_encoder(video_frames)
            else:
                with torch.no_grad():
                    cond_tokens = condition_encoder(video_frames)

            t = torch.randint(0, diffusion.timesteps, (mel.shape[0],), device=device, dtype=torch.long)

            with torch.autocast(device_type="cuda", enabled=autocast_enabled):
                loss = diffusion.p_losses(diffusion_model, mel, t, cond_tokens)

            if not torch.isfinite(loss):
                logger.warning("Skipping non-finite batch loss")
                continue

            if is_train:
                if autocast_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in diffusion_model.parameters() if p.requires_grad]
                        + ([p for p in condition_encoder.parameters() if p.requires_grad] if finetune_conditioner else []),
                        max_norm=1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in diffusion_model.parameters() if p.requires_grad]
                        + ([p for p in condition_encoder.parameters() if p.requires_grad] if finetune_conditioner else []),
                        max_norm=1.0,
                    )
                    optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1
        pbar.set_postfix(loss=f"{total_loss / max(total_steps, 1):.5f}")

    if total_steps == 0:
        return {"loss": float("inf")}
    return {"loss": total_loss / total_steps}


def save_history(rows: List[Dict[str, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train video-conditioned mel diffusion model")

    parser.add_argument("--audio-data-dir", type=str, default="data/ravdess_mels_0.5s")
    parser.add_argument("--video-data-dir", type=str, default="lip_crop_results_full")

    parser.add_argument("--video-checkpoint", type=str, default="aligned_results/stage2_best.pt")
    parser.add_argument("--alignment-checkpoint", type=str, default="aligned_results/stage2_best.pt")

    parser.add_argument("--output-dir", type=str, default="video_conditioned_mel_diffusion_results")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--finetune-conditioner", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    audio_data_dir = Path(args.audio_data_dir)
    video_data_dir = Path(args.video_data_dir)
    video_ckpt = Path(args.video_checkpoint)
    alignment_ckpt = Path(args.alignment_checkpoint) if args.alignment_checkpoint else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_audio_video_chunks(audio_data_dir, video_data_dir)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, seed=args.seed)

    if not train_pairs or not val_pairs or not test_pairs:
        raise RuntimeError(
            f"Invalid split sizes: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}"
        )

    train_ds = PairedMelVideoChunkDataset(train_pairs, chunk_seconds=args.chunk_seconds)
    val_ds = PairedMelVideoChunkDataset(val_pairs, chunk_seconds=args.chunk_seconds)
    test_ds = PairedMelVideoChunkDataset(test_pairs, chunk_seconds=args.chunk_seconds)

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

    logger.info("Split sizes | train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    video_order_model = load_video_order_model(video_ckpt)
    condition_encoder = VideoConditionEncoder(video_order_model)
    maybe_load_alignment_weights(condition_encoder, alignment_ckpt)
    condition_encoder = condition_encoder.to(device)

    diffusion_model = MelConditionedUNet(cond_dim=512, base_channels=64, time_dim=256).to(device)
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device)

    if not args.finetune_conditioner:
        set_requires_grad(condition_encoder, False)

    optimizer = build_optimizer(
        diffusion_model=diffusion_model,
        condition_encoder=condition_encoder,
        lr=args.lr,
        weight_decay=args.weight_decay,
        finetune_conditioner=args.finetune_conditioner,
    )

    amp_enabled = (not args.no_amp) and (device.type == "cuda")
    scaler = create_grad_scaler(enabled=amp_enabled)

    if args.dry_run:
        batch = next(iter(train_loader))
        mel = batch["mel"].to(device)
        video_frames = batch["video_frames"].to(device)

        with torch.no_grad():
            cond_tokens = condition_encoder(video_frames)
            t = torch.randint(0, diffusion.timesteps, (mel.shape[0],), device=device, dtype=torch.long)
            x_t = diffusion.q_sample(mel, t)
            pred_noise = diffusion_model(x_t, t, cond_tokens)

        logger.info("Dry run mel: %s", tuple(mel.shape))
        logger.info("Dry run video_frames: %s", tuple(video_frames.shape))
        logger.info("Dry run cond_tokens: %s", tuple(cond_tokens.shape))
        logger.info("Dry run x_t: %s", tuple(x_t.shape))
        logger.info("Dry run pred_noise: %s", tuple(pred_noise.shape))
        return

    history_rows: List[Dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            diffusion_model=diffusion_model,
            condition_encoder=condition_encoder,
            diffusion=diffusion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            desc=f"Train {epoch}/{args.epochs}",
            amp=amp_enabled,
            scaler=scaler,
            finetune_conditioner=args.finetune_conditioner,
        )

        val_metrics = run_epoch(
            diffusion_model=diffusion_model,
            condition_encoder=condition_encoder,
            diffusion=diffusion,
            loader=val_loader,
            optimizer=None,
            device=device,
            desc=f"Val {epoch}/{args.epochs}",
            amp=amp_enabled,
            scaler=None,
            finetune_conditioner=args.finetune_conditioner,
        )

        logger.info(
            "Epoch %03d/%03d | train_loss=%.5f | val_loss=%.5f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "diffusion_model_state_dict": diffusion_model.state_dict(),
                    "condition_encoder_state_dict": condition_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                output_dir / "best_model.pt",
            )

        torch.save(
            {
                "epoch": epoch,
                "diffusion_model_state_dict": diffusion_model.state_dict(),
                "condition_encoder_state_dict": condition_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
            },
            output_dir / "last_model.pt",
        )

    # Load best checkpoint for test-time evaluation if present.
    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        diffusion_model.load_state_dict(ckpt["diffusion_model_state_dict"])
        condition_encoder.load_state_dict(ckpt["condition_encoder_state_dict"])

    test_metrics = run_epoch(
        diffusion_model=diffusion_model,
        condition_encoder=condition_encoder,
        diffusion=diffusion,
        loader=test_loader,
        optimizer=None,
        device=device,
        desc="Test",
        amp=amp_enabled,
        scaler=None,
        finetune_conditioner=args.finetune_conditioner,
    )

    save_history(history_rows, output_dir / "training_history.csv")

    # Save a small qualitative sample batch.
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        sample_video = sample_batch["video_frames"].to(device)[: min(4, sample_batch["video_frames"].shape[0])]
        sample_cond = condition_encoder(sample_video)
        generated = diffusion.sample(
            model=diffusion_model,
            cond_tokens=sample_cond,
            shape=(sample_video.shape[0], 1, 128, 50),
        )

    np.save(output_dir / "sample_generated_mels.npy", generated.cpu().numpy())

    with (output_dir / "test_metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"test_loss={test_metrics['loss']:.6f}\n")

    logger.info("Training complete")
    logger.info("Best val loss: %.5f", best_val_loss)
    logger.info("Test loss: %.5f", test_metrics["loss"])


if __name__ == "__main__":
    main()
