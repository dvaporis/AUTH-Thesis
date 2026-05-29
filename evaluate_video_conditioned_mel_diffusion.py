"""Evaluate a video-conditioned mel diffusion model with a pretrained neural vocoder.

This script:
1. Loads the saved diffusion checkpoint from `video_conditioned_mel_diffusion_results/`.
2. Samples a small set of paired 0.5s video clips.
3. Generates mel spectrogram chunks conditioned on the video frames.
4. Compares generated mels against the corresponding target mels.
5. Saves the generated mels and compares them against the target mels.

Usage:
    python evaluate_video_conditioned_mel_diffusion.py --dry-run

    python evaluate_video_conditioned_mel_diffusion.py \
        --diffusion-checkpoint video_conditioned_mel_diffusion_results/best_model.pt \
        --video-checkpoint aligned_results/stage2_best.pt \
        --alignment-checkpoint aligned_results/stage2_best.pt \
        --num-samples 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_video_conditioned_mel_diffusion import (
    GaussianDiffusion,
    MelConditionedUNet,
    PairedMelVideoChunkDataset,
    VideoConditionEncoder,
    load_video_order_model,
    pair_audio_video_chunks,
    split_pairs,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def normalize_target_mel(mel_db: np.ndarray) -> np.ndarray:
    mean = float(mel_db.mean())
    std = float(mel_db.std())
    mel = (mel_db - mean) / (std + 1e-6)
    mel = np.clip(mel, -4.0, 4.0) / 4.0
    return mel.astype(np.float32)


def denormalize_generated_mel(generated_norm: np.ndarray, reference_mel_db: np.ndarray) -> np.ndarray:
    ref_mean = float(reference_mel_db.mean())
    ref_std = float(reference_mel_db.std())
    return (generated_norm * 4.0 * (ref_std + 1e-6) + ref_mean).astype(np.float32)


def mel_db_to_log_amplitude(mel_db: np.ndarray) -> torch.Tensor:
    mel = torch.from_numpy(np.asarray(mel_db, dtype=np.float32))
    return mel * (math.log(10.0) / 20.0)


def compute_mel_metrics(reference_norm: np.ndarray, candidate_norm: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    diff = reference_norm - candidate_norm
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))

    ref_std = float(np.std(reference_norm))
    cand_std = float(np.std(candidate_norm))
    if ref_std <= eps or cand_std <= eps:
        corr = 0.0
    else:
        corr = float(np.corrcoef(reference_norm.reshape(-1), candidate_norm.reshape(-1))[0, 1])

    return {"mse": mse, "mae": mae, "corr": corr}


def load_diffusion_bundle(
    diffusion_checkpoint: Path,
    video_checkpoint: Path,
    device: torch.device,
    timesteps_override: Optional[int] = None,
    beta_start_override: Optional[float] = None,
    beta_end_override: Optional[float] = None,
) -> Tuple[MelConditionedUNet, VideoConditionEncoder, GaussianDiffusion]:
    if not diffusion_checkpoint.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_checkpoint}")

    ckpt = torch.load(diffusion_checkpoint, map_location="cpu")
    diffusion_state = ckpt.get("diffusion_model_state_dict")
    condition_state = ckpt.get("condition_encoder_state_dict")
    if diffusion_state is None or condition_state is None:
        raise RuntimeError(f"Checkpoint does not contain expected model states: {diffusion_checkpoint}")

    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    timesteps = int(timesteps_override or saved_args.get("timesteps", 1000))
    beta_start = float(beta_start_override or saved_args.get("beta_start", 1e-4))
    beta_end = float(beta_end_override or saved_args.get("beta_end", 2e-2))

    video_order_model = load_video_order_model(video_checkpoint)
    condition_encoder = VideoConditionEncoder(video_order_model)
    missing, unexpected = condition_encoder.load_state_dict(condition_state, strict=False)
    if missing:
        logger.info("Condition encoder missing keys after load: %d", len(missing))
    if unexpected:
        logger.info("Condition encoder unexpected keys after load: %d", len(unexpected))
    condition_encoder = condition_encoder.to(device).eval()

    diffusion_model = MelConditionedUNet(cond_dim=512, base_channels=64, time_dim=256)
    missing, unexpected = diffusion_model.load_state_dict(diffusion_state, strict=False)
    if missing:
        logger.info("Diffusion model missing keys after load: %d", len(missing))
    if unexpected:
        logger.info("Diffusion model unexpected keys after load: %d", len(unexpected))
    diffusion_model = diffusion_model.to(device).eval()

    diffusion = GaussianDiffusion(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end).to(device)
    logger.info("Loaded diffusion model with timesteps=%d", timesteps)
    return diffusion_model, condition_encoder, diffusion


def plot_comparison(
    actual_mel_db: np.ndarray,
    generated_mel_db: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    mel_vmin = -80.0
    mel_vmax = 0.0

    im0 = axes[0].imshow(actual_mel_db, aspect="auto", origin="lower", cmap="magma", vmin=mel_vmin, vmax=mel_vmax)
    axes[0].set_title("Actual mel")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mel bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(generated_mel_db, aspect="auto", origin="lower", cmap="magma", vmin=mel_vmin, vmax=mel_vmax)
    axes[1].set_title("Generated mel")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Mel bin")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_samples(pairs: Sequence, num_samples: int, seed: int) -> List:
    if num_samples <= 0:
        return []
    if num_samples >= len(pairs):
        return list(pairs)

    rng = random.Random(seed)
    return rng.sample(list(pairs), k=num_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate video-conditioned mel diffusion with a neural vocoder")
    parser.add_argument("--audio-data-dir", type=str, default="data/ravdess_mels_0.5s")
    parser.add_argument("--video-data-dir", type=str, default="lip_crop_results_full")
    parser.add_argument("--diffusion-checkpoint", type=str, default="video_conditioned_mel_diffusion_results/best_model.pt")
    parser.add_argument("--video-checkpoint", type=str, default="aligned_results/stage2_best.pt")
    parser.add_argument("--alignment-checkpoint", type=str, default="aligned_results/stage2_best.pt")
    parser.add_argument("--output-dir", type=str, default="video_conditioned_mel_diffusion_eval")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--beta-start", type=float, default=None)
    parser.add_argument("--beta-end", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    audio_data_dir = Path(args.audio_data_dir)
    video_data_dir = Path(args.video_data_dir)
    diffusion_checkpoint = Path(args.diffusion_checkpoint)
    video_checkpoint = Path(args.video_checkpoint)
    alignment_checkpoint = Path(args.alignment_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_audio_video_chunks(audio_data_dir, video_data_dir)
    _, _, test_pairs = split_pairs(pairs, seed=args.split_seed)
    if not test_pairs:
        raise RuntimeError("No test pairs available after split")

    selected_pairs = select_samples(test_pairs, num_samples=args.num_samples, seed=args.seed)
    logger.info("Selected %d samples from test split of %d clips", len(selected_pairs), len(test_pairs))

    diffusion_model, condition_encoder, diffusion = load_diffusion_bundle(
        diffusion_checkpoint=diffusion_checkpoint,
        video_checkpoint=video_checkpoint,
        device=device,
        timesteps_override=args.timesteps,
        beta_start_override=args.beta_start,
        beta_end_override=args.beta_end,
    )

    if args.dry_run:
        sample = selected_pairs[0]
        logger.info("Dry run sample: %s chunk=%d", sample.clip_stem, sample.chunk_idx)
        return

    rows: List[Dict[str, float | str | int]] = []

    for index, sample in enumerate(selected_pairs, start=1):
        logger.info("Processing %d/%d: %s chunk=%d", index, len(selected_pairs), sample.clip_stem, sample.chunk_idx)

        with np.load(sample.audio_path) as data:
            actual_mel_db = data["mel"].astype(np.float32)

        actual_norm = normalize_target_mel(actual_mel_db)
        target_mel = torch.from_numpy(actual_norm).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            dataset = PairedMelVideoChunkDataset([sample])
            item = dataset[0]
            video_frames = item["video_frames"].unsqueeze(0).to(device)
            cond_tokens = condition_encoder(video_frames)
            t = torch.zeros((1,), device=device, dtype=torch.long)
            x_t = diffusion.q_sample(target_mel, t)
            _ = diffusion_model(x_t, t, cond_tokens)
            generated_norm = diffusion.sample(
                model=diffusion_model,
                cond_tokens=cond_tokens,
                shape=(1, 1, 128, 50),
            ).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        mel_metrics = compute_mel_metrics(actual_norm, generated_norm)

        generated_mel_db = denormalize_generated_mel(generated_norm, actual_mel_db)

        sample_prefix = f"{sample.clip_stem}__chunk{sample.chunk_idx:04d}"
        np.save(output_dir / f"{sample_prefix}_actual_mel_db.npy", actual_mel_db)
        np.save(output_dir / f"{sample_prefix}_generated_mel_db.npy", generated_mel_db)

        fig_path = output_dir / f"{sample_prefix}_comparison.png"
        plot_comparison(
            actual_mel_db=actual_mel_db,
            generated_mel_db=generated_mel_db,
            title=f"{sample.clip_stem} chunk {sample.chunk_idx:04d}",
            output_path=fig_path,
        )

        rows.append(
            {
                "clip_stem": sample.clip_stem,
                "chunk_idx": sample.chunk_idx,
                "audio_path": str(sample.audio_path),
                "video_path": str(sample.video_path),
                "mel_mse": mel_metrics["mse"],
                "mel_mae": mel_metrics["mae"],
                "mel_corr": mel_metrics["corr"],
            }
        )

        logger.info(
            "%s chunk=%04d | mel_corr=%.4f | figure=%s",
            sample.clip_stem,
            sample.chunk_idx,
            mel_metrics["corr"],
            fig_path.name,
        )

    summary_csv = output_dir / "evaluation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_stem",
                "chunk_idx",
                "audio_path",
                "video_path",
                "mel_mse",
                "mel_mae",
                "mel_corr",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_txt = output_dir / "evaluation_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as handle:
        handle.write(f"Diffusion checkpoint: {diffusion_checkpoint}\n")
        handle.write(f"Video checkpoint: {video_checkpoint}\n")
        handle.write(f"Alignment checkpoint: {alignment_checkpoint}\n")
        handle.write(f"Samples: {len(rows)}\n\n")
        for row in rows:
            handle.write(
                f"{row['clip_stem']} chunk {int(row['chunk_idx']):04d} | "
                f"mel_corr={float(row['mel_corr']):.4f}\n"
            )

    logger.info("Saved results to %s", output_dir)
    logger.info("Summary CSV: %s", summary_csv)
    logger.info("Summary TXT: %s", summary_txt)


if __name__ == "__main__":
    main()