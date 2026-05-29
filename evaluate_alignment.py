"""Evaluate trained audio-video alignment model.

Produces: top-1/top-5 accuracy, per-time confusion matrix, and cosine similarity stats.

Usage:
    python evaluate_alignment.py --model aligned_results/alignment_model_final.pt --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from train_audio_video_alignment import (
    PairedRavdessChunkDataset,
    pair_audio_video_chunks,
    load_audio_model,
    load_video_model,
    AudioVideoAlignmentModel,
    VideoTokenEncoder,
    TemporalUpsampler,
    VideoAlignmentTransformer,
    ProjectionHead,
)


def build_alignment_model(audio_ckpt: Path, video_ckpt: Path, device: torch.device) -> AudioVideoAlignmentModel:
    audio_model = load_audio_model(audio_ckpt)
    video_model = load_video_model(video_ckpt)

    model = AudioVideoAlignmentModel(
        audio_model=audio_model,
        video_token_encoder=VideoTokenEncoder(video_model),
        upsampler=TemporalUpsampler(dim=512, in_tokens=11, out_tokens=25),
        transformer=VideoAlignmentTransformer(dim=512, num_layers=3, num_heads=8, dropout=0.1),
        audio_projector=ProjectionHead(dim=512, out_dim=256),
        video_projector=ProjectionHead(dim=512, out_dim=256),
    )
    return model.to(device)


def evaluate_batch(model: AudioVideoAlignmentModel, mel_slices: torch.Tensor, video_frames: torch.Tensor) -> Tuple[int, int, int, np.ndarray, float, float]:
    # returns: correct, total, top5_correct, confusion(T,T), mean_true_sim, mean_max_neg_sim
    device = next(model.parameters()).device
    mel_slices = mel_slices.to(device)
    video_frames = video_frames.to(device)

    with torch.no_grad():
        audio_tokens, video_tokens = model(mel_slices, video_frames)  # [B, T, D]

    bsz, T, D = audio_tokens.shape

    audio_flat = torch.nn.functional.normalize(audio_tokens.reshape(bsz * T, D), dim=1)
    video_flat = torch.nn.functional.normalize(video_tokens.reshape(bsz * T, D), dim=1)

    sim = torch.matmul(video_flat, audio_flat.T)  # [N, N] where N = bsz*T

    N = sim.shape[0]
    device = sim.device
    idx = torch.arange(N, device=device)

    # Top-1 predictions
    preds = torch.argmax(sim, dim=1)
    correct = int((preds == idx).sum().item())

    # Top-5
    topk = 5
    topk_vals, topk_idx = torch.topk(sim, topk, dim=1)
    top5_correct = int(((topk_idx == idx.unsqueeze(1)).any(dim=1)).sum().item())

    # Confusion matrix by time index (T x T)
    conf = np.zeros((T, T), dtype=np.int64)
    preds_cpu = preds.cpu().numpy()
    idx_cpu = idx.cpu().numpy()
    for p, g in zip(preds_cpu, idx_cpu):
        gt_time = int(g % T)
        pred_time = int(p % T)
        conf[gt_time, pred_time] += 1

    # Similarity stats: true diagonal similarities and strongest negative per-row
    sims_diag = sim.diag()
    sim_masked = sim.clone()
    sim_masked[idx, idx] = float("-inf")
    max_neg, _ = sim_masked.max(dim=1)

    mean_true = float(sims_diag.mean().item())
    mean_max_neg = float(max_neg.mean().item())

    return correct, N, top5_correct, conf, mean_true, mean_max_neg


def save_confusion(conf: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gt_time"] + [f"pred_{i}" for i in range(conf.shape[1])])
        for i, row in enumerate(conf.tolist()):
            writer.writerow([i] + row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-data-dir", type=str, default="data/ravdess_mels_0.5s")
    parser.add_argument("--video-data-dir", type=str, default="lip_crop_results_full")
    parser.add_argument("--audio-checkpoint", type=str, default="ravdess_mel_cpc_results/best_model.pt")
    parser.add_argument("--video-checkpoint", type=str, default="video_frame_order_results/best_model.pt")
    parser.add_argument("--model", type=str, default="aligned_results/alignment_model_final.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=2, help="Stop after this many evaluation batches; use 0 or a negative value for the full dataset")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="aligned_results/eval")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    audio_data_dir = Path(args.audio_data_dir)
    video_data_dir = Path(args.video_data_dir)
    audio_ckpt = Path(args.audio_checkpoint)
    video_ckpt = Path(args.video_checkpoint)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_audio_video_chunks(audio_data_dir, video_data_dir)
    eval_ds = PairedRavdessChunkDataset(pairs)
    loader = torch.utils.data.DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # build model and load checkpoint
    model = build_alignment_model(audio_ckpt, video_ckpt, device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    total_correct = 0
    total_N = 0
    total_top5 = 0
    T = None
    agg_conf = None
    true_sims = []
    max_neg_sims = []

    for batch_idx, batch in enumerate(loader):
        if args.max_batches and args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        mel_slices = batch["mel_slices"]
        video_frames = batch["video_frames"]

        correct, N, top5_correct, conf, mean_true, mean_max_neg = evaluate_batch(model, mel_slices, video_frames)

        if agg_conf is None:
            T = conf.shape[0]
            agg_conf = np.zeros_like(conf)

        total_correct += correct
        total_N += N
        total_top5 += top5_correct
        agg_conf += conf
        true_sims.append(mean_true)
        max_neg_sims.append(mean_max_neg)

    acc = total_correct / total_N if total_N else 0.0
    top5 = total_top5 / total_N if total_N else 0.0

    mean_true_overall = float(np.mean(true_sims)) if true_sims else 0.0
    mean_max_neg_overall = float(np.mean(max_neg_sims)) if max_neg_sims else 0.0
    evaluated_batches = len(true_sims)

    print(f"Evaluated tokens: {total_N}")
    print(f"Evaluated batches: {evaluated_batches}")
    print(f"Top-1 accuracy: {acc:.6f}")
    print(f"Top-5 accuracy: {top5:.6f}")
    print(f"Mean true cosine similarity: {mean_true_overall:.6f}")
    print(f"Mean max negative cosine similarity: {mean_max_neg_overall:.6f}")

    save_confusion(agg_conf, out_dir / "confusion_by_time.csv")
    np.save(out_dir / "confusion_by_time.npy", agg_conf)

    # also save a small summary CSV
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["num_tokens", total_N])
        writer.writerow(["top1_acc", acc])
        writer.writerow(["top5_acc", top5])
        writer.writerow(["mean_true_cos", mean_true_overall])
        writer.writerow(["mean_max_neg_cos", mean_max_neg_overall])


if __name__ == "__main__":
    main()
