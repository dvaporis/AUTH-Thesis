#!/usr/bin/env python3
"""Train a visual Transformer + CTC model on lip crops and wav2vec2 phoneme targets.

This script reuses the same data pipeline, evaluation code, and diagnostics as
train_lip_lstm_ctc.py, but replaces the bidirectional LSTM with a Transformer
encoder over the per-frame visual features.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_lip_lstm_ctc import (
    DEFAULT_BLANK_TOKEN,
    FrameEncoder,
    LipPhonemeDataset,
    TrainingConfig,
    build_phoneme_diagnostics,
    build_samples,
    build_vocab,
    collate_batch,
    evaluate_loader_with_sequences,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
    split_samples,
    train_epoch,
    validate,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    feedforward_multiplier: int = 4


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = x.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}")

        device = x.device
        dtype = x.dtype
        positions = torch.arange(sequence_length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / hidden_dim)
        )

        positional = torch.zeros(sequence_length, hidden_dim, device=device, dtype=dtype)
        positional[:, 0::2] = torch.sin(positions * div_term)
        positional[:, 1::2] = torch.cos(positions * div_term)

        x = x + positional.unsqueeze(0)
        return self.dropout(x)


class VisualTransformerCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.1,
        feedforward_multiplier: int = 4,
        pretrained_backbone: bool = True,
        finetune_backbone: bool = True,
    ):
        super().__init__()
        self.frame_encoder = FrameEncoder(
            hidden_dim=hidden_dim,
            pretrained=pretrained_backbone,
            finetune=finetune_backbone,
        )

        self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim, dropout=transformer_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * feedforward_multiplier,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def _encode_frames(self, videos: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = videos.shape
        flat_frames = videos.reshape(batch_size * time_steps, channels, height, width)
        frame_features = self.frame_encoder(flat_frames)
        return frame_features.reshape(batch_size, time_steps, -1)

    def forward(self, videos: torch.Tensor, video_lengths: torch.Tensor) -> torch.Tensor:
        sequence_features = self._encode_frames(videos)
        sequence_features = self.positional_encoding(sequence_features)

        max_sequence_length = sequence_features.size(1)
        padding_mask = torch.arange(max_sequence_length, device=video_lengths.device).unsqueeze(0) >= video_lengths.unsqueeze(1)
        encoded_sequence = self.sequence_encoder(sequence_features, src_key_padding_mask=padding_mask)
        encoded_sequence = self.sequence_norm(encoded_sequence)
        logits = self.classifier(encoded_sequence)
        return logits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a visual Transformer CTC model on lip crops and phoneme targets")
    parser.add_argument("--video-dir", type=str, default="lip_crop_results_full")
    parser.add_argument("--phoneme-csv", type=str, default="phoneme_results/phoneme_predictions.csv")
    parser.add_argument("--output-dir", type=str, default="visual_transformer_ctc_results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-feedforward-multiplier", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linearly warm up the learning rate for this many epochs")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--class-aware-sampling",
        action="store_true",
        help="Use class-aware (weighted) sampling based on phoneme inverse frequency for training",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    video_dir = Path(args.video_dir)
    phoneme_csv = Path(args.phoneme_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(video_dir, phoneme_csv)
    train_samples, val_samples, test_samples = split_samples(
        samples,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=args.seed,
    )

    vocab = build_vocab(samples)
    id_to_token = {index + 1: token for index, token in enumerate(vocab)}
    id_to_token[0] = DEFAULT_BLANK_TOKEN

    with (output_dir / "phoneme_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump({"blank_id": 0, "blank_token": DEFAULT_BLANK_TOKEN, "tokens": vocab}, handle, ensure_ascii=False, indent=2)

    train_dataset = LipPhonemeDataset(
        train_samples,
        vocab=vocab,
        frame_size=args.frame_size,
        frame_stride=args.frame_stride,
        augment=True,
    )
    val_dataset = LipPhonemeDataset(
        val_samples,
        vocab=vocab,
        frame_size=args.frame_size,
        frame_stride=args.frame_stride,
        augment=False,
    )
    test_dataset = LipPhonemeDataset(
        test_samples,
        vocab=vocab,
        frame_size=args.frame_size,
        frame_stride=args.frame_stride,
        augment=False,
    )

    if args.class_aware_sampling:
        phoneme_counts: Dict[str, int] = {}
        for sample in train_samples:
            for token in sample.target_tokens:
                phoneme_counts[token] = phoneme_counts.get(token, 0) + 1

        inverse_frequency = {token: 1.0 / max(1, count) for token, count in phoneme_counts.items()}
        sample_weights: List[float] = []
        for sample in train_samples:
            tokens = [token for token in sample.target_tokens if token in inverse_frequency]
            if tokens:
                weight = float(sum(inverse_frequency[token] for token in tokens) / len(tokens))
            else:
                weight = 1.0
            sample_weights.append(weight)

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, collate_fn=collate_batch)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = VisualTransformerCTC(
        vocab_size=len(vocab) + 1,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.transformer_dropout,
        feedforward_multiplier=args.transformer_feedforward_multiplier,
        pretrained_backbone=not args.no_pretrained,
        finetune_backbone=not args.freeze_backbone,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=TrainingConfig.early_stopping_min_delta,
        min_lr=args.min_lr,
    )

    logger.info("Loaded %d samples (%d train / %d val / %d test)", len(samples), len(train_samples), len(val_samples), len(test_samples))
    logger.info("Phoneme vocabulary size: %d (+ blank)", len(vocab))
    if args.warmup_epochs > 0:
        logger.info("Learning rate schedule: linear warmup for %d epoch(s), then ReduceLROnPlateau", args.warmup_epochs)
    else:
        logger.info("Learning rate schedule: ReduceLROnPlateau")

    if args.dry_run:
        batch = next(iter(train_loader))
        videos = batch["videos"].to(device)
        video_lengths = batch["video_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        with torch.no_grad():
            logits = model(videos, video_lengths)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, video_lengths, target_lengths)
        logger.info("Dry run videos shape: %s", tuple(videos.shape))
        logger.info("Dry run logits shape: %s", tuple(logits.shape))
        logger.info("Dry run CTC loss: %.6f", float(loss.item()))
        return

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_ter": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warmup_factor = epoch / float(args.warmup_epochs)
            current_lr = args.lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device, id_to_token)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_ter"].append(val_metrics["token_error_rate"])
        if args.warmup_epochs <= 0 or epoch > args.warmup_epochs:
            scheduler.step(val_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | val_TER=%.4f | lr=%.2e",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["token_error_rate"],
            current_lr,
        )

        improved = val_metrics["loss"] < (best_val_loss - TrainingConfig.early_stopping_min_delta)
        if improved:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "vocab": vocab,
                    "config": vars(args),
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= TrainingConfig.early_stopping_patience:
            logger.info("Early stopping triggered")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": vars(args),
        },
        output_dir / "final_model.pt",
    )

    test_metrics, test_predictions, test_targets = evaluate_loader_with_sequences(
        model,
        test_loader,
        criterion,
        device,
        id_to_token,
    )
    matrix, _, per_phoneme_report, top_confusions = build_phoneme_diagnostics(test_targets, test_predictions, vocab)

    save_confusion_matrix_csv(matrix, vocab, output_dir / "test_confusion_matrix.csv")
    save_confusion_matrix_plot(matrix, vocab, output_dir / "test_confusion_matrix.png")

    completely_missed_phonemes = [
        item["phoneme"]
        for item in per_phoneme_report
        if int(item["support"]) > 0 and int(item["correct"]) == 0
    ]

    diagnostics = {
        "completely_missed_phonemes": completely_missed_phonemes,
        "per_phoneme_report": per_phoneme_report,
        "top_confusions": top_confusions,
    }
    with (output_dir / "test_phoneme_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)

    with (output_dir / "test_phoneme_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phoneme", "support", "predicted", "correct", "missed", "recall", "precision"])
        writer.writeheader()
        writer.writerows(per_phoneme_report)

    with (output_dir / "test_metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"test_loss={test_metrics['loss']:.6f}\n")
        handle.write(f"test_token_error_rate={test_metrics['token_error_rate']:.6f}\n")
        handle.write(f"completely_missed_phonemes={len(completely_missed_phonemes)}\n")
        if completely_missed_phonemes:
            handle.write("missed_phonemes=" + ", ".join(completely_missed_phonemes) + "\n")
        if top_confusions:
            handle.write("top_confusions=\n")
            for item in top_confusions[:10]:
                handle.write(f"  {item['actual']} -> {item['predicted']}: {item['count']}\n")

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Test loss: %.4f | Test TER: %.4f", test_metrics["loss"], test_metrics["token_error_rate"])


if __name__ == "__main__":
    main()