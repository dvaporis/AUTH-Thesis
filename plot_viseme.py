#!/usr/bin/env python3
"""Re-generate viseme training plots from a saved history.json file.

This script is meant for the output of train_lip_viseme.py. It reads the
single history file under a results directory and plots the metrics saved by
that training loop:

* train_total, train_frame, train_ctc
* val_total, val_ter
* lr

Usage:
    python plot_viseme.py --results-dir viseme_results
    python plot_viseme.py --results-dir viseme_results --output-dir viseme_plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(results_dir: Path) -> Dict[str, Any]:
    """Load the saved training history from a viseme results directory."""
    history_path = results_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.json in {results_dir}")

    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)

    if not isinstance(history, dict):
        raise ValueError(f"Expected a JSON object in {history_path}")

    return history


def _epochs(values: List[Any]) -> List[int]:
    return list(range(1, len(values) + 1))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_viseme_history(history: Dict[str, Any], output_dir: Path) -> None:
    """Plot the metrics saved by train_lip_viseme.py."""
    # Older histories may not contain every auxiliary-loss series.
    output_dir.mkdir(parents=True, exist_ok=True)

    train_total = history.get("train_total", [])
    train_frame = history.get("train_frame", [])
    train_ctc = history.get("train_ctc", [])
    val_total = history.get("val_total", [])
    val_ter = history.get("val_ter", [])
    lr = history.get("lr", [])

    # 1. Training losses
    fig, ax = plt.subplots(figsize=(10, 5))
    if train_total:
        ax.plot(_epochs(train_total), train_total, lw=2, label="Total")
    if train_frame:
        ax.plot(_epochs(train_frame), train_frame, lw=2, label="Frame CE")
    if train_ctc:
        ax.plot(_epochs(train_ctc), train_ctc, lw=2, label="CTC")
    ax.set_title("Training Loss Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, output_dir / "01_train_losses.png")

    # 2. Validation loss
    fig, ax = plt.subplots(figsize=(10, 5))
    if val_total:
        ax.plot(_epochs(val_total), val_total, color="#377eb8", lw=2, label="Val Total")
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, output_dir / "02_val_loss.png")

    # 3. Validation TER
    fig, ax = plt.subplots(figsize=(10, 5))
    if val_ter:
        ax.plot(_epochs(val_ter), val_ter, color="#e41a1c", lw=2, label="Val TER")
    ax.set_title("Validation TER")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Token Error Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, output_dir / "03_val_ter.png")

    # 4. Learning rate schedule
    fig, ax = plt.subplots(figsize=(10, 4))
    if lr:
        ax.plot(_epochs(lr), lr, color="#4daf4a", lw=1.8, label="LR")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, output_dir / "04_lr_schedule.png")

    # 5. Summary dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    if train_total:
        axes[0, 0].plot(_epochs(train_total), train_total, color="#377eb8", lw=2, label="Total")
    if train_frame:
        axes[0, 0].plot(_epochs(train_frame), train_frame, color="#4daf4a", lw=2, label="Frame CE")
    if train_ctc:
        axes[0, 0].plot(_epochs(train_ctc), train_ctc, color="#ff7f00", lw=2, label="CTC")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    if val_total:
        axes[0, 1].plot(_epochs(val_total), val_total, color="#377eb8", lw=2, label="Val Total")
    axes[0, 1].set_title("Val Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    if val_ter:
        axes[1, 0].plot(_epochs(val_ter), val_ter, color="#e41a1c", lw=2, label="Val TER")
    axes[1, 0].set_title("Val TER")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("TER")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    if lr:
        axes[1, 1].plot(_epochs(lr), lr, color="#4daf4a", lw=2, label="LR")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Viseme Training Summary", fontsize=14, fontweight="bold")
    _save(fig, output_dir / "00_dashboard.png")

    print(f"Saved {len(list(output_dir.glob('*.png')))} plots to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-plot viseme training results")
    parser.add_argument("--results-dir", default="viseme_results", help="Directory containing history.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save plots (default: the same directory as results-dir)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    history = load_history(results_dir)
    epochs = len(history.get("val_total", []) or history.get("train_total", []))
    print(f"Loaded history from {results_dir} ({epochs} epochs)")
    plot_viseme_history(history, output_dir)


if __name__ == "__main__":
    main()