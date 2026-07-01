#!/usr/bin/env python3
"""Re-generate ablation plots from already-saved history.json files.

Useful if you want to tweak plot styling without re-running training.

Usage:
    python plot_ablation.py --results-dir ablation_results
    python plot_ablation.py --results-dir ablation_results --output-dir custom_plots
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#ff7f00",  # orange
    "#984ea3",  # purple
    "#a65628",  # brown
]

VARIANT_LABELS = {
    "1_bilstm_full_loss":        "BiLSTM + Full Loss",
    "2_transformer_full_loss":   "Transformer + Full Loss",
    "3_no_seq_encoder":          "No Seq. Encoder (Proj. Only)",
    "4_bilstm_ctc_only":         "BiLSTM + CTC Only",
    "5_bilstm_framece_only":     "BiLSTM + Frame-CE Only",
    "6_bilstm_hard_targets":     "BiLSTM + Hard Targets (No Similarity)",
}


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load history.json and test_metrics.json for each variant subdirectory."""
    out = []
    for subdir in sorted(results_dir.iterdir()):
        hist_path = subdir / "history.json"
        test_path = subdir / "test_metrics.json"
        if not subdir.is_dir() or not hist_path.exists():
            continue
        with hist_path.open() as fh:
            history = json.load(fh)
        test_m: Dict[str, Any] = {}
        if test_path.exists():
            with test_path.open() as fh:
                test_m = json.load(fh)
        label = VARIANT_LABELS.get(subdir.name, subdir.name)
        val_acc = [1.0 - v for v in history.get("val_ter", [])]
        auc = float(np.trapz(val_acc) / max(1, len(val_acc))) if val_acc else 0.0
        out.append({
            "name":        subdir.name,
            "label":       label,
            "history":     history,
            "test":        test_m,
            "epochs_run":  len(history.get("val_ter", [])),
            "best_val_ter": min(history.get("val_ter", [1.0])),
            "auc_val_acc": auc,
        })
    if not out:
        raise FileNotFoundError(f"No variant result directories found in {results_dir}")
    return out


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(results: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in results]
    colors = COLORS[:len(results)]

    def ep(r):
        return list(range(1, r["epochs_run"] + 1))

    short = [r["label"].replace(" + ", "\n+\n") for r in results]
    test_ters   = [r["test"].get("ter", 1.0) for r in results]
    best_ters   = [r["best_val_ter"] for r in results]
    aucs        = [r["auc_val_acc"] for r in results]
    test_faccs  = [r["test"].get("frame_acc", None) for r in results]

    # 1. Train loss
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        if r["history"].get("train_total"):
            ax.plot(ep(r), r["history"]["train_total"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "01_train_loss.png")

    # 2. Val loss
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        if r["history"].get("val_total"):
            ax.plot(ep(r), r["history"]["val_total"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Validation Loss vs Epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "02_val_loss.png")

    # 3. Train vs Val loss grid
    ncols = 3
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, r, c in zip(axes_flat, results, colors):
        ep_ = ep(r)
        if r["history"].get("train_total"):
            ax.plot(ep_, r["history"]["train_total"], color=c, lw=2, label="Train")
        if r["history"].get("val_total"):
            ax.plot(ep_, r["history"]["val_total"], color=c, lw=2, ls="--", label="Val")
        ax.set_title(r["label"], fontsize=9); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(results):]:
        ax.set_visible(False)
    fig.suptitle("Train vs Validation Loss per Model", fontsize=12, fontweight="bold")
    _save(fig, output_dir / "03_train_val_loss_grid.png")

    # 4. Val TER
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        if r["history"].get("val_ter"):
            ax.plot(ep(r), r["history"]["val_ter"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Token Error Rate")
    ax.set_title("Validation TER vs Epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "04_val_ter.png")

    # 5. Val frame accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        if r["history"].get("val_frame_acc"):
            ax.plot(ep(r), r["history"]["val_frame_acc"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Frame Accuracy")
    ax.set_title("Validation Frame Accuracy vs Epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "05_val_frame_acc.png")

    # 6. Train frame accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, c in zip(results, colors):
        if r["history"].get("train_frame_acc"):
            ax.plot(ep(r), r["history"]["train_frame_acc"], color=c, lw=2, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Frame Accuracy")
    ax.set_title("Training Frame Accuracy vs Epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "06_train_frame_acc.png")

    # 7. LR schedule
    fig, ax = plt.subplots(figsize=(10, 4))
    for r, c in zip(results, colors):
        if r["history"].get("lr"):
            ax.plot(ep(r), r["history"]["lr"], color=c, lw=1.5, label=r["label"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule"); ax.legend(fontsize=8)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    _save(fig, output_dir / "07_lr_schedule.png")

    # 8. AUC bar
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short, aucs, color=colors, edgecolor="black", lw=0.8)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("AUC of Val (1 − TER)")
    ax.set_title("Area Under Val Accuracy Curve")
    ax.set_ylim(0, max(aucs)*1.15+0.01); ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(fontsize=8)
    _save(fig, output_dir / "08_auc_bar.png")

    # 9. Test TER bar
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short, test_ters, color=colors, edgecolor="black", lw=0.8)
    for bar, v in zip(bars, test_ters):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Token Error Rate"); ax.set_title("Test TER per Variant")
    ax.set_ylim(0, max(test_ters)*1.15+0.02); ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(fontsize=8)
    _save(fig, output_dir / "09_test_ter_bar.png")

    # 10. Best Val TER bar
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short, best_ters, color=colors, edgecolor="black", lw=0.8)
    for bar, v in zip(bars, best_ters):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Token Error Rate"); ax.set_title("Best Validation TER per Variant")
    ax.set_ylim(0, max(best_ters)*1.15+0.02); ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(fontsize=8)
    _save(fig, output_dir / "10_best_val_ter_bar.png")

    # 11. Test Frame Accuracy bar (if available)
    valid_faccs = [v for v in test_faccs if v is not None]
    if valid_faccs:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            [short[i] for i, v in enumerate(test_faccs) if v is not None],
            valid_faccs,
            color=[colors[i] for i, v in enumerate(test_faccs) if v is not None],
            edgecolor="black", lw=0.8
        )
        for bar, v in zip(bars, valid_faccs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Frame Accuracy"); ax.set_title("Test Frame Accuracy per Variant")
        ax.set_ylim(0, max(valid_faccs)*1.15+0.01); ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(fontsize=8)
        _save(fig, output_dir / "11_test_frame_acc_bar.png")

    # 12. CTC vs Frame loss split (where both exist)
    has_split = any(r["history"].get("val_frame") and r["history"].get("val_ctc") for r in results)
    if has_split:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for r, c in zip(results, colors):
            if r["history"].get("val_frame"):
                axes[0].plot(ep(r), r["history"]["val_frame"], color=c, lw=2, label=r["label"])
            if r["history"].get("val_ctc"):
                axes[1].plot(ep(r), r["history"]["val_ctc"], color=c, lw=2, label=r["label"])
        axes[0].set_title("Val Frame-CE Loss Component"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
        axes[1].set_title("Val CTC Loss Component"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
        for ax in axes:
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        _save(fig, output_dir / "12_val_loss_components.png")

    # 00. Dashboard (4-panel)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for r, c in zip(results, colors):
        if r["history"].get("val_total"):
            axes[0,0].plot(ep(r), r["history"]["val_total"], color=c, lw=2, label=r["label"])
        if r["history"].get("val_ter"):
            axes[0,1].plot(ep(r), r["history"]["val_ter"], color=c, lw=2, label=r["label"])
        if r["history"].get("val_frame_acc"):
            axes[1,0].plot(ep(r), r["history"]["val_frame_acc"], color=c, lw=2, label=r["label"])
    axes[0,0].set_title("Val Loss"); axes[0,0].legend(fontsize=7); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].set_title("Val TER");  axes[0,1].legend(fontsize=7); axes[0,1].grid(True, alpha=0.3)
    axes[1,0].set_title("Val Frame Acc"); axes[1,0].legend(fontsize=7); axes[1,0].grid(True, alpha=0.3)
    for ax in [axes[0,0], axes[0,1], axes[1,0]]:
        ax.set_xlabel("Epoch")
    # Test TER bar
    xi = range(len(results))
    bars = axes[1,1].bar(xi, test_ters, color=colors, edgecolor="black", lw=0.8)
    axes[1,1].set_xticks(list(xi))
    axes[1,1].set_xticklabels([r["label"] for r in results], rotation=20, ha="right", fontsize=7)
    axes[1,1].set_title("Test TER"); axes[1,1].set_ylabel("TER"); axes[1,1].grid(True, axis="y", alpha=0.3)
    for x_, v in zip(xi, test_ters):
        axes[1,1].text(x_, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("Ablation Study — Summary Dashboard", fontsize=14, fontweight="bold")
    _save(fig, output_dir / "00_dashboard.png")

    print(f"\nSaved {len(list(output_dir.glob('*.png')))} plots to {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Re-plot ablation results")
    p.add_argument("--results-dir", default="ablation_results")
    p.add_argument("--output-dir",  default=None,
                   help="Where to save plots (default: same as results-dir)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else results_dir

    results = load_results(results_dir)
    print(f"Loaded {len(results)} variants: {[r['label'] for r in results]}")
    plot_all(results, output_dir)


if __name__ == "__main__":
    main()