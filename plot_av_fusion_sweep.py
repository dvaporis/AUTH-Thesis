#!/usr/bin/env python3
"""Plot metrics from run_av_fusion_snr_experiment.py / run_av_fusion_video_experiment.py.

Two layers of data are produced by those sweep scripts, and this script plots
both of them:

  1. The sweep-level summary (`*_snr_sweep_summary.json` /
     `*_video_sweep_summary.json`). This only contains SCALARS per run --
     `final_train_loss`, `final_val_ter`, `best_epoch`, `best_val_ter`,
     `test_ter`, `test_loss`, `test_gate_video`, `test_gate_audio` -- because
     `summarise_run()` in the sweep scripts only pulls single numbers out of
     each run's history, it does not copy the per-epoch arrays up. This
     script turns those scalars into "metric vs SNR / vs video-obstruction"
     line plots, one line per modality (av / audio / video) if you pass more
     than one summary file.

  2. The full per-epoch curves. These were never copied into the summary --
     they still live in each individual run's own `<result_dir>/history.json`
     (train_total, val_total, val_ter, val_gate_video, val_gate_audio, lr).
     This script re-opens each run's history.json (using the `result_dir`
     path recorded in the summary row) and plots per-epoch train-loss / val
     TER curves, faceted by sweep setting, with one line per modality.

Usage
-----
    # SNR sweep, single modality
    python plot_av_fusion_sweep.py \
        --summary-json snr_sweep_results/av_snr_sweep_summary.json \
        --output-dir snr_sweep_plots

    # SNR sweep, compare av vs audio vs video
    python plot_av_fusion_sweep.py \
        --summary-json snr_sweep_results/av_snr_sweep_summary.json \
                        snr_sweep_results/audio_snr_sweep_summary.json \
                        snr_sweep_results/video_snr_sweep_summary.json \
        --output-dir snr_sweep_plots

    # Video-obstruction sweep
    python plot_av_fusion_sweep.py \
        --summary-json video_sweep_results/av_video_sweep_summary.json \
        --output-dir video_sweep_plots

    # Skip re-opening every run's history.json (summary-only plots, fast)
    python plot_av_fusion_sweep.py --summary-json ... --no-epoch-curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#377eb8", "#e41a1c", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"]

# Candidate x-axis keys written by the two sweep scripts, in priority order.
X_KEY_CANDIDATES: List[Tuple[str, str]] = [
    ("snr_db", "Audio SNR (dB)"),
    ("video_obstruct_frames", "Video obstruction length (frames)"),
]

# (summary key, y-axis label, lower_is_better)
SUMMARY_METRICS: List[Tuple[str, str, bool]] = [
    ("test_ter", "Test Token Error Rate", True),
    ("best_val_ter", "Best Validation TER", True),
    ("final_val_ter", "Final-epoch Validation TER", True),
    ("test_loss", "Test Loss", True),
    ("final_train_loss", "Final-epoch Train Loss", True),
    ("test_gate_video", "Test mean fusion gate (video)", None),
    ("test_gate_audio", "Test mean fusion gate (audio)", None),
    ("epochs_ran", "Epochs run before stopping", None),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_summary_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if not isinstance(loaded, list):
            raise ValueError(f"{path} does not look like a sweep summary JSON (expected a list of rows)")
        rows.extend(loaded)
    if not rows:
        raise ValueError("No rows found in any of the provided summary JSON files")
    return rows


def detect_x_key(rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    for key, label in X_KEY_CANDIDATES:
        if any(key in row and row[key] is not None for row in rows):
            return key, label
    raise ValueError(
        f"Could not find any of {[k for k, _ in X_KEY_CANDIDATES]} in the summary rows -- "
        "is this a summary produced by run_av_fusion_snr_experiment.py or "
        "run_av_fusion_video_experiment.py?"
    )


def group_by_modality(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        modality = row.get("modality", "unknown")
        grouped.setdefault(modality, []).append(row)
    for modality, group in grouped.items():
        group.sort(key=lambda r: r.get("_x", 0))
    return grouped


def load_run_history(row: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
    """Re-open the per-run history.json that the summary row points at.

    Returns None (rather than raising) if it's missing, so a partially
    finished sweep still plots whatever is available.
    """
    result_dir = row.get("result_dir")
    if not result_dir:
        return None
    history_path = Path(result_dir) / "history.json"
    if not history_path.exists():
        return None
    with history_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Summary-level plots (metric vs sweep setting, one line per modality)
# ---------------------------------------------------------------------------

def plot_summary_metric(
    grouped: Dict[str, List[Dict[str, Any]]],
    x_key: str,
    x_label: str,
    metric_key: str,
    metric_label: str,
    lower_is_better: Optional[bool],
    output_path: Path,
) -> bool:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_anything = False

    for color, (modality, rows) in zip(COLORS, sorted(grouped.items())):
        xs, ys = [], []
        for row in rows:
            value = row.get(metric_key)
            if value is None:
                continue
            xs.append(row[x_key])
            ys.append(value)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", color=color, linewidth=2, label=modality)
        plotted_anything = True

    if not plotted_anything:
        plt.close(fig)
        return False

    direction = ""
    if lower_is_better is True:
        direction = " (lower is better)"
    elif lower_is_better is False:
        direction = " (higher is better)"

    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_label + direction)
    ax.set_title(f"{metric_label} vs {x_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(title="modality")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Per-epoch curve plots (reconstructed from each run's own history.json)
# ---------------------------------------------------------------------------

def plot_epoch_curves_by_setting(
    grouped: Dict[str, List[Dict[str, Any]]],
    x_key: str,
    x_label: str,
    history_metric: str,
    metric_label: str,
    output_path: Path,
) -> bool:
    """One subplot per distinct sweep setting; each subplot overlays the
    per-epoch curve for every modality run at that setting."""

    all_x_values = sorted({row[x_key] for rows in grouped.values() for row in rows})
    if not all_x_values:
        return False

    import math
    ncols = min(3, len(all_x_values))
    nrows = math.ceil(len(all_x_values) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    plotted_anything = False
    for ax, x_value in zip(axes_flat, all_x_values):
        for color, (modality, rows) in zip(COLORS, sorted(grouped.items())):
            row = next((r for r in rows if r[x_key] == x_value), None)
            if row is None:
                continue
            history = load_run_history(row)
            if history is None or history_metric not in history or not history[history_metric]:
                continue
            series = history[history_metric]
            ax.plot(range(1, len(series) + 1), series, color=color, linewidth=2, label=modality)
            plotted_anything = True

        ax.set_title(f"{x_label} = {x_value}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes_flat[len(all_x_values):]:
        ax.set_visible(False)

    if not plotted_anything:
        plt.close(fig)
        return False

    fig.suptitle(f"Per-epoch {metric_label} across {x_label} settings", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot metrics from run_av_fusion_snr_experiment.py / run_av_fusion_video_experiment.py sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--summary-json", nargs="+", required=True,
        help="One or more *_sweep_summary.json files (e.g. one per modality: av / audio / video).",
    )
    p.add_argument("--output-dir", default="sweep_plots")
    p.add_argument(
        "--no-epoch-curves", action="store_true",
        help="Skip re-opening each run's history.json; only plot the summary-level scalars.",
    )
    return p


def main() -> None:
    args = make_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = [Path(p) for p in args.summary_json]
    for p in summary_paths:
        if not p.exists():
            raise FileNotFoundError(f"Summary file not found: {p}")

    rows = load_summary_rows(summary_paths)
    x_key, x_label = detect_x_key(rows)
    for row in rows:
        row["_x"] = row.get(x_key)
    grouped = group_by_modality(rows)

    modalities = sorted(grouped.keys())
    print(f"Detected sweep variable: {x_key} ({x_label})")
    print(f"Modalities found: {modalities}")
    for modality, group in grouped.items():
        settings = [r[x_key] for r in group]
        print(f"  {modality}: {len(group)} run(s), settings={settings}")

    # --- 1. Summary-level scalar plots -------------------------------------------------
    made_any_summary_plot = False
    for metric_key, metric_label, lower_is_better in SUMMARY_METRICS:
        out_path = output_dir / f"{metric_key}_vs_{x_key}.png"
        if plot_summary_metric(grouped, x_key, x_label, metric_key, metric_label, lower_is_better, out_path):
            print(f"Wrote {out_path}")
            made_any_summary_plot = True

    if not made_any_summary_plot:
        print(
            "Warning: none of the expected summary metrics were found in the rows -- "
            "double check this is a summary from the sweep scripts."
        )

    # --- 2. Per-epoch curves, reconstructed from each run's own history.json ----------
    if not args.no_epoch_curves:
        epoch_metric_specs = [
            ("val_ter", "Validation TER"),
            ("train_total", "Train Loss"),
            ("val_total", "Validation Loss"),
            ("val_gate_video", "Mean fusion gate (video)"),
            ("val_gate_audio", "Mean fusion gate (audio)"),
        ]
        made_any_epoch_plot = False
        for history_metric, metric_label in epoch_metric_specs:
            out_path = output_dir / f"epoch_{history_metric}_by_{x_key}.png"
            if plot_epoch_curves_by_setting(grouped, x_key, x_label, history_metric, metric_label, out_path):
                print(f"Wrote {out_path}")
                made_any_epoch_plot = True

        if not made_any_epoch_plot:
            print(
                "Note: no per-epoch curves were plotted. This means none of the runs' "
                "result_dir/history.json files could be found on disk from here -- "
                "run this script from wherever those result_dir paths resolve, or pass "
                "--no-epoch-curves to suppress this check."
            )

    print(f"\nAll plots written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()