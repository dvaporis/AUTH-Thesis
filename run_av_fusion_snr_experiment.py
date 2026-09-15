#!/usr/bin/env python3
"""Run an audio-visual learning experiment across multiple SNR settings.

This script orchestrates the existing pipeline in three stages for each SNR
value:

1. Generate a corrupted AV manifest with `add_av_noise.py`.
2. Precompute mel spectrograms for that manifest with
   `precompute_mel_spectograms.py` so training can reuse cached features.
3. Train `train_av_fusion.py` on the resulting dataset and collect the
   training / validation / test metrics.

The default sweep matches the requested experiment: -5, 0, 5, 10, and 15 dB.
Each run writes to its own output directory, which makes it easy to compare
how the model learns as audio quality changes.

Example
-------
    python run_av_fusion_snr_experiment.py --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv

    python run_av_fusion_snr_experiment.py --phase prepare --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


@dataclass
class SNRRun:
    snr_db: float
    modality: str
    tag: str
    noise_dir: Path
    result_dir: Path
    noise_manifest: Path
    mel_manifest: Path


def format_snr_tag(snr_db: float) -> str:
    text = f"{snr_db:g}".replace("-", "m").replace(".", "p")
    return f"snr_{text}db"


def run_command(command: Sequence[str]) -> None:
    log.info("Running: %s", subprocess.list2cmdline(list(command)))
    subprocess.run(list(command), check=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_run_plan(base_noise_dir: Path, base_result_dir: Path, snr_values: Sequence[float], modality: str) -> List[SNRRun]:
    # Build paths up front so each sweep run is independent and resumable.
    plan: List[SNRRun] = []
    for snr_db in snr_values:
        tag = f"{format_snr_tag(snr_db)}_{modality}"
        noise_dir = base_noise_dir / tag
        result_dir = base_result_dir / tag
        plan.append(SNRRun(
            snr_db=snr_db,
            modality=modality,
            tag=tag,
            noise_dir=noise_dir,
            result_dir=result_dir,
            noise_manifest=noise_dir / "noise_manifest.csv",
            mel_manifest=noise_dir / "noise_manifest_with_mel.csv",
        ))
    return plan


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep train_av_fusion.py across fixed audio SNR values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--phase", choices=["all", "prepare", "train"], default="all",
                   help="Prepare data only, train only, or do both.")
    p.add_argument("--modality", choices=["av", "audio", "video"], default="av",
                   help="Train the fused AV model, an audio-only model, or a video-only model.")
    p.add_argument("--snr-values", type=float, nargs="+", default=[-5, 0, 5, 10, 15],
                   help="SNR values in dB to sweep. The default matches the requested experiment.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a stage when its expected output already exists.")
    p.add_argument("--summary-csv", default="snr_sweep_results/snr_sweep_summary.csv")
    p.add_argument("--summary-json", default="snr_sweep_results/snr_sweep_summary.json")

    p.add_argument("--video-dir", default="s1_lip_crops")
    p.add_argument("--audio-source-dir", default="s1")
    p.add_argument("--phoneme-csv", default="phonemes_s1_aligned/phoneme_predictions.csv")
    p.add_argument("--noise-base-dir", default="snr_sweep_results/noise")
    p.add_argument("--result-base-dir", default="snr_sweep_results/train")

    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--video-obstruct-frames", type=int, default=30)
    p.add_argument("--video-obstruct-jitter", type=int, default=8)
    p.add_argument("--video-corruption-modes", nargs="+", default=["zero", "freeze", "blur"])
    p.add_argument("--audio-noise-types", nargs="+", default=["white", "babble"])
    p.add_argument("--audio-dropout-prob", type=float, default=0.3)
    p.add_argument("--audio-dropout-frames", type=int, default=15)
    p.add_argument("--audio-dropout-max-spans", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--dry-run-noise", action="store_true",
                   help="Pass --dry-run to add_av_noise.py so only one clip is generated per SNR.")

    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=0)
    p.add_argument("--max-files-precompute", type=int, default=0)
    p.add_argument("--dry-run-precompute", action="store_true",
                   help="Pass --dry-run to precompute_mel_spectograms.py.")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--min-phoneme-count", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--frame-size", type=int, default=96)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--encoder", choices=["bilstm", "transformer"], default="bilstm")
    p.add_argument("--lstm-layers", type=int, default=2)
    p.add_argument("--lstm-dropout", type=float, default=0.3)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--cross-attn-layers", type=int, default=1)
    p.add_argument("--cross-attn-heads", type=int, default=4)
    p.add_argument("--no-aux-heads", action="store_true")
    p.add_argument("--modality-dropout", type=float, default=0.15)
    p.add_argument("--ctc-weight", type=float, default=0.3)
    p.add_argument("--aux-ctc-weight", type=float, default=0.1)
    p.add_argument("--insertion-penalty", type=float, default=0.5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr-patience", type=int, default=6)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--early-stop", type=int, default=12)
    p.add_argument("--class-aware-sampling", action="store_true")

    p.add_argument("--skip-training", action="store_true",
                   help="Alias for --phase prepare.")
    return p


def build_noise_command(args: argparse.Namespace, run: SNRRun) -> List[str]:
    cmd = [
        sys.executable,
        "add_av_noise.py",
        "--video-dir", args.video_dir,
        "--audio-source-dir", args.audio_source_dir,
        "--output-dir", str(run.noise_dir),
        "--fps", str(args.fps),
        "--sample-rate", str(args.sample_rate),
        "--video-obstruct-frames", str(args.video_obstruct_frames),
        "--video-obstruct-jitter", str(args.video_obstruct_jitter),
        "--audio-snr-db-range", str(run.snr_db), str(run.snr_db),
        "--audio-noise-types", *args.audio_noise_types,
        "--audio-dropout-prob", str(args.audio_dropout_prob),
        "--audio-dropout-frames", str(args.audio_dropout_frames),
        "--audio-dropout-max-spans", str(args.audio_dropout_max_spans),
        "--seed", str(args.seed),
    ]
    if args.video_corruption_modes:
        cmd.extend(["--video-corruption-modes", *args.video_corruption_modes])
    if args.max_files > 0:
        cmd.extend(["--max-files", str(args.max_files)])
    if args.dry_run_noise:
        cmd.append("--dry-run")
    return cmd


def build_precompute_command(args: argparse.Namespace, run: SNRRun) -> List[str]:
    cmd = [
        sys.executable,
        "precompute_mel_spectograms.py",
        "--noise-manifest", str(run.noise_manifest),
        "--output-dir", str(run.noise_dir),
        "--fps", str(args.fps),
        "--sample-rate", str(args.sample_rate),
        "--n-mels", str(args.n_mels),
        "--n-fft", str(args.n_fft),
    ]
    if args.hop_length > 0:
        cmd.extend(["--hop-length", str(args.hop_length)])
    if args.max_files_precompute > 0:
        cmd.extend(["--max-files", str(args.max_files_precompute)])
    if args.dry_run_precompute:
        cmd.append("--dry-run")
    return cmd


def build_train_command(args: argparse.Namespace, run: SNRRun) -> List[str]:
    cmd = [
        sys.executable,
        "train_av_fusion.py",
        "--phoneme-csv", args.phoneme_csv,
        "--noise-manifest", str(run.mel_manifest),
        "--output-dir", str(run.result_dir),
        "--fps", str(args.fps),
        "--sample-rate", str(args.sample_rate),
        "--n-mels", str(args.n_mels),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--grad-clip", str(args.grad_clip),
        "--num-workers", str(args.num_workers),
        "--min-phoneme-count", str(args.min_phoneme_count),
        "--hidden-dim", str(args.hidden_dim),
        "--frame-size", str(args.frame_size),
        "--frame-stride", str(args.frame_stride),
        "--encoder", args.encoder,
        "--lstm-layers", str(args.lstm_layers),
        "--lstm-dropout", str(args.lstm_dropout),
        "--transformer-heads", str(args.transformer_heads),
        "--transformer-layers", str(args.transformer_layers),
        "--cross-attn-layers", str(args.cross_attn_layers),
        "--cross-attn-heads", str(args.cross_attn_heads),
        "--modality-dropout", str(args.modality_dropout),
        "--ctc-weight", str(args.ctc_weight),
        "--aux-ctc-weight", str(args.aux_ctc_weight),
        "--insertion-penalty", str(args.insertion_penalty),
        "--warmup-epochs", str(args.warmup_epochs),
        "--lr-patience", str(args.lr_patience),
        "--lr-factor", str(args.lr_factor),
        "--min-lr", str(args.min_lr),
        "--early-stop", str(args.early_stop),
    ]
    cmd.extend(["--modality", args.modality])
    if args.no_aux_heads:
        cmd.append("--no-aux-heads")
    if args.class_aware_sampling:
        cmd.append("--class-aware-sampling")
    return cmd


def summarise_run(run: SNRRun) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "snr_db": run.snr_db,
        "modality": run.modality,
        "tag": run.tag,
        "noise_dir": str(run.noise_dir),
        "result_dir": str(run.result_dir),
        "noise_manifest": str(run.noise_manifest),
        "mel_manifest": str(run.mel_manifest),
    }

    history_path = run.result_dir / "history.json"
    test_metrics_path = run.result_dir / "test_metrics.json"

    if history_path.exists():
        history = read_json(history_path)
        val_ter = history.get("val_ter", []) or []
        train_total = history.get("train_total", []) or []
        summary["epochs_ran"] = len(val_ter)
        summary["final_train_loss"] = train_total[-1] if train_total else None
        summary["final_val_ter"] = val_ter[-1] if val_ter else None
        if val_ter:
            best_epoch_idx = min(range(len(val_ter)), key=lambda i: val_ter[i])
            summary["best_epoch"] = best_epoch_idx + 1
            summary["best_val_ter"] = val_ter[best_epoch_idx]
    if test_metrics_path.exists():
        test_metrics = read_json(test_metrics_path)
        summary["test_ter"] = test_metrics.get("ter")
        summary["test_loss"] = test_metrics.get("total_loss")
        summary["test_gate_video"] = test_metrics.get("gate_video")
        summary["test_gate_audio"] = test_metrics.get("gate_audio")

    return summary


def write_summary(rows: List[Dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


def maybe_skip(expected_path: Path, enabled: bool) -> bool:
    return enabled and expected_path.exists()


def main() -> None:
    args = make_parser().parse_args()
    if args.skip_training:
        args.phase = "prepare"

    base_noise_dir = Path(args.noise_base_dir) / args.modality
    base_result_dir = Path(args.result_base_dir) / args.modality
    if args.summary_csv == "snr_sweep_results/snr_sweep_summary.csv":
        args.summary_csv = f"snr_sweep_results/{args.modality}_snr_sweep_summary.csv"
    if args.summary_json == "snr_sweep_results/snr_sweep_summary.json":
        args.summary_json = f"snr_sweep_results/{args.modality}_snr_sweep_summary.json"

    runs = build_run_plan(base_noise_dir, base_result_dir, args.snr_values, args.modality)

    summary_rows: List[Dict[str, Any]] = []

    for run in runs:
        log.info("=== SNR %.1f dB (%s, %s) ===", run.snr_db, run.tag, run.modality)

        if args.phase in {"all", "prepare"}:
            run.noise_dir.mkdir(parents=True, exist_ok=True)
            noise_command = build_noise_command(args, run)
            if maybe_skip(run.noise_manifest, args.skip_existing):
                log.info("Skipping noise generation; found %s", run.noise_manifest)
            else:
                run_command(noise_command)

            precompute_command = build_precompute_command(args, run)
            if maybe_skip(run.mel_manifest, args.skip_existing):
                log.info("Skipping mel precompute; found %s", run.mel_manifest)
            else:
                run_command(precompute_command)

        if args.phase in {"all", "train"} and not args.skip_training:
            run.result_dir.mkdir(parents=True, exist_ok=True)
            train_metrics_path = run.result_dir / "test_metrics.json"
            train_command = build_train_command(args, run)
            if maybe_skip(train_metrics_path, args.skip_existing):
                log.info("Skipping training; found %s", train_metrics_path)
            else:
                run_command(train_command)

        summary_rows.append(summarise_run(run))

    write_summary(summary_rows, Path(args.summary_csv), Path(args.summary_json))
    log.info("Wrote summary CSV to %s", Path(args.summary_csv))
    log.info("Wrote summary JSON to %s", Path(args.summary_json))
    log.info("Finished %d SNR run(s).", len(summary_rows))


if __name__ == "__main__":
    main()