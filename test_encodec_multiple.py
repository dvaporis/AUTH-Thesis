"""
Batch EnCodec test on multiple speech videos (01-*.mp4).

Runs encode/decode for multiple randomly selected Kaggle speech videos,
computes reconstruction metrics per sample, and saves a CSV summary.

Usage:
    python test_encodec_multiple.py --num-tests 10
"""

import argparse
import csv
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from encodec import EncodecModel

from test_encodec_with_kaggle import (
    compute_error_metrics,
    find_kaggle_audio_files,
    load_or_create_audio,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_single_encoding(
    model,
    device: torch.device,
    video_path: Path,
    sample_rate: int,
    num_video_frames: int,
) -> Dict[str, float]:
    waveform, _, duration_seconds, source_desc = load_or_create_audio(
        str(video_path), sample_rate=sample_rate, num_video_frames=num_video_frames
    )

    waveform = waveform.to(device)
    waveform_batch = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform

    with torch.no_grad():
        # Encode using audiocraft API
        encoded_frames = model.encode(waveform_batch)
        # Decode using audiocraft API
        reconstructed = model.decode(encoded_frames)

    original = waveform_batch.squeeze(0)
    reconstructed = reconstructed.squeeze(0)
    metrics = compute_error_metrics(original, reconstructed)

    result = {
        "file": video_path.name,
        "duration_s": round(duration_seconds, 6),
        "source": source_desc,
        **metrics,
    }
    return result


def summarize_results(results: List[Dict[str, float]]) -> Dict[str, float]:
    metric_names = ["MSE", "MAE", "RMSE", "SNR_dB", "PSNR_dB", "Cosine_Similarity"]
    summary = {}
    for metric in metric_names:
        values = [float(r[metric]) for r in results if metric in r]
        if values:
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"std_{metric}"] = float(np.std(values))
    return summary


def save_results_csv(results: List[Dict[str, float]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{timestamp}_multi_encodec_metrics.csv"

    fieldnames = [
        "file",
        "duration_s",
        "MSE",
        "MAE",
        "RMSE",
        "SNR_dB",
        "PSNR_dB",
        "Cosine_Similarity",
        "source",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EnCodec test on multiple speech videos (01-*.mp4)")
    parser.add_argument("--num-tests", type=int, default=5, help="Number of random videos to test")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Target sample rate")
    parser.add_argument("--num-video-frames", type=int, default=16,
                        help="Video frames represented by each audio bite (uses actual FPS per file)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="audio_encodec_results", help="Output directory")
    args = parser.parse_args()

    if args.sample_rate != 48000:
        logger.warning("This script is optimized for 48kHz stereo model.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_files = find_kaggle_audio_files()
    if not video_files:
        logger.error("No 01-* video files found. Run: python download_kaggle_dataset.py")
        return 1

    num_tests = min(args.num_tests, len(video_files))
    selected_files = random.sample(video_files, num_tests)
    logger.info(f"Selected {num_tests} video files for batch EnCodec test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading EnCodec model...")
    model = EncodecModel.encodec_model_48khz()
    # Set bandwidth to 24kbps (maximum quality)
    model.set_target_bandwidth(24.0)
    model.to(device)
    model.eval()
    logger.info("✓ Model loaded successfully (bandwidth: 24kbps)")

    results = []
    for idx, video_path in enumerate(selected_files, start=1):
        logger.info(f"[{idx}/{num_tests}] Processing: {video_path.name}")
        try:
            result = run_single_encoding(
                model=model,
                device=device,
                video_path=video_path,
                sample_rate=args.sample_rate,
                num_video_frames=args.num_video_frames,
            )
            results.append(result)
            logger.info(
                "  Metrics: "
                f"MSE={result['MSE']:.6f}, RMSE={result['RMSE']:.6f}, "
                f"SNR={result['SNR_dB']:.3f} dB, Cos={result['Cosine_Similarity']:.4f}"
            )
        except Exception as e:
            logger.warning(f"  Failed on {video_path.name}: {e}")

    if not results:
        logger.error("All encodings failed.")
        return 1

    output_dir = Path(args.output_dir)
    csv_path = save_results_csv(results, output_dir)
    summary = summarize_results(results)

    logger.info("=" * 70)
    logger.info(f"Completed {len(results)}/{num_tests} encodings")
    logger.info(f"Saved per-file metrics: {csv_path}")
    for key, value in summary.items():
        logger.info(f"{key}: {value:.6f}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
