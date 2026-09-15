# Quick Start

This guide runs the active GRID visual phoneme pipeline. Run commands from the repository root.

## Requirements

The active environment was tested on Python 3.14.0. See the root README for the supported range, pinned dependency files, FFmpeg/eSpeak-NG prerequisites, and legacy-environment warning.

### PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
```

For development and tests:

```powershell
python -m pip install -r requirements-dev.txt
```

### Linux/bash

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

The active pipeline also requires FFmpeg and eSpeak-NG on `PATH`. A compatible PyTorch build is required; CUDA is optional but recommended for training.

## Data preparation

Place GRID source videos under `s1/`. The active pipeline expects standard six-character GRID stems such as `bbaf2n` and uses 25 FPS.

### Crop lips

```bash
python crop_lips.py
```

For a small inspection run:

```bash
python crop_lips.py --max-videos 3
```

The default output is `s1_lip_crops/`. The MediaPipe model is downloaded automatically to `s1_lip_crops/models/`.

### Extract phoneme targets

```bash
python extract_phonemes.py \
    --video-dir s1_lip_crops \
    --output-dir phonemes_s1_aligned \
    --fps 25
```

Use `--dry-run` first to validate discovery and GRID filename decoding. The main output is `phonemes_s1_aligned/phoneme_predictions.csv`.

## Train and evaluate

```bash
python train_lip_viseme.py \
    --video-dir s1_lip_crops \
    --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
    --output-dir viseme_results
```

Evaluate a checkpoint:

```bash
python evaluate_lip_lstm.py \
    --checkpoint viseme_results/best_model.pt \
    --vocab-json viseme_results/phoneme_vocab.json \
    --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
    --video-dir s1_lip_crops \
    --output-dir eval_results \
    --num-videos 5
```

Plot visual training results:

```bash
python plot_viseme.py --results-dir viseme_results
python plot_ablation.py --results-dir ablation_results
```

## Audio-visual fusion

Create corrupted pairs:

```bash
python add_av_noise.py \
    --video-dir s1_lip_crops \
    --audio-source-dir s1 \
    --output-dir av_noise_s1 \
    --fps 25
```

Cache mel features and train:

```bash
python precompute_mel_spectograms.py \
    --noise-manifest av_noise_s1/noise_manifest.csv \
    --output-dir av_noise_s1

python train_av_fusion.py \
    --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
    --noise-manifest av_noise_s1/noise_manifest.csv \
    --output-dir av_fusion_results
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for ablations and SNR/video-corruption sweeps.
