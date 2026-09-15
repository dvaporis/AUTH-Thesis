# GRID Visual Phoneme Prediction

[![CI](https://github.com/dvapo/AUTH-Thesis/actions/workflows/ci.yml/badge.svg)](https://github.com/dvapo/AUTH-Thesis/actions/workflows/ci.yml)

This repository develops a **GRID-based visual phoneme prediction pipeline**: lip-crop video is converted into frame-aligned phoneme predictions, then decoded into words with GRID grammar. An optional audio-visual model studies robustness to controlled audio and video corruption.

Older RAVDESS, reconstruction, contrastive-learning, alignment, and diffusion experiments are retained for provenance, but are not required for the active GRID pipeline.

## Demo

![Subtitle-generation proof of concept](Report/figures/swwp2n_lipcrop_pred_subtitled.gif)

The model predicts **set white with p 2 now** and renders the words at frame-derived timings. [Open the original MP4](Report/figures/swwp2n_lipcrop_pred_subtitled.mp4) · [Predicted subtitles](Report/figures/swwp2n_lipcrop_pred.srt) · [Ground truth](Report/figures/swwp2n_lipcrop_gt.srt)

This is a qualitative single-clip proof of concept, not an additional aggregate metric.

## Results

The reported evaluation uses the known GRID vocabulary and sentence grammar, so these results are not unrestricted-vocabulary lip reading.

| Metric | Result | Scope |
| --- | ---: | --- |
| Word-correct prediction rate | **87.1%** | Main constrained test evaluation. |
| Exact sentence match | **38.0%** | Main constrained test evaluation. |
| Main-run phoneme TER | **12.41%** | Separate constrained visual evaluation run. |
| Best ablation frame accuracy | **84.74%** | BiLSTM full-loss baseline, before sequence collapse. |

### Figures

![Test phoneme confusion matrix](Report/figures/constrained_confusion_matrix.png)

![Test TER versus audio SNR](Report/figures/test_ter_vs_snr_db.png)

![Test TER versus video obstruction](Report/figures/test_ter_vs_video_obstruct_frames.png)

The full collection is in [`Report/figures`](Report/figures).

### Ablation summary

| Variant | Test TER | Test frame accuracy | Validation-accuracy AUC* |
| --- | ---: | ---: | ---: |
| BiLSTM + full loss | 0.1219 | 0.8474 | 0.7840 |
| Transformer + full loss | 0.1566 | 0.8394 | 0.7176 |
| No sequence encoder | 0.2464 | 0.8122 | 0.6006 |
| BiLSTM + CTC only | 0.1869 | 0.4743 | 0.6005 |
| BiLSTM + frame-CE only | 0.1419 | 0.8401 | 0.7402 |
| BiLSTM + hard targets | 0.1289 | 0.8456 | 0.7696 |

\* AUC is the trapezoidal area under the per-epoch validation-accuracy curve, normalized by the number of recorded epochs.

The 12.41% main-run TER and 0.1219 ablation-baseline TER are **different runs**, not contradictory values. The ablation baseline was retrained as V1 with its own output directory and conditions. Ablation results are single runs, so the 0.007-point difference between soft full loss and hard targets is not evidence of a statistically reliable advantage. The clearer result is the sequence-encoder ablation: BiLSTM TER 0.1219 versus projection-only TER 0.2464. CTC-only also illustrates an objective trade-off: sequence TER is 0.1869 while frame accuracy is 0.4743 because CTC optimizes collapsed sequence structure rather than framewise labels.

## Evaluation scope

The archived results use 1,000 paired clips from GRID speaker `s1`. The visual trainer uses a seeded random shuffle with `seed=42`, then a 70/15/15 within-speaker split:

| Split | Clips |
| --- | ---: |
| Train | 700 |
| Validation | 150 |
| Test | 150 |
| Total | 1,000 |

This split is not speaker-independent and can contain repeated GRID sentence structures across splits. GRID has 34 speakers; no held-out-speaker result is claimed yet. The package includes `grid_phoneme.split_by_speaker()` to support a future leakage-resistant experiment. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full scope and limitations.

Hardware and wall-clock training time were not recorded in a machine-readable manifest for the archived runs, so no hardware or timing claim is made here.

## How it works

```text
GRID videos in s1/
    -> crop_lips.py
GRID lip crops in s1_lip_crops/
    -> extract_phonemes.py
phonemes_s1_aligned/phoneme_predictions.csv
    -> train_lip_viseme.py or train_av_fusion.py
checkpoints, metrics, and plots
```

`extract_phonemes.py` decodes six-character GRID stems, creates canonical eSpeak phonemes, runs wav2vec2 forced alignment, and converts timings to 25 FPS frame labels. The CSV is the shared contract for visual training and audio-visual fusion.

The active entry points are `crop_lips.py`, `save_all_lip_videos.py`, `extract_phonemes.py`, `train_lip_viseme.py`, `train_av_fusion.py`, `add_av_noise.py`, `precompute_mel_spectograms.py`, `ablation_experiment.py`, `evaluate_lip_lstm.py`, the two AV sweep runners, and the plotting/augmentation helpers.

## Installation and usage

The active pipeline was tested on Python 3.14.0; the package metadata supports Python 3.11 through 3.14. Install pinned active dependencies with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
```

The active pipeline also needs FFmpeg, eSpeak-NG, a compatible PyTorch build, and GRID videos under `s1/`. For the complete visual and AV command sequence, see [`docs/QUICKSTART.md`](docs/QUICKSTART.md). For sweeps, ablations, and split details, see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). For debugging, see [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

The dependency files are intentionally split:

- [`requirements-core.txt`](requirements-core.txt): pinned active GRID dependencies.
- [`requirements-dev.txt`](requirements-dev.txt): core dependencies plus pytest.
- [`requirements-legacy.txt`](requirements-legacy.txt): unpinned historical package names only; it is not a lockfile. Install it in a separate environment and never combine it with core.

The active wav2vec2 extractor was checked under Transformers 5.2.0 and Hugging Face Hub 1.5.0: the three APIs it uses import successfully and `extract_phonemes.py --help` runs. A full extraction still requires model download, eSpeak-NG, and usable media.

## Repository hygiene

- [`grid_phoneme/`](grid_phoneme) is an importable package for shared GRID paths, stem normalization, and speaker grouping.
- [`tests/`](tests) contains focused domain tests for GRID decoding, viseme soft targets, Levenshtein/TER, and shared contracts.
- [`pyproject.toml`](pyproject.toml) defines package metadata and the supported Python range.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) installs the active requirements, compiles the repository, and runs tests on Python 3.11 and 3.14.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) documents local validation.
- [`LICENSE`](LICENSE) contains the MIT license.

The root scripts remain at the repository root for compatibility with legacy imports and subprocess entry points. Shared active utilities have been moved into `grid_phoneme`; future refactoring can migrate individual scripts behind compatibility wrappers.

## Legacy and retained work

These scripts are retained for completeness but are not part of the final GRID visual phoneme workflow:

- RAVDESS acquisition and preprocessing: `download_kaggle_dataset.py`, `download_only_videos.py`, `download_unzip_videos.py`, `extract_ravdess_phonemes.py`, `preprocess_ravdess_audio.py`, `preprocess_mels_full.py`.
- Audio representation experiments: `train_ravdess_mel_cpc.py`, `train_mfcc_contrastive.py`, `train_time_domain_contrastive.py`, `test_mel_reconstruction.py`, `test_mfcc_reconstruction.py`.
- Alignment and diffusion: `train_audio_video_alignment.py`, `evaluate_alignment.py`, `train_diffusion_mel.py`, `train_video_conditioned_mel_diffusion.py`, `evaluate_video_conditioned_mel_diffusion.py`.
- Earlier visual variants and decoders: `train_lip_lstm.py`, `train_lip_lstm_ctc.py`, `train_lip_transformer.py`, `extract_durations.py`, `wav_eval.py`, `wav_eval2.py`, `train_video_frame_order.py`, `train_video_contrastive.py`, and `visualize_video_augmentations.py`.

The old output folders remain as provenance. Their dependency list is deliberately separate and explicitly not historically reproducible because the original legacy environment was not preserved.
