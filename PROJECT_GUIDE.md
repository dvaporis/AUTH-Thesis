# Project guide

## Project focus

The current project is a **GRID-based visual phoneme prediction system**. It learns to predict a time-aligned phoneme label for each video frame from lip-crop videos, with optional audio-visual fusion for robustness to corrupted audio and video.

The main data path is:

```text
GRID videos in s1/
    -> crop_lips.py
GRID lip videos in s1_lip_crops/
    -> extract_phonemes.py
phonemes_s1_aligned/phoneme_predictions.csv
    -> train_lip_viseme.py or train_av_fusion.py
model checkpoints and metrics in viseme_results/ or av_fusion_results/
```

`extract_phonemes.py` decodes the six-character GRID filename, creates canonical eSpeak phonemes, aligns them to the audio, and converts the timings to 25 FPS video-frame labels. The training scripts consume that CSV contract rather than extracting phonemes themselves.

## Active final-version scripts

These are the scripts that belong to the final GRID/phoneme-prediction workflow.

### Data preparation and robustness

- `crop_lips.py`: crop GRID videos from `s1/` into `s1_lip_crops/`.
- `save_all_lip_videos.py`: batch version of the cropper for the same GRID folders.
- `extract_phonemes.py`: create timing-aware GRID phoneme targets.
- `add_av_noise.py`: create paired audio/video corruption manifests for fusion experiments.
- `precompute_mel_spectograms.py`: cache the mel features used by AV fusion. The filename keeps the historical `spectograms` spelling.

### Models and experiments

- `train_lip_viseme.py`: primary visual phoneme model with viseme-aware loss and auxiliary CTC loss.
- `train_av_fusion.py`: audio-visual phoneme model with reliability-aware fusion.
- `ablation_experiment.py`: compares alternative visual-model losses and sequence encoders.
- `evaluate_lip_lstm.py`: evaluates a visual checkpoint and synthesises duration-aligned audio examples.
- `run_av_fusion_snr_experiment.py`: sweeps audio corruption severity.
- `run_av_fusion_video_experiment.py`: sweeps video obstruction length.

### Plotting and augmentation inspection

- `plot_ablation.py`
- `plot_av_fusion_sweep.py`
- `plot_viseme.py`
- `visualize_lip_viseme_augmentations.py`

The repository also contains `visualize_video_augmentations.py`, which documents an older video-contrastive augmentation experiment and is kept as a reference rather than as part of the final phoneme pipeline.

## Legacy and reference work

The RAVDESS scripts and outputs are retained for reproducibility and completeness, but they are **not** part of the final GRID phoneme-prediction path. This includes scripts whose names or defaults mention RAVDESS, such as:

- `download_only_videos.py`, `download_unzip_videos.py`
- `extract_ravdess_phonemes.py`
- `preprocess_ravdess_audio.py`, `preprocess_mels_full.py`
- `train_ravdess_mel_cpc.py`, `train_audio_video_alignment.py`
- `train_mfcc_contrastive.py`, `train_time_domain_contrastive.py`
- `test_mel_reconstruction.py`, `test_mfcc_reconstruction.py`
- `evaluate_alignment.py`

Other exploratory models, including diffusion, video contrastive learning, frame-order prediction, and VideoMAE-related experiments, are also archived reference work unless listed in the active section above.

## Output folders

- `s1/`: source GRID videos used by the active pipeline.
- `s1_lip_crops/`: cropped GRID videos and optional crop inspection frames.
- `phonemes_s1_aligned/`: frame-aligned phoneme CSV and alignment metadata.
- `viseme_results/`: visual phoneme-model checkpoints, histories, reports, and plots.
- `av_noise_s1/`, `av_fusion_results/`: audio-visual corruption manifests and fusion runs.
- `ablation_results/`, `snr_sweep_results/`, `video_sweep_results/`: experiment outputs.
- `Report/`: thesis report material, separate from executable pipeline code.

## Why the Python files remain at the repository root

The scripts currently use root-relative paths, direct sibling imports, and subprocess calls such as `python train_av_fusion.py`. Moving them into folders without compatibility wrappers would make existing commands and imports fail. For now, conceptual organization is recorded here and in each script's module documentation; a future package refactor can move files safely in one coordinated change.

## Typical commands

```text
python crop_lips.py
python extract_phonemes.py --video-dir s1_lip_crops --output-dir phonemes_s1_aligned
python train_lip_viseme.py --video-dir s1_lip_crops --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv
python train_av_fusion.py --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv
```

Use `python <script> --help` for the complete set of options. On this machine, use the repository interpreter at `.venv/Scripts/python.exe` if the bare `python` command is not configured.
