# Experiments

## Reproducibility scope

The archived headline results use 1,000 paired clips from GRID speaker `s1`. The visual trainer performs a seeded random shuffle with `seed=42`, then takes 70% for training, 15% for validation, and the remainder for testing:

| Split | Clips |
| --- | ---: |
| Train | 700 |
| Validation | 150 |
| Test | 150 |
| Total | 1,000 |

This is a within-speaker split. It is not speaker-independent, and repeated GRID sentence structures can occur across the splits. The repository includes `grid_phoneme.split_by_speaker()` for future speaker-held-out experiments, but no held-out-speaker result is claimed here.

The archived runs do not record hardware or wall-clock training time in a machine-readable experiment manifest. Those details should be recorded before treating a new run as a full reproduction.

## Ablation

Run the six visual variants:

```bash
python ablation_experiment.py \
    --video-dir s1_lip_crops \
    --phoneme-csv phonemes_s1_aligned/phoneme_predictions.csv \
    --output-dir ablation_results
```

The tracked presentation figure is [`Report/figures/ablation_val_ter.png`](../Report/figures/ablation_val_ter.png). The numerical table in the README is sourced from the tracked [`Report/ablation_summary.json`](../Report/ablation_summary.json), copied from the historical ablation output. The generated `ablation_results/` directory remains ignored.

The ablation uses single runs, so small differences should not be treated as statistically established. The BiLSTM full-loss baseline has test TER 0.1219 versus 0.1289 for hard targets and 0.1419 for frame-CE only. Those margins are modest and may be affected by seed variance. The clearer result is the sequence-encoder ablation: the BiLSTM baseline at 0.1219 is substantially better than the projection-only variant at 0.2464.

The CTC-only variant is informative: it has relatively poor frame accuracy (0.4743) but a better sequence TER (0.1869), because CTC optimizes collapsed sequence structure rather than framewise label accuracy. This is an objective difference, not evidence that the two metrics measure the same behavior.

## Audio corruption sweeps

Audio SNR sweep:

```bash
python run_av_fusion_snr_experiment.py
```

Default SNR values are `-5, 0, 5, 10, 15` dB. Plot the summary:

```bash
python plot_av_fusion_sweep.py \
    --summary-json snr_sweep_results/snr_sweep_summary.json \
    --output-dir snr_sweep_plots
```

## Video obstruction sweeps

```bash
python run_av_fusion_video_experiment.py
```

Default obstruction lengths are 10, 15, 20, 25, and 30 frames. Use `--phase prepare` or `--phase train` to resume specific stages and `--skip-existing` to avoid repeating completed runs.

## Speaker-independent extension

A proper generalization experiment should select complete speakers for train, validation, and test without crossing speaker boundaries. For example, reserve one GRID speaker entirely for testing, train on several other speakers, and report the same word, sentence, phoneme TER, and frame metrics. Do not compare that result directly with the current s1 within-speaker result without labeling the split change.
