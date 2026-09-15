# Troubleshooting

## Python environment

Use Python 3.11 or newer within the supported range. On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe --version
```

On Linux/bash:

```bash
.venv/bin/python --version
```

Install `requirements-core.txt` for the active GRID pipeline. Install `requirements-legacy.txt` only in a separate environment; it is an unpinned historical package-name list and can overwrite core packages if combined with the active environment.

## CUDA

Run:

```bash
python diagnose_cuda.py
```

The training scripts can fall back to CPU, but CUDA is strongly recommended for full training.

## FFmpeg and audio decoding

FFmpeg must be available on `PATH` for reliable extraction from video containers. PyAV is also installed by the core requirements and is used as a decoder where possible.

## eSpeak-NG

`extract_phonemes.py` uses eSpeak-NG to produce canonical phonemes. The evaluation script also uses it for synthesis. Verify that `espeak-ng` runs from a shell before starting extraction.

## MediaPipe

`crop_lips.py` downloads the Face Landmarker model on first use. Confirm that the input videos are readable and that `s1_lip_crops/` is writable.

## Windows DataLoader workers

Use `--num-workers 0` if worker startup is slow or fails on Windows.

## Transformers compatibility

The active extractor was checked with Transformers 5.2.0 and Hugging Face Hub 1.5.0. The `Wav2Vec2FeatureExtractor`, `Wav2Vec2ForCTC`, and `Wav2Vec2CTCTokenizer` APIs import successfully, and `extract_phonemes.py --help` runs. A full extraction still requires model download, eSpeak-NG, and usable GRID media.

## Tests

Install development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The focused tests cover the GRID filename grammar, viseme similarity and soft-target normalization, Levenshtein distance, TER, and shared speaker/path contracts.
