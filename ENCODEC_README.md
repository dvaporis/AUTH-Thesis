# EnCodec Audio Encoder-Decoder Testing

This module tests Meta's EnCodec audio encoder-decoder for your contrastive learning research.

## Overview

The script applies the following workflow:
1. **Load or Generate Audio**: Either loads an audio file or generates a test signal with multiple frequency components
2. **Encode**: Uses EnCodec to compress audio into a latent representation
3. **Decode**: Reconstructs audio from the latent representation
4. **Error Analysis**: Computes multiple error metrics comparing original and reconstructed audio
5. **Visualization**: Creates plots showing waveforms, errors, and frequency spectra

## Installation

### 1. Activate your virtual environment
```bash
.venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements_encodec.txt
```

Note: This will install PyTorch. If you have GPU support and want CUDA-enabled PyTorch, you may need to adjust the PyTorch installation. See https://pytorch.org for details.

## Usage

### Basic usage (generates test signal):
```bash
python test_encodec_audio.py
```

### With a custom audio file:
```bash
python test_encodec_audio.py --audio path/to/your/audio.wav
```

### With different bandwidth settings:
```bash
python test_encodec_audio.py --bandwidth 24kbps
```

Available bandwidths: `1.5kbps`, `3kbps`, `6kbps`, `12kbps`, `24kbps`

Higher bandwidth = better quality but larger latent representation

## Output

The script creates an `audio_encodec_results/` directory containing:

- `original_audio.wav` - Original audio file
- `reconstructed_audio.wav` - Reconstructed audio after encode-decode
- `encodec_comparison.png` - Visualization with three subplots:
  - Original vs Reconstructed waveforms
  - Reconstruction error over time
  - Frequency spectrum comparison
- `metrics_summary.png` - Bar chart of all error metrics

## Error Metrics Explained

| Metric | Unit | Interpretation |
|--------|------|-----------------|
| **MSE** | - | Mean squared error (lower is better) |
| **MAE** | - | Mean absolute error (lower is better) |
| **RMSE** | - | Root mean squared error (lower is better) |
| **SNR_dB** | dB | Signal-to-noise ratio in decibels (higher is better) |
| **PSNR_dB** | dB | Peak SNR (higher is better, typically 40-50 dB is good) |
| **Cosine_Similarity** | [-1, 1] | Similarity between vectors (closer to 1 is better) |

## Next Steps for Contrastive Learning

After validating EnCodec, you'll want to:

1. **Extract embeddings** at different layers of EnCodec to use as audio representations
2. **Test video encoders** (e.g., ViT, ResNet, or 3D CNN models)
3. **Implement contrastive loss** to align video and audio embeddings in shared latent space
4. **Create dataset pairs** of matching video and audio clips
5. **Train the contrastive model** using InfoNCE or similar loss functions

## Troubleshooting

### CUDA/GPU issues:
- If using GPU, ensure you have the correct PyTorch version for your CUDA version
- Run on CPU: The script automatically falls back to CPU if CUDA unavailable

### EnCodec model download:
- First run will download the model (~100MB) - this may take a moment
- Subsequent runs will use cached model

### Memory issues with long audio:
- The script chunks audio appropriately for EnCodec
- For very long files, consider processing in segments
