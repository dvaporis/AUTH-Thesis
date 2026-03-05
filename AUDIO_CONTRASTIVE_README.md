# Audio Contrastive Learning with EnCodec

This implementation provides a complete pipeline for training contrastive learning models on audio data using a frozen EnCodec encoder.

## Features

### Architecture
- **Frozen EnCodec Encoder**: Pre-trained Meta EnCodec (48kHz stereo) for feature extraction
- **Trainable Projection Head**: 3-layer MLP with batch normalization and dropout
- **Overlapping Audio Chunks**: 50% overlap (25,600 samples per chunk, 12,800 sample stride)

### Contrastive Learning Objective

The model learns through two complementary objectives:

1. **Semantic Contrast**: Distinguishes between different videos/speakers
   - Different audio files should have distinct representations
   - Uses file index as semantic labels

2. **Temporal Contrast**: Captures temporal relationships within videos
   - Nearby temporal chunks should be similar
   - Distant chunks should be different
   - Uses temporal position (adjacent chunks as positives)

### Data Augmentation

The following augmentation techniques are implemented:

1. **Additive Controlled Noise**
   - White Gaussian noise (configurable SNR, default: -20 dB)
   - Colored noise (pink and brown noise with 1/f and 1/f² power spectra)

2. **Impulse Noise**
   - Random spike noise (default probability: 10%)
   - Simulates recording artifacts

3. **Time Stretching**
   - Stretches/compresses audio while maintaining output length
   - Range: 0.9x to 1.1x (configurable)
   - Uses resampling to adjust playback speed

4. **Phase Shift**
   - Applies random phase shift in frequency domain
   - Range: -π to π radians
   - Preserves magnitude spectrum

5. **Manifold Mixup**
   - Mixes features in embedding space (not raw audio)
   - Beta distribution parameter: α = 0.3
   - Applied with 30% probability during training

### Dataset Configuration

- **Train/Val/Test Split**: 60% / 20% / 20%
- **Audio Format**: 48kHz sample rate, stereo (2 channels)
- **Chunk Size**: 25,600 samples (~0.533s, equivalent to 16 video frames at 30fps)
- **Overlap**: 50% (12,800 samples stride)

## Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements_encodec.txt
```

Requirements:
- torch >= 2.0.0
- torchaudio >= 2.0.0
- audiocraft >= 1.0.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0

## Dataset Preparation

First, download the Kaggle audio-visual dataset:

```bash
python download_kaggle_dataset.py
```

This will download the Audio-Visual Database of Emotional Speech and Song dataset containing .mp4 video files with embedded audio.

## Usage

### Basic Training

```bash
python train_audio_contrastive.py --epochs 50 --batch-size 32 --lr 0.001
```

### Advanced Options

```bash
python train_audio_contrastive.py \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --temperature 0.07 \
    --seed 42 \
    --output-dir audio_contrastive_results
```

### Disable Augmentation

```bash
python train_audio_contrastive.py --no-augmentation
```

### Disable Manifold Mixup

```bash
python train_audio_contrastive.py --no-manifold-mixup
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 50 | Number of training epochs |
| `--batch-size` | int | 32 | Batch size for training |
| `--lr` | float | 0.001 | Learning rate |
| `--temperature` | float | 0.07 | Temperature for contrastive loss |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--output-dir` | str | audio_contrastive_results | Output directory for results |
| `--no-augmentation` | flag | False | Disable data augmentation |
| `--no-manifold-mixup` | flag | False | Disable manifold mixup |

## Output

The training script creates the following outputs in the `output-dir`:

1. **best_model.pt**: Checkpoint with best validation loss
   - Model state dict
   - Optimizer state dict
   - Training configuration
   - Best validation loss value

2. **final_model.pt**: Final model after all epochs
   - Complete training history
   - Final model state

3. **training_history.png**: Visualization of training progress
   - Total loss curves (train/val)
   - Semantic loss curves (train/val)
   - Temporal loss curves (train/val)

4. **test_metrics.txt**: Final evaluation metrics on test set
   - Test loss
   - Semantic loss component
   - Temporal loss component

## Configuration Details

### AudioConfig
```python
sample_rate: 48000          # EnCodec 48kHz stereo
num_samples: 25600          # ~16 video frames
overlapfloat = 0.5                # 50% overlap
stride: 12800               # samples (50% of num_samples)
encodec_bandwidth: 6.0      # kbps
```

### TrainingConfig
```python
batch_size: 32
num_epochs: 50
learning_rate: 0.001
weight_decay: 1e-4
temperature: 0.07           # Contrastive loss temperature

# Loss weights
semantic_weight: 1.0
temporal_weight: 0.5

# Projection head
projection_dim: 128
hidden_dim: 512
```

### Augmentation Parameters
```python
noise_level_db: 20.0        # SNR for additive noise (signal 100x stronger)
impulse_prob: 0.1           # Impulse noise probability
impulse_amplitude: 0.5      # Max amplitude for impulse noise
stretch_range: (0.9, 1.1)   # Time stretching range
phase_shift_range: (-π, π)  # Phase shift range (radians)
mixup_alpha: 0.3            # Manifold mixup parameter
```

## Architecture Details

### Projection Head

```
Input: EnCodec embeddings (varies based on encoder output)
↓
Linear(input_dim → 512) + BatchNorm + ReLU + Dropout(0.1)
↓
Linear(512 → 512) + BatchNorm + ReLU + Dropout(0.1)
↓
Linear(512 → 128)
↓
L2 Normalization
↓
Output: 128-dim normalized embeddings
```

### Loss Function

The total loss is a weighted combination:

```
L_total = w_semantic * L_semantic + w_temporal * L_temporal
```

Where:
- **L_semantic**: InfoNCE loss using file indices (different files = negatives)
- **L_temporal**: InfoNCE loss using temporal proximity (adjacent chunks = positives)
- Default weights: w_semantic = 1.0, w_temporal = 0.5

### InfoNCE Loss

For each sample i:
```
L_i = -log(Σ_j∈P exp(sim(i,j)/τ) / (Σ_j∈P exp(sim(i,j)/τ) + Σ_k∈N exp(sim(i,k)/τ)))
```

Where:
- P: positive pairs
- N: negative pairs
- τ: temperature (default: 0.07)
- sim: cosine similarity

## Training Tips

1. **Batch Size**: Larger batches provide more negative samples for contrastive learning
   - Recommended: 64-128 if GPU memory allows
   - Minimum: 16 for meaningful contrastive learning

2. **Temperature**: Lower values make the model more discriminative
   - Too low (< 0.01): May cause training instability
   - Too high (> 0.5): Model may not learn effectively
   - Recommended range: 0.05 - 0.1

3. **Learning Rate**: 
   - Start with 0.001 for Adam optimizer
   - Uses cosine annealing schedule
   - Will gradually decrease to 0 by final epoch

4. **Augmentation**:
   - During training, 1-3 random augmentations are applied per sample
   - Validation/test sets use no augmentation
   - Manifold mixup applied to 30% of batches

5. **GPU Memory**:
   - EnCodec encoder is frozen (no gradient computation)
   - Only projection head requires gradients
   - Can use larger batch sizes than end-to-end training

## Troubleshooting

### No Audio Files Found

```
ERROR: No audio/video files found! Please run download_kaggle_dataset.py first.
```

**Solution**: Run the dataset download script:
```bash
python download_kaggle_dataset.py
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size:
```bash
python train_audio_contrastive.py --batch-size 16
```

### Low Training Performance

If validation loss plateaus early:
1. Increase temperature (e.g., `--temperature 0.1`)
2. Adjust loss weights in code (increase semantic_weight)
3. Increase batch size for more negatives
4. Enable all augmentations

## Future Extensions

Potential improvements:
1. **Hard Negative Mining**: Focus on difficult negative pairs
2. **Supervised Contrastive Loss**: Use emotion labels from dataset
3. **Multi-view Contrastive Learning**: Combine with video features
4. **Momentum Encoder**: Implement MoCo-style momentum encoder
5. **Larger Projection Dimensions**: Try 256 or 512-dim projections

## References

- EnCodec: Meta's neural audio codec ([paper](https://arxiv.org/abs/2210.13438))
- SimCLR: Framework for contrastive learning ([paper](https://arxiv.org/abs/2002.05709))
- MoCo: Momentum contrast ([paper](https://arxiv.org/abs/1911.05722))
- Manifold Mixup: ([paper](https://arxiv.org/abs/1806.05236))

## Citation

If you use this code, please cite the original EnCodec paper:

```bibtex
@article{defossez2022encodec,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```
