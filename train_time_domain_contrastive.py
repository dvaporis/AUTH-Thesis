"""
Time-domain Audio Contrastive Learning with 1D Conv Backbone and NT-Xent Loss.

This script mirrors the MFCC-based implementation but operates on raw
audio chunks in the time domain. It uses a small 1D convolutional
backbone (several Conv1d layers + adaptive pooling) followed by the
same 2-layer MLP projection head used previously.

Usage:
    python train_time_domain_contrastive.py --epochs 50 --batch-size 32 --lr 0.001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional
import random
from dataclasses import dataclass
import argparse
from datetime import datetime
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import av

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for raw audio processing."""
    sample_rate: int = 16000
    chunk_duration: float = 0.5
    # Derived in __post_init__
    def __post_init__(self):
        self.num_samples = int(self.sample_rate * self.chunk_duration)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    temperature: float = 0.07
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    projection_hidden_dim: int = 512
    projection_output_dim: int = 128
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3


class AudioAugmentation:
    """Re-use augmentation approaches but return augmented raw audio.

    This is simplified compared to the MFCC script: augmentations operate
    on the waveform and the dataset will convert to tensors directly.
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        # probabilities and params chosen to match prior script's spirit
        self.gaussian_noise_snr_db = 20.0
        self.brown_noise_snr_db = 20.0
        self.pink_noise_snr_db = 20.0
        self.time_stretch_range = (0.9, 1.1)
        self.pitch_shift_steps_range = (-3, 3)

    def augment(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, torch.Tensor]:
        """Apply exactly one augmentation and return (aug_audio, tensor).

        The returned tensor is a float32 waveform of length `config.num_samples`.
        """
        methods = [
            self._augment_gaussian,
            self._augment_brown,
            self._augment_pink,
            lambda x: self._augment_time_stretch(x),
            lambda x: self._augment_pitch(x, sr),
        ]

        aug_func = random.choice(methods)
        audio_aug = aug_func(audio.copy())

        # Ensure length
        if len(audio_aug) > self.config.num_samples:
            audio_aug = audio_aug[:self.config.num_samples]
        elif len(audio_aug) < self.config.num_samples:
            pad_len = self.config.num_samples - len(audio_aug)
            audio_aug = np.pad(audio_aug, (0, pad_len), mode='constant')

        tensor = torch.from_numpy(audio_aug.astype(np.float32))
        return audio_aug, tensor

    def _augment_gaussian(self, audio: np.ndarray) -> np.ndarray:
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (self.gaussian_noise_snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
        return audio + noise

    def _augment_brown(self, audio: np.ndarray) -> np.ndarray:
        white_noise = np.random.randn(*audio.shape)
        brown_noise = np.cumsum(white_noise, axis=-1)
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(brown_noise ** 2)
        snr_linear = 10 ** (self.brown_noise_snr_db / 10)
        target_noise_power = signal_power / snr_linear
        brown_noise = brown_noise * np.sqrt(target_noise_power / noise_power)
        return audio + brown_noise

    def _augment_pink(self, audio: np.ndarray) -> np.ndarray:
        white_noise = np.random.randn(*audio.shape)
        kernel = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        pink_noise = scipy_signal.lfilter(kernel, 1.0, white_noise, axis=-1)
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(pink_noise ** 2)
        snr_linear = 10 ** (self.pink_noise_snr_db / 10)
        target_noise_power = signal_power / snr_linear
        pink_noise = pink_noise * np.sqrt(target_noise_power / noise_power)
        return audio + pink_noise

    def _augment_time_stretch(self, audio: np.ndarray) -> np.ndarray:
        rate = random.uniform(*self.time_stretch_range)
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched

    def _augment_pitch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        steps = random.randint(*self.pitch_shift_steps_range)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)


class TimeDomainDataset(torch.utils.data.Dataset):
    """Create non-overlapping raw audio chunks from video files."""

    def __init__(self, audio_files: List[Path], config: AudioConfig, augmentation: Optional[AudioAugmentation] = None):
        self.audio_files = audio_files
        self.config = config
        self.augmentation = augmentation
        self.chunks = []
        self._build_chunk_index()

    def _build_chunk_index(self):
        logger.info("Building time-domain chunk index from video files...")
        for file_idx, video_file in enumerate(self.audio_files):
            try:
                audio, sr = extract_audio_from_video(video_file, self.config.sample_rate)
                if audio is None:
                    logger.warning(f"Failed to extract audio from {video_file.name}")
                    continue
                audio_length = len(audio)
                num_chunks = audio_length // self.config.num_samples
                if num_chunks == 0:
                    logger.warning(f"Video file {video_file.name} has audio too short for any chunk")
                    continue
                for chunk_idx in range(num_chunks):
                    start_sample = chunk_idx * self.config.num_samples
                    end_sample = start_sample + self.config.num_samples
                    self.chunks.append({
                        'file_idx': file_idx,
                        'file_path': video_file,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'sample_rate': sr,
                        'chunk_idx': chunk_idx,
                    })
            except Exception as e:
                logger.warning(f"Failed to process {video_file}: {e}")
                continue
        logger.info(f"Built index with {len(self.chunks)} chunks from {len(self.audio_files)} video files")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk_info = self.chunks[idx]
        try:
            video_path = chunk_info['file_path']
            audio, sr = extract_audio_from_video(video_path, self.config.sample_rate)
            if audio is None:
                raise ValueError(f"Failed to extract audio from {video_path.name}")
            start = chunk_info['start_sample']
            end = chunk_info['end_sample']
            audio_chunk = audio[start:end]
            if len(audio_chunk) < self.config.num_samples:
                pad_len = self.config.num_samples - len(audio_chunk)
                audio_chunk = np.pad(audio_chunk, (0, pad_len), mode='constant')
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.mean(axis=1)

            audio_orig_tensor = torch.from_numpy(audio_chunk.astype(np.float32))

            result = {
                'audio_original': audio_orig_tensor,
                'file_idx': chunk_info['file_idx'],
                'chunk_idx': idx,
            }

            if self.augmentation is not None:
                _, aug_tensor = self.augmentation.augment(audio_chunk, sr)
                result['audio_augmented'] = aug_tensor

            return result
        except Exception as e:
            logger.error(f"Error loading chunk {idx}: {e}")
            return {
                'audio_original': torch.zeros(self.config.num_samples, dtype=torch.float32),
                'audio_augmented': torch.zeros(self.config.num_samples, dtype=torch.float32),
                'file_idx': -1,
                'chunk_idx': idx,
            }


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


class TimeDomainConvContrastiveModel(nn.Module):
    """Small 1D conv backbone followed by projection head."""

    def __init__(self, config: TrainingConfig, audio_config: AudioConfig):
        super().__init__()
        self.audio_config = audio_config

        # 1D conv stack
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(256, 512, kernel_size=9, padding=4),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        resnet_output_dim = 512

        self.projection_head = ProjectionHead(
            input_dim=resnet_output_dim,
            hidden_dim=config.projection_hidden_dim,
            output_dim=config.projection_output_dim,
        )

        logger.info(f"Time-domain model initialized: Conv1d stack -> {resnet_output_dim}D -> ProjectionHead")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [batch, num_samples] -> conv expects [batch, 1, num_samples]
        x = audio.unsqueeze(1)
        features = self.conv(x)  # [batch, 512, 1]
        features = features.view(features.size(0), -1)  # [batch, 512]
        projections = self.projection_head(features)
        return projections


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        device = z_i.device
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        logits = sim_matrix
        loss = F.cross_entropy(logits, labels)
        return loss


def extract_audio_from_video(video_path: Path, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int]:
    try:
        container = av.open(str(video_path))
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        if not audio_stream:
            container.close()
            logger.warning(f"No audio stream in {video_path.name}")
            return None, target_sr
        audio_sr = audio_stream.rate
        audio_frames = []
        for frame in container.decode(audio=0):
            audio_data = frame.to_ndarray()
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)
            elif audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T
            audio_frames.append(audio_data)
        container.close()
        if not audio_frames:
            logger.warning(f"No audio frames decoded from {video_path.name}")
            return None, audio_sr
        audio_full = np.concatenate(audio_frames, axis=0)
        if audio_full.dtype == np.int16:
            audio_full = audio_full.astype(np.float32) / 32768.0
        elif audio_full.dtype == np.int32:
            audio_full = audio_full.astype(np.float32) / 2147483648.0
        else:
            audio_full = audio_full.astype(np.float32)
        if audio_full.ndim > 1:
            audio_full = audio_full.mean(axis=1)
        if audio_sr != target_sr:
            num_samples_new = int(audio_full.shape[0] * target_sr / audio_sr)
            audio_resampled = scipy_signal.resample(audio_full, num_samples_new)
            audio_full = audio_resampled.astype(np.float32)
            audio_sr = target_sr
        return audio_full, audio_sr
    except Exception as e:
        logger.warning(f"Failed to extract audio from {video_path.name}: {e}")
        return None, target_sr


def find_ravdess_video_files() -> List[Path]:
    ravdess_path = Path("ravdess_videos_only")
    video_files = []
    if ravdess_path.exists():
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in video_extensions:
            video_files.extend(ravdess_path.rglob(f"*{ext}"))
        if video_files:
            logger.info(f"Found {len(video_files)} video files in {ravdess_path}")
            return video_files
    logger.warning(f"No video files found in {ravdess_path}!")
    return []


def split_dataset(audio_files: List[Path], train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    audio_files_shuffled = audio_files.copy()
    random.shuffle(audio_files_shuffled)
    n = len(audio_files_shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = audio_files_shuffled[:n_train]
    val_files = audio_files_shuffled[n_train:n_train + n_val]
    test_files = audio_files_shuffled[n_train + n_val:]
    logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    return train_files, val_files, test_files


def train_epoch(model: TimeDomainConvContrastiveModel, dataloader: torch.utils.data.DataLoader, criterion: NTXentLoss, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int = 1, total_epochs: int = 1) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch", leave=True)
    for batch in pbar:
        audio_original = batch['audio_original'].to(device)
        audio_augmented = batch['audio_augmented'].to(device)
        file_idx = batch['file_idx'].to(device)
        valid_mask = file_idx >= 0
        if not valid_mask.any():
            continue
        audio_original = audio_original[valid_mask]
        audio_augmented = audio_augmented[valid_mask]
        if audio_original.shape[0] < 2:
            continue
        projections_original = model(audio_original)
        projections_augmented = model(audio_augmented)
        loss = criterion(projections_original, projections_augmented)
        if not torch.isfinite(loss):
            logger.warning("Skipping batch with non-finite loss")
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': total_loss / num_batches})
    if num_batches == 0:
        return {'loss': float('inf')}
    return {'loss': total_loss / num_batches}


@torch.no_grad()
def validate(model: TimeDomainConvContrastiveModel, dataloader: torch.utils.data.DataLoader, criterion: NTXentLoss, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        audio_original = batch['audio_original'].to(device)
        audio_augmented = batch['audio_augmented'].to(device)
        file_idx = batch['file_idx'].to(device)
        valid_mask = file_idx >= 0
        if not valid_mask.any():
            continue
        audio_original = audio_original[valid_mask]
        audio_augmented = audio_augmented[valid_mask]
        if audio_original.shape[0] < 2:
            continue
        projections_original = model(audio_original)
        projections_augmented = model(audio_augmented)
        loss = criterion(projections_original, projections_augmented)
        if not torch.isfinite(loss):
            logger.warning("Skipping batch with non-finite validation loss")
            continue
        total_loss += loss.item()
        num_batches += 1
    if num_batches == 0:
        return {'loss': float('inf')}
    return {'loss': total_loss / num_batches}


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('NT-Xent Loss', fontsize=12)
    ax.set_title('Time-Domain Contrastive Learning - Training Progress', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training history saved to {save_path}")


def main():
    logger.info("="*80)
    logger.info("Time-domain Audio Contrastive Learning (1D conv backbone)")
    logger.info("="*80)
    parser = argparse.ArgumentParser(description='Time-domain contrastive learning')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='time_domain_contrastive_results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    audio_config = AudioConfig()
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
    )

    logger.info(f"Audio config: chunk_duration={audio_config.chunk_duration}s, num_samples={audio_config.num_samples}")
    logger.info(f"Training config: epochs={training_config.num_epochs}, batch_size={training_config.batch_size}, lr={training_config.learning_rate}")

    video_files = find_ravdess_video_files()
    if len(video_files) == 0:
        logger.error("No video files found! Please ensure ravdess_videos_only folder exists with video files.")
        return

    train_files, val_files, test_files = split_dataset(
        video_files,
        train_ratio=training_config.train_ratio,
        val_ratio=training_config.val_ratio,
        test_ratio=training_config.test_ratio,
        seed=args.seed
    )

    augmentation = AudioAugmentation(audio_config)

    train_dataset = TimeDomainDataset(train_files, audio_config, augmentation=augmentation)
    val_dataset = TimeDomainDataset(val_files, audio_config, augmentation=augmentation)
    test_dataset = TimeDomainDataset(test_files, audio_config, augmentation=augmentation)

    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    model = TimeDomainConvContrastiveModel(training_config, audio_config).to(device)
    criterion = NTXentLoss(temperature=training_config.temperature)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    logger.info("="*80)
    logger.info("STARTING TRAINING LOOP")
    logger.info("="*80)
    for epoch in range(training_config.num_epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch+1, total_epochs=training_config.num_epochs)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['loss'])
        logger.info(f"  Epoch {epoch+1:3d}/{training_config.num_epochs} │ Train: {train_metrics['loss']:.4f} │ Val: {val_metrics['loss']:.4f} │ No-improve: {epochs_without_improvement}/{training_config.early_stopping_patience}")
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        if val_metrics['loss'] < (best_val_loss - training_config.early_stopping_min_delta):
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_metrics['loss'], 'config': training_config.__dict__}, output_dir / 'best_model.pt')
            logger.info(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs (best val_loss: {best_val_loss:.4f})")
                break

    logger.info("="*80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*80)
    logger.info(f"Loading best model from: {output_dir / 'best_model.pt'}")
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = validate(model, test_loader, criterion, device)
    logger.info(f"✓ Test Loss: {test_metrics['loss']:.4f}")

    with open(output_dir / 'test_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")

    plot_training_history(history, output_dir / 'training_history.png')

    torch.save({'epoch': training_config.num_epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'history': history, 'config': training_config.__dict__}, output_dir / 'final_model.pt')
    logger.info("✓✓✓ TRAINING COMPLETE ✓✓✓")
    logger.info(f"Results saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
