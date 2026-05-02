"""
MFCC-based Audio Contrastive Learning with ResNet18 and NT-Xent Loss.

This script implements contrastive learning for audio using:
1. MFCC (Mel-Frequency Cepstral Coefficients) extraction from non-overlapping 0.5s chunks
2. ResNet18 backbone (pre-trained, fine-tunable) for feature extraction
3. 2-layer MLP projection head (input → 512 → 128)
4. NT-Xent loss with positive pairs (original + augmented) vs negatives (all others)
5. Rich data augmentation: Gaussian/pink/brown noise, time-stretching, pitch-shifting, frequency masking
6. RAVDESS Kaggle dataset

Architecture:
    Audio chunk (0.5s) → MFCC features [13, time_steps] → ResNet18 → Feature vector → 
    Projection Head (512 → 128) → NT-Xent Loss

Usage:
    python train_mfcc_contrastive.py --epochs 50 --batch-size 32 --lr 0.001
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
from torchvision import models

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for MFCC-based audio processing."""
    sample_rate: int = 16000  # Standard SR for speech
    chunk_duration: float = 0.5  # 0.5 second chunks (non-overlapping)
    num_mfcc: int = 13  # MFCC coefficients
    n_fft: int = 400  # ~25ms at 16kHz
    hop_length: int = 160  # ~10ms at 16kHz
    f_min: float = 80.0
    f_max: float = 7600.0
    
    # Augmentation parameters
    gaussian_noise_prob: float = 0.5
    gaussian_noise_snr_db: float = 20.0
    
    brown_noise_prob: float = 0.3
    brown_noise_snr_db: float = 20.0
    
    pink_noise_prob: float = 0.3
    pink_noise_snr_db: float = 20.0
    
    time_stretch_prob: float = 0.3
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    
    pitch_shift_prob: float = 0.3
    pitch_shift_steps_range: Tuple[int, int] = (-3, 3)  # semitones
    
    freq_mask_prob: float = 0.3
    freq_mask_max_width: float = 0.3  # as fraction of freq bins
    
    def __post_init__(self):
        """Compute derived parameters."""
        self.num_samples = int(self.sample_rate * self.chunk_duration)
        # Compute number of MFCC frames in a chunk
        self.num_frames = 1 + (self.num_samples - self.n_fft) // self.hop_length


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    temperature: float = 0.07  # NT-Xent temperature
    
    # Data split
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Model
    finetune_resnet: bool = True  # Allow ResNet18 fine-tuning
    projection_hidden_dim: int = 512
    projection_output_dim: int = 128
    
    # Convergence control
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3


class AudioAugmentation:
    """Data augmentation for audio signals."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        

    
    def frequency_mask(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to MFCC features.
        
        Args:
            mfcc: [num_mfcc, time_steps]
            
        Returns:
            Masked MFCC
        """
        if random.random() > self.config.freq_mask_prob:
            return mfcc
        
        num_mfcc = mfcc.shape[0]
        max_mask_width = max(1, int(num_mfcc * self.config.freq_mask_max_width))
        mask_width = random.randint(1, max_mask_width)
        mask_start = random.randint(0, max(0, num_mfcc - mask_width))
        
        mfcc_masked = mfcc.clone()
        mfcc_masked[mask_start:mask_start + mask_width, :] = 0
        
        return mfcc_masked
    
    def augment(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Apply exactly ONE augmentation to raw audio and extract MFCC.
        
        Augmentation strategy:
        - Select exactly ONE of 5 augmentation methods uniformly (20% probability each):
          * Gaussian noise
          * Brown noise
          * Pink noise
          * Time-stretching
          * Pitch-shifting
        
        This ensures clear positive pairs: (original MFCC, one-specific-augmentation MFCC)
        
        Args:
            audio: [num_samples] raw audio
            sr: sample rate
            
        Returns:
            Tuple of (augmented_audio, augmented_mfcc_tensor)
        """
        # Select exactly ONE augmentation method uniformly (each has 20% probability)
        augmentation_methods = [
            ('gaussian_noise', self._augment_gaussian),
            ('brown_noise', self._augment_brown),
            ('pink_noise', self._augment_pink),
            ('time_stretch', self._augment_time_stretch),
            ('pitch_shift', lambda x: self._augment_pitch(x, sr)),
        ]
        
        method_name, aug_func = random.choice(augmentation_methods)
        audio_aug = aug_func(audio.copy())
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_aug,
            sr=sr,
            n_mfcc=self.config.num_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            fmin=self.config.f_min,
            fmax=self.config.f_max,
        )
        
        # Apply frequency masking to MFCC (30% probability, independent of audio augmentation)
        mfcc_tensor = torch.from_numpy(mfcc).float()
        mfcc_tensor = self.frequency_mask(mfcc_tensor)
        
        return audio_aug, mfcc_tensor
    
    def _augment_gaussian(self, audio: np.ndarray) -> np.ndarray:
        """Helper: Apply Gaussian noise."""
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (self.config.gaussian_noise_snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
        return audio + noise
    
    def _augment_brown(self, audio: np.ndarray) -> np.ndarray:
        """Helper: Apply brown noise."""
        white_noise = np.random.randn(*audio.shape)
        brown_noise = np.cumsum(white_noise, axis=-1)
        
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(brown_noise ** 2)
        snr_linear = 10 ** (self.config.brown_noise_snr_db / 10)
        target_noise_power = signal_power / snr_linear
        
        brown_noise = brown_noise * np.sqrt(target_noise_power / noise_power)
        return audio + brown_noise
    
    def _augment_pink(self, audio: np.ndarray) -> np.ndarray:
        """Helper: Apply pink noise."""
        white_noise = np.random.randn(*audio.shape)
        kernel = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        pink_noise = scipy_signal.lfilter(kernel, 1.0, white_noise, axis=-1)
        
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(pink_noise ** 2)
        snr_linear = 10 ** (self.config.pink_noise_snr_db / 10)
        target_noise_power = signal_power / snr_linear
        
        pink_noise = pink_noise * np.sqrt(target_noise_power / noise_power)
        return audio + pink_noise
    
    def _augment_time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Helper: Apply time-stretching."""
        rate = random.uniform(*self.config.time_stretch_range)
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        elif len(stretched) < len(audio):
            pad_len = len(audio) - len(stretched)
            stretched = np.pad(stretched, (0, pad_len), mode='edge')
        
        return stretched
    
    def _augment_pitch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Helper: Apply pitch shifting."""
        steps = random.randint(*self.config.pitch_shift_steps_range)
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        return shifted


class MFCCDataset(torch.utils.data.Dataset):
    """
    Dataset for non-overlapping MFCC chunks from RAVDESS dataset.
    """
    
    def __init__(
        self,
        audio_files: List[Path],
        config: AudioConfig,
        augmentation: Optional[AudioAugmentation] = None,
        return_original: bool = False,
    ):
        """
        Args:
            audio_files: List of audio file paths
            config: Audio configuration
            augmentation: AudioAugmentation instance (optional)
            return_original: If True, return both original and augmented MFCCs
        """
        self.audio_files = audio_files
        self.config = config
        self.augmentation = augmentation
        self.return_original = return_original
        
        # Build index of all chunks
        self.chunks = []
        self._build_chunk_index()
    
    def _build_chunk_index(self):
        """Build index of all non-overlapping audio chunks extracted from videos."""
        logger.info("Building MFCC chunk index from video files...")
        
        for file_idx, video_file in enumerate(self.audio_files):
            try:
                # Extract audio from video
                audio, sr = extract_audio_from_video(video_file, self.config.sample_rate)
                
                if audio is None:
                    logger.warning(f"Failed to extract audio from {video_file.name}")
                    continue
                
                audio_length = len(audio)
                
                # Compute chunk boundaries (non-overlapping)
                num_chunks = audio_length // self.config.num_samples
                
                if num_chunks == 0:
                    logger.warning(f"Video file {video_file.name} has audio too short for any chunk")
                    continue
                
                # Add chunks to index
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
        """
        Get a chunk and return MFCC features.
        
        Returns:
            dict with keys:
                - 'mfcc_original': [num_mfcc, time_steps]
                - 'mfcc_augmented': [num_mfcc, time_steps] (if augmentation available)
                - 'file_idx': file identifier
                - 'chunk_idx': chunk index
        """
        chunk_info = self.chunks[idx]
        
        try:
            # Extract audio from video
            video_path = chunk_info['file_path']
            audio, sr = extract_audio_from_video(video_path, self.config.sample_rate)
            
            if audio is None:
                raise ValueError(f"Failed to extract audio from {video_path.name}")
            
            # Extract chunk
            start_sample = chunk_info['start_sample']
            end_sample = chunk_info['end_sample']
            audio_chunk = audio[start_sample:end_sample]
            
            # Pad if necessary
            if len(audio_chunk) < self.config.num_samples:
                pad_len = self.config.num_samples - len(audio_chunk)
                audio_chunk = np.pad(audio_chunk, (0, pad_len), mode='constant')
            
            # Ensure audio_chunk is 1D (mono)
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.mean(axis=1)
            
            # Extract MFCC from original
            mfcc_original = librosa.feature.mfcc(
                y=audio_chunk,
                sr=sr,
                n_mfcc=self.config.num_mfcc,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                fmin=self.config.f_min,
                fmax=self.config.f_max,
            )
            mfcc_original = torch.from_numpy(mfcc_original).float()
            
            result = {
                'mfcc_original': mfcc_original,
                'file_idx': chunk_info['file_idx'],
                'chunk_idx': idx,
            }
            
            # Apply augmentation if available
            if self.augmentation is not None:
                _, mfcc_augmented = self.augmentation.augment(audio_chunk, sr)
                result['mfcc_augmented'] = mfcc_augmented
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading chunk {idx}: {e}")
            # Return zero tensors as fallback
            return {
                'mfcc_original': torch.zeros(self.config.num_mfcc, self.config.num_frames),
                'mfcc_augmented': torch.zeros(self.config.num_mfcc, self.config.num_frames),
                'file_idx': -1,
                'chunk_idx': idx,
            }


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning."""
    
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
        """Return normalized projection."""
        return F.normalize(self.projection(x), dim=1)


class MFCCResNetContrastiveModel(nn.Module):
    """
    MFCC-based contrastive learning model with ResNet18 backbone.
    
    Architecture:
        MFCC [num_mfcc, time_steps] → Reshape to image-like → 
        ResNet18 backbone → Feature vector → Projection head → Normalized output
    """
    
    def __init__(self, config: TrainingConfig, audio_config: AudioConfig):
        super().__init__()
        
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer to accept 1-channel input (instead of 3)
        # MFCC will be treated as [batch, 1, num_mfcc, time_steps]
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # Copy weights averaged across channels for initialization
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Option to freeze ResNet backbone initially
        if not config.finetune_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
            logger.info("ResNet18 backbone frozen (no fine-tuning)")
        else:
            logger.info("ResNet18 backbone fine-tuning enabled")
        
        # Get ResNet output dimension (typically 512 for ResNet18)
        resnet_output_dim = self.resnet.fc.in_features
        
        # Remove ResNet's classification head and add projection head
        self.resnet.fc = nn.Identity()
        
        self.projection_head = ProjectionHead(
            input_dim=resnet_output_dim,
            hidden_dim=config.projection_hidden_dim,
            output_dim=config.projection_output_dim
        )
        
        logger.info(
            f"Model initialized: ResNet18 → {resnet_output_dim}D → "
            f"ProjectionHead(hidden={config.projection_hidden_dim}, output={config.projection_output_dim})"
        )
    
    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            mfcc: [batch, num_mfcc, time_steps] MFCC features
            
        Returns:
            projections: [batch, projection_output_dim] normalized projections
        """
        # Add channel dimension for ResNet: [batch, 1, num_mfcc, time_steps]
        mfcc_reshaped = mfcc.unsqueeze(1)
        
        # Pass through ResNet
        features = self.resnet(mfcc_reshaped)  # [batch, 512]
        
        # Project to contrastive space
        projections = self.projection_head(features)  # [batch, 128]
        
        return projections


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.
    
    For batch with augmented pairs [view1, view2], encourages:
    - Each sample's two views to be similar (positive pair)
    - Each sample to be dissimilar from all other samples (negative pairs)
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for two views.
        
        Args:
            z_i: [batch, dim] normalized projections of original
            z_j: [batch, dim] normalized projections of augmented
            
        Returns:
            loss: scalar
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Concatenate views: [2*batch, dim]
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute cosine similarity matrix [2*batch, 2*batch]
        # z is already normalized, so dot product = cosine similarity
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create labels for positive pairs
        # Positive pairs: (i, batch+i) and (batch+i, i)
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Compute cross-entropy loss
        logits = sim_matrix
        loss = F.cross_entropy(logits, labels)
        
        return loss


def extract_audio_from_video(video_path: Path, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int]:
    """
    Extract audio from a video file using PyAV.
    
    Args:
        video_path: Path to video file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        container = av.open(str(video_path))
        
        # Get audio stream
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        if not audio_stream:
            container.close()
            logger.warning(f"No audio stream in {video_path.name}")
            return None, target_sr
        
        audio_sr = audio_stream.rate
        
        # Decode all audio frames
        audio_frames = []
        for frame in container.decode(audio=0):
            audio_data = frame.to_ndarray()
            
            # Ensure [samples, channels] format
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)
            elif audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T
            
            audio_frames.append(audio_data)
        
        container.close()
        
        if not audio_frames:
            logger.warning(f"No audio frames decoded from {video_path.name}")
            return None, audio_sr
        
        # Concatenate all frames
        audio_full = np.concatenate(audio_frames, axis=0)
        
        # Normalize to float32
        if audio_full.dtype == np.int16:
            audio_full = audio_full.astype(np.float32) / 32768.0
        elif audio_full.dtype == np.int32:
            audio_full = audio_full.astype(np.float32) / 2147483648.0
        else:
            audio_full = audio_full.astype(np.float32)
        
        # Convert to mono if multi-channel
        if audio_full.ndim > 1:
            audio_full = audio_full.mean(axis=1)
        
        # Resample if needed
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
    """Find video files from RAVDESS dataset in ravdess_videos_only folder."""
    ravdess_path = Path("ravdess_videos_only")
    
    video_files = []
    
    if ravdess_path.exists():
        # Find video files (.mp4, .avi, .mov, .mkv)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in video_extensions:
            video_files.extend(ravdess_path.rglob(f"*{ext}"))
        
        if video_files:
            logger.info(f"Found {len(video_files)} video files in {ravdess_path}")
            return video_files
    
    logger.warning(f"No video files found in {ravdess_path}!")
    return []


def split_dataset(
    audio_files: List[Path],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split audio files into train/val/test sets."""
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


def train_epoch(
    model: MFCCResNetContrastiveModel,
    dataloader: torch.utils.data.DataLoader,
    criterion: NTXentLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch", leave=True)
    
    for batch in pbar:
        mfcc_original = batch['mfcc_original'].to(device)
        mfcc_augmented = batch['mfcc_augmented'].to(device)
        file_idx = batch['file_idx'].to(device)
        
        # Filter out invalid samples
        valid_mask = file_idx >= 0
        if not valid_mask.any():
            continue
        
        mfcc_original = mfcc_original[valid_mask]
        mfcc_augmented = mfcc_augmented[valid_mask]
        
        if mfcc_original.shape[0] < 2:
            continue
        
        # Forward pass for both views
        projections_original = model(mfcc_original)
        projections_augmented = model(mfcc_augmented)
        
        # Compute NT-Xent loss
        loss = criterion(projections_original, projections_augmented)
        
        # Skip numerically invalid batches
        if not torch.isfinite(loss):
            logger.warning("Skipping batch with non-finite loss")
            continue
        
        # Backward pass
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
def validate(
    model: MFCCResNetContrastiveModel,
    dataloader: torch.utils.data.DataLoader,
    criterion: NTXentLoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        mfcc_original = batch['mfcc_original'].to(device)
        mfcc_augmented = batch['mfcc_augmented'].to(device)
        file_idx = batch['file_idx'].to(device)
        
        valid_mask = file_idx >= 0
        if not valid_mask.any():
            continue
        
        mfcc_original = mfcc_original[valid_mask]
        mfcc_augmented = mfcc_augmented[valid_mask]
        
        if mfcc_original.shape[0] < 2:
            continue
        
        # Forward pass
        projections_original = model(mfcc_original)
        projections_augmented = model(mfcc_augmented)
        
        # Compute loss
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
    """Plot and save training history."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('NT-Xent Loss', fontsize=12)
    ax.set_title('MFCC Contrastive Learning - Training Progress', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training history saved to {save_path}")


def main():
    logger.info("="*80)
    logger.info("MFCC-based Audio Contrastive Learning with ResNet18")
    logger.info("="*80)
    
    parser = argparse.ArgumentParser(description='MFCC-based contrastive learning with ResNet18')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help='NT-Xent temperature')
    parser.add_argument('--finetune-resnet', action='store_true', default=True,
                        help='Allow ResNet18 fine-tuning')
    parser.add_argument('--freeze-resnet', action='store_true',
                        help='Freeze ResNet18 backbone')
    parser.add_argument('--early-stopping-patience', type=int, default=8,
                        help='Early stopping patience')
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-3,
                        help='Minimum improvement for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='mfcc_contrastive_results',
                        help='Output directory')
    
    args = parser.parse_args()
    logger.info(f"[1/9] Configuration loaded")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.info(f"[2/9] Random seeds set (seed={args.seed})")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[3/9] Device initialized: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"[4/9] Output directory created: {output_dir.absolute()}")
    
    # Configurations
    audio_config = AudioConfig()
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        finetune_resnet=not args.freeze_resnet,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    
    logger.info(f"Audio config: chunk_duration={audio_config.chunk_duration}s, "
                f"num_mfcc={audio_config.num_mfcc}, num_frames={audio_config.num_frames}")
    logger.info(f"Training config: epochs={training_config.num_epochs}, batch_size={training_config.batch_size}, "
                f"lr={training_config.learning_rate}, temperature={training_config.temperature}")
    logger.info(f"Early stopping: patience={training_config.early_stopping_patience}, "
                f"min_delta={training_config.early_stopping_min_delta}")
    
    # Find video files
    logger.info("[5a/9] Searching for RAVDESS video files in ravdess_videos_only...")
    video_files = find_ravdess_video_files()
    
    if len(video_files) == 0:
        logger.error("No video files found! Please ensure ravdess_videos_only folder exists with video files.")
        return
    
    logger.info(f"[5b/9] Found {len(video_files)} total video files")
    
    # Split dataset
    logger.info("[5c/9] Splitting dataset into train/val/test...")
    train_files, val_files, test_files = split_dataset(
        video_files,
        train_ratio=training_config.train_ratio,
        val_ratio=training_config.val_ratio,
        test_ratio=training_config.test_ratio,
        seed=args.seed
    )
    
    # Create augmentation
    logger.info("[5d/9] Initializing augmentation pipeline...")
    augmentation = AudioAugmentation(audio_config)
    
    # Create datasets
    logger.info("[5e/9] Creating MFCC datasets and chunk index...")
    train_dataset = MFCCDataset(train_files, audio_config, augmentation=augmentation)
    val_dataset = MFCCDataset(val_files, audio_config, augmentation=augmentation)
    test_dataset = MFCCDataset(test_files, audio_config, augmentation=augmentation)
    
    logger.info(f"[5f/9] Dataset sizes - Train: {len(train_dataset)} chunks, Val: {len(val_dataset)} chunks, Test: {len(test_dataset)} chunks")
    
    # Create dataloaders
    logger.info("[5g/9] Creating data loaders...")
    num_workers = 0  # Keep 0 for Windows stability
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    logger.info(f"Data loaders ready: {len(train_loader)} train batches, "
                f"{len(val_loader)} val batches, {len(test_loader)} test batches")
    
    # Create model
    logger.info("[6/9] Creating ResNet18 model with projection head...")
    model = MFCCResNetContrastiveModel(training_config, audio_config).to(device)
    
    # Create loss and optimizer
    logger.info("[7/9] Setting up loss function and optimizer...")
    criterion = NTXentLoss(temperature=training_config.temperature)
    logger.info(f"Loss function: NT-Xent (temperature={training_config.temperature})")
    
    # Only optimize projection head if ResNet is frozen, otherwise optimize all
    if not training_config.finetune_resnet:
        optimizer = torch.optim.Adam(
            model.projection_head.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        logger.info("Optimizer: Adam (projection head only)")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        logger.info("Optimizer: Adam (all parameters including ResNet fine-tuning)")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    logger.info("Learning rate scheduler: ReduceLROnPlateau")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Training loop
    logger.info("="*80)
    logger.info("[8/9] STARTING TRAINING LOOP")
    logger.info("="*80)
    for epoch in range(training_config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch + 1, total_epochs=training_config.num_epochs
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        logger.info(f"  Epoch {epoch+1:3d}/{training_config.num_epochs} │ "
                   f"Train: {train_metrics['loss']:.4f} │ Val: {val_metrics['loss']:.4f} │ "
                   f"No-improve: {epochs_without_improvement}/{training_config.early_stopping_patience}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < (best_val_loss - training_config.early_stopping_min_delta):
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': training_config.__dict__,
            }, output_dir / 'best_model.pt')
            logger.info(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(best val_loss: {best_val_loss:.4f})"
                )
                break
    
    # Test on best model
    logger.info("="*80)
    logger.info("[9/9] EVALUATING ON TEST SET")
    logger.info("="*80)
    logger.info(f"Loading best model from: {output_dir / 'best_model.pt'}")
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Evaluating on {len(test_loader)} test batches...")
    test_metrics = validate(model, test_loader, criterion, device)
    logger.info(f"✓ Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Best Val Loss: {best_val_loss:.4f}")
    
    # Save test metrics
    logger.info("Saving test metrics...")
    with open(output_dir / 'test_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
    logger.info(f"✓ Test metrics saved to: {output_dir / 'test_metrics.txt'}")
    
    # Plot training history
    logger.info("Generating training history plot...")
    plot_training_history(history, output_dir / 'training_history.png')
    logger.info(f"✓ Training plot saved to: {output_dir / 'training_history.png'}")
    
    # Save final model
    logger.info("Saving final model...")
    torch.save({
        'epoch': training_config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': training_config.__dict__,
    }, output_dir / 'final_model.pt')
    logger.info(f"✓ Final model saved to: {output_dir / 'final_model.pt'}")
    
    logger.info("="*80)
    logger.info("✓✓✓ TRAINING COMPLETE ✓✓✓")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir.absolute()}")
    logger.info(f"Files created:")
    logger.info(f"  - best_model.pt (best validation checkpoint)")
    logger.info(f"  - final_model.pt (final checkpoint)")
    logger.info(f"  - test_metrics.txt (test performance)")
    logger.info(f"  - training_history.png (loss curves)")
    logger.info("="*80)


if __name__ == '__main__':
    main()
