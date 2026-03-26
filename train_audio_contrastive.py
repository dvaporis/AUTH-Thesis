"""
Audio Contrastive Learning with EnCodec.

This script implements contrastive learning for audio using:
1. Frozen EnCodec encoder for feature extraction
2. Trainable classifier head
3. Contrastive loss with semantic and temporal components
4. Fast data augmentation: white/colored noise, impulse noise
   (Time stretching and phase shift removed - too slow for training)
5. 50% overlapping audio chunks (equivalent to 16 video frames)
6. 60-20-20 train/val/test split

Usage:
    python train_audio_contrastive.py --epochs 50 --batch-size 64 --lr 0.001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional
import random
from dataclasses import dataclass
import argparse
from datetime import datetime
from scipy.io import wavfile
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import av

# Suppress scipy WAV file warnings
warnings.filterwarnings('ignore', category=wavfile.WavFileWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 48000  # EnCodec 48kHz stereo
    num_video_frames: int = 16  # target bite length in video frames
    reference_video_fps: float = 30000 / 1001  # NTSC-like fps used by many videos (~29.97)
    num_samples: Optional[int] = None  # derived from frames/fps if None
    overlap: float = 0.5  # 50% overlap
    stride: Optional[int] = None  # derived from overlap if None
    num_channels: int = 2  # stereo

    # EnCodec config
    encodec_bandwidth: float = 6.0  # kbps

    # Augmentation parameters
    noise_level_db: float = 20.0  # SNR for additive noise (signal 100x stronger than noise)
    impulse_prob: float = 0.1
    impulse_amplitude: float = 0.5  # Max amplitude for impulse noise
    stretch_range: Tuple[float, float] = (0.9, 1.1)  # Time stretching range
    phase_shift_range: Tuple[float, float] = (-np.pi, np.pi)
    mixup_alpha: float = 0.3  # For manifold mixup

    def __post_init__(self):
        self.update_derived_params()

    def update_derived_params(self):
        """Recompute derived timing parameters from frames/fps settings."""
        if self.num_samples is None:
            self.num_samples = int(round(self.sample_rate * self.num_video_frames / self.reference_video_fps))

        # Keep overlap-driven chunking valid and deterministic.
        self.overlap = float(np.clip(self.overlap, 0.0, 0.99))

        if self.stride is None:
            self.stride = int(round(self.num_samples * (1.0 - self.overlap)))
        self.stride = max(1, int(self.stride))

    @property
    def chunk_duration_seconds(self) -> float:
        """Audio segment duration in seconds (target is about 0.534s for 16/30 fps)."""
        return self.num_samples / float(self.sample_rate)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    temperature: float = 0.07  # Contrastive loss temperature
    
    # Loss weights
    semantic_weight: float = 1.0
    temporal_weight: float = 0.5
    temporal_window: int = 2  # Chunks within this distance in the same file are positives
    acoustic_top_k: int = 2  # Extra cross-file positives per sample from acoustic neighbors
    acoustic_sim_threshold: float = 0.30  # Minimum cosine similarity for acoustic positives
    
    # Data split
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Classifier head
    projection_dim: int = 128
    hidden_dim: int = 512

    # Convergence control
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3


class AudioAugmentation:
    """Data augmentation for audio signals."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
    def add_white_noise(self, audio: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        """Add white Gaussian noise at specified SNR."""
        if snr_db is None:
            snr_db = self.config.noise_level_db
            
        signal_power = torch.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        return audio + noise
    
    def add_colored_noise(self, audio: torch.Tensor, color: str = 'pink', snr_db: float = None) -> torch.Tensor:
        """
        Add colored noise (pink or brown).
        Pink noise: 1/f power spectrum
        Brown noise: 1/f^2 power spectrum
        """
        if snr_db is None:
            snr_db = self.config.noise_level_db
        
        # Generate white noise
        white_noise = torch.randn_like(audio)
        
        # Apply coloring filter in frequency domain
        if color == 'pink':
            # Pink noise filter (approximate)
            kernel = torch.tensor([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        elif color == 'brown':
            # Brown noise filter (approximate)
            kernel = torch.tensor([0.02, -0.06, 0.06, -0.02])
        else:
            kernel = torch.tensor([1.0])  # White noise
        
        kernel = kernel.to(audio.device)
        
        # Apply filter
        if len(kernel) > 1:
            colored_noise = F.conv1d(
                white_noise.unsqueeze(1), 
                kernel.view(1, 1, -1),
                padding=len(kernel)//2
            ).squeeze(1)
            # Ensure same length as original
            if colored_noise.shape[-1] != audio.shape[-1]:
                colored_noise = colored_noise[..., :audio.shape[-1]]
        else:
            colored_noise = white_noise
        
        # Adjust SNR
        signal_power = torch.mean(audio ** 2)
        noise_power_current = torch.mean(colored_noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear
        
        colored_noise = colored_noise * torch.sqrt(target_noise_power / noise_power_current)
        return audio + colored_noise
    
    def add_impulse_noise(self, audio: torch.Tensor, prob: float = None) -> torch.Tensor:
        """Add random impulse (spike) noise."""
        if prob is None:
            prob = self.config.impulse_prob
        
        mask = torch.rand_like(audio) < prob
        impulses = torch.randn_like(audio) * self.config.impulse_amplitude  # Controlled amplitude
        return audio + mask.float() * impulses
    
    def time_stretch(self, audio: torch.Tensor, rate: float = None) -> torch.Tensor:
        """
        Time stretching while maintaining the same output length.
        Uses resampling to achieve stretching effect.
        """
        if rate is None:
            rate = random.uniform(*self.config.stretch_range)
        
        # Convert to numpy for scipy processing
        audio_np = audio.cpu().numpy()
        
        # Resample to stretch/compress
        original_length = audio_np.shape[-1]
        stretched_length = int(original_length / rate)
        
        # Resample
        resampled = scipy_signal.resample(audio_np, stretched_length, axis=-1)
        
        # Crop or pad to original length
        if stretched_length > original_length:
            # Crop
            resampled = resampled[..., :original_length]
        else:
            # Stretch should not inject silence: pad with edge values instead of zeros.
            pad_length = original_length - stretched_length
            resampled = np.pad(resampled, ((0, 0), (0, pad_length)), mode='edge')
        
        return torch.from_numpy(resampled).to(audio.device)

    def time_squeeze(self, audio: torch.Tensor, squeeze_factor: float = None) -> torch.Tensor:
        """
        Squeeze audio in time, then place it in the center with zeros on both sides.

        Args:
            audio: Audio tensor [channels, samples]
            squeeze_factor: Compression factor in (0, 1). If None, sampled randomly.

        Returns:
            Audio tensor with original length and centered zero-padding.
        """
        if squeeze_factor is None:
            squeeze_factor = random.uniform(0.6, 0.95)

        squeeze_factor = float(np.clip(squeeze_factor, 0.05, 0.99))

        audio_np = audio.cpu().numpy()
        original_length = audio_np.shape[-1]
        squeezed_length = max(1, int(round(original_length * squeeze_factor)))

        squeezed = scipy_signal.resample(audio_np, squeezed_length, axis=-1)

        pad_length = original_length - squeezed_length
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        squeezed_padded = np.pad(squeezed, ((0, 0), (left_pad, right_pad)), mode='constant')

        return torch.from_numpy(squeezed_padded).to(audio.device)
    
    def phase_shift(self, audio: torch.Tensor, shift: float = None) -> torch.Tensor:
        """Apply phase shift in frequency domain."""
        if shift is None:
            shift = random.uniform(*self.config.phase_shift_range)
        
        # Convert to frequency domain
        audio_fft = torch.fft.rfft(audio, dim=-1)
        
        # Create phase shift
        freqs = torch.fft.rfftfreq(audio.shape[-1], device=audio.device)
        phase_shift_tensor = torch.exp(1j * shift * freqs)
        
        # Apply phase shift
        audio_fft_shifted = audio_fft * phase_shift_tensor
        
        # Convert back to time domain
        audio_shifted = torch.fft.irfft(audio_fft_shifted, n=audio.shape[-1], dim=-1)
        
        return audio_shifted.real
    
    def manifold_mixup(self, features: torch.Tensor, labels: torch.Tensor, alpha: float = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Manifold mixup: mix features in the embedding space.
        
        Args:
            features: Feature tensor [batch, ...]
            labels: Label tensor [batch]
            alpha: Beta distribution parameter
            
        Returns:
            Mixed features, mixed labels, and lambda value
        """
        if alpha is None:
            alpha = self.config.mixup_alpha
        
        batch_size = features.size(0)
        
        # Sample lambda from Beta distribution
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        # Random permutation
        index = torch.randperm(batch_size, device=features.device)
        
        # Mix features
        mixed_features = lam * features + (1 - lam) * features[index]
        
        # Mix labels (for contrastive learning, we'll handle this differently)
        mixed_labels = labels  # Keep original labels for now
        
        return mixed_features, mixed_labels, lam
    
    def augment(self, audio: torch.Tensor, apply_all: bool = False) -> torch.Tensor:
        """
        Apply random augmentation to audio (faster version).
        Time stretching is SKIPPED during training (too slow).
        Phase shift is SKIPPED during training (expensive FFT).
        
        Args:
            audio: Audio tensor [channels, samples]
            apply_all: If True, apply all augmentations; if False, apply random subset
            
        Returns:
            Augmented audio
        """
        if apply_all:
            # Apply only fast augmentations
            audio = self.add_white_noise(audio)
            audio = self.add_colored_noise(audio, color=random.choice(['pink', 'brown']))
            audio = self.add_impulse_noise(audio)
        else:
            # Randomly select FAST augmentations only
            # Time stretching and phase shift are too slow for training
            aug_funcs = [
                lambda x: self.add_white_noise(x),
                lambda x: self.add_colored_noise(x, color='pink'),
                lambda x: self.add_colored_noise(x, color='brown'),
                lambda x: self.add_impulse_noise(x),
            ]
            
            # Apply 0-2 random augmentations (sparse augmentation)
            num_augs = random.randint(0, 2)
            selected_augs = random.sample(aug_funcs, min(num_augs, len(aug_funcs)))
            
            for aug_func in selected_augs:
                audio = aug_func(audio)
        
        return audio


class AudioChunkDataset(torch.utils.data.Dataset):
    """
    Dataset for overlapping audio chunks from Kaggle dataset.
    """
    
    def __init__(self, audio_files: List[Path], config: AudioConfig, 
                 augment: bool = False, augmentation: Optional[AudioAugmentation] = None):
        """
        Args:
            audio_files: List of audio file paths
            config: Audio configuration
            augment: Whether to apply augmentation
            augmentation: AudioAugmentation instance
        """
        self.audio_files = audio_files
        self.config = config
        self.augment = augment
        self.augmentation = augmentation
        
        # Build index of all chunks
        self.chunks = []
        self._build_chunk_index()
        
    def _build_chunk_index(self):
        """Build index of all valid audio chunks."""
        logger.info("Building chunk index...")
        
        for file_idx, audio_file in enumerate(self.audio_files):
            try:
                # Check if it's a video file
                is_video = audio_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
                
                if is_video:
                    # Extract audio from video
                    audio_data, sr = extract_audio_from_video(audio_file, self.config.sample_rate)
                    if audio_data is None:
                        continue
                    audio_length = audio_data.shape[0]
                else:
                    # Load WAV file to get length
                    sr, audio_data = wavfile.read(str(audio_file))
                    audio_length = audio_data.shape[0] if audio_data.ndim > 1 else len(audio_data)
                
                start_samples = self._compute_chunk_starts(audio_length)

                if not start_samples:
                    continue

                # Add chunks to index
                for temporal_pos, start_sample in enumerate(start_samples):
                    self.chunks.append({
                        'file_idx': file_idx,
                        'file_path': audio_file,
                        'start_sample': start_sample,
                        'temporal_pos': temporal_pos,
                        'sample_rate': sr,
                        'is_video': is_video,
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")
                continue
        
        logger.info(f"Built index with {len(self.chunks)} chunks from {len(self.audio_files)} files")

    def _compute_chunk_starts(self, audio_length: int) -> List[int]:
        """Compute chunk start indices using overlap-defined stride (about 50% by default)."""
        if audio_length < self.config.num_samples:
            return []

        max_start = audio_length - self.config.num_samples
        starts = list(range(0, max_start + 1, self.config.stride))

        # Ensure tail coverage even when length is not divisible by stride.
        if starts and starts[-1] != max_start:
            starts.append(max_start)

        return starts
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        """
        Get a chunk and return:
        - audio: [channels, samples]
        - file_idx: video/file identifier (for semantic contrast)
        - chunk_idx: temporal position (for temporal contrast)
        """
        chunk_info = self.chunks[idx]
        
        try:
            if chunk_info['is_video']:
                # Extract full audio from video
                audio_data, sr = extract_audio_from_video(chunk_info['file_path'], self.config.sample_rate)
                if audio_data is None:
                    raise ValueError("Failed to extract audio from video")
                start_sample = chunk_info['start_sample']
            else:
                # Load WAV file
                sr, audio_data = wavfile.read(str(chunk_info['file_path']))
                
                # Resample if needed
                if sr != self.config.sample_rate:
                    audio_data = scipy_signal.resample(
                        audio_data, 
                        int(len(audio_data) * self.config.sample_rate / sr)
                    )
                    # Recalculate start sample for new sample rate
                    start_sample = int(chunk_info['start_sample'] * self.config.sample_rate / sr)
                else:
                    start_sample = chunk_info['start_sample']
                
                # Ensure shape is [samples, channels] for consistent slicing
                if audio_data.ndim == 1:
                    audio_data = audio_data[:, np.newaxis]
            
            # Ensure audio_data is [samples, channels]
            if audio_data.ndim == 1:
                audio_data = audio_data[:, np.newaxis]
            elif audio_data.shape[0] < audio_data.shape[1]:
                # If it's [channels, samples], transpose it
                audio_data = audio_data.T
            
            # Keep exactly configured number of channels
            if audio_data.shape[1] < self.config.num_channels:
                pad_channels = self.config.num_channels - audio_data.shape[1]
                audio_data = np.pad(audio_data, ((0, 0), (0, pad_channels)), mode='constant')
            elif audio_data.shape[1] > self.config.num_channels:
                audio_data = audio_data[:, :self.config.num_channels]
            
            # Extract chunk [samples, channels]
            audio_chunk = audio_data[start_sample:start_sample + self.config.num_samples]
            
            # Ensure chunk has correct length (pad if needed)
            if audio_chunk.shape[0] < self.config.num_samples:
                pad_samples = self.config.num_samples - audio_chunk.shape[0]
                audio_chunk = np.pad(audio_chunk, ((0, pad_samples), (0, 0)), mode='constant')
            
            # Normalize to float32 [-1, 1]
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            elif audio_chunk.dtype == np.int32:
                audio_chunk = audio_chunk.astype(np.float32) / 2147483648.0
            elif audio_chunk.dtype == np.uint8:
                audio_chunk = (audio_chunk.astype(np.float32) - 128.0) / 128.0
            else:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Convert to tensor [channels, samples]
            audio_tensor = torch.from_numpy(audio_chunk).transpose(0, 1).float()
            
            # Apply augmentation with 60% probability (not every sample)
            if self.augment and self.augmentation is not None and random.random() < 0.6:
                audio_tensor = self.augmentation.augment(audio_tensor)
            
            # Return audio, file_idx (for semantic), and temporal position
            temporal_pos = chunk_info.get('temporal_pos', chunk_info['start_sample'] // self.config.stride)
            
            return {
                'audio': audio_tensor,
                'file_idx': chunk_info['file_idx'],
                'temporal_pos': temporal_pos,
                'chunk_idx': idx,
            }
            
        except Exception as e:
            logger.error(f"Error loading chunk {idx}: {e}")
            # Return a zero tensor as fallback
            return {
                'audio': torch.zeros(self.config.num_channels, self.config.num_samples),
                'file_idx': -1,
                'temporal_pos': -1,
                'chunk_idx': idx,
            }


class ProjectionHead(nn.Module):
    """Classifier/Projection head for contrastive learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


class AudioContrastiveModel(nn.Module):
    """
    Audio contrastive learning model with frozen EnCodec encoder and trainable projection head.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        # Load frozen EnCodec encoder from encodec library
        try:
            from encodec import EncodecModel
            self.encodec = EncodecModel.encodec_model_48khz()
            # Set bandwidth to 24kbps for best reconstruction quality
            self.encodec.set_target_bandwidth(24.0)
            logger.info("Loaded EnCodec 48kHz model (bandwidth: 24kbps)")
        except Exception as e:
            logger.error(f"Failed to load EnCodec: {e}")
            logger.error("Make sure encodec library is installed: pip install encodec")
            raise
        
        # Freeze EnCodec parameters
        for param in self.encodec.parameters():
            param.requires_grad = False
        
        # Determine encoder output dimension by running a dummy forward pass
        # EnCodec produces quantized codes with shape [batch, codebooks, sequence_length]
        # We'll flatten these to get a fixed-size representation
        with torch.no_grad():
            dummy_num_samples = AudioConfig().num_samples
            dummy_audio = torch.randn(1, 2, dummy_num_samples)
            
            # Encode to get codes
            encoded_frames = self.encodec.encode(dummy_audio)
            logger.info(f"EnCodec encoding produces {len(encoded_frames)} frame(s)")
            
            # Flatten all codes and scales
            total_codes = 0
            for frame in encoded_frames:
                codes = frame[0]  # [batch, codebooks, seq_len]
                total_codes += codes.shape[1] * codes.shape[2]  # codebooks * seq_len
            
            encoder_dim = total_codes
            logger.info(f"Total codes per sample: {encoder_dim}")
        
        # Trainable projection head
        self.projection_head = ProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.projection_dim
        )
        
        logger.info(f"Encoder output dim: {encoder_dim}, Projection dim: {config.projection_dim}")
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to embeddings using frozen EnCodec.
        
        Args:
            audio: [batch, channels, samples]
            
        Returns:
            embeddings: [batch, total_codes] - flattened quantized codes (float)
        """
        with torch.no_grad():
            # Use EnCodec to encode audio to quantized codes
            # Returns list of encoded frames, each containing (codes, scales)
            encoded_frames = self.encodec.encode(audio)
            logger.debug(f"Number of encoded frames: {len(encoded_frames)}")
            
            # Flatten all codes from all frames
            batch_size = audio.shape[0]
            codes_list = []
            
            for frame in encoded_frames:
                codes = frame[0]  # [batch, codebooks, seq_len]
                # Convert codes to float for projection head
                codes = codes.float()
                # Flatten codebooks and seq_len dimensions
                codes_flat = codes.reshape(batch_size, -1)  # [batch, codebooks*seq_len]
                codes_list.append(codes_flat)
            
            # Concatenate codes from all frames
            embeddings = torch.cat(codes_list, dim=1)  # [batch, total_codes]
            logger.debug(f"Flattened embeddings shape: {embeddings.shape} (dim={embeddings.shape[1]})")
        
        return embeddings
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode audio and project to contrastive space.
        
        Args:
            audio: [batch, channels, samples]
            
        Returns:
            projections: [batch, projection_dim] - normalized projections
        """
        embeddings = self.encode(audio)
        projections = self.projection_head(embeddings)
        logger.debug(f"Projection head output shape: {projections.shape} (dim={projections.shape[1]})")
        return projections


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss with semantic and temporal components.
    
    Semantic contrast: Different videos should have different representations
    Temporal contrast: Different parts of the same video may have different/similar representations
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        semantic_weight: float = 1.0,
        temporal_weight: float = 0.5,
        temporal_window: int = 2,
        acoustic_top_k: int = 2,
        acoustic_sim_threshold: float = 0.30,
    ):
        super().__init__()
        self.temperature = temperature
        self.semantic_weight = semantic_weight
        self.temporal_weight = temporal_weight
        self.temporal_window = temporal_window
        self.acoustic_top_k = max(0, int(acoustic_top_k))
        self.acoustic_sim_threshold = float(acoustic_sim_threshold)

    def build_acoustic_positive_mask(
        self,
        encoder_embeddings: torch.Tensor,
        file_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build cross-file positives from acoustic nearest neighbors in frozen EnCodec space.

        This encourages chunks with similar sound content (e.g., same phoneme by
        different actors) to be considered positives for temporal learning.
        """
        batch_size = encoder_embeddings.shape[0]
        device = encoder_embeddings.device
        if batch_size <= 1 or self.acoustic_top_k <= 0:
            return torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)

        emb = F.normalize(encoder_embeddings.detach(), dim=1)
        sim = torch.matmul(emb, emb.T)  # cosine since embeddings are normalized

        eye = torch.eye(batch_size, dtype=torch.bool, device=device)
        same_file = file_indices.unsqueeze(1) == file_indices.unsqueeze(0)

        # Only mine cross-file neighbors; self-pairs are invalid.
        candidate_mask = (~eye) & (~same_file)
        sim_masked = sim.masked_fill(~candidate_mask, float('-inf'))

        k = min(self.acoustic_top_k, max(1, batch_size - 1))
        topk_vals, topk_idx = torch.topk(sim_masked, k=k, dim=1)

        acoustic_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        valid_topk = torch.isfinite(topk_vals) & (topk_vals >= self.acoustic_sim_threshold)

        row_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(topk_idx)
        if valid_topk.any():
            acoustic_mask[row_idx[valid_topk], topk_idx[valid_topk]] = True

        # Symmetrize to stabilize pair assignment.
        acoustic_mask = acoustic_mask | acoustic_mask.T
        return acoustic_mask
    
    def info_nce_from_positive_mask(self, features: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        """
        Multi-positive InfoNCE with explicit positive mask (Diff-Foley compatible).
        
        Args:
            features: [batch, dim] - normalized features
            positive_mask: [batch, batch] bool mask where True indicates a positive pair
            
        Returns:
            loss: scalar
        """
        batch_size = features.shape[0]
        device = features.device

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # Remove self-pairs from both positives and denominator candidates.
        eye = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = positive_mask.bool() & (~eye)
        denominator_mask = ~eye

        # Compute logits and apply masked log-softmax over non-self entries.
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits.masked_fill(~denominator_mask, float('-inf'))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # Average log-prob over available positives per anchor.
        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0

        if not valid.any():
            return torch.tensor(0.0, device=device)

        # Avoid 0 * (-inf) -> NaN by masking with where instead of multiplication.
        log_prob_pos = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
        mean_log_prob_pos = log_prob_pos.sum(dim=1) / positive_counts.clamp(min=1)
        loss = -mean_log_prob_pos[valid].mean()
        return loss
    
    def compute_augmentation_positives(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute augmentation-based positives for improved training stability.
        Assumes features are organized as [v1_0, v1_1, ..., v2_0, v2_1, ...]
        where v1_i and v2_i are augmented views of the same chunk i.
        
        Args:
            features: [2*batch, dim] - concatenated v1 and v2 projections
            
        Returns:
            loss: scalar
        """
        batch_size = features.shape[0] // 2
        device = features.device
        
        if batch_size <= 0:
            return torch.tensor(0.0, device=device)
        
        # Create positive mask: each v1_i is positive with v2_i
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = torch.block_diag(positive_mask, positive_mask)  # Block diagonal structure
        # Swap blocks: make (i, batch+i) and (batch+i, i) true
        for i in range(batch_size):
            positive_mask[i, batch_size + i] = True
            positive_mask[batch_size + i, i] = True
        
        return self.info_nce_from_positive_mask(features, positive_mask)
    
    def forward(
        self,
        projections: torch.Tensor,
        file_indices: torch.Tensor,
        temporal_positions: torch.Tensor,
        encoder_embeddings: Optional[torch.Tensor] = None,
        augmentation_loss_weight: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Diff-Foley compatible contrastive loss with semantic and temporal components.
        Also includes augmentation-based positives for improved training stability.
        
        Args:
            projections: [batch, projection_dim] - normalized projections
            file_indices: [batch] - file/video identifiers
            temporal_positions: [batch] - temporal positions within videos
            encoder_embeddings: Optional encoder embeddings for acoustic similarity
            augmentation_loss_weight: Weight for augmentation-based positive pairs loss
            
        Returns:
            Dictionary with total loss and component losses
        """
        same_file = file_indices.unsqueeze(1) == file_indices.unsqueeze(0)
        temporal_distance = torch.abs(temporal_positions.unsqueeze(1) - temporal_positions.unsqueeze(0))

        # Semantic positives: chunks from the same file/video (Diff-Foley style).
        semantic_positive_mask = same_file
        semantic_loss = self.info_nce_from_positive_mask(projections, semantic_positive_mask)

        # Temporal positives: nearby chunks in same file OR acoustically similar chunks across files (Diff-Foley style).
        temporal_positive_mask = same_file & (temporal_distance <= self.temporal_window)
        if encoder_embeddings is not None:
            acoustic_positive_mask = self.build_acoustic_positive_mask(encoder_embeddings, file_indices)
            temporal_positive_mask = temporal_positive_mask | acoustic_positive_mask

        temporal_loss = self.info_nce_from_positive_mask(projections, temporal_positive_mask)
        
        # Augmentation-based positives: improve training stability
        # Only compute if batch is doubled (from two augmented views)
        augmentation_loss = torch.tensor(0.0, device=projections.device)
        if projections.shape[0] > 1 and projections.shape[0] % 2 == 0:
            augmentation_loss = self.compute_augmentation_positives(projections)
        
        # Total loss: Diff-Foley base + augmentation regularization
        base_loss = self.semantic_weight * semantic_loss + self.temporal_weight * temporal_loss
        total_loss = base_loss + augmentation_loss_weight * augmentation_loss
        
        return {
            'total_loss': total_loss,
            'semantic_loss': semantic_loss,
            'temporal_loss': temporal_loss,
            'augmentation_loss': augmentation_loss,
        }


def extract_audio_from_video(video_path: Path, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """
    Extract complete audio from a video file using PyAV.
    
    Args:
        video_path: Path to video file (.mp4, .avi, etc.)
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
        - audio_data: [samples, channels] numpy array
        - sample_rate: Sample rate in Hz
    """
    try:
        container = av.open(str(video_path))
        
        # Get audio stream
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        if not audio_stream:
            container.close()
            logger.warning(f"No audio stream in {video_path.name}")
            return None, None
        
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
            return None, None
        
        # Concatenate all frames
        audio_full = np.concatenate(audio_frames, axis=0)
        
        # Normalize to float32
        if audio_full.dtype == np.int16:
            audio_full = audio_full.astype(np.float32) / 32768.0
        elif audio_full.dtype == np.int32:
            audio_full = audio_full.astype(np.float32) / 2147483648.0
        else:
            audio_full = audio_full.astype(np.float32)
        
        # Resample if needed
        if audio_sr != target_sr:
            num_samples_new = int(audio_full.shape[0] * target_sr / audio_sr)
            audio_resampled = scipy_signal.resample(audio_full, num_samples_new, axis=0)
            audio_full = audio_resampled.astype(np.float32)
            audio_sr = target_sr
        
        return audio_full, audio_sr
        
    except Exception as e:
        logger.warning(f"Failed to extract audio from {video_path.name}: {e}")
        return None, None


def get_video_fps(video_path: Path, fallback_fps: float = 30000 / 1001) -> float:
    """Read FPS from video stream using PyAV, matching alignment extraction logic."""
    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0] if container.streams.video else None
        if video_stream and video_stream.average_rate:
            fps = float(video_stream.average_rate)
            container.close()
            return fps
        container.close()
    except Exception as e:
        logger.warning(f"Could not read FPS from {video_path.name}: {e}")

    return float(fallback_fps)


def find_kaggle_audio_files() -> List[Path]:
    """Find video files (01*.mp4) from Kaggle dataset - extracts audio from videos only."""
    import os
    
    possible_paths = [
        Path("kaggle_datasets"),
        Path(os.path.expanduser("~/.cache/kagglehub/datasets")),
        Path(os.environ.get("USERPROFILE", "~")) / ".cache/kagglehub/datasets",
    ]
    
    video_files = []
    
    for kaggle_path in possible_paths:
        if kaggle_path.exists():
            # Find video files starting with "01" (speech videos with audio)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(kaggle_path.rglob(f"*{ext}"))
            
            # Filter to only "01" prefix (speech videos with embedded audio)
            video_files = [v for v in all_videos if v.name.startswith('01')]
            
            if video_files:
                logger.info(f"Found {len(video_files)} speech video files (01*.mp4, will extract audio) in {kaggle_path}")
                break
    
    if not video_files:
        logger.warning("No video files with '01' prefix found! Ensure Kaggle dataset is downloaded.")
    
    return video_files


def split_dataset(audio_files: List[Path], train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split audio files into train/val/test sets.
    
    Args:
        audio_files: List of audio file paths
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
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


def train_epoch(model: AudioContrastiveModel, dataloader: torch.utils.data.DataLoader, 
                criterion: ContrastiveLoss, optimizer: torch.optim.Optimizer, 
                device: torch.device, augmentation: AudioAugmentation,
                apply_manifold_mixup: bool = True, epoch: int = 1, total_epochs: int = 1) -> Dict[str, float]:
    """Train for one epoch with progress bar."""
    model.train()
    
    total_loss = 0.0
    total_semantic = 0.0
    total_temporal = 0.0
    total_augmentation = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch", leave=True)
    
    for batch in pbar:
        audio = batch['audio'].to(device)  # [batch, channels, samples]
        file_idx = batch['file_idx'].to(device)
        temporal_pos = batch['temporal_pos'].to(device)
        
        # Create two augmented views of the audio for augmentation-based positive pairs
        audio_v1 = augmentation.augment(audio)
        audio_v2 = augmentation.augment(audio)
        
        # Forward pass for both views
        encoder_embeddings_v1 = model.encode(audio_v1)
        projections_v1 = model.projection_head(encoder_embeddings_v1)
        
        encoder_embeddings_v2 = model.encode(audio_v2)
        projections_v2 = model.projection_head(encoder_embeddings_v2)
        
        # Concatenate projections: [batch_v1, batch_v2]
        # This allows loss to compute: (1) Diff-Foley semantic/temporal, (2) augmentation-based positives
        projections = torch.cat([projections_v1, projections_v2], dim=0)
        file_idx_doubled = torch.cat([file_idx, file_idx], dim=0)
        temporal_pos_doubled = torch.cat([temporal_pos, temporal_pos], dim=0)
        encoder_embeddings_doubled = torch.cat([encoder_embeddings_v1, encoder_embeddings_v2], dim=0)
        
        # Apply manifold mixup with some probability
        if apply_manifold_mixup and random.random() < 0.3:
            projections, file_idx_doubled, lam = augmentation.manifold_mixup(projections, file_idx_doubled)
        
        # Compute loss: Diff-Foley style + augmentation regularization
        loss_dict = criterion(
            projections, file_idx_doubled, temporal_pos_doubled, 
            encoder_embeddings=encoder_embeddings_doubled,
            augmentation_loss_weight=0.2
        )
        loss = loss_dict['total_loss']

        # Skip numerically invalid batches to keep optimizer state healthy.
        if not torch.isfinite(loss):
            logger.warning("Skipping batch with non-finite training loss")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_semantic += loss_dict['semantic_loss'].item()
        total_temporal += loss_dict['temporal_loss'].item()
        total_augmentation += loss_dict.get('augmentation_loss', torch.tensor(0.0)).item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / num_batches,
            'semantic': total_semantic / num_batches,
            'temporal': total_temporal / num_batches,
            'aug': total_augmentation / num_batches,
        })
    
    if num_batches == 0:
        return {
            'loss': float('inf'),
            'semantic_loss': float('inf'),
            'temporal_loss': float('inf'),
            'augmentation_loss': 0.0,
        }

    return {
        'loss': total_loss / num_batches,
        'semantic_loss': total_semantic / num_batches,
        'temporal_loss': total_temporal / num_batches,
        'augmentation_loss': total_augmentation / num_batches,
    }


@torch.no_grad()
def validate(model: AudioContrastiveModel, dataloader: torch.utils.data.DataLoader,
             criterion: ContrastiveLoss, device: torch.device) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_semantic = 0.0
    total_temporal = 0.0
    num_batches = 0
    
    for batch in dataloader:
        audio = batch['audio'].to(device)
        file_idx = batch['file_idx'].to(device)
        temporal_pos = batch['temporal_pos'].to(device)
        
        # Forward pass
        encoder_embeddings = model.encode(audio)
        projections = model.projection_head(encoder_embeddings)
        
        # Compute loss
        loss_dict = criterion(projections, file_idx, temporal_pos, encoder_embeddings=encoder_embeddings)

        if not torch.isfinite(loss_dict['total_loss']):
            logger.warning("Skipping batch with non-finite validation loss")
            continue
        
        # Accumulate metrics
        total_loss += loss_dict['total_loss'].item()
        total_semantic += loss_dict['semantic_loss'].item()
        total_temporal += loss_dict['temporal_loss'].item()
        num_batches += 1
    
    if num_batches == 0:
        return {
            'loss': float('inf'),
            'semantic_loss': float('inf'),
            'temporal_loss': float('inf'),
        }

    return {
        'loss': total_loss / num_batches,
        'semantic_loss': total_semantic / num_batches,
        'temporal_loss': total_temporal / num_batches,
    }


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Semantic loss
    axes[1].plot(epochs, history['train_semantic'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_semantic'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Semantic Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # Temporal loss
    axes[2].plot(epochs, history['train_temporal'], 'b-', label='Train')
    axes[2].plot(epochs, history['val_temporal'], 'r-', label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Temporal Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training history saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train audio contrastive learning model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help='Contrastive loss temperature')
    parser.add_argument('--temporal-window', type=int, default=2,
                        help='Max chunk distance for temporal positive pairs (same file only)')
    parser.add_argument('--acoustic-top-k', type=int, default=2,
                        help='Number of cross-file acoustic neighbors used as temporal positives')
    parser.add_argument('--acoustic-sim-threshold', type=float, default=0.30,
                        help='Minimum cosine similarity for acoustic temporal positives')
    parser.add_argument('--early-stopping-patience', type=int, default=8,
                        help='Stop after this many epochs without meaningful val improvement')
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-3,
                        help='Minimum val loss improvement to reset early-stopping counter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='audio_contrastive_results', help='Output directory')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--no-manifold-mixup', action='store_true', help='Disable manifold mixup')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configurations
    audio_config = AudioConfig()
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        temperature=args.temperature,
        temporal_window=args.temporal_window,
        acoustic_top_k=args.acoustic_top_k,
        acoustic_sim_threshold=args.acoustic_sim_threshold,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    
    # Find audio files
    logger.info("Finding audio files...")
    audio_files = find_kaggle_audio_files()
    
    if len(audio_files) == 0:
        logger.error("No audio files found! Please run download_kaggle_dataset.py first.")
        return

    # Match alignment duration rule: segment duration = num_video_frames / fps_from_video.
    aligned_fps = get_video_fps(audio_files[0], fallback_fps=audio_config.reference_video_fps)
    audio_config.reference_video_fps = aligned_fps
    audio_config.num_samples = int(round(audio_config.sample_rate * audio_config.num_video_frames / aligned_fps))
    audio_config.stride = int(round(audio_config.num_samples * (1.0 - audio_config.overlap)))
    audio_config.update_derived_params()
    logger.info(
        f"Aligned timing config from video FPS {aligned_fps:.5f}: "
        f"duration={audio_config.chunk_duration_seconds:.6f}s, "
        f"num_samples={audio_config.num_samples}, stride={audio_config.stride}"
    )
    
    # Split dataset
    train_files, val_files, test_files = split_dataset(
        audio_files,
        train_ratio=training_config.train_ratio,
        val_ratio=training_config.val_ratio,
        test_ratio=training_config.test_ratio,
        seed=args.seed
    )
    
    # Create augmentation
    augmentation = AudioAugmentation(audio_config)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = AudioChunkDataset(
        train_files, 
        audio_config, 
        augment=not args.no_augmentation,
        augmentation=augmentation
    )
    val_dataset = AudioChunkDataset(
        val_files,
        audio_config,
        augment=False,
        augmentation=None
    )
    test_dataset = AudioChunkDataset(
        test_files,
        audio_config,
        augment=False,
        augmentation=None
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    # Use fewer workers on Windows (multiprocessing has overhead)
    num_workers = 0 if device.type == 'cpu' else 2
    
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
    
    # Create model
    logger.info("Creating model...")
    model = AudioContrastiveModel(training_config).to(device)
    
    # Create loss and optimizer
    criterion = ContrastiveLoss(
        temperature=training_config.temperature,
        semantic_weight=training_config.semantic_weight,
        temporal_weight=training_config.temporal_weight,
        temporal_window=training_config.temporal_window,
        acoustic_top_k=training_config.acoustic_top_k,
        acoustic_sim_threshold=training_config.acoustic_sim_threshold,
    )
    
    optimizer = torch.optim.Adam(
        model.projection_head.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_semantic': [],
        'train_temporal': [],
        'val_loss': [],
        'val_semantic': [],
        'val_temporal': [],
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(training_config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            augmentation, apply_manifold_mixup=not args.no_manifold_mixup,
            epoch=epoch + 1, total_epochs=training_config.num_epochs
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler based on validation loss
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Semantic: {train_metrics['semantic_loss']:.4f}, "
                   f"Temporal: {train_metrics['temporal_loss']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Semantic: {val_metrics['semantic_loss']:.4f}, "
                   f"Temporal: {val_metrics['temporal_loss']:.4f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_semantic'].append(train_metrics['semantic_loss'])
        history['train_temporal'].append(train_metrics['temporal_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_semantic'].append(val_metrics['semantic_loss'])
        history['val_temporal'].append(val_metrics['temporal_loss'])
        
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
            logger.info(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(best val_loss: {best_val_loss:.4f})"
                )
                break
    
    # Test on best model
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = validate(model, test_loader, criterion, device)
    logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, "
               f"Semantic: {test_metrics['semantic_loss']:.4f}, "
               f"Temporal: {test_metrics['temporal_loss']:.4f}")
    
    # Save test metrics
    with open(output_dir / 'test_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Semantic Loss: {test_metrics['semantic_loss']:.4f}\n")
        f.write(f"Temporal Loss: {test_metrics['temporal_loss']:.4f}\n")
    
    # Plot training history
    plot_training_history(history, output_dir / 'training_history.png')
    
    # Save final model
    torch.save({
        'epoch': training_config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': training_config.__dict__,
    }, output_dir / 'final_model.pt')
    
    logger.info(f"\n✓ Training complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
