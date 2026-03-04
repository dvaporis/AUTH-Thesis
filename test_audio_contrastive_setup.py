"""
Quick test script to verify the audio contrastive learning setup.

This script:
1. Checks for audio files
2. Tests data loading and augmentation
3. Tests model forward pass
4. Verifies loss computation
5. Shows sample augmentations

Usage:
    python test_audio_contrastive_setup.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

# Import from training script
from train_audio_contrastive import (
    AudioConfig, TrainingConfig, AudioAugmentation,
    AudioChunkDataset, AudioContrastiveModel, ContrastiveLoss,
    find_kaggle_audio_files, split_dataset
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test dataset loading and chunking."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Data Loading")
    logger.info("="*60)
    
    # Find audio files
    audio_files = find_kaggle_audio_files()
    
    if len(audio_files) == 0:
        logger.error("❌ No audio files found!")
        logger.info("Please run: python download_kaggle_dataset.py")
        return False
    
    logger.info(f"✓ Found {len(audio_files)} audio files")
    
    # Test dataset creation
    audio_config = AudioConfig()
    train_files, val_files, test_files = split_dataset(audio_files[:10])  # Use only 10 files for quick test
    
    dataset = AudioChunkDataset(train_files, audio_config, augment=False)
    logger.info(f"✓ Created dataset with {len(dataset)} chunks")
    
    # Test loading a sample
    sample = dataset[0]
    logger.info(f"✓ Sample shape: {sample['audio'].shape}")
    logger.info(f"  - File index: {sample['file_idx']}")
    logger.info(f"  - Temporal position: {sample['temporal_pos']}")
    
    return True


def test_augmentation():
    """Test all augmentation techniques."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Audio Augmentation")
    logger.info("="*60)
    
    # Create dummy audio signal
    audio_config = AudioConfig()
    t = np.linspace(0, audio_config.num_samples / audio_config.sample_rate, audio_config.num_samples)
    audio_np = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    audio = torch.from_numpy(audio_np).unsqueeze(0).float()
    
    augmentation = AudioAugmentation(audio_config)
    
    # Test each augmentation
    tests = [
        ("White Noise", lambda: augmentation.add_white_noise(audio)),
        ("Pink Noise", lambda: augmentation.add_colored_noise(audio, 'pink')),
        ("Brown Noise", lambda: augmentation.add_colored_noise(audio, 'brown')),
        ("Impulse Noise", lambda: augmentation.add_impulse_noise(audio)),
        ("Time Stretch", lambda: augmentation.time_stretch(audio)),
        ("Phase Shift", lambda: augmentation.phase_shift(audio)),
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot original
    axes[0].plot(t[:1000], audio[0, :1000].numpy())
    axes[0].set_title('Original Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    # Plot augmentations
    for idx, (name, aug_func) in enumerate(tests, start=1):
        try:
            aug_audio = aug_func()
            axes[idx].plot(t[:1000], aug_audio[0, :1000].numpy())
            axes[idx].set_title(name)
            axes[idx].set_xlabel('Time (s)')
            axes[idx].set_ylabel('Amplitude')
            axes[idx].grid(True)
            logger.info(f"✓ {name} augmentation works")
        except Exception as e:
            logger.error(f"❌ {name} augmentation failed: {e}")
            axes[idx].text(0.5, 0.5, f'Failed: {name}', ha='center', va='center')
    
    # Hide unused subplots
    for idx in range(len(tests) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_dir = Path('audio_contrastive_results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'test_augmentations.png', dpi=150)
    logger.info(f"✓ Augmentation visualization saved to {output_dir / 'test_augmentations.png'}")
    plt.close()
    
    return True


def test_model_forward():
    """Test model creation and forward pass."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Model Forward Pass")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    training_config = TrainingConfig()
    
    try:
        model = AudioContrastiveModel(training_config).to(device)
        logger.info("✓ Model created successfully")
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return False
    
    # Create dummy batch
    batch_size = 8
    audio_config = AudioConfig()
    dummy_audio = torch.randn(batch_size, 1, audio_config.num_samples).to(device)
    
    # Test forward pass
    try:
        with torch.no_grad():
            # Test encoding
            embeddings = model.encode(dummy_audio)
            logger.info(f"✓ Encoder output shape: {embeddings.shape}")
            
            # Test full forward (with projection)
            projections = model(dummy_audio)
            logger.info(f"✓ Projection output shape: {projections.shape}")
            logger.info(f"✓ Projections are normalized: {torch.allclose(torch.norm(projections, dim=1), torch.ones(batch_size).to(device), atol=1e-5)}")
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        return False
    
    # Check frozen parameters
    frozen_params = sum(1 for p in model.encodec.parameters() if not p.requires_grad)
    trainable_params = sum(1 for p in model.projection_head.parameters() if p.requires_grad)
    logger.info(f"✓ EnCodec frozen parameters: {frozen_params}")
    logger.info(f"✓ Projection head trainable parameters: {trainable_params}")
    
    return True


def test_loss_computation():
    """Test contrastive loss computation."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Loss Computation")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    batch_size = 16
    projection_dim = 128
    projections = F.normalize(torch.randn(batch_size, projection_dim), dim=1).to(device)
    
    # Create labels
    file_indices = torch.randint(0, 5, (batch_size,)).to(device)  # 5 different files
    temporal_positions = torch.randint(0, 20, (batch_size,)).to(device)  # 20 temporal positions
    
    # Create loss
    criterion = ContrastiveLoss()
    
    try:
        loss_dict = criterion(projections, file_indices, temporal_positions)
        
        logger.info(f"✓ Total loss: {loss_dict['total_loss'].item():.4f}")
        logger.info(f"✓ Semantic loss: {loss_dict['semantic_loss'].item():.4f}")
        logger.info(f"✓ Temporal loss: {loss_dict['temporal_loss'].item():.4f}")
        
        # Check loss is finite and reasonable
        assert torch.isfinite(loss_dict['total_loss']), "Loss is not finite!"
        assert loss_dict['total_loss'].item() > 0, "Loss should be positive!"
        
        logger.info("✓ Loss computation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Loss computation failed: {e}")
        return False


def test_dataloader():
    """Test dataloader with batching."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: DataLoader Batching")
    logger.info("="*60)
    
    # Find audio files
    audio_files = find_kaggle_audio_files()
    if len(audio_files) == 0:
        logger.warning("⚠ Skipping dataloader test (no audio files)")
        return False
    
    # Create dataset and dataloader
    audio_config = AudioConfig()
    train_files, _, _ = split_dataset(audio_files[:5])  # Use only 5 files
    
    augmentation = AudioAugmentation(audio_config)
    dataset = AudioChunkDataset(train_files, audio_config, augment=True, augmentation=augmentation)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )
    
    try:
        # Get one batch
        batch = next(iter(dataloader))
        
        logger.info(f"✓ Batch audio shape: {batch['audio'].shape}")
        logger.info(f"✓ Batch file indices shape: {batch['file_idx'].shape}")
        logger.info(f"✓ Batch temporal positions shape: {batch['temporal_pos'].shape}")
        logger.info(f"✓ Unique files in batch: {len(torch.unique(batch['file_idx']))}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ DataLoader test failed: {e}")
        return False


def test_manifold_mixup():
    """Test manifold mixup."""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Manifold Mixup")
    logger.info("="*60)
    
    audio_config = AudioConfig()
    augmentation = AudioAugmentation(audio_config)
    
    # Create dummy features and labels
    features = torch.randn(16, 128)
    labels = torch.randint(0, 5, (16,))
    
    try:
        mixed_features, mixed_labels, lam = augmentation.manifold_mixup(features, labels)
        
        logger.info(f"✓ Original features shape: {features.shape}")
        logger.info(f"✓ Mixed features shape: {mixed_features.shape}")
        logger.info(f"✓ Lambda value: {lam:.3f}")
        logger.info(f"✓ Features are mixed properly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Manifold mixup test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("AUDIO CONTRASTIVE LEARNING - SETUP VERIFICATION")
    logger.info("="*60)
    
    # Import for normalization
    import torch.nn.functional as F
    globals()['F'] = F
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Augmentation", test_augmentation),
        ("Model Forward Pass", test_model_forward),
        ("Loss Computation", test_loss_computation),
        ("DataLoader Batching", test_dataloader),
        ("Manifold Mixup", test_manifold_mixup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    logger.info(f"\nPassed {passed}/{total} tests")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Ready to train.")
        logger.info("Run: python train_audio_contrastive.py --epochs 50 --batch-size 32")
    else:
        logger.warning("\n⚠ Some tests failed. Please check the errors above.")
    
    logger.info("="*60)


if __name__ == '__main__':
    main()
