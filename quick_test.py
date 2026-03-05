from train_audio_contrastive import AudioContrastiveModel, TrainingConfig, ContrastiveLoss, AudioConfig
import torch
import logging

# Enable debug logging to see encoder dimensions
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

config = TrainingConfig()
audio_config = AudioConfig()
print('Creating model...')
model = AudioContrastiveModel(config)
print('[OK] Model created successfully')

dummy_audio = torch.randn(2, 2, audio_config.num_samples)
print('Testing forward pass...')
output = model(dummy_audio)
print(f'[OK] Output shape: {output.shape}')

print('\nTesting encode separately...')
embeddings = model.encode(dummy_audio)
print(f'[OK] Embeddings shape: {embeddings.shape}')

print('Testing loss...')
file_idx = torch.tensor([0, 1])
temporal_pos = torch.tensor([0, 1])
criterion = ContrastiveLoss()
loss_dict = criterion(output, file_idx, temporal_pos)
print(f'[OK] Loss: {loss_dict["total_loss"].item():.4f}')

print('[SUCCESS] ALL TESTS PASSED')
