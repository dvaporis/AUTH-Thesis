import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

model_id = "MCG-NJU/videomae-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing VideoMAE on {device}...")
processor = VideoMAEImageProcessor.from_pretrained(model_id)
model = VideoMAEModel.from_pretrained(model_id).to(device).eval()
print("Model loaded successfully.")


def get_video_embedding(frames):
    # frames: list of 16 HWC uint8 frames
    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


dummy_frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
embedding = get_video_embedding(dummy_frames)
print(f"Embedding shape: {tuple(embedding.shape)}")