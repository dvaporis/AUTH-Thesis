"""Extract precise frame-level phoneme durations from a trained CTC model.

This script loads the best trained model, runs inference on a lip video,
and applies Viterbi decoding to map the unaligned target phoneme sequence
to exact video frame boundaries.
"""

import argparse
import json
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any

# Import the exact model class from your training script
from train_lip_lstm import VisualLSTMFrameCE, DEFAULT_BLANK_TOKEN

def load_video_frames(video_path: Path, frame_size: int = 224) -> torch.Tensor:
    """Loads video and prepares it exactly like the validation transform."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frames = []
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    video = torch.stack(frames, dim=0).float()  # [T, C, H, W]
    
    # Apply standard validation normalization
    video = F.interpolate(video, size=(frame_size, frame_size), mode="bilinear", align_corners=False)
    video /= 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    video = (video - mean) / std
    return video

def viterbi_forced_alignment(
    log_probs: torch.Tensor, 
    target_ids: List[int], 
    blank_id: int = 0
) -> List[int]:
    """Computes the optimal Viterbi path forcing alignment to the target sequence.
    
    Args:
        log_probs: Tensor of shape [Time, Vocab_Size] containing log-softmax outputs.
        target_ids: List of integer IDs representing the unaligned ground-truth text sequence.
        blank_id: Index of the CTC blank token.
        
    Returns:
        List of length Time, where each element is the assigned phoneme/blank token ID per frame.
    """
    num_frames = log_probs.size(0)
    
    # Construct the CTC state sequence graph: alternating blanks and targets
    # e.g., target [p, o] becomes state sequence [blank, p, blank, o, blank]
    states = [blank_id]
    for tid in target_ids:
        states.append(tid)
        states.append(blank_id)
    num_states = len(states)

    # Initialize Viterbi trellis and backpointer matrix
    # trellis[t, s] = max log probability at time t in state s
    trellis = torch.full((num_frames, num_states), float("-inf"))
    backpointers = torch.zeros((num_frames, num_states), dtype=torch.long)

    # Time step 0 initialization (can start at state 0 or state 1)
    trellis[0, 0] = log_probs[0, states[0]]
    if num_states > 1:
        trellis[0, 1] = log_probs[0, states[1]]

    # Forward dynamic programming pass
    for t in range(1, num_frames):
        for s in range(num_states):
            current_token = states[s]
            log_p = log_probs[t, current_token]

            # Option 1: Stay in the same state (looping)
            best_prev_s = s
            best_prob = trellis[t - 1, s]

            # Option 2: Transition from the immediate previous state
            if s > 0 and trellis[t - 1, s - 1] > best_prob:
                best_prev_s = s - 1
                best_prob = trellis[t - 1, s - 1]

            # Option 3: Skip a blank token if transitioning between two distinct phonemes
            if (
                s > 1 
                and states[s] != blank_id 
                and states[s - 1] == blank_id 
                and states[s] != states[s - 2]
            ):
                if trellis[t - 1, s - 2] > best_prob:
                    best_prev_s = s - 2
                    best_prob = trellis[t - 1, s - 2]

            trellis[t, s] = best_prob + log_p
            backpointers[t, s] = best_prev_s

    # Backward tracking pass to locate the optimal path
    # CTC allows the sequence to finish at either the final blank state or the final phoneme state
    best_final_s = num_states - 1 if trellis[-1, -1] > trellis[-1, -2] else num_states - 2
    
    path = []
    current_s = best_final_s
    for t in range(num_frames - 1, -1, -1):
        path.append(states[current_s])
        current_s = int(backpointers[t, current_s])
    
    path.reverse()
    return path

def extract_durations_from_path(path: List[int], blank_id: int = 0) -> List[Dict[str, Any]]:
    """Aggregates the frame-by-frame path into explicit start/end boundaries and frame counts."""
    segments = []
    if not path:
        return segments

    current_token_id = path[0]
    start_frame = 0

    for idx, token_id in enumerate(path):
        if token_id != current_token_id:
            # Save the completed segment
            segments.append({
                "token_id": current_token_id,
                "start_frame": start_frame,
                "end_frame": idx - 1,
                "duration_frames": idx - start_frame
            })
            current_token_id = token_id
            start_frame = idx

    # Append the final remaining active segment
    segments.append({
        "token_id": current_token_id,
        "start_frame": start_frame,
        "end_frame": len(path) - 1,
        "duration_frames": len(path) - start_frame
    })
    
    return segments

def main():
    parser = argparse.ArgumentParser(description="Force-align targets to video frames via Viterbi decoding.")
    parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--vocab-json", type=str, required=True, help="Path to phoneme_vocab.json")
    parser.add_argument("--video-path", type=str, required=True, help="Path to input video clip")
    parser.add_argument("--target-phonemes", type=str, required=True, help="Space-separated target phonemes (e.g. 'p a s t')")
    args = parser.parse_args()

    # Load configuration parameters and model architecture definitions
    with open(args.vocab_json, "r", encoding="utf-8") as handle:
        vocab_data = json.load(handle)
    vocab = vocab_data["tokens"]
    
    token_to_id = {token: index + 1 for index, token in enumerate(vocab)}
    id_to_token = {index + 1: token for index, token in enumerate(vocab)}
    id_to_token[0] = DEFAULT_BLANK_TOKEN

    checkpoint = torch.load(args.model_checkpoint, map_map={"cuda": "cpu"})
    
    # Initialize architecture
    model = VisualLSTMFrameCE(
        vocab_size=len(vocab) + 1,
        hidden_dim=checkpoint["config"]["hidden_dim"],
        lstm_layers=checkpoint["config"]["lstm_layers"],
        pretrained_backbone=False,
        finetune_backbone=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Prepare inputs
    video_tensor = load_video_frames(Path(args.video_path), frame_size=checkpoint["config"]["frame_size"])
    video_length = torch.tensor([video_tensor.size(0)], dtype=torch.long)
    video_tensor = video_tensor.unsqueeze(0) # Add batch dimension -> [1, T, C, H, W]

    # Convert word/sentence input phoneme targets to IDs
    target_tokens = args.target_phonemes.strip().split()
    target_ids = [token_to_id[tok] for tok in target_tokens if tok in token_to_id]

    # Model Pass
    with torch.no_grad():
        logits = model(video_tensor, video_length)  # [1, T, Vocab_Size + 1]
        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [T, Vocab_Size + 1]

    # Run Viterbi alignment step
    frame_path = viterbi_forced_alignment(log_probs, target_ids, blank_id=0)
    segments = extract_durations_from_path(frame_path, blank_id=0)

    # Print results to console for structural analysis
    print(f"\n--- Alignment Summary for {Path(args.video_path).name} ---")
    print(f"Total Frames: {video_length.item()}")
    print(f"{'Phoneme':<12} | {'Start Frame':<12} | {'End Frame':<12} | {'Duration (Frames)':<18}")
    print("-" * 62)
    
    # Filter out blank tokens when passing timing vectors straight to your speech synthesis head
    for seg in segments:
        token_str = id_to_token[seg["token_id"]]
        print(f"{token_str:<12} | {seg['start_frame']:<12} | {seg['end_frame']:<12} | {seg['duration_frames']:<18}")

if __name__ == "__main__":
    main()