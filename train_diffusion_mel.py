#!/usr/bin/env python3
"""Train a simple conditional diffusion model to generate mel-spectrograms conditioned on
the visual LSTM hidden state from `train_lip_lstm_ctc`.

This trainer supports two conditioning modes:
 - `--cond-dir`: use precomputed conditioning vectors (.npy/.npz per-sample)
 - `--lstm-checkpoint` + `--video-dir`: load the VisualLSTMCTC model and compute per-sample
    conditioning vectors on-the-fly from videos.

The mel inputs must be .npz files with key `mel` (produced by `preprocess_mels_full.py`).

This is a minimal, extensible trainer intended as a starting point.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2


class MelDataset(Dataset):
    def __init__(self, mel_dir: Path, cond_dir: Optional[Path] = None, lip_crop_dir: Optional[Path] = None, max_frames: int = 400):
        self.mel_dir = Path(mel_dir)
        self.cond_dir = Path(cond_dir) if cond_dir is not None else None
        self.lip_crop_dir = Path(lip_crop_dir) if lip_crop_dir is not None else None
        self.files = sorted([p for p in self.mel_dir.rglob('*.npz')])
        self.max_frames = int(max_frames)

    def __len__(self):
        return len(self.files)

    def _find_video(self, stem: str):
        # prefer lip-cropped videos (expected for LSTM conditioning)
        if self.lip_crop_dir is None:
            return None

        for ext in ('.mp4', '.avi', '.mov', '.mkv'):
            candidate = self.lip_crop_dir / (stem + ext)
            if candidate.exists():
                return candidate
        # fallback: try any file starting with stem
        for p in self.lip_crop_dir.rglob(stem + '*'):
            if p.is_file():
                return p
        return None

    def _load_mel(self, path: Path):
        data = np.load(str(path))
        mel = data['mel'].astype(np.float32)
        return mel

    def __getitem__(self, idx: int):
        mel_path = self.files[idx]
        stem = mel_path.stem
        mel = self._load_mel(mel_path)
        # pad / truncate in time axis
        n_mels, T = mel.shape
        if T > self.max_frames:
            mel = mel[:, : self.max_frames]
        elif T < self.max_frames:
            pad = self.max_frames - T
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant', constant_values=mel.min())

        sample = {'mel': torch.from_numpy(mel)}

        # condition vector
        if self.cond_dir is not None:
            cond_path = self.cond_dir / (stem + '.npy')
            if cond_path.exists():
                cond = np.load(str(cond_path)).astype(np.float32)
                sample['cond'] = torch.from_numpy(cond)
            else:
                sample['cond'] = torch.zeros(512, dtype=torch.float32)
        else:
            sample['video_path'] = self._find_video(stem)

        sample['stem'] = stem
        return sample


def load_video_frames(video_path: Path, frame_size: int = 224, max_frames: Optional[int] = None):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size))
        frame = frame.astype('float32') / 255.0
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f'No frames read from {video_path}')
    arr = np.stack(frames, axis=0)  # [T,H,W,C]
    arr = np.transpose(arr, (0, 3, 1, 2))  # [T,C,H,W]
    return torch.from_numpy(arr)


class SimpleUNet1D(nn.Module):
    def __init__(self, n_mels: int = 80, cond_dim: int = 512, base_chan: int = 64):
        super().__init__()
        self.n_mels = n_mels
        self.cond_dim = cond_dim
        self.enc1 = nn.Sequential(
            nn.Conv1d(n_mels, base_chan, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_chan, base_chan, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.down1 = nn.Conv1d(base_chan, base_chan * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.ReLU(), nn.Conv1d(base_chan * 2, base_chan * 2, 3, padding=1), nn.ReLU())
        self.down2 = nn.Conv1d(base_chan * 2, base_chan * 4, kernel_size=4, stride=2, padding=1)

        self.mid = nn.Sequential(nn.ReLU(), nn.Conv1d(base_chan * 4, base_chan * 4, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose1d(base_chan * 4, base_chan * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.ReLU(), nn.Conv1d(base_chan * 4, base_chan * 2, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose1d(base_chan * 2, base_chan, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.ReLU(), nn.Conv1d(base_chan * 2, base_chan, 3, padding=1), nn.ReLU())
        self.out = nn.Conv1d(base_chan, n_mels, kernel_size=1)

        # conditioning projection
        self.cond_proj = nn.Linear(cond_dim, base_chan * 4)

    def forward(self, x, cond):
        # x: [B, n_mels, T]
        # cond: [B, cond_dim]
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        m = self.mid(d2)

        # add conditioning as broadcasted bias
        c = self.cond_proj(cond).unsqueeze(-1)  # [B, C, 1]
        if c.shape[1] != m.shape[1]:
            # adapt
            c = c[:, : m.shape[1], :]
        m = m + c

        u2 = self.up2(m)
        d2_cat = torch.cat([u2, e2], dim=1)
        dec2 = self.dec2(d2_cat)
        u1 = self.up1(dec2)
        d1_cat = torch.cat([u1, e1], dim=1)
        dec1 = self.dec1(d1_cat)
        out = self.out(dec1)
        return out


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ds = MelDataset(
        Path(args.mel_dir),
        cond_dir=Path(args.cond_dir) if args.cond_dir else None,
        lip_crop_dir=Path(args.lip_crop_dir) if getattr(args, 'lip_crop_dir', None) else None,
        max_frames=args.max_frames,
    )

    # custom collate: stack tensors/ndarrays but keep non-tensor items (paths/strings) as lists
    def _custom_collate(batch):
        out = {}
        for key in batch[0].keys():
            vals = [d[key] for d in batch]
            # torch tensors
            if all(isinstance(v, torch.Tensor) for v in vals):
                out[key] = torch.stack(vals, dim=0)
            # numpy arrays
            elif all(isinstance(v, np.ndarray) for v in vals):
                out[key] = torch.stack([torch.from_numpy(v) for v in vals], dim=0)
            # numbers
            elif all(isinstance(v, (int, float, complex, np.number)) for v in vals):
                out[key] = torch.tensor(vals)
            else:
                out[key] = vals
        return out

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=_custom_collate)

    # conditioning dim: may be overridden if using LSTM online mode
    cond_dim = args.cond_dim

    # optional: load LSTM for online conditioning (infer cond_dim from it)
    lstm_model = None
    if args.lstm_checkpoint and args.lip_crop_dir:
        from train_lip_lstm_ctc import VisualLSTMCTC
        lstm_model = VisualLSTMCTC(vocab_size=2)
        ckpt = torch.load(args.lstm_checkpoint, map_location='cpu')
        try:
            lstm_model.load_state_dict(ckpt)
        except Exception:
            lstm_model.load_state_dict(ckpt, strict=False)
        lstm_model.eval()
        lstm_model.to(device)
        cond_dim = lstm_model.lstm.hidden_size * 2
        print('Using LSTM online conditioning, cond_dim=', cond_dim)

    # instantiate model after determining `cond_dim`
    model = SimpleUNet1D(n_mels=args.n_mels, cond_dim=cond_dim, base_chan=args.base_chan).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    timesteps = args.timesteps
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    # dry-run: perform a single forward pass to verify shapes and data flow
    if args.dry_run:
        model.eval()
        try:
            batch = next(iter(loader))
        except Exception as e:
            print('Dry-run failed: unable to read batch from dataset:', e)
            return

        with torch.no_grad():
            mels = batch['mel'].to(device)
            if mels.ndim != 3:
                mels = mels.unsqueeze(1)

            B = mels.shape[0]

            # compute conditioning for dry run
            if 'cond' in batch:
                cond = batch['cond'].to(device)
            else:
                if lstm_model is None:
                    print('Dry-run: no precomputed cond and no LSTM provided; aborting dry-run')
                    return
                cond_list = []
                for i in range(B):
                    vpath = batch['video_path'][i]
                    if vpath is None:
                        cond_list.append(torch.zeros(cond_dim, dtype=torch.float32))
                        continue
                    video = load_video_frames(Path(vpath), frame_size=args.frame_size, max_frames=args.frame_size)
                    video = video.to(device)
                    flat = video.reshape(-1, video.shape[1], video.shape[2], video.shape[3])
                    feats = lstm_model.frame_encoder(flat)
                    seq = feats.reshape(1, feats.shape[0], feats.shape[1])
                    lstm_out, _ = lstm_model.lstm(seq)
                    hid = lstm_out.mean(dim=1).squeeze(0)
                    cond_list.append(hid.cpu())
                cond = torch.stack(cond_list, dim=0).to(device)

            # normalize and make noisy input
            mels_mean = mels.mean(dim=[1, 2], keepdim=True)
            mels_std = mels.std(dim=[1, 2], keepdim=True) + 1e-6
            mels_norm = (mels - mels_mean) / mels_std

            t = torch.randint(0, timesteps, (B,), device=device).long()
            a_t = alphas_cumprod[t].view(B, 1, 1)
            noise = torch.randn_like(mels_norm)
            noisy = torch.sqrt(a_t) * mels_norm + torch.sqrt(1 - a_t) * noise

            pred_noise = model(noisy, cond)

            print('Dry-run successful')
            print('mels:', tuple(mels.shape), mels.dtype, mels.device)
            print('cond:', tuple(cond.shape), cond.dtype, cond.device)
            print('noisy input:', tuple(noisy.shape), noisy.dtype, noisy.device)
            print('pred_noise:', tuple(pred_noise.shape), pred_noise.dtype, pred_noise.device)
        return

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            mels = batch['mel'].to(device)  # [B, n_mels, T]
            if mels.ndim == 3:
                pass
            else:
                mels = mels.unsqueeze(1)

            B = mels.shape[0]

            # compute conditioning
            if 'cond' in batch:
                cond = batch['cond'].to(device)
            else:
                cond_list = []
                for i in range(B):
                    vpath = batch['video_path'][i]
                    if vpath is None:
                        cond_list.append(torch.zeros(cond_dim, dtype=torch.float32))
                        continue
                    video = load_video_frames(Path(vpath), frame_size=args.frame_size, max_frames=args.frame_size)
                    video = video.to(device)
                    # compute frame features
                    with torch.no_grad():
                        flat = video.reshape(-1, video.shape[1], video.shape[2], video.shape[3])
                        feats = lstm_model.frame_encoder(flat)
                        seq = feats.reshape(1, feats.shape[0], feats.shape[1])
                        lstm_out, _ = lstm_model.lstm(seq)
                        hid = lstm_out.mean(dim=1).squeeze(0)
                        cond_list.append(hid.cpu())
                cond = torch.stack(cond_list, dim=0).to(device)

            # normalize mels per-sample
            mels_mean = mels.mean(dim=[1, 2], keepdim=True)
            mels_std = mels.std(dim=[1, 2], keepdim=True) + 1e-6
            mels_norm = (mels - mels_mean) / mels_std

            # sample noise and timesteps
            t = torch.randint(0, timesteps, (B,), device=device).long()
            a_t = alphas_cumprod[t].view(B, 1, 1)
            noise = torch.randn_like(mels_norm)
            noisy = torch.sqrt(a_t) * mels_norm + torch.sqrt(1 - a_t) * noise

            pred_noise = model(noisy, cond)
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f}")

        # checkpoint
        out_ckpt = Path(args.out_dir) / f"diffusion_epoch{epoch+1}.pt"
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1}, str(out_ckpt))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mel-dir', type=str, required=True)
    p.add_argument('--video-dir', type=str, default=None)
    p.add_argument('--cond-dir', type=str, default=None)
    p.add_argument('--lstm-checkpoint', type=str, default=None)
    p.add_argument('--out-dir', type=str, default='diffusion_results')
    p.add_argument('--n-mels', type=int, default=80)
    p.add_argument('--max-frames', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--cond-dim', type=int, default=512)
    p.add_argument('--base-chan', type=int, default=64)
    p.add_argument('--frame-size', type=int, default=224)
    p.add_argument('--dry-run', action='store_true', help='Run a single forward pass and exit')
    p.add_argument('--lip-crop-dir', type=str, default='lip_crop_results_full', help='Directory with lip-cropped videos (preferred for LSTM conditioning)')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
