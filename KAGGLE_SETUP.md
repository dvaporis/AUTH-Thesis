# Kaggle Dataset Integration Guide

This guide walks you through setting up access to the Kaggle dataset for audio testing.

## Dataset Information

**Dataset**: Audio-Visual Database of Emotional Speech and Song  
**ID**: `thbdh5765/audio-visual-database-of-emotional-speech-and-song`  
**Type**: Speech and Song recordings with emotional labels  
**Perfect for**: Audio-visual contrastive learning research

## Setup Instructions

### Step 1: Create a Kaggle Account

1. Visit: https://www.kaggle.com/signup
2. Sign up with email or Google/GitHub account
3. Verify your email

### Step 2: Get API Credentials

1. Go to: https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New API Token"
   - This downloads `kaggle.json` file
4. Keep this file safe (contains your credentials)

### Step 3: Configure Credentials

#### On Windows (Current Setup):

**Option A: Manual Setup (Recommended)**

1. Create directory: `C:\Users\{YourUsername}\.kaggle`
2. Move `kaggle.json` there
3. Make sure the file is not accessible to others:
   - Right-click → Properties → Security
   - Remove "Users" group if present
   - Only your user should have access

**Option B: Environment Variable**

```powershell
# Set environment variable
[Environment]::SetEnvironmentVariable("KAGGLE_CONFIG_DIR", "$env:USERPROFILE\.kaggle", "User")

# Restart PowerShell/Terminal
```

#### On macOS/Linux:

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Move credentials file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions (important for security)
chmod 600 ~/.kaggle/kaggle.json
```

### Step 4: Verify Setup

Test your credentials:

```bash
# Activate environment
.venv\Scripts\activate

# Test kaggle access
python -c "import kagglehub; print('✓ Kaggle setup working!')"
```

If this works, you're ready to download!

## Download the Dataset

### Quick Download

```bash
# Activate environment
.venv\Scripts\activate

# Download dataset
python download_kaggle_dataset.py
```

This will:
- Download ~2-5 GB of data (depending on your internet)
- Cache it in `kaggle_datasets/` folder
- List all audio and video files found
- Show you file locations

### What Gets Downloaded

The dataset includes:
- **Audio files**: Speech and song recordings
- **Video files**: Video versions of the above
- **Metadata**: Emotion labels, speaker info, etc.
- **Organized by**: Emotion category, speaker, or content type

First download takes: **5-30 minutes** depending on internet speed  
Subsequent runs: **Instant** (uses cache)

## Using the Dataset for Testing

### Option 1: Test with Random Kaggle Audio

```bash
python test_encodec_with_kaggle.py --use-kaggle
```

This:
- Randomly selects an audio file from Kaggle dataset
- Tests EnCodec on real speech/song audio
- Generates comparison plots with actual data

### Option 2: Test Specific Audio File

```bash
python test_encodec_with_kaggle.py --audio path/to/audio.wav
# Or use video file
python test_encodec_with_kaggle.py --audio path/to/video.mp4
```

### Option 3: Generate vs Real Comparison

```bash
# Test with generated signal (reproducible)
python test_encodec_simple.py

# Test with real Kaggle audio (realistic)
python test_encodec_with_kaggle.py --use-kaggle
```

Compare the metrics to see how EnCodec performs on:
- **Generated signal**: Clean, artificial
- **Real speech**: Natural variations, noise, emotion
- **Real song**: Music, pitch variations, harmony

## Expected Metrics Comparison

### Generated Test Signal
```
MSE: ~0.010
SNR: ~10 dB
PSNR: ~19.5 dB
Cosine Similarity: ~0.95
```

### Real Kaggle Audio (Speech/Song)
```
MSE: ~0.015-0.025  (slightly worse - more complex)
SNR: ~8-12 dB      (depends on content)
PSNR: ~18-20 dB    (similar range)
Cosine Similarity: ~0.92-0.96  (slightly lower)
```

Real audio will likely show slightly higher reconstruction error because:
- Natural speech/song has more complexity
- Background noise/reverb
- Wider frequency range used
- Emotional expression variations

This is **expected and normal** - it shows the codec performing realistically.

## Troubleshooting

### "Authentication Failed"

```
Error: 403 Unauthorized - you haven't accepted the dataset's terms
```

**Solution**:
1. Go to: https://www.kaggle.com/datasets/thbdh5765/audio-visual-database-of-emotional-speech-and-song
2. Click "Join" or accept the terms
3. Try download again

### "Dataset Not Found"

Possible causes:
1. Dataset ID is wrong
2. You don't have access
3. Internet connection issue

**Solution**:
```bash
# Try manual download from web
# Visit: https://www.kaggle.com/datasets/thbdh5765/audio-visual-database-of-emotional-speech-and-song
# Click download button
# Extract to: kaggle_datasets/
```

### No Audio Files Found

If `python download_kaggle_dataset.py` shows 0 audio files:

```bash
# Check downloaded folder structure
dir /s kaggle_datasets\

# List all files (including extensions)
dir /s /b kaggle_datasets\ > file_list.txt
```

The dataset might have different structure than expected. Check the output.

### Out of Disk Space

The dataset can be 2-5 GB. Ensure you have:
- At least 10 GB free space
- SSD preferred for faster extraction

```bash
# Check available space
wmic logicaldisk get name,freespace  # Windows
df -h                                 # Linux/Mac
```

### Slow Download

Normal download speeds:
- **Good**: 1-5 MB/s (finishes in 10-50 minutes)
- **Slow**: <1 MB/s (may timeout after 30+ mins)

If too slow:
1. Try at different time (less network congestion)
2. Use VPN if in restricted region
3. Download manually from web and extract locally

## File Organization

After download, dataset structure looks like:

```
kaggle_datasets/
├── models--thbdh5765--audio-visual-database-of-emotional-speech-and-song/
│   ├── snapshots/
│   │   └── {hash}/
│   │       ├── audio_files/
│   │       │   ├── happy/
│   │       │   │   ├── speaker1.wav
│   │       │   │   └── speaker2.wav
│   │       │   ├── sad/
│   │       │   └── ...
│   │       ├── video_files/
│   │       ├── metadata.csv
│   │       └── README.md
```

All audio and video files will be auto-discovered by scripts.

## Next Steps

### 1. Test Both Types of Audio

```bash
# Test generated signal
python test_encodec_simple.py

# Test real Kaggle audio (multiple runs get different files)
python test_encodec_with_kaggle.py --use-kaggle
python test_encodec_with_kaggle.py --use-kaggle
python test_encodec_with_kaggle.py --use-kaggle
```

Compare metrics and visualizations.

### 2. Extract Embeddings for Contrastive Learning

```python
# Example: Extract audio embeddings from Kaggle dataset
from transformers import AutoModel
import torch
from pathlib import Path

model = AutoModel.from_pretrained("facebook/encodec_48khz")
encoder = model.encoder

# Load all Kaggle audio
kaggle_files = list(Path("kaggle_datasets").rglob("*.wav"))

embeddings = []
for audio_file in kaggle_files[:100]:  # First 100 files
    waveform, sr = torchaudio.load(audio_file)
   # Resample to 48kHz and ensure stereo
   if sr != 48000:
      waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
   if waveform.shape[0] == 1:
      waveform = waveform.repeat(2, 1)
   elif waveform.shape[0] > 2:
      waveform = waveform[:2, :]
   waveform = waveform.unsqueeze(0)  # [1, 2, samples]
    with torch.no_grad():
      embedding = encoder(waveform)  # [1, 128, time]
    embeddings.append(embedding.mean(dim=-1))  # Pool time dimension

# Now you have audio embeddings for contrastive learning!
audio_embeddings = torch.cat(embeddings)  # [100, 128]
```

### 3. Get Video Files Too

If dataset has video:

```python
from pathlib import Path

video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
video_files = [f for f in Path("kaggle_datasets").rglob("*") 
               if f.suffix.lower() in video_extensions]

print(f"Found {len(video_files)} video files")
```

Next step: Test on matching video files and compare embeddings!

## Important Notes

- ✅ Dataset is free to use (requires account)
- ✅ Credentials are stored locally (never sent to code)
- ✅ Cache is persistent (download once, use many times)
- ⚠️ For research use, cite the original dataset creators
- ⚠️ Keep `kaggle.json` secure - don't share or commit to GitHub

## Getting Help

If you encounter issues:

1. Check Kaggle FAQ: https://www.kaggle.com/docs/api
2. Verify dataset access: https://www.kaggle.com/settings/account
3. Check credentials file exists in `~/.kaggle/kaggle.json`
4. Run test without authentication:
   ```bash
   python test_encodec_simple.py  # Works offline
   python test_encodec_with_kaggle.py --audio manual_file.wav  # Works without Kaggle
   ```

---

**Ready?** Run: `python download_kaggle_dataset.py`
