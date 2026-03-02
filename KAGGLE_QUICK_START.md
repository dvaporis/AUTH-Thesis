# Quick Start: Kaggle Dataset with EnCodec

## TL;DR - Get Started in 5 Minutes

### 1. Set Up Kaggle Credentials (One-time)

```bash
# Get API credentials from: https://www.kaggle.com/settings/account
# Save kaggle.json to: C:\Users\YourUsername\.kaggle\kaggle.json
```

### 2. Download Dataset

```bash
.venv\Scripts\activate
python download_kaggle_dataset.py
```

**First time**: 5-30 minutes (downloads ~2-5 GB)  
**Later times**: Instant (uses cache)

### 3. Test with Real Audio

```bash
# Test random Kaggle audio file
python test_encodec_with_kaggle.py --use-kaggle

# Test specific audio file
python test_encodec_with_kaggle.py --audio C:\path\to\audio.wav
```

### 4. View Results

Results saved to: `audio_encodec_results/`
- `original_audio.wav` - Your test audio
- `reconstructed_audio.wav` - After encoding/decoding
- `*_comparison.png` - Visual comparison
- `*_metrics.png` - Performance metrics

---

## Setup Kaggle Access

### Step 1: Create Account
Visit: https://www.kaggle.com/signup

### Step 2: Get API Token
1. Go to: https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. This downloads `kaggle.json`

### Step 3: Save Credentials
Save `kaggle.json` to: **`C:\Users\YourUsername\.kaggle\kaggle.json`**

(Where `YourUsername` is your Windows username)

### Step 4: Verify
```bash
python -c "import kagglehub; print('✓ Setup OK!')"
```

---

## Using the Dataset

### Download All Data

```bash
python download_kaggle_dataset.py
```

Output shows:
- Number of audio/video files found
- Total storage used
- File locations
- Sample file list

### Test Options

**Option A: Random file from dataset**
```bash
python test_encodec_with_kaggle.py --use-kaggle
```
Picks different file each time.

**Option B: Specific file**
```bash
python test_encodec_with_kaggle.py --audio your_audio.wav
```

**Option C: Generated test (no download needed)**
```bash
python test_encodec_simple.py
```

---

## What Happens When You Run

```
1. Loads EnCodec model from HuggingFace (~30s first time, instant after)
2. Loads your audio file
3. Compresses it (encoding)
4. Decompresses it (decoding)
5. Measures how different they are (error metrics)
6. Creates visualization plots
7. Saves results to audio_encodec_results/
```

Expected output:
```
Testing EnCodec Audio Codec
===================================================================
Loading audio from: kaggle_datasets/...speaker1.wav
Audio shape: torch.Size([1, 240000]), Sample rate: 24000Hz
...
Processing audio...
  Encoding...
  Decoding...

Reconstruction Quality Metrics:
--------------------------------------------------
  MSE                 :     0.010054
  MAE                 :     0.079996
  RMSE                :     0.100270
  SNR_dB              :    10.051548
  PSNR_dB             :    19.531021
  Cosine_Similarity   :     0.953989
--------------------------------------------------

✓ Saved results to audio_encodec_results/
✓ EnCodec test completed successfully!
```

---

## Understanding the Metrics

| Metric | Good Value | Interpretation |
|--------|-----------|-----------------|
| **RMSE** | < 0.15 | Reconstruction error (smaller = better) |
| **SNR** | 8-12 dB | Signal quality (higher = better) |
| **Cosine Sim** | > 0.95 | How similar (closer to 1 = better) |

For lossy codecs like EnCodec, these small errors are **expected and normal**.

---

## Dataset Contents

The dataset includes:
- **Audio recordings** of speech and singing
- **Emotion labels** (happy, sad, angry, etc.)
- **Video files** (optional)
- **Metadata** (speaker info, duration, etc.)

Total size: 2-5 GB depending on which files you download

---

## Troubleshooting

### "404: Dataset Not Found"
```
→ Make sure you accepted terms: 
  https://www.kaggle.com/datasets/thbdh5765/audio-visual-database-of-emotional-speech-and-song
→ Click "Join" or "I Understand" button
```

### "401: Unauthorized"
```
→ Check credentials file exists at: C:\Users\YourUsername\.kaggle\kaggle.json
→ Make sure it's readable only by you (check permissions)
```

### "No audio files found"
```
→ Check if download completed
→ Try: dir kaggle_datasets /s
→ Download is still in progress (wait a bit)
```

### Very slow download
```
→ Try at different time (less network load)
→ Manual download: 
   https://www.kaggle.com/datasets/thbdh5765/audio-visual-database-of-emotional-speech-and-song
   Extract to kaggle_datasets/
```

---

## What's Next After Testing?

### Test with Multiple Files
```bash
# Create a test script
for i in 1..5:
    python test_encodec_with_kaggle.py --use-kaggle
```

Compare metrics across different speakers/emotions.

### Extract Embeddings for Contrastive Learning
```python
# Get audio feature vectors for contrastive loss
embeddings = model.encoder(audio)  # [batch, 128, time]
```

### Test Video Encoder
Similar to this audio test, but for video files.

### Combine Audio + Video
```python
audio_embedding = audio_model.encoder(audio)
video_embedding = video_model(video)
# Align them with contrastive loss
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `download_kaggle_dataset.py` | Download dataset from Kaggle |
| `test_encodec_with_kaggle.py` | Test with Kaggle audio files |
| `test_encodec_simple.py` | Test with generated audio (no download) |
| `KAGGLE_SETUP.md` | Detailed setup instructions |
| `ENCODEC_README.md` | EnCodec documentation |
| `KAGGLE_QUICK_START.md` | This file |

---

## Summary

```
1. Set up Kaggle credentials (5 min, one-time)
2. Download dataset (10-30 min, first-time only)
3. Test with real audio (2-5 seconds per test)
4. Review results (automatic visualization)
5. Extract embeddings for your contrastive model
```

**Total time to first working test: ~1-2 hours**

---

**Ready?** 

```bash
# Step 1: Set up Kaggle (follow KAGGLE_SETUP.md)

# Step 2: Download
python download_kaggle_dataset.py

# Step 3: Test!
python test_encodec_with_kaggle.py --use-kaggle
```
