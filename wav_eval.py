import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


# ==========================
# CONFIG
# ==========================
CTC_JSON = "viseme_results/ctc_outputs.json"
OUTPUT_DIR = Path("ctc_audio_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS = 25.0
SAMPLE_RATE = 22050

# Set this if espeak-ng is not in PATH
ESPEAK_EXE = "espeak-ng"


# ==========================
# PHONEME -> ESPEAK IPA MAP
# ==========================
# Your model vocab uses simplified IPA.
# eSpeak expects slightly different IPA symbols sometimes.
PHONEME_MAP = {
    "a": "a",
    "b": "b",
    "d": "d",
    "e": "e",
    "f": "f",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "z": "z",
    "ð": "ð",
    "ɐ": "ɐ",
    "ɑ": "ɑ",
    "ɒ": "ɒ",
    "ɔ": "ɔ",
    "ə": "ə",
    "ɛ": "ɛ",
    "ɡ": "g",   # espeak usually wants ASCII g
    "ɪ": "ɪ",
    "ɹ": "r",
    "ʃ": "ʃ",
    "ʊ": "ʊ",
    "ʒ": "ʒ",
    "θ": "θ",
}


# ==========================
# UTIL
# ==========================
def collapse_ctc(tokens):
    """
    Collapse repeats and remove blanks.
    Returns list of (token, start_frame, end_frame)
    """
    collapsed = []
    prev = None
    start = None

    for i, tok in enumerate(tokens):
        if tok == "<blank>":
            if prev is not None:
                collapsed.append((prev, start, i - 1))
                prev = None
                start = None
            continue

        if tok == prev:
            continue

        if prev is not None:
            collapsed.append((prev, start, i - 1))

        prev = tok
        start = i

    if prev is not None:
        collapsed.append((prev, start, len(tokens) - 1))

    return collapsed


def generate_phoneme_wav(phoneme):
    """
    Generate robotic phoneme sound using espeak-ng.
    Returns numpy waveform.
    """
    ipa = PHONEME_MAP.get(phoneme, phoneme)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        cmd = [
            ESPEAK_EXE,
            "-v", "en",
            "-s", "120",
            "-p", "50",
            "--ipa",
            ipa,
            "-w",
            wav_path,
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        audio, sr = sf.read(wav_path)

        if sr != SAMPLE_RATE:
            # crude resample if needed
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, int(len(audio) * SAMPLE_RATE / sr))
            audio = np.interp(x_new, x_old, audio)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio.astype(np.float32)

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def time_stretch(audio, target_samples):
    """
    Simple resampler to fit audio exactly into desired duration.
    """
    if len(audio) == target_samples:
        return audio

    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_samples)
    stretched = np.interp(x_new, x_old, audio)

    return stretched.astype(np.float32)


def synthesize_sample(sample):
    stem = sample["stem"]
    frames = sample["frame_predictions"]

    collapsed = collapse_ctc(frames)

    total_duration = len(frames) / FPS
    total_samples = int(total_duration * SAMPLE_RATE)

    output = np.zeros(total_samples, dtype=np.float32)

    print(f"\nProcessing {stem}")
    print("Collapsed sequence:")
    print([x[0] for x in collapsed])

    for phoneme, start_f, end_f in collapsed:
        duration_sec = (end_f - start_f + 1) / FPS
        target_samples = max(1, int(duration_sec * SAMPLE_RATE))

        try:
            ph_audio = generate_phoneme_wav(phoneme)
        except Exception as e:
            print(f"Skipping {phoneme}: {e}")
            continue

        ph_audio = time_stretch(ph_audio, target_samples)

        start_sample = int(start_f / FPS * SAMPLE_RATE)
        end_sample = min(start_sample + len(ph_audio), len(output))

        output[start_sample:end_sample] += ph_audio[:end_sample - start_sample]

    # Normalize
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp * 0.9

    out_path = OUTPUT_DIR / f"{stem}.wav"
    sf.write(out_path, output, SAMPLE_RATE)
    print(f"Saved: {out_path}")


def main():
    with open(CTC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        synthesize_sample(sample)


if __name__ == "__main__":
    main()