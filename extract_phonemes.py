#!/usr/bin/env python3
"""Extract timing-aware phoneme targets for GRID corpus clips.

Pipeline per clip
─────────────────
1. Decode the GRID filename  →  canonical sentence  (e.g. "bin blue at f 2 now")
2. Convert each word to its canonical eSpeak IPA phoneme sequence using
   the same phoneme inventory as facebook/wav2vec2-lv-60-espeak-cv-ft so
   that token identities match at training time.
3. Run wav2vec2 forced-alignment on the audio to get per-phoneme
   (start_sec, end_sec) timings.
4. Align the forced-alignment result against the canonical phoneme list
   with a DTW / edit-distance pass so that any acoustic mishap or
   deletion by the actor is corrected while preserving real timings.
5. Convert timings to video frame indices (25 FPS by default).
6. Write one CSV row per clip with columns:
       stem, sentence, canonical_phonemes, per_frame_labels

Output CSV columns
──────────────────
stem               : filename stem, e.g. bbaf2n
sentence           : decoded sentence, e.g. "bin blue at f 2 now"
canonical_phonemes : space-separated phoneme list from text, e.g. "b ɪ n b l uː ..."
per_frame_labels   : JSON list of length T (number of video frames), each
                     element is the canonical phoneme assigned to that frame,
                     or "" for silence / unassigned frames.

Dependencies
────────────
    pip install transformers torchaudio librosa espeak-ng phonemizer

    System: espeak-ng must be on PATH  (apt install espeak-ng  /  brew install espeak)

Usage
─────
    python extract_grid_phonemes_aligned.py --video-dir s1_lip_crops --output-dir phonemes_s1
    python extract_grid_phonemes_aligned.py --video-dir s1_lip_crops --output-dir phonemes_s1 --dry-run
    python extract_grid_phonemes_aligned.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# GRID filename decoding
# ──────────────────────────────────────────────────────────────────────────────

# Filename format: [cmd][col][prep][letter][digit][adv]
# Each slot is one character (letter or digit) EXCEPT digit which is a single digit.

GRID_COMMANDS = {
    "b": "bin", "l": "lay", "p": "place", "s": "set",
}
GRID_COLORS = {
    "b": "blue", "g": "green", "r": "red", "w": "white",
}
GRID_PREPS = {
    "a": "at", "b": "by", "i": "in", "w": "with",
}
# Letters a-z excluding 'w' (not used in GRID)
GRID_LETTERS = {c: c for c in "abcdefghijklmnopqrstuvxyz"}
GRID_DIGITS = {str(d): str(d) for d in range(1, 10)}
GRID_DIGITS["z"] = "zero"  # "0" is encoded as "z" in GRID filenames
GRID_ADVS = {
    "a": "again", "n": "now", "p": "please", "s": "soon",
}


def decode_grid_stem(stem: str) -> Optional[str]:
    """Turn a GRID filename stem into a plain-English sentence.

    Format: c c p l d a   (6 characters, digit may be 0-9)
    E.g.  bbaf2n  →  "bin blue at f 2 now"
    """
    # Strip any trailing _lipcrop suffix added by crop scripts
    stem = re.sub(r"_lipcrop$", "", stem)
    if len(stem) != 6:
        return None
    c, col, prep, letter, digit, adv = stem[0], stem[1], stem[2], stem[3], stem[4], stem[5]
    try:
        return " ".join([
            GRID_COMMANDS[c],
            GRID_COLORS[col],
            GRID_PREPS[prep],
            GRID_LETTERS[letter],
            GRID_DIGITS[digit],
            GRID_ADVS[adv],
        ])
    except KeyError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Canonical phoneme sequence from text via eSpeak-NG
# ──────────────────────────────────────────────────────────────────────────────

# Mapping from eSpeak IPA tokens → wav2vec2-lv-60-espeak token strings.
# wav2vec2-lv-60-espeak uses IPA symbols directly (space-separated), so in
# most cases the eSpeak output already matches.  We add a small normalisation
# table for the handful of surface differences observed in practice.
_ESPEAK_NORM: Dict[str, str] = {
    "ː": "",        # length mark: fold into the base vowel (wav2vec2 omits it)
    "\u0361": "",   # tie-bar ligature
    "ˈ": "",        # primary stress mark – wav2vec2 strips it
    "ˌ": "",        # secondary stress mark
    "ʔ": "",        # glottal stop – not in the model vocabulary, drop it
}

# Characters / tokens that should be dropped entirely from the eSpeak stream
_ESPEAK_DROP = {"(", ")", "[", "]", "'", ",", ".", "!", "?", ";", ":"}


def _run_espeak(text: str, voice: str = "en-gb") -> str:
    """Call espeak-ng and return the IPA string."""
    cmd = ["espeak-ng", "--ipa=1", "-q", f"-v{voice}", text]
    # capture raw bytes and decode as UTF-8 explicitly so that Windows
    # codepage settings (e.g. cp1253 on Greek locales) don't corrupt IPA output
    result = subprocess.run(cmd, capture_output=True, text=False, check=True)
    return result.stdout.decode("utf-8", errors="replace")


def text_to_canonical_phonemes(sentence: str, voice: str = "en-gb") -> List[str]:
    """Return a list of IPA phoneme tokens matching the wav2vec2 vocabulary.

    The eSpeak-NG IPA output is cleaned so that each token in the returned
    list appears in the wav2vec2-lv-60-espeak-cv-ft vocabulary.
    """
    raw = _run_espeak(sentence, voice=voice)
    # eSpeak returns newlines between words; collapse to spaces
    raw = raw.replace("\n", " ").strip()

    tokens: List[str] = []
    for ch in raw:
        if ch == " ":
            continue
        if ch in _ESPEAK_DROP:
            continue
        normalised = _ESPEAK_NORM.get(ch, ch)
        if normalised:
            tokens.append(normalised)

    # Some eSpeak versions emit multi-character digraphs (e.g. "dʒ").
    # wav2vec2-lv-60-espeak represents these as two separate tokens.
    # The split above already handles this because we iterate character-by-
    # character, which naturally decomposes every digraph.
    return [t for t in tokens if t.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
TARGET_SR = 16_000


def load_audio_16k(path: Path) -> torch.Tensor:
    """Load audio from a video or audio file, resample to 16 kHz mono.

    Tries backends in order:
      1. torchaudio with soundfile backend  (fast, works for .wav/.flac)
      2. torchaudio with ffmpeg backend     (needed for .mp4/.avi etc.)
      3. librosa + soundfile fallback       (most compatible on Windows)
      4. ffmpeg subprocess → temp wav       (last resort)
    """
    suffix = path.suffix.lower()

    # ── attempt 1: torchaudio with explicit ffmpeg backend (2.x API) ────────────
    try:
        waveform, sr = torchaudio.load(str(path.resolve()), backend="ffmpeg")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        return waveform.squeeze(0)
    except Exception:
        pass

    # ── attempt 2: PyAV (pip install av) — direct ffmpeg binding, no PATH needed
    try:
        import av
        import numpy as np
        container = av.open(str(path))
        audio_frames = []
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SR)
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                audio_frames.append(resampled.to_ndarray()[0])
        container.close()
        if audio_frames:
            audio_np = np.concatenate(audio_frames).astype(np.float32)
            return torch.from_numpy(audio_np)
    except Exception:
        pass

    # ── attempt 3: torchaudio legacy backend API ──────────────────────────────
    try:
        torchaudio.set_audio_backend("ffmpeg")
        waveform, sr = torchaudio.load(str(path.resolve()))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        return waveform.squeeze(0)
    except Exception:
        pass

    raise RuntimeError(
        f"All audio backends failed for {path}. "
        "Install PyAV as a fallback: pip install av"
    )


# ──────────────────────────────────────────────────────────────────────────────
# wav2vec2 forced alignment
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PhonemeSpan:
    token: str
    start_sec: float
    end_sec: float
    score: float = 0.0


def load_wav2vec2(model_name: str, device: torch.device):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()
    return extractor, model


def get_log_probs(
    waveform: torch.Tensor,
    extractor,
    model,
    device: torch.device,
) -> torch.Tensor:
    """Run wav2vec2 and return log-softmax probabilities [T_frames, V]."""
    inputs = extractor(
        waveform.numpy(), return_tensors="pt", sampling_rate=TARGET_SR
    )
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits  # [1, T, V]
    log_probs = torch.log_softmax(logits[0], dim=-1)  # [T, V]
    return log_probs.cpu()


def forced_align_phonemes(
    log_probs: torch.Tensor,
    target_token_ids: List[int],
    blank_id: int,
    sample_rate: int = TARGET_SR,
    hop_length_samples: int = 320,   # wav2vec2 feature stride: 20 ms at 16 kHz
) -> List[PhonemeSpan]:
    """Run CTC forced alignment using torchaudio.functional.forced_align.

    Returns a list of PhonemeSpan with timing in seconds.
    torchaudio ≥ 2.1 ships forced_align as a stable API.
    """
    try:
        from torchaudio.functional import forced_align as ta_forced_align
        from torchaudio.functional import merge_tokens
    except ImportError:
        raise ImportError(
            "torchaudio >= 2.1 is required for forced_align. "
            "Install with: pip install torchaudio --upgrade"
        )

    # forced_align expects [B, T, V] log_probs and [B, S] targets (batch-first)
    log_probs_batched = log_probs.unsqueeze(0)                                    # [1, T, V]
    targets_tensor = torch.tensor(target_token_ids, dtype=torch.int32).unsqueeze(0)  # [1, S]
    input_lengths = torch.tensor([log_probs.shape[0]], dtype=torch.int32)
    target_lengths = torch.tensor([len(target_token_ids)], dtype=torch.int32)

    # Returns (paths, scores) — both Tensors of shape [B, T]
    paths, scores = ta_forced_align(
        log_probs_batched, targets_tensor, input_lengths, target_lengths, blank=blank_id
    )

    # merge_tokens collapses the per-frame path into per-token TokenSpan objects
    # with .token (int), .start (int frame), .end (int frame, inclusive), .score
    token_spans = merge_tokens(paths[0], scores[0])

    spans_out: List[PhonemeSpan] = []
    sec_per_frame = hop_length_samples / sample_rate
    for span in token_spans:
        start_sec = float(span.start) * sec_per_frame
        end_sec = float(span.end + 1) * sec_per_frame  # end is inclusive frame index
        spans_out.append(PhonemeSpan(
            token=str(span.token),  # integer token id stored as string for later lookup
            start_sec=start_sec,
            end_sec=end_sec,
            score=float(span.score),
        ))
    return spans_out


# ──────────────────────────────────────────────────────────────────────────────
# Alignment correction: map canonical phoneme list onto audio-derived timings
# ──────────────────────────────────────────────────────────────────────────────

def _edit_distance_alignment(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Return list of (ref_idx, hyp_idx) pairs via standard edit-distance alignment.

    None in ref_idx  = insertion in hypothesis (no canonical match).
    None in hyp_idx  = deletion in hypothesis  (canonical phoneme not found in audio).
    """
    R, H = len(reference), len(hypothesis)
    # DP table
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        dp[i][0] = i
    for j in range(H + 1):
        dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrack
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))   # deletion
            i -= 1
        else:
            pairs.append((None, j - 1))   # insertion
            j -= 1
    pairs.reverse()
    return pairs


def map_canonical_to_spans(
    canonical_tokens: List[str],
    audio_spans: List[PhonemeSpan],
    id_to_token: Dict[int, str],
) -> List[PhonemeSpan]:
    """Return one PhonemeSpan per canonical token, with timing from the audio.

    Strategy
    ────────
    • Convert audio span token IDs back to IPA strings.
    • Edit-distance-align canonical list vs. audio list.
    • For each canonical token:
        – If matched to an audio span → use that span's timing.
        – If deleted (audio missed it) → interpolate between neighbours.
    • Insertions in the audio (spurious phonemes) are discarded.
    """
    audio_tokens = [id_to_token.get(int(s.token), "") for s in audio_spans]

    pairs = _edit_distance_alignment(canonical_tokens, audio_tokens)

    result: List[PhonemeSpan] = []
    for ref_idx, hyp_idx in pairs:
        if ref_idx is None:
            # Insertion in audio – skip
            continue
        canonical = canonical_tokens[ref_idx]
        if hyp_idx is not None:
            span = audio_spans[hyp_idx]
            result.append(PhonemeSpan(
                token=canonical,
                start_sec=span.start_sec,
                end_sec=span.end_sec,
                score=span.score,
            ))
        else:
            # Deletion – interpolate timing from neighbours
            result.append(PhonemeSpan(
                token=canonical,
                start_sec=-1.0,   # will be filled below
                end_sec=-1.0,
                score=0.0,
            ))

    # Fill interpolated timings for deleted phonemes
    _fill_missing_timings(result)
    return result


def _fill_missing_timings(spans: List[PhonemeSpan]) -> None:
    """In-place linear interpolation for spans with start_sec == -1."""
    n = len(spans)
    if n == 0:
        return

    # Forward pass: propagate last known end time
    last_known_end = 0.0
    missing_start: List[int] = []
    for i, span in enumerate(spans):
        if span.start_sec < 0:
            missing_start.append(i)
        else:
            # Fill any preceding missing spans linearly between last_known_end
            # and this span's start
            if missing_start:
                seg_dur = (span.start_sec - last_known_end) / (len(missing_start) + 1)
                for k, mi in enumerate(missing_start, start=1):
                    s = last_known_end + k * seg_dur
                    spans[mi].start_sec = round(s, 6)
                    spans[mi].end_sec = round(s + seg_dur, 6)
                missing_start.clear()
            last_known_end = span.end_sec

    # Any remaining missing spans sit at the very end of the utterance
    if missing_start:
        seg_dur = 0.04  # 40 ms fallback
        for k, mi in enumerate(missing_start):
            s = last_known_end + k * seg_dur
            spans[mi].start_sec = round(s, 6)
            spans[mi].end_sec = round(s + seg_dur, 6)


# ──────────────────────────────────────────────────────────────────────────────
# Convert phoneme spans → per-frame label sequence
# ──────────────────────────────────────────────────────────────────────────────

def spans_to_frame_labels(
    spans: List[PhonemeSpan],
    num_frames: int,
    fps: float = 25.0,
) -> List[str]:
    """Assign a canonical phoneme label to each video frame index.

    Frame i covers the time interval [i/fps, (i+1)/fps).
    A phoneme is assigned to a frame if its span overlaps the frame's interval
    by more than half the frame duration (majority-overlap rule).
    Frames with no overlapping phoneme get the empty string "".
    """
    frame_dur = 1.0 / fps
    labels = [""] * num_frames

    for span in spans:
        # First and last frame indices that overlap with this span
        first_frame = int(span.start_sec * fps)
        last_frame = int(span.end_sec * fps)   # exclusive-ish

        for f in range(max(0, first_frame), min(num_frames, last_frame + 1)):
            frame_start = f * frame_dur
            frame_end = frame_start + frame_dur
            overlap = min(span.end_sec, frame_end) - max(span.start_sec, frame_start)
            if overlap > 0.5 * frame_dur:
                labels[f] = span.token

    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Video frame count utility
# ──────────────────────────────────────────────────────────────────────────────

def get_frame_count(video_path: Path) -> Optional[int]:
    """Return total frame count for a video file using OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count if count > 0 else None
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

ALL_MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS


def find_media_files(directory: Path) -> List[Path]:
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in ALL_MEDIA_EXTS
    )


def normalize_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_lipcrop"):
        return stem[: -len("_lipcrop")]
    return stem


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract timing-aware phoneme targets for GRID corpus clips."
    )
    p.add_argument("--video-dir", default="s1_lip_crops",
                   help="Directory containing lip-crop videos (silent crops are fine).")
    p.add_argument("--source-dir", default="",
                   help=(
                       "Directory containing the original GRID .mpg files with audio. "
                       "Required when --video-dir contains silent lip crops. "
                       "The script matches each crop stem (e.g. bbaf2n) to the "
                       "corresponding source file (e.g. bbaf2n.mpg) in this directory."
                   ))
    p.add_argument("--output-dir", default="phonemes_s1_aligned",
                   help="Where to write the output CSV.")
    p.add_argument("--model-name", default="facebook/wav2vec2-lv-60-espeak-cv-ft",
                   help="Hugging Face model ID for wav2vec2 phoneme CTC.")
    p.add_argument("--fps", type=float, default=25.0,
                   help="Video frame rate used to compute per-frame labels.")
    p.add_argument("--espeak-voice", default="en-gb",
                   help="eSpeak-NG voice for canonical phonemisation.")
    p.add_argument("--dry-run", action="store_true",
                   help="Process only the first file and print results; do not write CSV.")
    p.add_argument("--max-files", type=int, default=0,
                   help="Cap on number of files to process (0 = all).")
    p.add_argument("--device", default="",
                   help="Force torch device ('cpu', 'cuda', 'mps'). Auto-detected if empty.")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Using device: %s", device)

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a stem → source-file lookup when a separate audio source dir is given.
    # GRID .mpg files live in source_dir (possibly in per-speaker sub-folders).
    source_index: Dict[str, Path] = {}
    if args.source_dir:
        source_dir = Path(args.source_dir)
        for src_file in source_dir.rglob("*"):
            if src_file.is_file() and src_file.suffix.lower() in ALL_MEDIA_EXTS:
                source_index[src_file.stem.lower()] = src_file
        log.info("Built source-file index: %d file(s) in %s", len(source_index), source_dir)

    files = find_media_files(video_dir)
    if not files:
        raise FileNotFoundError(f"No media files found in {video_dir}")

    if args.dry_run:
        files = files[:1]
    elif args.max_files > 0:
        files = files[: args.max_files]

    log.info("Found %d file(s) to process.", len(files))

    # Load model once
    log.info("Loading wav2vec2 model: %s", args.model_name)
    extractor, model = load_wav2vec2(args.model_name, device)

    # Build token lookup tables from the model vocab
    from transformers import Wav2Vec2CTCTokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_name)
    blank_id: int = tokenizer.pad_token_id  # wav2vec2 CTC blank == pad token
    vocab: Dict[str, int] = tokenizer.get_vocab()
    id_to_token: Dict[int, str] = {v: k for k, v in vocab.items()}
    # Remove special tokens that don't appear in eSpeak output
    _SPECIAL = {"<pad>", "<s>", "</s>", "<unk>", "|"}

    # ── per-file processing ───────────────────────────────────────────────────
    rows: List[Dict] = []
    failed: List[str] = []

    for path in files:
        stem = normalize_stem(path)
        sentence = decode_grid_stem(stem)
        if sentence is None:
            log.warning("Cannot decode GRID stem '%s' – skipping.", stem)
            failed.append(stem)
            continue

        log.info("Processing %s  →  \"%s\"", path.name, sentence)

        # 1. Canonical phoneme sequence from text
        try:
            canonical = text_to_canonical_phonemes(sentence, voice=args.espeak_voice)
        except Exception as exc:
            log.error("eSpeak failed for '%s': %s", stem, exc)
            failed.append(stem)
            continue

        if not canonical:
            log.warning("No canonical phonemes for '%s' – skipping.", stem)
            failed.append(stem)
            continue

        # 2. Load audio — use the original source file if crops are silent
        audio_path = path
        if source_index:
            matched = source_index.get(stem.lower())
            if matched is not None:
                audio_path = matched
                log.debug("  audio source: %s", audio_path.name)
            else:
                log.warning("  No source file found for stem '%s' in source-dir – "
                            "trying lip-crop file directly.", stem)
        try:
            waveform = load_audio_16k(audio_path)
        except Exception as exc:
            log.error("Audio load failed for '%s' (tried %s): %s", stem, audio_path, exc)
            failed.append(stem)
            continue

        # 3. Run wav2vec2 to get log-probs
        try:
            log_probs = get_log_probs(waveform, extractor, model, device)
        except Exception as exc:
            log.error("wav2vec2 inference failed for '%s': %s", stem, exc)
            failed.append(stem)
            continue

        # 4. Forced alignment – convert canonical phoneme strings → token IDs
        canonical_ids: List[int] = []
        canonical_valid: List[str] = []
        for ph in canonical:
            if ph in _SPECIAL:
                continue
            tid = vocab.get(ph)
            if tid is None:
                log.debug("  phoneme '%s' not in vocab – dropping from target", ph)
                continue
            canonical_ids.append(tid)
            canonical_valid.append(ph)

        if not canonical_ids:
            log.warning("All canonical phonemes absent from vocab for '%s' – skipping.", stem)
            failed.append(stem)
            continue

        try:
            audio_spans = forced_align_phonemes(log_probs, canonical_ids, blank_id)
        except Exception as exc:
            log.error("Forced alignment failed for '%s': %s", stem, exc)
            failed.append(stem)
            continue

        # 5. Map canonical tokens onto audio-derived timings (correcting actor deviations)
        try:
            final_spans = map_canonical_to_spans(canonical_valid, audio_spans, id_to_token)
        except Exception as exc:
            log.error("Span mapping failed for '%s': %s", stem, exc)
            failed.append(stem)
            continue

        # 6. Convert to per-frame labels
        num_frames: Optional[int] = None
        if path.suffix.lower() in VIDEO_EXTS:
            num_frames = get_frame_count(path)
        if num_frames is None:
            # Estimate from audio duration and FPS
            duration_sec = waveform.shape[0] / TARGET_SR
            num_frames = max(1, int(round(duration_sec * args.fps)))

        frame_labels = spans_to_frame_labels(final_spans, num_frames, fps=args.fps)

        # Diagnostics
        if args.dry_run:
            print(f"\n{'─'*60}")
            print(f"File    : {path.name}")
            print(f"Sentence: {sentence}")
            print(f"Canonical phonemes ({len(canonical_valid)}): {' '.join(canonical_valid)}")
            print(f"Audio spans ({len(audio_spans)}):")
            for sp in audio_spans:
                print(f"  {id_to_token.get(int(sp.token), '?'):>6}  "
                      f"{sp.start_sec:.3f}s – {sp.end_sec:.3f}s  "
                      f"score={sp.score:.3f}")
            print(f"Final aligned spans ({len(final_spans)}):")
            for sp in final_spans:
                print(f"  {sp.token:>6}  {sp.start_sec:.3f}s – {sp.end_sec:.3f}s")
            print(f"Frame labels (first 30 of {num_frames}): {frame_labels[:30]}")

        rows.append({
            "stem": stem,
            "sentence": sentence,
            "canonical_phonemes": " ".join(canonical_valid),
            "num_frames": num_frames,
            "per_frame_labels": json.dumps(frame_labels),
            "spans_json": json.dumps([
                {"token": sp.token,
                 "start_sec": round(sp.start_sec, 6),
                 "end_sec": round(sp.end_sec, 6),
                 "score": round(sp.score, 4)}
                for sp in final_spans
            ]),
        })

    # ── write CSV ─────────────────────────────────────────────────────────────
    if rows and not args.dry_run:
        csv_path = output_dir / "phoneme_predictions.csv"
        fieldnames = ["stem", "sentence", "canonical_phonemes",
                      "num_frames", "per_frame_labels", "spans_json"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Wrote %d rows → %s", len(rows), csv_path)
    elif args.dry_run:
        log.info("Dry run complete – no CSV written.")
    else:
        log.warning("No rows were produced – check errors above.")

    if failed:
        fail_path = output_dir / "failed_stems.txt"
        if not args.dry_run:
            fail_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
        log.warning("%d file(s) failed – stems: %s", len(failed), ", ".join(failed[:10]))


if __name__ == "__main__":
    main()