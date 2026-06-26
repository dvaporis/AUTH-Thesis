"""decode_grid_ctc.py — Improved GRID corpus CTC decoder.

Improvements over the baseline:
  1. Phoneme-feature-weighted edit distance: partial credit for articulatorily
     similar phonemes (e.g. ð/θ, p/b, s/ʃ) so rare-but-distinctive phonemes
     actually influence word selection.
  2. Rare-phoneme bonus: if a phonetically distinctive phoneme (one that occurs
     in few GRID words) is detected, candidate words that contain it get a
     strong additive bonus, biasing the match toward the rare-phoneme word.
  3. Ground-truth extraction from GRID filename stems
     (e.g. bbaf2n -> bin blue at f 2 now).
  4. Per-slot and overall accuracy metrics, plus a per-sample breakdown CSV.
  5. Subtitle burning: for 3 automatically-selected videos (one perfect, one
     near-perfect, one bad) the script creates two sets of .srt subtitles and
     uses ffmpeg to burn them onto the source video.

Usage:
    python decode_grid_ctc.py --ctc-json viseme_results/ctc_outputs.json \
                              --video-dir s1 \
                              --output-dir decode_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# GRID vocabulary and grammar
# ---------------------------------------------------------------------------

BLANK = "<blank>"

VOCAB: Dict[str, Dict[str, List[str]]] = {
    "command": {
        "bin":   ["b", "ɪ", "n"],
        "lay":   ["l", "e", "ɪ"],
        "place": ["p", "l", "e", "ɪ", "s"],
        "set":   ["s", "ɛ", "t"],
    },
    "color": {
        "blue":  ["b", "l", "u"],
        "green": ["ɡ", "ɹ", "i", "n"],
        "red":   ["ɹ", "ɛ", "d"],
        "white": ["w", "a", "ɪ", "t"],
    },
    "prep": {
        "at":   ["a", "t"],
        "by":   ["b", "a", "ɪ"],
        "in":   ["ɪ", "n"],
        "with": ["w", "ɪ", "ð"],
    },
    "adverb": {
        "again":  ["ɐ", "ɡ", "ɛ", "n"],
        "now":    ["n", "a", "ʊ"],
        "please": ["p", "l", "i", "z"],
        "soon":   ["s", "u", "n"],
    },
}

# Letters (a-z, their IPA pronunciations)
_LETTERS: Dict[str, List[str]] = {
    "a": ["ɐ"],
    "b": ["b", "i"],
    "c": ["s", "i"],
    "d": ["d", "i"],
    "e": ["i"],
    "f": ["ɛ", "f"],
    "g": ["d", "ʒ", "i"],
    "h": ["e", "ɪ", "t", "ʃ"],
    "i": ["a", "ɪ"],
    "j": ["d", "ʒ", "e", "ɪ"],
    "k": ["k", "e", "ɪ"],
    "l": ["ɛ", "l"],
    "m": ["ɛ", "m"],
    "n": ["ɛ", "n"],
    "o": ["ə", "ʊ"],
    "p": ["p", "i"],
    "q": ["k", "j", "u"],
    "r": ["ɑ", "ɹ"],
    "s": ["ɛ", "s"],
    "t": ["t", "i"],
    "u": ["j", "ʊ"],
    "v": ["v", "i"],
    "w": ["d", "ʌ", "b", "ə", "ɫ", "j", "u"],
    "x": ["ɛ", "k", "s"],
    "y": ["w", "a", "ɪ"],
    "z": ["z", "ɛ", "d"],
}
VOCAB["letter"] = _LETTERS

VOCAB["digit"] = {
    "zero": ["z", "i", "ə", "ɹ", "ə", "ʊ"],
    "1":    ["w", "ɒ", "n"],
    "2":    ["t", "u"],
    "3":    ["θ", "ɹ", "i"],
    "4":    ["f", "ɔ"],
    "5":    ["f", "a", "ɪ", "v"],
    "6":    ["s", "ɪ", "k", "s"],
    "7":    ["s", "ɛ", "v", "ə", "n"],
    "8":    ["e", "ɪ", "t"],
    "9":    ["n", "a", "ɪ", "n"],
}

GRAMMAR_ORDER = ["command", "color", "prep", "letter", "digit", "adverb"]

# ---------------------------------------------------------------------------
# GRID filename parser → ground-truth words
# ---------------------------------------------------------------------------

_GRID_CMD  = {"b": "bin",   "l": "lay",   "p": "place", "s": "set"}
_GRID_COL  = {"b": "blue",  "g": "green", "r": "red",   "w": "white"}
_GRID_PREP = {"a": "at",    "b": "by",    "i": "in",    "w": "with"}
_GRID_ADV  = {"a": "again", "n": "now",   "p": "please","s": "soon"}
_GRID_DIG  = {
    "z": "zero", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5",    "6": "6", "7": "7", "8": "8", "9": "9",
}


def parse_grid_stem(stem: str) -> Optional[List[str]]:
    """Return the 6-word GRID sentence encoded in *stem*, or None if unparseable.

    GRID stem format: [cmd][color][prep][letter][digit][adverb]
    Examples: bbaf2n → bin blue at f 2 now
              prwz9p → place red with z 9 please
    """
    s = stem.lower().strip()
    s = re.sub(r"^s\d+_", "", s)      # strip speaker prefix  s1_...
    s = s.replace("_lipcrop", "")     # strip lipcrop suffix

    if len(s) < 6:
        return None

    cmd  = _GRID_CMD.get(s[0])
    col  = _GRID_COL.get(s[1])
    prep = _GRID_PREP.get(s[2])
    let  = s[3] if s[3] in _LETTERS else None
    dig  = _GRID_DIG.get(s[4])
    adv  = _GRID_ADV.get(s[5])

    if None in (cmd, col, prep, let, dig, adv):
        return None
    return [cmd, col, prep, let, dig, adv]

# ---------------------------------------------------------------------------
# Phoneme feature tables for weighted edit distance
# ---------------------------------------------------------------------------

# (place, manner, voiced, rounded)
# Consonants: place 0-7 as before.
# Vowels: place encodes height+backness to distinguish e.g. e from ɛ, ɪ from ɛ.
#   8  = close-front  (i, ɪ)
#   9  = close-mid front (e, eɪ)
#   10 = open-mid front  (ɛ)
#   11 = open front      (æ, a, aɪ, aʊ)
#   12 = central         (ə, ɐ, ʌ, ɜ, ɚ)
#   13 = close back      (u, ʊ)
#   14 = mid back        (o, ɔ, oʊ, ɔɪ)
#   15 = open back       (ɑ, ɒ)
# manner: 0=plosive 1=fricative 2=affricate 3=nasal 4=approximant 5=lateral 6=vowel
PHONEME_FEATURES: Dict[str, Tuple[int,int,int,int]] = {
    "p":(0,0,0,0),"b":(0,0,1,0),"m":(0,3,1,0),
    "f":(1,1,0,0),"v":(1,1,1,0),
    "θ":(2,1,0,0),"ð":(2,1,1,0),
    "t":(3,0,0,0),"d":(3,0,1,0),"n":(3,3,1,0),
    "s":(3,1,0,0),"z":(3,1,1,0),"l":(3,5,1,0),"ɫ":(3,5,1,0),
    "ɹ":(3,4,1,0),"r":(3,4,1,0),
    "ʃ":(4,1,0,0),"ʒ":(4,1,1,0),"tʃ":(4,2,0,0),"dʒ":(4,2,1,0),
    "j":(5,4,1,0),
    "k":(6,0,0,0),"g":(6,0,1,0),"ŋ":(6,3,1,0),"ɡ":(6,0,1,0),
    "h":(7,1,0,0),
    "w":(0,4,1,1),
    # Vowels — height-aware place index prevents false identity matches
    "i":(8,6,1,0),"ɪ":(8,6,1,0),"iː":(8,6,1,0),
    "e":(9,6,1,0),"eɪ":(9,6,1,0),
    "ɛ":(10,6,1,0),
    "æ":(11,6,1,0),"a":(11,6,1,0),"aɪ":(11,6,1,0),"aʊ":(11,6,1,0),
    "ə":(12,6,1,0),"ɐ":(12,6,1,0),"ʌ":(12,6,1,0),"ɜ":(12,6,1,0),"ɚ":(12,6,1,0),
    "u":(13,6,1,1),"uː":(13,6,1,1),"ʊ":(13,6,1,1),
    "o":(14,6,1,1),"oʊ":(14,6,1,1),"ɔ":(14,6,1,1),"ɔɪ":(14,6,1,1),
    "ɑ":(15,6,1,0),"ɒ":(15,6,1,0),
}

# Viseme groups: phonemes that look the same on lips.
# Vowel groups are coarser (lip aperture / rounding) since fine height
# differences are invisible; the feature table handles sub-group scoring.
VISEME_GROUPS: Dict[str, set] = {
    "bilabial":     {"p","b","m"},
    "labiodental":  {"f","v"},
    "dental":       {"θ","ð"},
    "alveolar":     {"t","d","n","s","z","l","ɹ","r","ɫ"},
    "postalveolar": {"ʃ","ʒ","tʃ","dʒ"},
    "velar":        {"k","g","ŋ","ɡ"},
    "glottal":      {"h"},
    "approx_w":     {"w"},
    "palatal":      {"j"},
    # Vowels grouped by visible lip shape (spread/neutral vs rounded, aperture)
    "close_vowel":  {"i","ɪ","iː","e","eɪ","u","uː","ʊ"},
    "mid_vowel":    {"ɛ","æ","ə","ɐ","ʌ","ɜ","ɚ","ɔ","o","oʊ"},
    "open_vowel":   {"a","ɑ","ɒ","aɪ","aʊ","ɔɪ"},
}

_PHONEME_TO_VISEME: Dict[str, str] = {
    ph: grp for grp, phones in VISEME_GROUPS.items() for ph in phones
}


def phoneme_substitution_cost(a: str, b: str) -> float:
    """Cost in [0, 1] for substituting phoneme *a* with phoneme *b*."""
    if a == b:
        return 0.0

    va = _PHONEME_TO_VISEME.get(a)
    vb = _PHONEME_TO_VISEME.get(b)
    fa = PHONEME_FEATURES.get(a)
    fb = PHONEME_FEATURES.get(b)

    # Same viseme class → visually indistinguishable
    if va and va == vb:
        if fa and fb and fa[:2] == fb[:2]:
            return 0.15   # same place+manner, only voicing differs
        return 0.35       # same lip shape but different articulation

    # Feature-based cost
    if fa and fb:
        # Consonant ↔ vowel boundary is very expensive.
        # Vowels now use place indices 8-15; consonants use 0-7.
        cv_a = (fa[0] >= 8)
        cv_b = (fb[0] >= 8)
        if cv_a != cv_b:
            return 0.95

        diffs = sum(int(x != y) for x, y in zip(fa, fb))
        return min(1.0, diffs / 4.0)

    return 1.0


def weighted_edit_distance(observed: List[str], target: List[str]) -> float:
    """Length-normalised edit distance with phoneme-feature substitution costs.

    Gap cost = 1.0 (insertion / deletion), substitution cost ∈ [0, 1].

    The raw distance is divided by max(len(observed), len(target)) so that
    a short word never wins simply because it requires fewer gap insertions
    than a longer correct word.  Without normalisation, e.g. ["ɹ","i","n"]
    (green with its initial ɡ dropped by CTC) scores lower against "red"
    (3 phones) than against "green" (4 phones), causing systematic errors.
    """
    n, m = len(observed), len(target)
    if n == 0 and m == 0:
        return 0.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = phoneme_substitution_cost(observed[i - 1], target[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + sub_cost,
            )
    raw = dp[n][m]
    return raw / max(n, m)


# ---------------------------------------------------------------------------
# Rare-phoneme bonus
#
# Some phonemes appear in very few GRID words.  If the model predicts such a
# phoneme we should trust it strongly.  We compute, for every phoneme, how
# many distinct words across the full GRID vocabulary contain it, then define
# a bonus inversely proportional to that count (capped at a maximum value).
# ---------------------------------------------------------------------------

def _build_rare_phoneme_index() -> Dict[str, Dict[str, set]]:
    """For each slot category, map phoneme → set of words that contain it."""
    index: Dict[str, Dict[str, set]] = {}
    for cat, words in VOCAB.items():
        ph_to_words: Dict[str, set] = defaultdict(set)
        for word, phones in words.items():
            for ph in set(phones):
                ph_to_words[ph].add(word)
        index[cat] = dict(ph_to_words)
    return index


_RARE_INDEX = _build_rare_phoneme_index()


def _build_anchor_index() -> Dict[str, Dict[str, List[str]]]:
    """For each slot category, map word → list of phonemes that appear in
    that word but in NO other word in the same category.

    These "anchor" phonemes are the most reliable signal: if the model predicts
    one of them, the corresponding word should be strongly preferred even if
    the overall edit distance is not the lowest.
    """
    index: Dict[str, Dict[str, List[str]]] = {}
    for cat, words in VOCAB.items():
        from collections import Counter as _Counter
        all_phones: List[str] = []
        for phones in words.values():
            all_phones.extend(phones)
        counts = _Counter(all_phones)
        cat_anchors: Dict[str, List[str]] = {}
        for word, phones in words.items():
            cat_anchors[word] = [p for p in set(phones) if counts[p] == 1]
        index[cat] = cat_anchors
    return index


_ANCHOR_INDEX = _build_anchor_index()


def anchor_bonus(
    observed_phones: List[str],
    candidate_word: str,
    category: str,
) -> float:
    """Return a score reduction when anchor phonemes of *candidate_word* are
    detected in *observed_phones*.

    Anchor phonemes are those that appear in exactly one word within their
    category (e.g. ɡ only in "green", ð only in "with", u only in "blue").
    Detecting one is near-conclusive evidence for the corresponding word.

    Two-tier matching:
      Exact match (cost = 0.0)            → +0.50 per anchor phoneme
      Voicing-only near-match (cost ≤ 0.15) → +0.25 per anchor phoneme
        (e.g. p↔b, t↔d, s↔z, f↔v, θ↔ð, i↔ɪ/e)

    The threshold is deliberately tight (≤ 0.15) to avoid false positives
    from phonemes that share only broad place of articulation — e.g. ɹ and d
    are both alveolar but cost 0.35, so ɹ does not fire the anchor for "red".
    """
    cat_anchors = _ANCHOR_INDEX.get(category, {})
    anchors = cat_anchors.get(candidate_word, [])
    if not anchors:
        return 0.0

    total = 0.0
    for anchor_ph in anchors:
        for obs_ph in observed_phones:
            c = phoneme_substitution_cost(obs_ph, anchor_ph)
            if c == 0.0:
                total += 0.50
                break
            elif c <= 0.15:
                total += 0.25
                break
    return total


def rare_phoneme_bonus(
    observed_phones: List[str],
    candidate_word: str,
    category: str,
    max_bonus: float = 1.5,
) -> float:
    """Return a cost *reduction* to apply when the candidate word contains
    phonemes that are distinctively rare across its category vocabulary.

    The bonus rewards candidates whose rare phonemes are actually present in
    the observed sequence, helping rare words like "with" (contains ð) beat
    more common-looking candidates like "in".
    """
    cat_index = _RARE_INDEX.get(category, {})
    total_words_in_cat = len(VOCAB.get(category, {}))
    if total_words_in_cat == 0:
        return 0.0

    candidate_phones = set(VOCAB[category][candidate_word])
    obs_set = set(observed_phones)

    bonus = 0.0
    for ph in candidate_phones:
        words_with_ph = cat_index.get(ph, set())
        rarity = 1.0 - len(words_with_ph) / total_words_in_cat  # 0=common 1=unique
        if rarity < 0.01:
            continue   # phoneme is in every word, no discriminative value
        # Reward matching: check if observed sequence contains this rare phoneme
        # or a phonetically very close one (same viseme group)
        matched = False
        for obs_ph in obs_set:
            if obs_ph == ph:
                matched = True
                break
            # Accept near-match within same place+manner class (voicing only)
            if phoneme_substitution_cost(obs_ph, ph) <= 0.15:
                matched = True
                break
        if matched:
            bonus += rarity * max_bonus

    return min(bonus, max_bonus)


# ---------------------------------------------------------------------------
# Best word matching within a GRID category
# ---------------------------------------------------------------------------

def best_word(
    segment: List[str],
    category: str,
) -> Tuple[str, float]:
    """Return (best_word, adjusted_score) for *segment* in *category*.

    Score = normalised_weighted_edit_distance(segment, candidate_phones)
              − anchor_bonus(...)      # strong: unique phoneme detected
              − rare_phoneme_bonus(...)  # weaker: rare phoneme detected
    Lower is better.

    Using the normalised distance prevents shorter candidate words from
    winning simply because they need fewer gap insertions than longer ones.
    The anchor bonus gives a decisive nudge when a phoneme that uniquely
    identifies a candidate (e.g. ɡ for "green", ð for "with", u for "blue")
    is present in the observed sequence or a near-match is.
    """
    best_word_str = ""
    best_score = math.inf

    for word, phones in VOCAB[category].items():
        dist  = weighted_edit_distance(segment, phones)   # already normalised
        ab    = anchor_bonus(segment, word, category)
        rb    = rare_phoneme_bonus(segment, word, category)
        score = dist - ab - rb * 0.15  # anchor dominates; rare is a very soft nudge
        if score < best_score:
            best_score = score
            best_word_str = word

    return best_word_str, best_score


# ---------------------------------------------------------------------------
# CTC collapse
# ---------------------------------------------------------------------------

@dataclass
class TokenSpan:
    phoneme: str
    start: int
    end: int

    @property
    def num_frames(self) -> int:
        return self.end - self.start


def collapse_ctc(frame_preds: List[str]) -> List[TokenSpan]:
    """Collapse CTC frame predictions into (phoneme, start, end) spans."""
    spans: List[TokenSpan] = []
    prev: Optional[str] = None
    start: Optional[int] = None

    for i, token in enumerate(frame_preds):
        if token == prev:
            continue
        if prev is not None and prev != BLANK:
            spans.append(TokenSpan(prev, start, i))
        start = i if token != BLANK else None
        prev = token

    if prev is not None and prev != BLANK and start is not None:
        spans.append(TokenSpan(prev, start, len(frame_preds)))

    return spans


# ---------------------------------------------------------------------------
# Segmentation via DP
# ---------------------------------------------------------------------------

def segment_into_words(
    spans: List[TokenSpan],
) -> Optional[List[Tuple[str, int, int]]]:
    """Partition *spans* into exactly 6 word slots via DP.

    Returns list of (word, span_start_idx, span_end_idx_exclusive) tuples,
    or None if *spans* is empty.
    """
    phonemes = [s.phoneme for s in spans]
    N = len(phonemes)
    K = len(GRAMMAR_ORDER)

    if N == 0:
        # Return best single-phoneme match for each slot with empty segment
        return [
            (best_word([], cat)[0], 0, 0)
            for cat in GRAMMAR_ORDER
        ]

    memo: Dict[Tuple[int,int], Tuple[float, Optional[list]]] = {}

    def dp(pos: int, k: int) -> Tuple[float, Optional[list]]:
        if (pos, k) in memo:
            return memo[(pos, k)]

        if k == K:
            result = (0.0, []) if pos == N else (math.inf, None)
            memo[(pos, k)] = result
            return result

        remaining_slots = K - k - 1
        best_cost = math.inf
        best_path = None

        # Ensure at least 1 phoneme per remaining slot
        max_end = N - remaining_slots
        for end in range(pos + 1, max_end + 1):
            segment = phonemes[pos:end]
            word, cost = best_word(segment, GRAMMAR_ORDER[k])
            future_cost, future_path = dp(end, k + 1)
            total = cost + future_cost
            if total < best_cost and future_path is not None:
                best_cost = total
                best_path = [(word, pos, end)] + future_path

        memo[(pos, k)] = (best_cost, best_path)
        return memo[(pos, k)]

    _, path = dp(0, 0)
    return path


# ---------------------------------------------------------------------------
# Decode one entry from ctc_outputs.json
# ---------------------------------------------------------------------------

def decode_entry(
    entry: Dict,
) -> List[Dict]:
    """Decode a single CTC output into 6 timed word predictions."""
    frame_preds: List[str] = entry.get("frame_predictions", [])
    spans = collapse_ctc(frame_preds)
    path = segment_into_words(spans)

    words = []
    if path is None or not spans:
        for cat in GRAMMAR_ORDER:
            words.append({"word": best_word([], cat)[0], "start_frame": 0, "end_frame": 0})
        return words

    for word, start_idx, end_idx in path:
        if start_idx < end_idx and start_idx < len(spans):
            start_frame = spans[start_idx].start
            end_frame = spans[min(end_idx, len(spans)) - 1].end
        else:
            start_frame = spans[0].start if spans else 0
            end_frame = spans[-1].end if spans else 0
        words.append({"word": word, "start_frame": start_frame, "end_frame": end_frame})

    return words


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    results: List[Dict],
) -> Dict:
    """Compute accuracy metrics across all samples and per grammar slot."""
    slot_correct = [0] * 6
    slot_total   = [0] * 6
    sample_scores = []

    for r in results:
        gt_words   = r.get("gt_words")
        pred_words = [w["word"] for w in r["decoded_words"]]
        if gt_words is None:
            continue

        correct = sum(int(g == p) for g, p in zip(gt_words, pred_words))
        sample_scores.append({
            "stem":     r["stem"],
            "gt":       " ".join(gt_words),
            "pred":     " ".join(pred_words),
            "correct":  correct,
            "total":    6,
            "accuracy": correct / 6,
            "errors":   6 - correct,
        })
        for k, (g, p) in enumerate(zip(gt_words, pred_words)):
            slot_total[k] += 1
            if g == p:
                slot_correct[k] += 1

    n = len(sample_scores)
    overall_acc = sum(s["accuracy"] for s in sample_scores) / max(1, n)
    exact_match = sum(1 for s in sample_scores if s["errors"] == 0) / max(1, n)

    per_slot = [
        {
            "slot": GRAMMAR_ORDER[k],
            "correct": slot_correct[k],
            "total": slot_total[k],
            "accuracy": slot_correct[k] / max(1, slot_total[k]),
        }
        for k in range(6)
    ]

    return {
        "overall_word_accuracy": overall_acc,
        "exact_sentence_match":  exact_match,
        "per_slot": per_slot,
        "per_sample": sample_scores,
    }


# ---------------------------------------------------------------------------
# SRT subtitle generation
# ---------------------------------------------------------------------------

def frames_to_srt_time(frame: int, fps: float) -> str:
    total_sec = frame / max(1.0, fps)
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    ms = int((total_sec - int(total_sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(words: List[Dict], fps: float, total_frames: int) -> str:
    """Build an SRT string with one subtitle cue per word."""
    lines = []
    n_frames = max(total_frames, 1)

    for i, w in enumerate(words, start=1):
        sf = max(0, w["start_frame"])
        ef = min(n_frames - 1, w["end_frame"])
        if ef <= sf:
            ef = min(n_frames - 1, sf + max(1, int(fps * 0.3)))

        t_start = frames_to_srt_time(sf, fps)
        t_end   = frames_to_srt_time(ef, fps)
        lines.append(f"{i}\n{t_start} --> {t_end}\n{w['word']}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe resolution (handles Windows PATH gaps)
# ---------------------------------------------------------------------------

# Populated once by resolve_ffmpeg_exe(); None if not found.
_FFMPEG_EXE:  Optional[str] = None
_FFPROBE_EXE: Optional[str] = None


def _candidate_paths(binary: str) -> List[str]:
    """Return a list of paths to try for *binary* (ffmpeg or ffprobe)."""
    candidates = [binary]   # relies on PATH first

    # Common Windows install locations
    win_roots = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
    ]
    # Also check every directory on PATH explicitly — shutil.which can miss
    # entries when the venv is active and PATH has been modified.
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for d in win_roots + path_dirs:
        for ext in ("", ".exe"):
            candidates.append(os.path.join(d, binary + ext))

    # imageio-ffmpeg ships its own binary
    try:
        import imageio_ffmpeg  # type: ignore
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            candidates.append(bundled)
            # ffprobe usually lives next to the ffmpeg binary
            probe = os.path.join(os.path.dirname(bundled), "ffprobe" + os.path.splitext(bundled)[1])
            candidates.append(probe)
    except Exception:
        pass

    return candidates


def resolve_ffmpeg_exe(override: Optional[str] = None) -> Optional[str]:
    """Return the path to ffmpeg, or None if it cannot be located."""
    global _FFMPEG_EXE
    if override:
        _FFMPEG_EXE = override
        return _FFMPEG_EXE
    if _FFMPEG_EXE is not None:
        return _FFMPEG_EXE
    for p in _candidate_paths("ffmpeg"):
        if shutil.which(p) or (os.path.isfile(p) and os.access(p, os.X_OK)):
            _FFMPEG_EXE = p
            return p
    return None


def resolve_ffprobe_exe() -> Optional[str]:
    """Return the path to ffprobe, or None if it cannot be located."""
    global _FFPROBE_EXE
    if _FFPROBE_EXE is not None:
        return _FFPROBE_EXE
    # If we already know where ffmpeg is, look for ffprobe beside it
    ffmpeg_path = resolve_ffmpeg_exe()
    if ffmpeg_path:
        probe_candidate = os.path.join(
            os.path.dirname(ffmpeg_path),
            "ffprobe" + os.path.splitext(ffmpeg_path)[1],
        )
        if os.path.isfile(probe_candidate):
            _FFPROBE_EXE = probe_candidate
            return _FFPROBE_EXE
    for p in _candidate_paths("ffprobe"):
        if shutil.which(p) or (os.path.isfile(p) and os.access(p, os.X_OK)):
            _FFPROBE_EXE = p
            return p
    return None


# ---------------------------------------------------------------------------
# Video processing with ffmpeg
# ---------------------------------------------------------------------------

def find_source_video(stem: str, video_dir: Path) -> Optional[Path]:
    """Locate the source video for *stem* in *video_dir* (any extension)."""
    # strip lipcrop / speaker prefix for searching
    clean = re.sub(r"^s\d+_", "", stem).replace("_lipcrop", "")
    for ext in (".mpg", ".mp4", ".avi", ".mov", ".mkv"):
        for candidate in [
            video_dir / f"{stem}{ext}",
            video_dir / f"{clean}{ext}",
        ]:
            if candidate.exists():
                return candidate
    return None


def get_video_fps(video_path: Path) -> float:
    ffprobe = resolve_ffprobe_exe()
    if ffprobe is None:
        return 25.0
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    raw = result.stdout.strip()
    if "/" in raw:
        num, den = raw.split("/")
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(raw)
    except ValueError:
        return 25.0


def get_video_frame_count(video_path: Path) -> int:
    ffprobe = resolve_ffprobe_exe()
    if ffprobe is None:
        return 0
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    label: str = "",
) -> bool:
    """Burn *srt_path* subtitles into *video_path* and save to *output_path*.

    Windows-safe: ffmpeg's subtitles filter cannot handle paths that contain
    a colon (C:\\...) even with escaping.  We work around this by copying the
    SRT to a temp file in the current working directory and passing only the
    bare filename — which contains no colon — to ffmpeg.
    """
    ffmpeg = resolve_ffmpeg_exe()
    if ffmpeg is None:
        print(
            "  [WARNING] ffmpeg not found.\n"
            "  Try passing --ffmpeg-path 'C:\\path\\to\\ffmpeg.exe' on the command line,\n"
            "  or add ffmpeg to your system PATH and restart the terminal."
        )
        return False

    style = "FontSize=22,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=3"

    # Copy SRT to cwd under a short safe name with no drive-letter colon.
    tmp_srt_name = f"_tmp_sub_{output_path.stem}.srt"
    tmp_srt = Path(tmp_srt_name)
    try:
        shutil.copy2(srt_path, tmp_srt)
        vf = f"subtitles={tmp_srt_name}:force_style='{style}'"

        def _run(extra_audio_flags):
            return subprocess.run(
                [ffmpeg, "-y", "-i", str(video_path.resolve()),
                 "-vf", vf,
                 "-c:v", "libx264", "-crf", "23", "-preset", "fast"]
                + extra_audio_flags
                + [str(output_path.resolve())],
                capture_output=True,
                text=True,
                cwd=str(Path.cwd()),
            )

        result = _run(["-c:a", "aac", "-b:a", "128k"])
        if result.returncode != 0:
            # mp2 audio in GRID .mpg files can trip up the aac encoder
            result = _run(["-an"])
        if result.returncode != 0:
            print(f"  [ERROR] ffmpeg failed for {output_path.name}:")
            print(result.stderr[-600:])
            return False
        return True
    finally:
        try:
            tmp_srt.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Select three "showcase" videos: perfect / near-perfect / bad
# ---------------------------------------------------------------------------

def select_showcase_videos(
    metrics: Dict,
    n_each: int = 1,
) -> Dict[str, List[str]]:
    """Return stems categorised as perfect / near_perfect / bad."""
    perfect, near, bad = [], [], []
    for s in metrics["per_sample"]:
        e = s["errors"]
        if e == 0:
            perfect.append(s["stem"])
        elif e <= 1:
            near.append(s["stem"])
        else:
            bad.append(s["stem"])
    return {"perfect": perfect[:n_each], "near_perfect": near[:n_each], "bad": bad[:n_each]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Improved GRID CTC decoder with subtitles")
    parser.add_argument("--ctc-json",   default="viseme_results/ctc_outputs.json",
                        help="Path to the ctc_outputs.json produced by train_lip_viseme.py")
    parser.add_argument("--video-dir",  default="s1",
                        help="Directory containing the original .mpg source videos")
    parser.add_argument("--output-dir", default="decode_results")
    parser.add_argument("--fps",        type=float, default=0.0,
                        help="Source video FPS override (0 = auto-detect). "
                             "Used only for frame-count detection, not subtitle timing.")
    parser.add_argument("--model-fps",  type=float, default=25.0,
                        help="FPS of the lip-crop frames fed to the model "
                             "(default 25 for GRID corpus). "
                             "This controls subtitle timing — set it to match "
                             "the frame rate your lip crops were extracted at.")
    parser.add_argument("--ffmpeg-path", default="",
                        help="Explicit path to ffmpeg.exe if it is not on PATH "
                             r"(e.g. C:\ffmpeg\bin\ffmpeg.exe)")
    args = parser.parse_args()

    # Resolve ffmpeg early so all helpers use the right binary
    resolve_ffmpeg_exe(override=args.ffmpeg_path if args.ffmpeg_path else None)
    ffmpeg_found = resolve_ffmpeg_exe()
    if ffmpeg_found:
        print(f"Using ffmpeg: {ffmpeg_found}")
    else:
        print(
            "[WARNING] ffmpeg could not be located automatically.\n"
            "  Subtitle burning will be skipped.\n"
            "  Pass --ffmpeg-path 'C:\\path\\to\\ffmpeg.exe' to enable it."
        )

    ctc_json_path = Path(args.ctc_json)
    video_dir     = Path(args.video_dir)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load CTC outputs
    # ------------------------------------------------------------------
    with ctc_json_path.open("r", encoding="utf-8") as fh:
        data: List[Dict] = json.load(fh)

    print(f"Loaded {len(data)} samples from {ctc_json_path}")

    # ------------------------------------------------------------------
    # Decode each sample
    # ------------------------------------------------------------------
    results = []
    for entry in data:
        stem     = entry.get("stem", "unknown")
        gt_words = parse_grid_stem(stem)
        decoded  = decode_entry(entry)
        results.append({
            "stem":         stem,
            "gt_words":     gt_words,
            "decoded_words": decoded,
        })

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(results)
    print(f"\nOverall word accuracy : {metrics['overall_word_accuracy']:.3f}")
    print(f"Exact sentence match  : {metrics['exact_sentence_match']:.3f}")
    print("\nPer-slot accuracy:")
    for slot in metrics["per_slot"]:
        print(f"  {slot['slot']:10s}: {slot['accuracy']:.3f}  ({slot['correct']}/{slot['total']})")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "overall_word_accuracy": metrics["overall_word_accuracy"],
            "exact_sentence_match":  metrics["exact_sentence_match"],
            "per_slot":              metrics["per_slot"],
        }, fh, indent=2)
    print(f"\nSaved metrics → {metrics_path}")

    # Per-sample CSV
    sample_csv_path = output_dir / "per_sample_accuracy.csv"
    fieldnames = ["stem", "gt", "pred", "correct", "total", "accuracy", "errors"]
    with sample_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics["per_sample"])
    print(f"Saved per-sample CSV → {sample_csv_path}")

    # ------------------------------------------------------------------
    # Save full decoded output JSON
    # ------------------------------------------------------------------
    decoded_json = output_dir / "decoded_sentences.json"
    with decoded_json.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"Saved decoded sentences → {decoded_json}")

    # ------------------------------------------------------------------
    # Select showcase videos
    # ------------------------------------------------------------------
    showcases = select_showcase_videos(metrics)
    print("\nShowcase video selection:")
    for cat, stems in showcases.items():
        print(f"  {cat}: {stems}")

    # ------------------------------------------------------------------
    # Subtitle burning for showcase videos
    # ------------------------------------------------------------------
    subs_dir = output_dir / "subtitled_videos"
    subs_dir.mkdir(exist_ok=True)

    # Build a quick lookup: stem → result
    result_by_stem = {r["stem"]: r for r in results}

    # Find a mapping from per_sample metrics stem → result
    sample_by_stem = {s["stem"]: s for s in metrics["per_sample"]}

    for category, stems in showcases.items():
        for stem in stems:
            result = result_by_stem.get(stem)
            if result is None:
                print(f"  [SKIP] {stem} not found in results")
                continue

            # Find source video
            source_video = find_source_video(stem, video_dir)
            if source_video is None:
                print(f"  [SKIP] no source video found for {stem} in {video_dir}")
                continue

            # model_fps: the frame rate of the lip-crop frames the model
            # processed.  For GRID this is 25 fps.  Subtitle timestamps must
            # be derived from this rate, NOT from the source video FPS, because
            # frame_predictions indices index into model input frames.
            model_fps = args.model_fps

            # Total model frames for this clip comes from the CTC output itself
            # so we are never mis-aligned with the source video frame count.
            entry = next((e for e in data if e.get("stem") == stem), None)
            ctc_n_frames = len(entry.get("frame_predictions", [])) if entry else 0

            # Source video FPS / frame count are only used for the ffprobe call
            # that lets us sanity-check the video length; not for timing.
            src_fps = args.fps if args.fps > 0 else get_video_fps(source_video)
            src_n_frames = get_video_frame_count(source_video)

            # If the source video and model have different frame counts, compute
            # a scale factor so CTC frame indices map to source video frames.
            # For GRID (25 fps clips, no subsampling) this should be ~1.0.
            if ctc_n_frames > 0 and src_n_frames > 0:
                frame_scale = src_n_frames / ctc_n_frames
            else:
                frame_scale = 1.0

            def scale_word(w: Dict) -> Dict:
                return {
                    "word":        w["word"],
                    "start_frame": int(w["start_frame"] * frame_scale),
                    "end_frame":   int(w["end_frame"]   * frame_scale),
                }

            gt_words      = result.get("gt_words") or []
            decoded_words = result["decoded_words"]
            scaled_words  = [scale_word(w) for w in decoded_words]

            # Build GT subtitle cues: predicted timing, ground-truth text
            gt_sub_cues = [
                {
                    "word":        gt_words[i] if i < len(gt_words) else "?",
                    "start_frame": scaled_words[i]["start_frame"],
                    "end_frame":   scaled_words[i]["end_frame"],
                }
                for i in range(len(scaled_words))
            ]

            # Use model_fps for SRT timing so 1 model frame = 1/model_fps seconds
            gt_srt   = build_srt(gt_sub_cues, model_fps, src_n_frames or ctc_n_frames)
            pred_srt = build_srt(scaled_words, model_fps, src_n_frames or ctc_n_frames)

            gt_srt_path   = subs_dir / f"{stem}_gt.srt"
            pred_srt_path = subs_dir / f"{stem}_pred.srt"
            gt_srt_path.write_text(gt_srt,   encoding="utf-8")
            pred_srt_path.write_text(pred_srt, encoding="utf-8")

            sample_info = sample_by_stem.get(stem)
            err_count   = sample_info["errors"] if sample_info else "?"
            print(f"\n  [{category.upper()}] {stem}  errors={err_count}")
            print(f"    GT : {' '.join(gt_words)}")
            print(f"    Pred: {' '.join(w['word'] for w in decoded_words)}")

            gt_out   = subs_dir / f"{stem}_gt_subtitled.mp4"
            pred_out = subs_dir / f"{stem}_pred_subtitled.mp4"

            print(f"    Burning GT subtitles   → {gt_out.name}")
            ok = burn_subtitles(source_video, gt_srt_path, gt_out, label="GT")
            print(f"    {'OK' if ok else 'FAILED'}")

            print(f"    Burning PRED subtitles → {pred_out.name}")
            ok = burn_subtitles(source_video, pred_srt_path, pred_out, label="PRED")
            print(f"    {'OK' if ok else 'FAILED'}")

    print(f"\nDone. All outputs in {output_dir}")


if __name__ == "__main__":
    main()