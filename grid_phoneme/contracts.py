"""Shared paths and filename contracts for the active GRID pipeline."""

from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple

GRID_FPS = 25.0
GRID_SOURCE_DIR = Path("s1")
GRID_LIP_CROP_DIR = Path("s1_lip_crops")


def normalize_stem(path: Path) -> str:
    """Return the GRID clip stem without the cropper-added suffix."""

    stem = path.stem
    if stem.endswith("_lipcrop"):
        return stem[: -len("_lipcrop")]
    return stem


def speaker_id(path: Path) -> str:
    """Extract a GRID speaker identifier from a path or return ``unknown``."""

    for part in (path.stem, *path.parts):
        match = re.search(r"(?:^|[_\\/])s(\d+)(?:$|[_\\/])", part.lower())
        if match:
            return f"s{match.group(1)}"
    return "unknown"


def split_by_speaker(paths: Sequence[Path]) -> List[Tuple[str, List[Path]]]:
    """Group paths by speaker for leakage-resistant evaluation splits."""

    grouped: dict[str, List[Path]] = {}
    for path in paths:
        grouped.setdefault(speaker_id(path), []).append(path)
    return sorted((speaker, sorted(items)) for speaker, items in grouped.items())
