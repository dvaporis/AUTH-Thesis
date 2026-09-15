from pathlib import Path

from grid_phoneme import (
    GRID_FPS,
    GRID_LIP_CROP_DIR,
    GRID_SOURCE_DIR,
    normalize_stem,
    speaker_id,
    split_by_speaker,
)


def test_active_grid_defaults():
    assert GRID_FPS == 25.0
    assert GRID_SOURCE_DIR == Path("s1")
    assert GRID_LIP_CROP_DIR == Path("s1_lip_crops")


def test_normalize_stem_removes_lip_crop_suffix():
    assert normalize_stem(Path("bbaf2n_lipcrop.mp4")) == "bbaf2n"
    assert normalize_stem(Path("bbaf2n.mp4")) == "bbaf2n"


def test_speaker_split_groups_paths_without_cross_speaker_mixing():
    paths = [Path("s1/bbaf2n.mpg"), Path("s2/bbaf2n.mpg"), Path("s1/bbal6n.mpg")]
    assert speaker_id(paths[0]) == "s1"
    assert [(speaker, len(items)) for speaker, items in split_by_speaker(paths)] == [("s1", 2), ("s2", 1)]
