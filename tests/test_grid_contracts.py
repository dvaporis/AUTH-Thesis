from pathlib import Path

import pytest
import torch

from extract_phonemes import decode_grid_stem
from grid_phoneme import (
    GRID_FPS,
    GRID_LIP_CROP_DIR,
    GRID_SOURCE_DIR,
    normalize_stem,
    speaker_id,
    split_by_speaker,
)
from train_lip_viseme import (
    PHONEME_TO_VISEME,
    build_similarity_matrix,
    levenshtein,
    similarity_matrix_to_soft_targets,
    token_error_rate,
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


@pytest.mark.parametrize(
    ("stem", "sentence"),
    [
        ("bbaf2n", "bin blue at f 2 now"),
        ("swwp2n_lipcrop", "set white with p 2 now"),
        ("pbwxzn", "place blue with x zero now"),
        ("swiz9a", "set white in z 9 again"),
    ],
)
def test_decode_grid_stem_handles_standard_and_edge_fields(stem: str, sentence: str):
    assert decode_grid_stem(stem) == sentence


@pytest.mark.parametrize("stem", ["bbaf", "bbaf2x", "xbaf2n", "bbaw2n"])
def test_decode_grid_stem_rejects_invalid_grid_stems(stem: str):
    assert decode_grid_stem(stem) is None


def test_similarity_matrix_is_symmetric_and_viseme_aware():
    vocab = ["p", "b", "t", "k", "i", "u"]
    matrix = build_similarity_matrix(vocab)

    assert matrix.shape == (len(vocab) + 1, len(vocab) + 1)
    assert torch.allclose(matrix, matrix.T)
    assert torch.allclose(torch.diag(matrix), torch.ones(len(vocab) + 1))
    assert matrix[0, 0] == 1.0
    assert torch.all(matrix[0, 1:] == 0.0)
    assert torch.all(matrix[1:, 0] == 0.0)

    bilabial_p = 1 + vocab.index("p")
    bilabial_b = 1 + vocab.index("b")
    alveolar_t = 1 + vocab.index("t")
    assert PHONEME_TO_VISEME["p"] == PHONEME_TO_VISEME["b"]
    assert matrix[bilabial_p, bilabial_b] > matrix[bilabial_p, alveolar_t]


def test_soft_targets_are_normalized_and_keep_blank_one_hot():
    matrix = build_similarity_matrix(["p", "b", "t"])
    soft_targets = similarity_matrix_to_soft_targets(matrix)

    assert torch.allclose(soft_targets.sum(dim=1), torch.ones(soft_targets.shape[0]))
    assert torch.equal(soft_targets[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.all(soft_targets[1:, 0] == 0.0)
    assert torch.allclose(torch.diag(soft_targets)[1:], torch.full((3,), 0.85))


def test_levenshtein_and_token_error_rate():
    assert levenshtein([], []) == 0
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([1, 2, 3], [1, 4, 3, 5]) == 2
    assert levenshtein([1, 2], []) == 2

    predictions = [[1, 2, 3], [4], []]
    references = [[1, 9, 3], [4, 5], [7]]
    assert token_error_rate(predictions, references) == pytest.approx(3 / 6)
