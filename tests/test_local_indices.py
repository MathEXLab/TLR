import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts.local_indices import compute_alphat


@pytest.fixture
def dist_matrix():
    """Minimal distance matrix with large diagonals and small off-diagonals."""
    return np.array([
        [100.0, 1.0, 2.0],
        [1.0, 100.0, 1.5],
        [2.0, 1.5, 100.0],
    ])


@pytest.fixture
def exceeds_bool():
    """Boolean mask with consistent neighbour counts across rows."""
    return np.array([
        [False, True, True],
        [True, False, True],
        [True, True, False],
    ], dtype=bool)


@pytest.fixture
def time_lag():
    return [1]


@pytest.fixture
def alphat_path(tmp_path: Path):
    filename = "toy"
    return tmp_path, filename


def _load_alphat(tmp_dir: Path, filename: str, l: int):
    pickle_path = tmp_dir / f"{filename}_alphat_max1_0.99_0_{l}.pkl"
    with pickle_path.open("rb") as handle:
        return pickle.load(handle)


def test_compute_alphat_counts(dist_matrix, exceeds_bool, time_lag, alphat_path):
    tmp_dir, filename = alphat_path
    compute_alphat(dist_matrix, exceeds_bool, tmp_dir, filename, time_lag, l=0)

    alphat = _load_alphat(tmp_dir, filename, l=0)
    expected = np.array([0.5, 0.5])
    np.testing.assert_allclose(alphat[time_lag[0]], expected)


def test_compute_alphat_distance_weighted(dist_matrix, exceeds_bool, time_lag, alphat_path):
    tmp_dir, filename = alphat_path
    compute_alphat(dist_matrix, exceeds_bool, tmp_dir, filename, time_lag, l=1)

    alphat = _load_alphat(tmp_dir, filename, l=1)
    expected = np.array([1.0, 1.0])
    np.testing.assert_allclose(alphat[time_lag[0]], expected)
