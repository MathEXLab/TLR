import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

import numpy as np

from scripts import local_indices as li


@pytest.fixture(scope="module")
def lorenz_subset(tmp_path_factory):
    data_path = Path("data/datasets/lorenz_chaotic.txt")
    if not data_path.exists():
        pytest.skip("Lorenz dataset not available")

    # Load subset of the Lorenz dataset to keep regression test tractable.
    subset = np.loadtxt(data_path, max_rows=600)
    time = subset[:, 0]
    states = subset[:, 1:]

    # Temporary directory for any side-effect files produced by the library.
    workdir = tmp_path_factory.mktemp("lorenz_regression")
    return workdir, time, states


def test_lorenz_alphat_regression(lorenz_subset):
    workdir, time, states = lorenz_subset
    filename = "lorenz_subset"
    ql = 0.99
    theiler_window = 0
    lag_list = [11, 22, 44, 220, 440]

    dist, exceeds, exceeds_idx, exceeds_bool = li.compute_exceeds(
        states,
        filepath=str(workdir),
        filename=filename,
        ql=ql,
        n_jobs=1,
        theiler_len=theiler_window,
        save_full=False,
    )

    alphat = li.compute_alphat(
        dist,
        exceeds_bool,
        filepath=str(workdir),
        filename=filename,
        time_lag=lag_list,
        ql=ql,
        theiler_len=theiler_window,
        l=1,
    )

    expected_path = Path("tests/data/lorenz_subset_alphat.json")
    with expected_path.open() as f:
        expected = {int(k): np.array(v) for k, v in json.load(f).items()}

    assert set(alphat.keys()) == set(expected.keys())

    for lag in sorted(expected):
        np.testing.assert_allclose(
            alphat[lag],
            expected[lag],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Mismatch detected for lag {lag}",
        )

    # Additional sanity checks on intermediate arrays to guard against
    # structural regressions.
    assert dist.shape[0] == dist.shape[1] == states.shape[0]
    assert exceeds.shape[0] == states.shape[0]
    assert exceeds_bool.shape == dist.shape
    assert exceeds_idx.shape[0] == states.shape[0]
