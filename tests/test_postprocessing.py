"""Regression tests for helper utilities in scripts/postprocessing.py."""

import os
import numpy as np
import pytest

# Ensure scripts directory on path for importing postprocessing module
import sys
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPTS_DIR = os.path.join(CURRENT_DIR, "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import postprocessing as pp


def test_load_exponential_test(tmp_path):
    """Regression test for loading exponential statistics."""
    filepath = tmp_path
    filename = "example"
    ql = 0.95
    theiler = 0
    sr = slice(None)

    data = np.array([-1.0, 0.25, -0.5, 0.0, 0.75])
    exp_path = filepath / f"{filename}_exp_stat_{ql}_{theiler}.txt"
    np.savetxt(exp_path, data)

    flag_expo, E, color_negative, color_positive = pp._load_exponential_test(
        str(filepath), filename, ql, theiler, sr
    )

    assert flag_expo is True
    np.testing.assert_allclose(E, data)
    assert color_negative == ["red", "red"]
    assert color_positive == ["green", "green", "green"]

    exp_path.unlink()
    flag_expo, E, color_negative, color_positive = pp._load_exponential_test(
        str(filepath), filename, ql, theiler, sr
    )

    assert flag_expo is False
    assert E is None
    assert color_negative is None
    assert color_positive is None


def test_extract_neighbours_behaviour():
    """Regression test capturing neighbour extraction semantics."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    X = np.array(
        [
            [10.0, -1.0],
            [11.0, -0.5],
            [12.0, 0.25],
            [13.5, 1.5],
        ]
    )
    l_ind = np.array([0.2, 0.5, 0.35, 0.9])
    exceeds_idx = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [0, 2, 3],
            [1, 2, 3],
        ]
    )

    t_neigh, X_neigh, t_c, X_c, n_interest_pts = pp._extract_neighbours(
        t, X, l_ind, exceeds_idx, ["max", 2]
    )

    assert n_interest_pts == 2
    assert t_neigh.shape == (3, 2)
    assert X_neigh.shape == (3, 2, 2)
    np.testing.assert_allclose(t_neigh[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(t_neigh[:, 1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(X_neigh[:, :, 0], [[11.0, -0.5], [12.0, 0.25], [13.5, 1.5]])
    np.testing.assert_allclose(X_neigh[:, :, 1], [[11.0, -0.5], [12.0, 0.25], [13.5, 1.5]])
    np.testing.assert_allclose(t_c[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(X_c, [[10.0, -1.0], [11.0, -0.5]])


def test_slice_and_label_helpers():
    """Verify helper utilities used for plotting metadata."""
    shorter_row_var = np.ones(5)

    sr_alpha = pp._get_slice_rows(shorter_row_var, "alphat_example")
    sr_d1 = pp._get_slice_rows(shorter_row_var, "d1")

    assert isinstance(sr_alpha, slice)
    assert sr_alpha.start is None and sr_alpha.stop == shorter_row_var.shape[0]
    assert sr_d1 == slice(None)

    assert pp._get_label_str("d1_value") == "$d_1$"
    assert pp._get_label_str("theta_value") == "$\\theta$"
    assert pp._get_label_str("alphat_value") == "$\\alpha_\\eta$"

    assert pp._get_title_str("d1", tau_l=1.0) == "$d_1$"
    assert pp._get_title_str("theta", tau_l=1.5) == "$\\theta$"
    assert pp._get_title_str("alphat", tau_l=0.5) == "$\\eta = $0.5 $\\tau_l$"

    vmin, vmax = pp._get_vmin_and_max(None, None, np.array([0.2, 0.6, 0.4]))
    assert vmin == pytest.approx(0.2)
    assert vmax == pytest.approx(0.6)

