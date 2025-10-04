#!/usr/bin/env python3
"""
Pytest-based regression test for TLR repository using lorenz_small.txt
"""

import json
import os
import sys
import numpy as np
import xarray as xr
import tempfile
import shutil
import pytest


def load_regression_checks():
    """Load sampled Lorenz regression reference values from the JSON fixture."""
    fixture_path = os.path.join(CFD, "fixtures", "lorenz_small_regression_checks.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Current file path
CWD = os.getcwd()
CFD = os.path.dirname(os.path.realpath(__file__))

# Add scripts directory to path
sys.path.append(os.path.join(CFD, "..", "scripts"))
import local_indices as li


@pytest.fixture
def test_data():
    """Load test data from lorenz_small.txt"""
    test_data_path = os.path.join(CFD, "../data/datasets/lorenz_small.txt")
    assert os.path.exists(test_data_path), f"Test data file not found: {test_data_path}"
    
    Y = np.loadtxt(test_data_path)
    assert Y.shape[1] >= 4, f"Invalid data shape: {Y.shape}"
    
    t = Y[:, 0]
    X = Y[:, 1:]
    return t, X

@pytest.fixture
def test_params():
    """Test parameters"""
    return {
        'ql': 0.99,
        'theiler_len': 0,
        'l': 0,
        'tau_list': [5, 10, 20],
        'filename': 'lorenz_small_test'
    }

@pytest.fixture
def temp_dir():
    """Create temporary directory for test output"""
    test_filepath = "test_output"
    if os.path.exists(test_filepath):
        shutil.rmtree(test_filepath)
    os.makedirs(test_filepath)
    yield test_filepath
    # Cleanup
    if os.path.exists(test_filepath):
        shutil.rmtree(test_filepath)

def test_data_loading(test_data):
    """Test that test data can be loaded correctly"""
    _, X = test_data
    assert X.shape[0] > 0, "Data should have at least one row"
    assert X.shape[1] == 3, "Lorenz data should have 3 spatial dimensions"
    assert len(t) == X.shape[0], "Time and spatial data should have same length"

def test_tlr_computation(test_data, test_params, temp_dir):
    """Test complete TLR computation pipeline"""
    t, X = test_data
    params = test_params
    test_filepath = temp_dir
    
    # Compute exceedances
    dist, exceeds, exceeds_idx, exceeds_bool = li.compute_exceeds(
        X, filepath=test_filepath, filename=params['filename'], 
        ql=params['ql'], n_jobs=1, theiler_len=params['theiler_len'], save_full=False
    )
    
    # Verify exceedances
    assert exceeds.shape[0] == X.shape[0], "Exceeds should have same number of rows as input data"
    assert exceeds_idx.shape[0] == X.shape[0], "Exceeds_idx should have same number of rows as input data"
    
    # Compute local dimension d1
    d1 = li.compute_d1(exceeds, test_filepath, params['filename'], 
                      ql=params['ql'], theiler_len=params['theiler_len'])
    
    # Verify d1
    assert len(d1) == X.shape[0], "d1 should have same length as input data"
    assert np.all(d1 > 0), "d1 values should be positive"
    
    # Compute persistence theta
    theta = li.compute_theta(exceeds_idx, test_filepath, params['filename'], 
                           ql=params['ql'], theiler_len=params['theiler_len'], method='sueveges')
    
    # Verify theta
    assert len(theta) == X.shape[0], "theta should have same length as input data"
    assert np.all(theta >= 0) and np.all(theta <= 1), "theta values should be in [0,1]"
    
    # Compute alphat
    li.compute_alphat(dist, exceeds_bool, test_filepath, params['filename'], 
                     params['tau_list'], ql=params['ql'], 
                     theiler_len=params['theiler_len'], l=params['l'])
    
    # Verify results
    results_path = os.path.join(test_filepath, f"results_{params['filename']}")
    assert os.path.exists(results_path), "Results directory should be created"
    
    # Check NetCDF file
    nc_file = os.path.join(
        results_path,
        f"{params['filename']}_alphat_max{max(params['tau_list'])}_{params['ql']}"
        f"_{params['theiler_len']}_{params['l']}.nc",
    )
    assert os.path.exists(nc_file), "NetCDF file should be created"

    # Load and verify NetCDF content
    ds = xr.open_dataset(nc_file)
    assert "alphat" in ds, "NetCDF file should contain 'alphat' variable"

    # Verify all tau values are present
    for tau in params['tau_list']:
        assert tau in ds["lag"].values, f"Missing tau value {tau} in results"
        alpha_values = ds["alphat"].sel(lag=tau).dropna("time_index").values
        assert alpha_values.size > 0, f"Empty data for tau value {tau}"

        # Check alpha values are in valid range
        assert np.all(alpha_values >= 0) and np.all(alpha_values <= 1), \
            f"Alpha values out of range [0,1] for tau {tau}"

    ds.close()

def test_alpha_statistics(test_data, test_params, temp_dir):
    """Test that alpha statistics are reasonable"""
    t, X = test_data
    params = test_params
    test_filepath = temp_dir
    
    # Run computation
    dist, exceeds, exceeds_idx, exceeds_bool = li.compute_exceeds(
        X, filepath=test_filepath, filename=params['filename'], 
        ql=params['ql'], n_jobs=1, theiler_len=params['theiler_len'], save_full=False
    )
    
    li.compute_alphat(dist, exceeds_bool, test_filepath, params['filename'], 
                     params['tau_list'], ql=params['ql'], 
                     theiler_len=params['theiler_len'], l=params['l'])
    
    # Load results
    results_path = os.path.join(test_filepath, f"results_{params['filename']}")
    nc_file = os.path.join(
        results_path,
        f"{params['filename']}_alphat_max{max(params['tau_list'])}_{params['ql']}"
        f"_{params['theiler_len']}_{params['l']}.nc",
    )
    ds = xr.open_dataset(nc_file)
    
    # Check statistics for each tau
    for tau in params['tau_list']:
        alpha_values = ds["alphat"].sel(lag=tau).dropna("time_index").values
        mean_alpha = np.mean(alpha_values)
        std_alpha = np.std(alpha_values)
        
        # Basic sanity checks
        assert 0 < mean_alpha < 1, f"Mean alpha should be in (0,1) for tau {tau}, got {mean_alpha}"
        assert std_alpha > 0, f"Standard deviation should be positive for tau {tau}, got {std_alpha}"
        
        # Check that alpha decreases with increasing tau (typical behavior)
        if tau > params['tau_list'][0]:
            prev_tau = params['tau_list'][params['tau_list'].index(tau) - 1]
            prev_values = ds["alphat"].sel(lag=prev_tau).dropna("time_index").values
            prev_mean = np.mean(prev_values)
            assert mean_alpha <= prev_mean, \
                f"Alpha should generally decrease with tau: tau {prev_tau} mean={prev_mean:.4f}, tau {tau} mean={mean_alpha:.4f}"

    ds.close()


def test_tlr_results_are_reproducible(test_data, test_params, temp_dir):
    """Ensure the Lorenz pipeline reproduces the stored regression results."""
    t, X = test_data
    params = test_params
    test_filepath = temp_dir

    expected = load_regression_checks()

    dist, exceeds, exceeds_idx, exceeds_bool = li.compute_exceeds(
        X, filepath=test_filepath, filename=params['filename'],
        ql=params['ql'], n_jobs=1, theiler_len=params['theiler_len'], save_full=False
    )

    d1 = li.compute_d1(
        exceeds, test_filepath, params['filename'],
        ql=params['ql'], theiler_len=params['theiler_len']
    )
    theta = li.compute_theta(
        exceeds_idx, test_filepath, params['filename'],
        ql=params['ql'], theiler_len=params['theiler_len'], method='sueveges'
    )

    li.compute_alphat(
        dist, exceeds_bool, test_filepath, params['filename'],
        params['tau_list'], ql=params['ql'],
        theiler_len=params['theiler_len'], l=params['l']
    )

    results_path = os.path.join(test_filepath, f"results_{params['filename']}")
    nc_file = os.path.join(
        results_path,
        f"{params['filename']}_alphat_max{max(params['tau_list'])}_{params['ql']}"
        f"_{params['theiler_len']}_{params['l']}.nc",
    )

    ds = xr.open_dataset(nc_file)
    try:
        alphat = ds["alphat"].values
        lag = ds["lag"].values
        time_index = ds["time_index"].values
    finally:
        ds.close()

    d1_expected = expected["d1"]
    d1_indices = d1_expected["indices"]
    np.testing.assert_allclose(
        d1[d1_indices], d1_expected["values"], rtol=1e-6, atol=1e-6
    )
    assert float(np.mean(d1)) == pytest.approx(
        d1_expected["mean"], rel=1e-6, abs=1e-6
    )
    assert float(np.std(d1)) == pytest.approx(
        d1_expected["std"], rel=1e-6, abs=1e-6
    )

    theta_expected = expected["theta"]
    theta_indices = theta_expected["indices"]
    np.testing.assert_allclose(
        theta[theta_indices], theta_expected["values"], rtol=1e-6, atol=1e-6
    )
    assert float(np.mean(theta)) == pytest.approx(
        theta_expected["mean"], rel=1e-6, abs=1e-6
    )
    assert float(np.std(theta)) == pytest.approx(
        theta_expected["std"], rel=1e-6, abs=1e-6
    )

    alphat_expected = expected["alphat"]
    for sample in alphat_expected["samples"]:
        lag_idx = sample["lag_index"]
        time_idx = sample["time_index"]
        assert float(alphat[lag_idx, time_idx]) == pytest.approx(
            sample["value"], rel=1e-6, abs=1e-6
        )

    for lag_idx, expected_mean in enumerate(alphat_expected["mean_by_lag"]):
        assert float(np.nanmean(alphat[lag_idx])) == pytest.approx(
            expected_mean, rel=1e-6, abs=1e-6
        )
        assert float(np.nanstd(alphat[lag_idx])) == pytest.approx(
            alphat_expected["std_by_lag"][lag_idx], rel=1e-6, abs=1e-6
        )

    assert lag.tolist() == expected["lag"]

    time_index_expected = expected["time_index"]
    assert int(time_index[0]) == time_index_expected["first"]
    assert int(time_index[-1]) == time_index_expected["last"]
    assert int(time_index.size) == time_index_expected["count"]

# Pytest will automatically discover and run test functions
# No need for main block when using pytest
