#!/usr/bin/env python3
"""
Pytest-based regression test for TLR repository using lorenz_small.txt
"""

import os
import sys
import numpy as np
import json
import tempfile
import shutil
import pytest

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import local_indices as li

@pytest.fixture
def test_data():
    """Load test data from lorenz_small.txt"""
    test_data_path = "../data/datasets/lorenz_small.txt"
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
    t, X = test_data
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
    
    # Check JSON file
    json_file = os.path.join(results_path, 
                           f"{params['filename']}_alphat_max{max(params['tau_list'])}_{params['ql']}_{params['theiler_len']}_{params['l']}.json")
    assert os.path.exists(json_file), "JSON file should be created"
    
    # Load and verify JSON content
    with open(json_file, 'r') as f:
        alpha_dict = json.load(f)
    
    # Verify all tau values are present
    for tau in params['tau_list']:
        assert str(tau) in alpha_dict, f"Missing tau value {tau} in results"
        assert len(alpha_dict[str(tau)]) > 0, f"Empty data for tau value {tau}"
        
        # Check alpha values are in valid range
        alpha_values = np.array(alpha_dict[str(tau)])
        assert np.all(alpha_values >= 0) and np.all(alpha_values <= 1), \
            f"Alpha values out of range [0,1] for tau {tau}"

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
    json_file = os.path.join(results_path, 
                           f"{params['filename']}_alphat_max{max(params['tau_list'])}_{params['ql']}_{params['theiler_len']}_{params['l']}.json")
    
    with open(json_file, 'r') as f:
        alpha_dict = json.load(f)
    
    # Check statistics for each tau
    for tau in params['tau_list']:
        alpha_values = np.array(alpha_dict[str(tau)])
        mean_alpha = np.mean(alpha_values)
        std_alpha = np.std(alpha_values)
        
        # Basic sanity checks
        assert 0 < mean_alpha < 1, f"Mean alpha should be in (0,1) for tau {tau}, got {mean_alpha}"
        assert std_alpha > 0, f"Standard deviation should be positive for tau {tau}, got {std_alpha}"
        
        # Check that alpha decreases with increasing tau (typical behavior)
        if tau > params['tau_list'][0]:
            prev_tau = params['tau_list'][params['tau_list'].index(tau) - 1]
            prev_mean = np.mean(alpha_dict[str(prev_tau)])
            assert mean_alpha <= prev_mean, \
                f"Alpha should generally decrease with tau: tau {prev_tau} mean={prev_mean:.4f}, tau {tau} mean={mean_alpha:.4f}"

# Pytest will automatically discover and run test functions
# No need for main block when using pytest
