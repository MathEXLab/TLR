import os
import sys
import time
import numpy as np
import xarray as xr
from tqdm import tqdm

import scipy.stats as sc
import statsmodels.api as sm
import multiprocessing as mp
import sklearn.metrics.pairwise as skmp
import json
import logging
logging.captureWarnings(True)


def compute_exceeds(X, filepath, filename, 
                    ql = 0.99, n_jobs=1, theiler_len=0, p_value_exp=None,
                    exp_test_method='anderson', save_full=False, **kwargs):
    """
    Compute the predictability of a collection of states: X
    save_full: save the full matrix of exceedances (True) or only the indices (False)
    """

    # Create results directory if it doesn't exist   
    results_path = os.path.join(filepath, f"results_{filename}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ## Read kwargs
    metric = kwargs.get("metric")
    if metric is None: metric = "euclidean"
    
    ## check data format 
    if X.ndim != 2:	print("data is not in correct shape.")

    n_samples, n_features = X.shape  

    # initialize distances 
    dist = np.zeros((n_samples, n_samples))

    # compute pairwise distances (can take time)
    st = time.time()
    print(f'Computing pairwise_distances using {n_jobs} threads...')
    dist[:,:] = skmp.pairwise_distances(X, X, metric=metric, n_jobs=n_jobs)
    print(f'Elapsed time: {time.time()-st} seconds')

    # Theiler window
    for i in range(1, 1+theiler_len):       
        np.fill_diagonal(dist[:-i, i:], sys.float_info.max)  
        np.fill_diagonal(dist[i:, :-i], sys.float_info.max)
    np.fill_diagonal(dist, sys.float_info.max)

    if save_full:
        fname = f'{results_path}/{filename}_dist_{ql}_{theiler_len}.npy'
        np.save(fname, dist)
        print(f'Saved distance matrix to {fname}')
    else:
        print('Distance matrix not saved.')

    dist_log = -np.log(dist)
    # initialize quantile and neighbours index
    q     		 = np.zeros((n_samples)) + np.nan
    exceeds_bool = np.zeros((n_samples, n_samples), dtype = bool)

    ## calc quantile as with mquantiles with alphap=0.5 and betap=0.5
    # last 'time_lag' states do not have this index
    q = np.quantile(dist_log, ql, axis=1, method='midpoint')

    # neighbors: 1; non-neighbors: 0
    # n_neigh = int(np.round(n_samples*(1-ql))) #
    exceeds_bool = dist_log > q.reshape(-1,1)
    n_neigh = np.sum(exceeds_bool, axis=1).min()
    exceeds_bool = _correct_n_neigh(exceeds_bool, dist_log, q, n_neigh)

    exceeds_idx = np.argwhere(exceeds_bool).reshape(n_samples,n_neigh,2)[:,:,1]
    row_idx = np.arange(n_samples)[:,None]
    exceeds = dist_log[row_idx,exceeds_idx] - q[:, None]
    
    # save matrix
    fname = f'{results_path}/{filename}_exceeds_idx_{ql}_{theiler_len}.npy'
    np.save(fname, exceeds_idx)
    fname = f'{results_path}/{filename}_exceeds_{ql}_{theiler_len}.npy'
    np.save(fname, exceeds)

    return dist, exceeds, exceeds_idx, exceeds_bool


def compute_d1(exceeds, filepath, filename, ql=0.99, theiler_len=0):
    """
    Compute the local dimension d1. 
    """
    d1 = 1 / np.mean(exceeds, axis=1)
    # Create results directory if it doesn't exist
    results_path = os.path.join(filepath, f"results_{filename}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # save d1
    fname = f'{results_path}/{filename}_d1_{ql}_{theiler_len}.npy'
    np.save(fname, d1)
    return d1


def compute_theta(idx, filepath, filename, ql=0.99, theiler_len=0,
                    method='sueveges'):
    """
    Compute the extremal index \theta, also known as inverse persistence. 
    """
    def _calc_sueveges(idx, ql=0.99):
        q = 1 - ql
        # import pdb
        # pdb.set_trace()
        Ti = idx[:,1:] - idx[:,:-1]
        Si = Ti - 1
        Nc = np.sum(Si > 0, axis=1)
        K  = np.sum(q * Si, axis=1)
        N  = Ti.shape[1]
        theta = (K + N + Nc - np.sqrt((K + N + Nc)**2 - 8 * Nc * K)) / (2 * K)
        return theta
    
    def _calc_ferro(idx):
            Ti = idx[:,1:] - idx[:,:-1]
            theta = 2 * (np.sum(Ti, axis=1)**2) / ((Ti.shape[1] - 1) * \
                        np.sum(Ti**2, axis=1))
            theta2 = 2 * (np.sum(Ti - 1, axis=1)**2) / ((Ti.shape[1] - 1) * \
                        np.sum((Ti - 1) * (Ti - 2), axis=1))
            idx_Timax_larger_than_2 = np.max(Ti, axis=1) <= 2
            theta[idx_Timax_larger_than_2] = theta2[idx_Timax_larger_than_2]
            theta[theta>1] = 1
            return theta
    if method == 'sueveges':
        theta = _calc_sueveges(idx, ql)
    elif method == 'ferro':
        theta = _calc_ferro(idx)
    else:
        print(f'Method {method} to compute theta not recognized.')
        print('Using default Sueveges method.')
        theta = _calc_sueveges(idx, ql)
    
    # Create results directory if it doesn't exist
    results_path = os.path.join(filepath, f"results_{filename}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # save theta
    fname = f'{results_path}/{filename}_theta_{ql}_{theiler_len}.npy'
    np.save(fname, theta)
    return theta



def compute_alphat(dist, exceeds_bool, filepath, filename, time_lag, 
                   ql=0.99, theiler_len=0, l=1):
    """
    Compute the lagged co-recurrence of the binary matrix 'neigh'.
    Requiring trajectories to stick to the reference trajectory. 
    'exceeds_idx' is the matrix of neighbour indices (i.e., exceedances).
    'time_lag' is the sorted list of time lag of interest
    """
    # find neighbour indices of the state of interest now
    exceeds_bool_now = exceeds_bool.copy()
    
    n_samples = exceeds_bool.shape[0]   # number of states
    n_neigh = np.sum(exceeds_bool[0,:]) # number of neighbours
    alphat_dict = {}

    for lag in tqdm(time_lag):
        
        # find forward recurrences shifting by lag exceeds_bool_now
        exceeds_bool_lag = np.zeros([n_samples-lag, n_samples], dtype=bool)
        exceeds_bool_lag[:, lag:] = exceeds_bool_now[:-lag, :-lag]

        # find forward-reference-state recurrences
        exceeds_bool_lag_ref = exceeds_bool_now[lag:, :]
        
        # compute the intersection of forward recurrences and
        # forward-reference-state recurrences and use a flag of -1 where there
        # is no intersection
        exceeds_bool_intersect = np.logical_and(exceeds_bool_lag_ref, exceeds_bool_lag)
        
        if l == 0:
            alphat_dict[lag] = np.sum(exceeds_bool_intersect, axis=1) / n_neigh
        else:
            dist_sum_in = np.nansum(np.where(exceeds_bool_intersect, dist[lag:, :], np.nan) ** l, axis=1) # Neighbors sticking
            dist_sum_all = np.nansum(np.where(exceeds_bool_lag, dist[lag:, :], np.nan) ** l, axis=1)      # All forward neighbors
            alphat_dict[lag] = dist_sum_in / dist_sum_all

    # save alphat_dict
    # Create results directory if it doesn't exist
    results_path = os.path.join(filepath, f"results_{filename}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        else:
            return obj
    
    json_alphat_dict = convert_numpy_to_list(alphat_dict)
    
    with open(f'{results_path}/{filename}_alphat_max{np.array(time_lag).max()}_{ql}_{theiler_len}_{l}.json', 'w') as f:
        json.dump(json_alphat_dict, f, indent=2)
        


def create_bool_from_idx(exceeds_idx):
    """
    Create a boolean mask matrix from an index matrix without using for loops.
    
    Parameters:
    - exceeds_idx (ndarray): A matrix where each row contains column indices
        to be marked as True.
    - num_cols (int): The number of columns in the boolean mask matrix.
    
    Returns:
    - mask_matrix (ndarray): The boolean mask matrix.
    """
    n_samples = exceeds_idx.shape[0]
    exceeds_bool = np.zeros((n_samples, n_samples), dtype=bool)
    row_indices = np.arange(n_samples)[:, None]
    exceeds_bool[row_indices, exceeds_idx] = True
    return exceeds_bool



def _correct_n_neigh(exceeds_bool, dist, q, n_neigh):
    '''
    Correct the matrix `exceeds_bool` in case some neighbours fall on the edge
    of the ball.
    '''
    exceeds_bool_geq = dist >= q.reshape(-1, 1)
    print(f'Neighbors selected: {n_neigh}, checking for neighbors falling on the edge...')
    idx_edge = np.where(np.sum(exceeds_bool, axis=1)!=n_neigh)[0]
    for i in idx_edge:
        idx_add = np.where(exceeds_bool[i,:]!=exceeds_bool_geq[i,:])[0][0]
        exceeds_bool[i,idx_add] = True
    
    return exceeds_bool
