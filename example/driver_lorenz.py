import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(624)
# custom libraries
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, "../scripts"))
import local_indices as li

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Parse two strings.")
	parser.add_argument("filepath", type=str, help="filepath")
	parser.add_argument("filename", type=str, help="filename")
	parser.add_argument("ql", type=float, help="quantile")
	parser.add_argument("win", type=int, help="theiler_window")
	parser.add_argument("l", type=int, help="l")

	args = parser.parse_args()
    
	filepath = args.filepath
	filename = args.filename		
	ql = args.ql				# quantile	
	win = args.win				# Theiler window
	l = args.l					# l is the power of the distance, l = 0 for original alpha.

	tau_list = [11, 22, 44, 220, 440] # 0.05, 0.1, 0.2, 1, 2, 4 Lyapunov times for Lorenz

	# load data
	fname = os.path.join(filepath, filename+'.txt')
	Y = np.loadtxt(fname)
	t = Y[:, 0]
	X = Y[:, 1:]
	print('Start time: ', t[0], 'End time: ', t[-1], 'Time step: ', t[1]-t[0])
	print("Loaded data shape: ", X.shape)

	# compute indices
	# dist, exceeds, exceeds_idx, exceeds_bool = li.compute_exceeds(
	# 	X, filepath=filepath, filename=filename, ql=ql, n_jobs=20,
	# 	theiler_len=win, save_full=True)
	
	# print('Shape of dist:', dist.shape)
	# print('Shape of exceeds:', exceeds.shape)
	# print('Shape of exceeds_idx:', exceeds_idx.shape)
	# print('Shape of exceeds_bool:', exceeds_bool.shape)

	# if you do not want to recompute exceedances because they are too
	# expensive to calculate and already saved earlier...
	
	dist = 0
	exceeds_idx = np.load(f'{filepath}/results_{filename}/{filename}_exceeds_idx_{ql}_{win}.npy')
	exceeds_bool = li.create_bool_from_idx(exceeds_idx)
	print('Convert exceeds from index. Shape: ', exceeds_bool.shape)

	# The other two dynamical indices (local dimension and persistence)
	# d1 = li.compute_d1(exceeds, filepath, filename, ql=ql, theiler_len=win)
	# theta = li.compute_theta(exceeds_idx, filepath, filename, ql=ql,
	# 		  theiler_len=win, method='sueveges')

	alphat = li.compute_alphat(dist, exceeds_bool, filepath, filename, tau_list,
			    ql=ql, theiler_len=win, l=l)
	
