import os
import sys
import argparse
import numpy as np
import xarray as xr
np.random.seed(624)

# custom libraries
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, "../scripts"))
import postprocessing as post


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Parse two strings.")
	parser.add_argument("filepath", type=str, help="filepath")
	parser.add_argument("filename", type=str, help="filename")
	parser.add_argument("ql", type=float, help="quantile")
	parser.add_argument("win", type=int, help="theiler_window")
	parser.add_argument("l", type=int, help="distance metric")

	args = parser.parse_args()
    
	filepath = args.filepath
	filename = args.filename
	ql = args.ql				# quantile	
	win = args.win				# Theiler window
	l = args.l					# l is the power of the distance, l = 0 for original alpha.

	# For Lorenz
	tau_list = np.array([11, 22, 44, 220, 440])
	tau_l_list = [0.05, 0.1, 0.2, 1, 2] 		# 0.05, 0.1, 0.2, 1, 2 Lyapunov times for Lorenz
	max_tau = np.max(tau_list)
	
	# # load local index (new version)
        results_path = os.path.join(filepath, f"results_{filename}")
        nc_path = os.path.join(results_path,
                f"{filename}_alphat_max{max_tau}_{ql}_{win}_{l}.nc")
        ds = xr.open_dataset(nc_path)
        alphat_da = ds["alphat"].transpose("time_index", "lag")
        alphat = np.asarray(alphat_da.sel(lag=tau_list).values)
        ds.close()

        post.plot_attractor_pdf(
        filepath=filepath, filename=filename, ql=ql, alphat=alphat,
                tau_list=tau_list, tau_l_list=tau_l_list, l=l)
