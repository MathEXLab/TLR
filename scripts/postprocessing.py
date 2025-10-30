import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

# Use LaTeX for text rendering
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

## ------------------------
## Main methods for users
## ------------------------

def plot_indices_scatter(filepath, filename, ql, theiler, 
                         tau_list, tau_l_list, figx=8, figy=6, 
                         savefig="save"):
	"""
	Scatter plot of \theta vs d1 colored by \alphat
	"""
	print(f"Plotting [plot_indices_scatter] for {filename}, {ql}, {theiler}")
	# load local index
	fname = filename+'_d1_'+str(ql)+'_'+str(theiler)+'.txt'
	d1 = np.loadtxt(os.path.join(filepath, fname))
	fname = filename+'_theta_'+str(ql)+'_'+str(theiler)+'.txt'
	theta = np.loadtxt(os.path.join(filepath, fname))
	n_samples = len(d1)
	alphat = np.empty([n_samples-tau_list[0],len(tau_list)])
	for (i, tau) in enumerate(tau_list):
		fname = filename+'_alphat'+str(tau)+'_'+str(ql)+'_'+str(theiler)+'.txt'
		n_rows = np.loadtxt(os.path.join(filepath, fname)).shape[0]
		alphat[:n_rows,i] = np.loadtxt(os.path.join(filepath, fname))
	# Scatter plot
	fig, axes = plt.subplots(3,2, figsize=(figx,figy))
	i = -1
	for (tau, ax) in zip(tau_list, axes.flat):
		i = i+1
		cax = ax.scatter(d1[:-tau_list[0]], theta[:-tau_list[0]], 1, \
		   alphat[:,i], cmap='magma', vmin=0, vmax=1)
		ax.set_title(r'$\eta = $'+str(tau_l_list[i])+r' $\tau_l$')
		if i == 4 or i == 5:
			ax.set_xlabel(r'$d_1$')
		else:
			ax.tick_params(
				axis='x',     # changes apply to the x-axis
				which='both', # both major and minor ticks are affected
				bottom=False, # ticks along the bottom edge are off
				top=False,    # ticks along the top edge are off
				labelbottom=False)
		if np.mod(i,2)==0:
			ax.set_ylabel(r'$\theta$')
		else:
			ax.tick_params(
				axis='y',     # changes apply to the y-axis
				which='both', # both major and minor ticks are affected
				left=False,   # ticks along the left edge are off
				right=False,  # ticks along the right edge are off
				labelleft=False)
	# deploy colorbar
	plt.tight_layout()
	cbar = fig.colorbar(
     	cax, ax=axes, orientation='horizontal', pad=0.08, fraction=0.02)
	cbar.set_label(r'$\alpha_\eta$')
	# save figure
	fpath = os.path.join(filepath, "figures")
	if not os.path.exists(fpath): os.makedirs(fpath)
	fout = os.path.join(fpath, filename+'_scatter'\
     	+str(ql)+'_'+str(theiler)+'.png')
	_close_fig(savefig, fout, plt)
 
 
 
def plot_indices_horizon(filepath, filename, ql, theiler, 
						 tau_list, tau_l_list, figx=12, figy=8, 
       					 savefig="save"):
	"""
	TODO: Bug - plots random values time to time (intermittent behaviour)
	Plot of max, min and average \alphat vs \eta
	"""
	print(f"Plotting [plot_indices_horizon] for {filename}, {ql}, {theiler}")
 	# load local index d1 to get n_samples for loading alpha
	fname = filename+'_d1_'+str(ql)+'_'+str(theiler)+'.txt'
	d1 = np.loadtxt(os.path.join(filepath, fname))
	n_samples = len(d1)
	# load alpha and compute min, max and average
	alphat      = np.empty([n_samples-tau_list[0],len(tau_list)])
	alphat_min  = np.empty([len(tau_list),])
	alphat_mean = np.empty([len(tau_list),])
	alphat_max  = np.empty([len(tau_list),])
	for (i, tau) in enumerate(tau_list):
		fname = filename+'_alphat'+str(tau)+'_'+str(ql)+'_'+str(theiler)+'.txt'
		n_rows = np.loadtxt(os.path.join(filepath, fname)).shape[0]
		alphat[:n_rows,i] = np.loadtxt(os.path.join(filepath, fname))
		alphat_min [i] = np.min (alphat[:,i])
		alphat_max [i] = np.max (alphat[:,i])
		alphat_mean[i] = np.mean(alphat[:,i])
		# epsilon_t = 1e-16
		# if alphat_min [i] == 0: alphat_min [i] = epsilon_t
		# if alphat_max [i] == 0: alphat_max [i] = epsilon_t
		# if alphat_mean[i] == 0: alphat_mean[i] = epsilon_t
	# plot alpha min, max, and average vs tau -- loglog
	fig, axes = plt.subplots(1,1,figsize=(figx,figy))	
	plt.plot(tau_l_list, alphat_min , '.-b')
	plt.plot(tau_l_list, alphat_mean, '.-k')
	plt.plot(tau_l_list, alphat_max , '.-r')
	plt.xscale('log')	
	plt.yscale('log')
	plt.xlabel(r'$t~(\tau_l)$')
	plt.ylabel(r'$\alpha_\eta$')
	plt.tight_layout()
	# save figure
	fpath = os.path.join(filepath, "figures")
	if not os.path.exists(fpath): os.makedirs(fpath)
	fout = os.path.join(fpath, filename+'_indices_horizon_loglog'\
		+str(ql)+'_'+str(theiler)+'.png')
	plt.savefig(fout, dpi=300)
	plt.show()
 	# plot alpha min, max, and average vs tau -- non-loglog
	fig, axes = plt.subplots(1,1,figsize=(figx,figy))	
	plt.plot(tau_l_list, alphat_min , '.-b')
	plt.plot(tau_l_list, alphat_mean, '.-k')
	plt.plot(tau_l_list, alphat_max , '.-r')
	plt.xlabel(r'$t~(\tau_l)$')
	plt.ylabel(r'$\alpha_\eta$')
	plt.tight_layout()
	# save figure
	fpath = os.path.join(filepath, "figures")
	if not os.path.exists(fpath): os.makedirs(fpath)
	fout = os.path.join(fpath, filename+'_indices_horizon'\
     	+str(ql)+'_'+str(theiler)+'.png')
	_close_fig(savefig, fout, plt)



def plot_ts(filepath, filename, local_index, ql, theiler, 
            tau_l=None, figx=12, figy=8, vmin=None, vmax=None, 
			cmap='magma', savefig="save"):
	print(f"Plotting [plot_ts] ")
	print(f"for {filename}, {local_index}, {ql}, {theiler}")
	# load local index
	fname = filename+'_'+local_index+'_'+str(ql)+'_'+str(theiler)+'.txt'
	l_ind = np.loadtxt(os.path.join(filepath, fname))
	# load data
	Y = np.loadtxt(os.path.join(filepath, filename+'.txt'))
	t = Y[:,0]
	X = Y[:,1:]
	n_vars = X.shape[1]
	# load exponential test data
	sr = _get_slice_rows(shorter_row_var=l_ind, local_index=local_index)
	flag_expo, E, color_negative, color_positive = \
     	_load_exponential_test(filepath, filename, ql, theiler, sr)
      
	# plot ts with l_ind color
	fig, axes = plt.subplots(n_vars,1,figsize=(figx,figy))
	for (i, ax) in zip(range(n_vars), axes.flat):
		vmin, vmax = _get_vmin_and_max(c_val=l_ind, vmin=vmin, vmax=vmax)
		cax = ax.scatter(
			t[sr], X[sr,i], c=l_ind, cmap=cmap, vmin=vmin, vmax=vmax, s=0.3)
		if flag_expo:
			min_val = np.ones(E[sr][E[sr]<0].shape)*np.min(X[sr,i])
			ax.scatter(t[sr][E[sr]<0], min_val[sr], c=color_negative[sr], 
          		alpha=1.0, s=0.2)
		# axes title
		if i == 0:
			title_str = _get_title_str(local_index, tau_l)
			ax.set_title(title_str)
		if i == n_vars-1: 
			ax.set_xlabel(r'$t$')
	# deploy colorbar
	plt.tight_layout()
	cbar = fig.colorbar(
     	cax, ax=axes, orientation='horizontal', pad=0.08, fraction=0.02)
	label_str = _get_label_str(local_index)
	cbar.set_label(label_str)
	# save figure
	fpath = os.path.join(filepath, "figures")
	if not os.path.exists(fpath): os.makedirs(fpath)
	fout = os.path.join(fpath, filename+'_ts_'+local_index\
      	+'_'+str(ql)+'_'+str(theiler)+'.png')
	_close_fig(savefig, fout, plt)
  
  
  
def plot_ts_neighbours(filepath, filename, local_index, ql, theiler, 
                       interest_pts=['min', 10], tau_l=None, figx=12, 
                       figy=8, vmin=None, vmax=None, cmap='magma',
					   alpha=0.8, savefig="save"):
	print(f"Plotting [plot_ts_neighbours] "\
     	f"for {filename}, {local_index}, {ql}, {theiler}")
	# load local index
	fname = filename+'_'+local_index+'_'+str(ql)+'_'+str(theiler)+'.txt'
	l_ind = np.loadtxt(os.path.join(filepath, fname))
	sr = _get_slice_rows(shorter_row_var=l_ind, local_index=local_index)
	# load data
	Y = np.loadtxt(os.path.join(filepath, filename+'.txt'))
	t = Y[:,0]
	X = Y[:,1:]
	n_vars = X.shape[1]
	# load exceeds_idx matrix
	ftmp = f'{filename}_exceeds_idx_{ql}_{theiler}.npy'
	exceeds_idx = np.load(os.path.join(filepath, ftmp))
	# load exponential test data
	flag_expo, E, color_negative, color_positive = \
     	_load_exponential_test(filepath, filename, ql, theiler, sr)

	# extract neighbours of the points of interest
	t_neigh, X_neigh, t_c, X_c, n_interest_pts = _extract_neighbours(
		t, X, l_ind, exceeds_idx, interest_pts)

	# scatter plot of time series with neighbours
	for j in range(n_interest_pts):
		fig, axes = plt.subplots(n_vars,1,figsize=(figx,figy))
		for (i, ax) in zip(range(n_vars), axes.flat):
			# slice depending on lag for \alphat
			vmin, vmax = _get_vmin_and_max(c_val=l_ind, vmin=vmin, vmax=vmax)
			cax = ax.scatter(
       			t[sr], X[sr,i], c=l_ind, cmap=cmap, vmin=vmin, vmax=vmax,
				alpha=alpha, s=0.3)
			if flag_expo:
				min_val = np.ones(E[sr][E[sr]<0].shape)*np.min(X[sr,i])
				ax.scatter(t[sr][E[sr]<0], min_val[sr], c=color_negative[sr], 
              		alpha=1.0, s=0.2)
			# set titles
			if i == 0:
				title_str = _get_neigh_title_str(
        			local_index, tau_l, j, t_c[j], X_c[j])
				ax.set_title(title_str)
			if i == n_vars-1: ax.set_xlabel(r'$t$')
			# plot neighbours for point of interest j
			ax.scatter(t_neigh[:,j], X_neigh[:,i,j], 20, edgecolors='g')
		# deploy colorbar
		plt.tight_layout()
		cbar = fig.colorbar(
			cax, ax=axes, orientation='horizontal', pad=0.08, fraction=0.02)
		label_str = _get_label_str(local_index)
		cbar.set_label(label_str)
   	# save figure
	fpath = os.path.join(filepath, "figures")
	if not os.path.exists(fpath): os.makedirs(filepath)
	fout = os.path.join(fpath, filename+'_ts_'+local_index\
     	+'_ip'+str(j)+'_'+str(interest_pts[0])\
		+'_'+str(ql)+'_'+str(theiler)+'.png')
	_close_fig(savefig, fout, plt)
 

 
def plot_attractor(filepath, filename, local_index, ql, theiler, 
                   tau_l=None, figx=12, figy=8, vmin=None, vmax=None, 
                   cmap='magma', savefig="save"):
	print(f"Plotting [plot_attractor] "\
		f"for {filename}, {local_index}, {ql}, {theiler}")
	# load local index
	fname = filename+'_'+local_index+'_'+str(ql)+'_'+str(theiler)+'.txt'
	l_ind = np.loadtxt(os.path.join(filepath, fname))
	# load data
	Y = np.loadtxt(os.path.join(filepath, filename+'.txt'))
	X = Y[:,1:]
	n_vars = X.shape[1]
 
	# plot attractor with l_ind color
	fig = plt.figure(figsize=(figx,figy))
	ax = fig.add_subplot(111, projection='3d')
	# slice depending on lag for \alphat
	sr = _get_slice_rows(shorter_row_var=l_ind, local_index=local_index)
	vmin, vmax = _get_vmin_and_max(c_val=l_ind, vmin=vmin, vmax=vmax)
	# scatter plot
	if n_vars == 2:
		cax = ax.scatter(
			X[sr,0], X[sr,1], c=l_ind, cmap=cmap, vmin=vmin, vmax=vmax)		
	elif n_vars == 3:
		cax = ax.scatter(
			X[sr,0], X[sr,1], X[sr,2], c=l_ind, cmap=cmap, vmin=vmin, vmax=vmax)
	else:
		print(f'dimension n_vars={n_vars} not supported - exiting')
		return True
	# set title 
	title_str = _get_title_str(local_index, tau_l)
	ax.set_title(title_str, pad=0.04)
	# deploy colorbar
	cbar = fig.colorbar(cax, orientation='horizontal', pad=0.01, fraction=0.04)
	label_str = _get_label_str(local_index)
	cbar.set_label(label_str)
	plt.tight_layout()
	# close figure
	fpath = os.path.join(filepath, "figures")
	if not os.path.exists(fpath): os.makedirs(fpath)
	fout = os.path.join(fpath, filename+'_attractor_'+local_index\
      	+'_'+str(ql)+'_'+str(theiler)+'.png')
	_close_fig(savefig, fout, plt)
 
 
 
def plot_attractor_neighbours(filepath, filename, local_index, 
	ql=0.99, theiler=0, tau_l=0.1, interest_pts=['max', 2], 
	figx=12, figy=8, vmin=None, vmax=None, cmap='magma', 
	alpha=0.04, savefig="save"):
	print(f"Plotting [plot_attractor_neighbours] "\
		f"for {filename}, {local_index}, {ql}, {theiler}")
	# load local index
	fname = filename+'_'+local_index+'_'+str(ql)+'_'+str(theiler)+'.txt'
	l_ind = np.loadtxt(os.path.join(filepath, fname))
	# load data
	Y = np.loadtxt(os.path.join(filepath, filename+'.txt'))
	t = Y[:,0]
	X = Y[:,1:]
	n_vars = X.shape[1]
	# load exceeds_idx matrix
	fname = os.path.join(filepath, f'{filename}_exceeds_idx_{ql}_{theiler}.npy')
	exceeds_idx = np.load(fname)

	# extract neighbours of the points of interest
	_, X_neigh, t_c, X_c, n_interest_pts = _extract_neighbours(
		t, X, l_ind, exceeds_idx, interest_pts)
	
	# plot neighbours of the points of interest on the attractor
	for j in range(n_interest_pts):		
		fig = plt.figure(figsize=(figx,figy))
		ax = fig.add_subplot(111, projection='3d')
		# slice depending on lag for \alphat
		sr = _get_slice_rows(shorter_row_var=l_ind, local_index=local_index)
		vmin, vmax = _get_vmin_and_max(c_val=l_ind, vmin=vmin, vmax=vmax)
		# scatter plots
		if n_vars == 2:
			cax = ax.scatter(X[sr,0], X[sr,1], 
				c=l_ind, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
			ax.scatter(X_neigh[:,0,j], X_neigh[:,1,j], alpha=1, c='g', s=40)
		elif n_vars == 3: 
			cax = ax.scatter(X[sr,0], X[sr,1], X[sr,2], 
				c=l_ind, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
			ax.scatter(X_neigh[:,0,j], X_neigh[:,1,j], X_neigh[:,2,j], 
				alpha=1, c='g', s=40)
		else:
			print(f'dimension n_vars={n_vars} not supported - exiting')
			return True

		# set title
		title_str = _get_neigh_title_str(local_index, tau_l, j, t_c[j], X_c[j])
		ax.set_title(title_str, pad=0.04)
		# deploy colorbar
		cbar = fig.colorbar(
      		cax, orientation='horizontal', pad=0.01, fraction=0.04)
		label_str = _get_label_str(local_index)
		cbar.set_label(label_str)
		plt.tight_layout()
		# save figure
		fpath = os.path.join(filepath, "figures")
		if not os.path.exists(fpath): os.makedirs(fpath)
		fout = os.path.join(fpath, filename+'_attractor_'\
      		+local_index+'_ip'+str(j)+'_'+str(interest_pts[0])+'_'\
            +str(ql)+'_'+str(theiler)+'.png')
		_close_fig(savefig, fout, plt)
		
def plot_pdf(filepath, filename, ql, theiler, 
                        tau_list, tau_l_list):	
	"""
	Draw the distributuion of \alphat for different \eta.
	"""
	# load data
	Y = np.loadtxt(os.path.join(filepath, filename+'.txt'))
	t = Y[:,0]
	n_samples = len(t)
	assert len(tau_list) == 5, 'The tau_list must have 5 elements'
	alphat = np.empty([n_samples-tau_list[0], len(tau_list)])
	for (i, tau) in enumerate(tau_list):
		fname = filename+'_alphat'+str(tau)+'_'+str(ql)+'_'+str(theiler)+'.npy'
		n_rows = np.load(os.path.join(filepath, fname)).shape[0]
		alphat[:n_rows,i] = np.load(os.path.join(filepath, fname))
	font_size = 13
	fig = plt.figure(figsize=(5, 4), dpi=300)
	ax = fig.add_subplot(111)
	ax.set_ylim(0, 8)
	ax.set_xlim(0, 1)
	ax.set_xticks(np.linspace(0, 1, 6))
	ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
	ax.set_yticks(np.linspace(0, 7, 8))
	ax.set_xticks(np.linspace(0, 1, 21), minor=True)
	ax.set_yticks(np.linspace(0, 7.8, 40), minor=True)
	c_list = ['k', 'r', 'b', 'g', 'm']
	style = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1))]

	for i, t_l in enumerate(tau_l_list):
		data = alphat[:, i]
		kde  = gaussian_kde(data)
		x_values = np.linspace(0, 1, 201)  
		density  = kde(x_values) 
		ax.plot(x_values, density, label=rf'$\mathdefault{{\eta = {t_l}\tau_\ell}}$', color=c_list[i], lw=1, linestyle=style[i])

	ax.legend(fontsize=font_size-2, loc='upper center', frameon=False, ncol=2)
	ax.set_xlabel(r'$\mathdefault{\alpha_\eta}$', fontsize=font_size+2)
	ax.set_ylabel('PDF', fontsize=font_size+2)
	plt.tight_layout()
	# Save the figure
	fpath = os.path.join(filepath, "figures")
	if not os.path.exists(fpath): os.makedirs(fpath)
	fout = os.path.join(fpath, filename+'_pdf_'+str(ql)+'.png')
	plt.savefig(fout, dpi=300)

def plot_attractor_pdf_2D(filepath, filename, ql, alphat, 
                        tau_list, tau_l_list, l):      
    """
    Draw the attractor together with the PDF of \alphat.
    """
    # Set font type
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    
    # load data
    Y = np.loadtxt(os.path.join(filepath, filename+'.txt'))
    t = Y[:,0]
    X = Y[:,1:]
    n_vars = X.shape[1]
    n_samples = len(t)
    assert len(tau_list) == 5, 'The tau_list must have 5 elements'
    assert len(tau_list) == len(tau_l_list), 'tau_list and tau_l_list must have the same length'
    fig = plt.figure(figsize=(8, 10), dpi=450)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    # Create three 3D axes
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    axes = [ax1, ax2, ax3, ax4, ax5]
    # Key parameters, including the colormap
    font_size = 13
    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0, 1, 21))  
    segmented_cmap = LinearSegmentedColormap.from_list("segmented_inferno", colors, N=20)
    # Plot the attractor
    for i, ax in enumerate(axes):
        im = ax.scatter(X[:-tau_list[i], 0], X[:-tau_list[i], 1], 
           c=alphat[:-tau_list[i], i], s=0.2, cmap=segmented_cmap, vmin=0, vmax=1)
        ax.tick_params(labelsize=font_size, pad=0.08)
        ax.set_title(rf'$\mathdefault{{\eta = {tau_l_list[i]}}}$; $\mathdefault{{\overline{{\alpha}}_\eta={np.round(alphat[:, i].mean(), 2)}}}$',
                    fontsize=font_size+3)
        cbar = ax.figure.colorbar(im, orientation='vertical', pad=0.1, shrink=0.65)
        cbar.ax.set_yticks(np.linspace(0, 1, 6))
        cbar.ax.set_yticks(np.linspace(0, 1, 21), minor=True)
        cbar.ax.tick_params(labelsize=font_size)
    # ax6 (PDF)
    ax6 = fig.add_subplot(326)
    ax6.set_position([0.56, 0.06, 0.33, 0.25])
    ax6.tick_params(labelsize=font_size)
    ax6.set_xlim(0, 1)
    ax6.set_xticks(np.linspace(0, 1, 6))
    ax6.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    ax6.set_xticks(np.linspace(0, 1, 21), minor=True)
    c_list = ['k', 'r', 'b', 'g', 'm']
    style = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1))]
    ymax = -np.inf
    for i, t_l in enumerate(tau_l_list):
        data = alphat[:-tau_list[i], i]
        kde  = gaussian_kde(data)
        x_values = np.linspace(0, 1, 201)  
        density  = kde(x_values) 
        if density.max() > ymax:
            ymax = density.max()
        ax6.plot(x_values, density, label=rf'$\mathdefault{{\eta = {t_l}}}$', color=c_list[i], lw=1, linestyle=style[i])
    ax6.legend(fontsize=font_size-2, loc='upper left', frameon=False, ncol=1)
    ax6.set_xlabel(r'$\mathdefault{\alpha_\eta}$', fontsize=font_size+2)
    ax6.set_ylabel('PDF', fontsize=font_size+2)
    
    # Set the y-axis limits
    ax6.set_ylim(0, np.ceil(ymax))
    # ax6.set_yticks(np.linspace(0, np.ceil(ymax), int(np.ceil(ymax)*5)+1), minor=True)
    ax6.set_yticks(np.arange(0, 3500, 500))
    # Add index for each subfigure
    a = 0.02
    fig.text(a, 0.965, '(a)', ha='center', va='center', fontsize=font_size+5)
    fig.text(0.49+a, 0.965, '(b)', ha='center', va='center', fontsize=font_size+5)
    fig.text(a, 0.641, '(c)', ha='center', va='center', fontsize=font_size+5)
    fig.text(0.49+a, 0.641, '(d)', ha='center', va='center', fontsize=font_size+5)
    fig.text(a, 0.322, '(e)', ha='center', va='center', fontsize=font_size+5)
    fig.text(0.49+a, 0.322, '(f)', ha='center', va='center', fontsize=font_size+5)
    # Save the figure
    # plt.tight_layout()
    fpath = os.path.join(filepath, "figures")
    if not os.path.exists(fpath): os.makedirs(fpath)
    fout = os.path.join(fpath, filename+'_attractor_pdf_'+str(ql)+'_'+str(l)+'.jpg')
    plt.savefig(fout, dpi=450)


def plot_attractor_pdf(filepath, filename, ql, alphat, 
                        tau_list, tau_l_list, l):		
	"""
	Draw the attractor together with the PDF of \alphat.
	"""
	# Set font type
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	
	# load data
	Y = np.loadtxt(os.path.join(filepath, filename+'.txt'))
	t = Y[:,0]
	X = Y[:,1:]

	tau_list = np.array(tau_list)
	assert len(tau_list) == 5, 'The tau_list must have 5 elements'
	assert len(tau_list) == len(tau_l_list), 'tau_list and tau_l_list must have the same length'

	fig = plt.figure(figsize=(8, 10), dpi=450)
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05, hspace=0.05)
	# Create three 3D axes
	ax1 = fig.add_subplot(321, projection='3d')
	ax2 = fig.add_subplot(322, projection='3d')
	ax3 = fig.add_subplot(323, projection='3d')
	ax4 = fig.add_subplot(324, projection='3d')
	ax5 = fig.add_subplot(325, projection='3d')
	axes = [ax1, ax2, ax3, ax4, ax5]

	# Key parameters, including the colormap
	font_size = 13
	cmap = plt.cm.inferno
	colors = cmap(np.linspace(0, 1, 21))  
	segmented_cmap = LinearSegmentedColormap.from_list("segmented_inferno", colors, N=20)

	# Plot the attractor
	for i, ax in enumerate(axes):
		
		end_idx = - (tau_list[i] - np.min(tau_list)) if tau_list[i] != np.min(tau_list) else X.shape[0]
		im = ax.scatter(X[:-tau_list[i], 0], X[:-tau_list[i], 1], X[:-tau_list[i], 2], 
		   c=alphat[:end_idx, i], s=0.2, cmap=segmented_cmap, vmin=0, vmax=1)
		ax.tick_params(labelsize=font_size, pad=0.08)
		ax.text(0, 0, X[:-tau_list[i], 2].max()*1.6, rf'$\mathdefault{{\eta = {tau_l_list[i]}\eta_\ell}}$; $\mathdefault{{\overline{{\alpha}}_\eta={np.round(alphat[:, i].mean(), 2)}}}$',
					fontsize=font_size+3, ha='center', va='center')
		cbar = ax.figure.colorbar(im, orientation='vertical', pad=0.1, shrink=0.65)
		cbar.ax.set_yticks(np.linspace(0, 1, 6))
		cbar.ax.set_yticks(np.linspace(0, 1, 21), minor=True)
		cbar.ax.tick_params(labelsize=font_size)

	# ax6 (PDF)
	ax6 = fig.add_subplot(326)
	ax6.set_position([0.57, 0.055, 0.4, 0.25])
	ax6.tick_params(labelsize=font_size)
	ax6.set_xlim(0, 1)
	ax6.set_xticks(np.linspace(0, 1, 6))
	ax6.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
	ax6.set_xticks(np.linspace(0, 1, 21), minor=True)

	c_list = ['k', 'r', 'b', 'g', 'm']
	style = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1))]
	ymax = -np.inf

	for i, t_l in enumerate(tau_l_list):
		end_idx = - (tau_list[i] - np.min(tau_list)) if tau_list[i] != np.min(tau_list) else X.shape[0]
		data = alphat[:end_idx, i]
		kde  = gaussian_kde(data)
		x_values = np.linspace(0, 1, 201)  
		density  = kde(x_values) 
		if density.max() > ymax:
			ymax = density.max()
		ax6.plot(x_values, density, label=rf'$\mathdefault{{\eta = {t_l}\eta_\ell}}$', color=c_list[i], lw=1, linestyle=style[i])
	ax6.legend(fontsize=font_size-2, loc='upper center', frameon=False, ncol=1)
	ax6.set_xlabel(r'$\mathdefault{\alpha_\eta}$', fontsize=font_size+2)
	ax6.set_ylabel('PDF', fontsize=font_size+2)
	
	# Set the y-axis limits
	ax6.set_ylim(0, np.ceil(ymax))
	# ax6.set_yticks(np.linspace(0, np.ceil(ymax), int(np.ceil(ymax)*5)+1), minor=True)
	ax6.set_yticks(np.linspace(0, ((np.ceil(ymax) + 4) // 5) * 5, 6))

	# Add index for each subfigure
	a = 0.02
	fig.text(a, 0.97, '(a)', ha='center', va='center', fontsize=font_size+5)
	fig.text(0.49+a, 0.97, '(b)', ha='center', va='center', fontsize=font_size+5)
	fig.text(a, 0.632, '(c)', ha='center', va='center', fontsize=font_size+5)
	fig.text(0.49+a, 0.632, '(d)', ha='center', va='center', fontsize=font_size+5)
	fig.text(a, 0.294, '(e)', ha='center', va='center', fontsize=font_size+5)
	fig.text(0.49+a, 0.294, '(f)', ha='center', va='center', fontsize=font_size+5)

	# Save the figure
	# plt.tight_layout()
	fpath = os.path.join(filepath, "figures")
	if not os.path.exists(fpath): os.makedirs(fpath)
	fout = os.path.join(fpath, filename+'_attractor_pdf_'+str(ql)+'_'+str(l)+'.jpg')
	fig.savefig(fout, dpi=450, bbox_inches='tight')

def plot_attractor_pdf_sup(filepath, filename, ql, alphat, 
                        tau_list, tau_l_list, l):		
	"""
	Draw the attractor together with the PDF of \alphat.
	"""
	# Set font type
	plt.rcParams["font.family"] = "serif"
	plt.rcParams["font.serif"] = "Times New Roman"
	
	# load data
	Y = np.loadtxt(os.path.join(filepath, filename+'.txt'))
	t = Y[:,0]
	X = Y[:,1:-1]
	n_vars = X.shape[1]
	n_samples = len(t)
	assert len(tau_list) == 5, 'The tau_list must have 5 elements'
	assert len(tau_list) == len(tau_l_list), 'tau_list and tau_l_list must have the same length'

	fig = plt.figure(figsize=(8, 10), dpi=450)
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05, hspace=0.05)
	# Create three 3D axes
	ax1 = fig.add_subplot(321, projection='3d')
	ax2 = fig.add_subplot(322, projection='3d')
	ax3 = fig.add_subplot(323, projection='3d')
	ax4 = fig.add_subplot(324, projection='3d')
	ax5 = fig.add_subplot(325, projection='3d')
	axes = [ax1, ax2, ax3, ax4, ax5]

	# Key parameters, including the colormap
	font_size = 13
	cmap = plt.cm.inferno
	colors = cmap(np.linspace(0, 1, 21))  
	segmented_cmap = LinearSegmentedColormap.from_list("segmented_inferno", colors, N=20)

	# Plot the attractor
	for i, ax in enumerate(axes):
		im = ax.scatter(X[:-tau_list[i], 0], X[:-tau_list[i], 1], X[:-tau_list[i], 2], 
		   c=alphat[:-tau_list[i], i], s=0.2, cmap=segmented_cmap, vmin=0, vmax=1)
		ax.tick_params(labelsize=font_size, pad=0.08)
		ax.text(0, 0, 0.75, rf'$\mathdefault{{\eta = {tau_l_list[i]}}}$; $\mathdefault{{\overline{{\alpha}}_\eta={np.round(alphat[:, i].mean(), 2)}}}$',
					fontsize=font_size+3, ha='center', va='center')
		cbar = ax.figure.colorbar(im, orientation='vertical', pad=0.1, shrink=0.65)
		cbar.ax.set_yticks(np.linspace(0, 1, 6))
		cbar.ax.set_yticks(np.linspace(0, 1, 21), minor=True)
		cbar.ax.tick_params(labelsize=font_size)

	# ax6 (PDF)
	ax6 = fig.add_subplot(326)
	ax6.set_position([0.57, 0.055, 0.4, 0.25])
	ax6.tick_params(labelsize=font_size)
	ax6.set_xlim(0, 1)
	ax6.set_xticks(np.linspace(0, 1, 6))
	ax6.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
	ax6.set_xticks(np.linspace(0, 1, 21), minor=True)

	c_list = ['k', 'r', 'b', 'g', 'm']
	style = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1))]
	ymax = -np.inf
	for i, t_l in enumerate(tau_l_list):
		data = alphat[:-tau_list[i], i]
		kde  = gaussian_kde(data)
		x_values = np.linspace(0, 1, 201)  
		density  = kde(x_values) 
		if density.max() > ymax:
			ymax = density.max()
		ax6.plot(x_values, density, label=rf'$\mathdefault{{\eta = {t_l}}}$', color=c_list[i], lw=1, linestyle=style[i])
	ax6.legend(fontsize=font_size-2, loc='upper left', frameon=False, ncol=1)
	ax6.set_xlabel(r'$\mathdefault{\alpha_\eta}$', fontsize=font_size+2)
	ax6.set_ylabel('PDF', fontsize=font_size+2)
	
	# Set the y-axis limits
	ax6.set_ylim(0, np.ceil(ymax))
	# ax6.set_yticks(np.linspace(0, np.ceil(ymax), int(np.ceil(ymax)*5)+1), minor=True)
	ax6.set_yticks(np.linspace(0, 10, 6))

	# Add index for each subfigure
	a = 0.02
	fig.text(a, 0.97, '(a)', ha='center', va='center', fontsize=font_size+5)
	fig.text(0.49+a, 0.97, '(b)', ha='center', va='center', fontsize=font_size+5)
	fig.text(a, 0.632, '(c)', ha='center', va='center', fontsize=font_size+5)
	fig.text(0.49+a, 0.632, '(d)', ha='center', va='center', fontsize=font_size+5)
	fig.text(a, 0.294, '(e)', ha='center', va='center', fontsize=font_size+5)
	fig.text(0.49+a, 0.294, '(f)', ha='center', va='center', fontsize=font_size+5)

	# Save the figure
	# plt.tight_layout()
	fpath = os.path.join(filepath, "figures")
	if not os.path.exists(fpath): os.makedirs(fpath)
	fout = os.path.join(fpath, filename+'_attractor_pdf_'+str(ql)+'_'+str(l)+'.jpg')
	fig.savefig(fout, dpi=450, bbox_inches='tight')

## ----------------------------------
## Auxialiary methods for developers
## ----------------------------------
def _load_exponential_test(filepath, filename, ql, theiler, sr):
	# load exponential test data
	fexpo = os.path.join(filepath, f'{filename}_exp_stat_{ql}_{theiler}.txt')
	if os.path.exists(fexpo):
		E = np.loadtxt(fexpo)
		negative_values = E[sr][E[sr] <0]
		positive_values = E[sr][E[sr]>=0]
		color_negative = ['red'   for value in negative_values]
		color_positive = ['green' for value in positive_values]
		flag_expo = True
		return flag_expo, E, color_negative, color_positive 
	else:
		flag_expo = False
		E = None
		color_negative = None
		color_positive = None 
		return flag_expo, E, color_negative, color_positive
  
  
  
def _extract_neighbours(t, X, l_ind, exceeds_idx, interest_pts):
	"""
	Extract the neighbours for plotting.
	"""
	# number of variables and neighbours
	n_vars = X.shape[1]
	n_neigh = exceeds_idx.shape[1]

	# unpack interest points
	method = interest_pts[0]
	n_interest_pts = interest_pts[1]
 
	# initialize time and var vectors for neighbours
	t_neigh = np.empty([n_neigh, n_interest_pts])
	X_neigh = np.empty([n_neigh, n_vars, n_interest_pts])
	t_c = np.empty([n_interest_pts, 1])
	X_c = np.empty([n_interest_pts, n_vars])
	if method == "max":
		idx_l_ind_sorted = np.argsort(l_ind)
		idx_interest_points = idx_l_ind_sorted[-n_interest_pts:]
	elif method == "min":
		idx_l_ind_sorted = np.argsort(l_ind)
		idx_interest_points = idx_l_ind_sorted[:n_interest_pts]
	for j in range(n_interest_pts):
		idx = exceeds_idx[idx_interest_points[j],:]
		n_neigh_idx = len(idx)
		t_neigh[:n_neigh_idx,j] = t[idx]
		X_neigh[:n_neigh_idx,:,j] = X[idx,:]
		# get coordinates of interest point
		t_c[j] = t[j]
		X_c[j,:] = X[j,:]
	return t_neigh, X_neigh, t_c, X_c, n_interest_pts



def _get_title_str(local_index, tau_l):
	if local_index[:2] == "d1":
		title_str = r'$d_1$'
	if local_index[:5] == "theta":
		title_str = r'$\theta$'
	if local_index[:6] == "alphat":
		title_str = r'$\eta = $'+str(tau_l)+r' $\tau_l$'
	return title_str



def _get_neigh_title_str(local_index, tau_l, interest_pt, t_c, X_c):
	if local_index[:2] == "d1":
		title_str = r'$d_1$ for interest point '\
        	+str(interest_pt)+': t='+str(t_c)+', X='+str(X_c)
	if local_index[:5] == "theta":
		title_str = r'$\theta$ for interest point '\
        	+str(interest_pt)+': t='+str(t_c)+', X='+str(X_c)
	if local_index[:6] == "alphat":
		title_str = r'$\eta = $'+str(tau_l)+r' $\tau_l$ for interest point '\
  			+str(interest_pt)+': t='+str(t_c)+', X='+str(X_c)
	return title_str


	
def _get_label_str(local_index):
	if   local_index[:2] == "d1"    : label_str = r'$d_1$'
	elif local_index[:5] == "theta" : label_str = r'$\theta$'
	elif local_index[:6] == "alphat": label_str = r'$\alpha_\eta$'
	return label_str



def _get_slice_rows(shorter_row_var, local_index):
	cut_row = shorter_row_var.shape[0]
	if local_index[:6] == "alphat": 
		sr = slice(cut_row)
	else: 
		sr = slice(None)
	return sr



def _get_vmin_and_max(vmin, vmax, c_val):
	if vmin is None: vmin = np.min(c_val)
	if vmax is None: vmax = np.max(c_val)
	print(f'Contour colored by: {vmin=:}, {vmax=:}')
	return vmin, vmax



def _close_fig(savefig, fout, plt):
	# save figure
	if savefig=="show":
		plt.show()
		plt.close()
	elif savefig=="save_and_show":
		plt.savefig(fout, dpi=300)
		plt.show()
		plt.close()
	elif savefig=="save":
		plt.savefig(fout, dpi=300)
		plt.close()
	else:
		print(f"parameter {savefig} unknown.")
		exit(0)
