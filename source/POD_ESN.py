"""
python script for POD.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path> --modes=<modes> --hyperparam_file=<hyperparam_file> --config_number=<config_number> --reduce_domain=<reduce_domain> --w_threshold=<w_threshold> --reduce_domain2=<reduce_domain2>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --modes=<modes>                      number of modes for POD 
    --hyperparam_file=<hyperparam_file>  hyperparameters for ESN
    --config_number=<config_number>      config_number [default: 0]
    --reduce_domain=<reduce_domain>      reduce size of domain [default: False]
    --reduce_domain2=<reduce_domain2>    reduce size of domain keep time period [default: False]
    --w_threshold=<w_threshold>          w_threshold [defualt: False]
"""

import os
import sys
sys.path.append(os.getcwd())
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.stdout.reconfigure(line_buffering=True)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
from scipy.io import loadmat, savemat
from skopt.plots import plot_convergence
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from Eval_Functions import *
from Plotting_Functions import *
from POD_functions import *

import json
import time as time

from docopt import docopt
args = docopt(__doc__)

exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('run functions files')

input_path = args['--input_path']
output_path = args['--output_path']
modes = int(args['--modes'])
hyperparam_file = args['--hyperparam_file']
config_number = int(args['--config_number'])


reduce_domain = args['--reduce_domain']
if reduce_domain == 'False':
    reduce_domain = False
    print('domain not reduced', reduce_domain)
elif reduce_domain == 'True':
    reduce_domain = True
    print('domain reduced', reduce_domain)

reduce_domain2 = args['--reduce_domain2']
if reduce_domain2 == 'False':
    reduce_domain2 = False
    print('domain not reduced', reduce_domain2)
elif reduce_domain2 == 'True':
    reduce_domain2 = True
    print('domain reduced', reduce_domain2)

w_threshold = args['--w_threshold']
if w_threshold == 'False':
    w_threshold = False
    print('no threshold', w_threshold)
elif w_threshold == 'True':
    w_threshold = True
    print('threshold', w_threshold)


with open(hyperparam_file, "r") as f:
    hyperparams = json.load(f)
    Nr = hyperparams["Nr"]
    train_len = hyperparams["N_train"]
    val_len = hyperparams["N_val"]
    test_len = hyperparams["N_test"]
    washout_len = hyperparams["N_washout"]
    washout_len_val = hyperparams.get("N_washout_val", washout_len)
    t_lyap = hyperparams["t_lyap"]
    normalisation = hyperparams["normalisation"]
    ens = hyperparams["ens"]
    ensemble_test = hyperparams["ensemble_test"]
    n_tests = hyperparams["n_tests"]
    grid_x = hyperparams["grid_x"]
    grid_y = hyperparams["grid_y"]
    added_points = hyperparams["added_points"]
    val = hyperparams["val"]
    noise = hyperparams["noise"]
    alpha = hyperparams["alpha"]
    alpha0 = hyperparams["alpha0"]
    n_forward = hyperparams["n_forward"]
    threshold_ph = hyperparams.get("threshold_ph", 0.2)

def load_data_set_TD(file, names, snapshots):
    with h5py.File(file, 'r') as hf:
        print(hf.keys())
        x = hf['x'][:]  # 1D x-axis
        z = hf['z'][:]  # 1D z-axis
        time_vals = hf['time'][:]  # 1D time vector
        
        data = np.zeros((len(time_vals), len(x), len(z), len(names)))
        
        index=0
        for name in names:
            print(name)
            print(hf[name])
            Var = np.array(hf[name])
            data[:,:,:,index] = Var[:snapshots,:,:]
            index+=1

    return data, x, z, time_vals

def load_data_set_RB(file, names, snapshots):
    with h5py.File(file, 'r') as hf:
        print(hf.keys())
        time_vals = np.array(hf['total_time_all'][:snapshots])
        
        data = np.zeros((len(time_vals), len(x), len(z), len(names)))
        
        index=0
        for name in names:
            print(name)
            print(hf[name])
            Var = np.array(hf[name])
            data[:,:,:,index] = Var[:snapshots,:,0,:]
            index+=1

    return data, time_vals

def load_data_set_RB_act(file, names, snapshots):
    with h5py.File(file, 'r') as hf:
        print(hf.keys())
        time_vals = np.array(hf['total_time_all'][:snapshots])
        
        data           = np.zeros((len(time_vals), len(x), len(z), len(names)-1))
        plume_features = np.zeros((len(time_vals), 4))

        index=0
        for name in names:
            print(name)
            print(hf[name])
            Var = np.array(hf[name])
            print(np.shape(Var))
            if index == 4:
                plume_features = Var[:snapshots]
            else:
                data[:,:,:,index] = Var[:snapshots,:,:]
            index+=1

    return data, time_vals, plume_features


#### LOAD DATA AND POD ####
Data = 'RB'
if Data == 'ToyData':
    name = names = variables = ['combined']
    n_components = 3
    num_variables = 1
    snapshots = 25000
    data_set, x, z, time_vals = load_data_set_TD(input_path+'/plume_wave_dataset_smallergrid_longertime.h5', name, snapshots)
    print('shape of dataset', np.shape(data_set))
    dt = 0.05

elif Data == 'RB':
    variables = ['q_all', 'w_all', 'u_all', 'b_all']
    names = ['q', 'w', 'u', 'b']
    x = np.load(input_path+'/x.npy')
    z = np.load(input_path+'/z.npy')
    snapshots_load = 16000
    data_set, time_vals = load_data_set_RB(input_path+'/data_4var_5000_48000.h5', variables, snapshots_load)
    print('shape of dataset', np.shape(data_set))
    dt = 2

elif Data == 'RB_plume':
    variables = ['q_all', 'w_all', 'u_all', 'b_all']
    variables_plus_act = ['q_all', 'w_all', 'u_all', 'b_all', 'plume_features']
    names = ['q_all', 'w_all', 'u_all', 'b_all']
    names_plus_act = ['q', 'w', 'u', 'b', 'active']
    x = np.load(input_path+'/x.npy')
    z = np.load(input_path+'/z.npy')
    snapshots_load = 16000
    data_set, time_vals, plume_features = load_data_set_RB_act(input_path+'/data_4var_5000_48000_plumes.h5', variables_plus_act, snapshots_load)
    print('shape of dataset', np.shape(data_set))
    dt = 2
    print('shape of plume_features', np.shape(plume_features))

reduce_domain  = reduce_domain
reduce_domain2 = reduce_domain2

if reduce_domain:
    data_set = data_set[200:392,60:80,:,:] # 408 so we have 13 batches 12 for training and 1 for 'validation'
    x = x[60:80]
    time_vals = time_vals[200:392]
    print('reduced domain shape', np.shape(data_set))
    print('reduced x domain', np.shape(x))
    print('reduced x domain', len(x))
    print(x[0], x[-1])

if reduce_domain2:
    # data_set = data_set[:4650,128:160,:,:] # 10LTs washout, 200LTs train, 1000LTs test
    # x = x[128:160]
    # time_vals = time_vals[:4650]
    # print('reduced domain shape', np.shape(data_set))
    # print('reduced x domain', np.shape(x))
    # print('reduced x domain', len(x))
    # print(x[0], x[-1])

    data_set = data_set[:,128:160,:,:] # 10LTs washout, 200LTs train, 1000LTs test
    x = x[128:160]
    time_vals = time_vals[:]
    print('reduced domain shape', np.shape(data_set))
    print('reduced x domain', np.shape(x))
    print('reduced x domain', len(x))
    print(x[0], x[-1])

data_reshape = data_set.reshape(-1, data_set.shape[-1])
print('shape of data reshaped', data_reshape)

# fit the scaler
scaling = 'SS'
if scaling == 'SS':
    print('applying standard scaler')
    scaler = StandardScaler()
    scaler.fit(data_reshape)
    print('means', scaler.mean_)

    print('shape of data before scaling', np.shape(data_reshape))
    data_scaled_reshape = scaler.transform(data_reshape)
    #reshape 
    data_scaled = data_scaled_reshape.reshape(data_set.shape)
else:
    print('no scaling')
    data_scaled = data_set

if Data == 'RB':
    snapshots_POD = 11200
    data_scaled = data_scaled[:snapshots_POD]
elif Data == 'RB_plume':
    snapshots_POD = 11200
    data_scaled = data_scaled[:snapshots_POD]
    plume_features = plume_features[:snapshots_POD]
n_components = modes
data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_, cev = POD(data_scaled, n_components, x, z, names, f"modes{n_components}")
print('cumulative equiv ratio', cev)

if Data == 'RB_plume':
    U = np.concatenate([data_reduced, plume_features], axis=1)  # shape (time, 68)
else:
    U = data_reduced 

centre_of_energy = False
if centre_of_energy:

    plot_rbf = True
    def compute_center_of_energy(w_slice, x_coords=None):
        """
        Computes the center of energy in x-direction for each time step.
        
        Parameters:
        - w_slice: np.ndarray of shape (T, X), the vertical velocity at a single z level across time.
        - x_coords: Optional 1D array of x positions (length X). Defaults to 0 to X-1.
        
        Returns:
        - center: np.ndarray of shape (T,), the energy-weighted center position at each time.
        """
        T, X = w_slice.shape
        if x_coords is None:
            x_coords = np.arange(X)

        energy = np.abs(w_slice)**2  # energy at each x
        weighted_sum = (energy * x_coords[None, :]).sum(axis=1)  # sum_x (x * |w|^2)
        total_energy = energy.sum(axis=1) + 1e-8  # avoid division by zero
        center = weighted_sum / total_energy
        return center

    def rbf_encode(center_positions, num_centers=10, domain_length=256, sigma=None):
        """
        Apply RBF encoding to the center positions.
        
        Parameters:
        - center_positions: np.ndarray of shape (T,), the center positions of energy for each time step.
        - num_centers: Number of RBF centers.
        - domain_length: The length of the domain (e.g., the x-dimension size).
        - sigma: The width (spread) of the RBFs. If None, it will be set to a default value.

        Returns:
        - np.ndarray of shape (T, num_centers) containing the RBF-encoded features for each time step.
        """
        T = center_positions.shape[0]
        centers = np.linspace(0, domain_length - 1, num_centers)  # Evenly spaced centers in the domain
        if sigma is None:
            sigma = (domain_length / num_centers) / 2  # Default spread based on number of centers

        # RBF encoding: exp(-((x - c)^2) / (2 * sigma^2))
        rbf_features = np.exp(-((center_positions[:, None] - centers[None, :])**2) / (2 * sigma**2))
        return rbf_features  # shape: (T, num_centers)

    w_slice = data_scaled[:, :, 47, 1]
    center_positions = compute_center_of_energy(w_slice, x)
    
    rbf_features = rbf_encode(center_positions, num_centers=10, domain_length=w_slice.shape[1], sigma=0.5)

    U = np.hstack((data_reduced, rbf_features))

    if plot_rbf:
        time_step = 175
        rbf_at_timestep = rbf_features[time_step,:]
        # Generate x values for the plot (the RBF centers)
        centers = np.linspace(x[0], x[-1], 10)  # Assuming domain length is 256
        
        # Plot the RBF features
        fig, ax = plt.subplots(2, figsize=(8,6))
        ax[0].plot(centers, rbf_at_timestep, marker='o', linestyle='-', color='b')
        ax[0].set_xlabel('RBF Center Position (x)')
        ax[0].set_ylabel('RBF Response')

        ax[1].contourf(x, z, data_scaled[time_step, :, :, 1].T)
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('z')
        fig.suptitle(time_vals[time_step])
        fig.savefig(output_path+'/rbf_features.png')
        
if w_threshold:
    def detect_multiple_regions(w_slice, threshold=0.5, num_regions=3):
        """
        Detect multiple plumes by dividing the domain into regions and finding all regions with plumes.
        
        Parameters:
        - w_slice: np.ndarray of shape (T, X), vertical velocity across time and space.
        - threshold: Minimum height to consider a region as containing a plume.
        - num_regions: Number of regions to divide the domain into.
        
        Returns:
        - region_encoding: np.ndarray of shape (T, num_regions), encoding for the presence of plumes in each region.
        """
        T, X = w_slice.shape
        region_encoding = np.zeros((T, num_regions))
        
        # Define the boundaries of the regions
        region_width = X // num_regions
        
        for t in range(T):
            # Detect the indices where vertical velocity exceeds the threshold
            plume_region = np.where(np.abs(w_slice[t, :]) > threshold)[0]
            
            for region_idx in range(num_regions):
                # Define the start and end indices of the current region
                region_start = region_idx * region_width
                region_end = (region_idx + 1) * region_width
                
                # If any plume points are within this region, mark it as 1
                if np.any((plume_region >= region_start) & (plume_region < region_end)):
                    region_encoding[t, region_idx] = 1
        
        return region_encoding
    
    w_slice = data_scaled[:,:,47,1]
    threshold_w = np.percentile(w_slice, 99)
    region_encoding = detect_multiple_regions(w_slice, threshold=threshold_w, num_regions=8)

    fig, ax = plt.subplots(2, figsize=(8,6))
    ax[0].pcolormesh(time_vals[:750], x[::32], region_encoding[:750].T, cmap='Reds', shading='auto')
    ax[1].contourf(time_vals[:750], x, w_slice[:750].T)
    fig.savefig(output_path+'/regions.png')
    
    time_step = 175
    fig, ax =plt.subplots(2, figsize=(8,6))
    #ax[0].pcolormesh(x[::32], [0,1], region_encoding[time_step].T)
    ax[0].plot(x[::32], region_encoding[time_step])
    ax[1].plot(x, w_slice[time_step])
    fig.savefig(output_path+'/region_timestep.png')

    U = np.hstack((data_reduced, region_encoding))


print('shape of data for ESN', np.shape(U))

# number of time steps for washout, train, validation, test
t_lyap    = t_lyap
dt        = dt
N_lyap    = int(t_lyap//dt)
print('N_lyap', N_lyap)
N_washout = int(washout_len*N_lyap) #75
N_washout_val = int(washout_len_val*N_lyap)
N_train   = train_len*N_lyap #600
N_val     = val_len*N_lyap #45
N_test    = test_len*N_lyap #45
dim       = U.shape[1]

train_data = data_set[:int(N_washout+N_train)]
global_stds = [np.std(train_data[..., c]) for c in range(train_data.shape[-1])]

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)

# standardisation 
norm_std = U_data.std(axis=0)
normalisation = normalisation #on, off, standard

#standardisation_plusregions
norm_std_pr = U_data[:, :n_components].std(axis=0)
u_mean_pr   = U_data[:, :n_components].mean(axis=0)

print('norm', norm)
print('u_mean', u_mean)
print('norm_std', norm_std)

# washout
U_washout = U[:N_washout].copy()
# data to be used for training + validation
U_tv  = U[N_washout:N_washout+N_train-1].copy() #inputs
Y_tv  = U[N_washout+1:N_washout+N_train].copy() #data to match at next timestep

if Data == 'ToyData':
    indexes_to_plot = np.array([1, 2, 3, 4] ) -1
else:
    indexes_to_plot = np.array([1, 2, 10, 32, 64] ) -1
indexes_to_plot = indexes_to_plot[indexes_to_plot <= (n_components-1)]

# adding noise to training set inputs with sigma_n the noise of the data
# improves performance and regularizes the error as a function of the hyperparameters
fig,ax = plt.subplots(len(indexes_to_plot), figsize=(12,6), tight_layout=True, sharex=True)
for m in range(len(indexes_to_plot)):
    index = indexes_to_plot[m]
    ax[m].plot(U_tv[:N_val,index], c='b', label='Non-noisy')
seed = 42   #to be able to recreate the data, set also seed for initial condition u0
rnd1  = np.random.RandomState(seed)
noisy = True
if noisy:
    data_std = np.std(U,axis=0)
    sigma_n = noise #1e-3     #change this to increase/decrease noise in training inputs (up to 1e-1)
    for i in range(n_components):
        U_tv[:,i] = U_tv[:,i] \
                        + rnd1.normal(0, sigma_n*data_std[i], N_train-1)
    for m in range(len(indexes_to_plot)):
        index = indexes_to_plot[m]
        ax[m].plot(U_tv[:N_val,index], 'r--', label='Noisy')
        ax[m].grid()
        ax[m].set_title('mode %i' % (index+1))
ax[0].legend()
fig.savefig(output_path + '/noise_addition_sigman%.4f.png' % sigma_n)
plt.close()

#### ESN hyperparameters #####
if normalisation == 'on':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm))]) #input bias (average absolute value of the inputs)
elif normalisation == 'standard':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm_std))]) #input bias (average absolute value of the inputs)
elif normalisation == 'off':
    bias_in   = np.array([np.mean(np.abs(U_data))]) #input bias (average absolute value of the inputs)
elif normalisation == 'standard_plusregions':
    bias_in   = np.array([np.mean(np.abs((U_data[:,:n_components]-u_mean_pr)/norm_std_pr))]) #input bias (average absolute value of the inputs)
bias_out  = np.array([1.]) #output bias

N_units      = Nr #neurons
connectivity = 3
sparseness   = 1 - connectivity/(N_units-1)

tikh = np.array([1e-1,1e-3,1e-6,1e-9]) #np.array([1e-3,1e-6,1e-9,1e-12])  # Tikhonov factor (optimize among the values in this list)

print('tikh:', tikh)
print('N_r:', N_units, 'sparsity:', sparseness)
print('bias_in:', bias_in, 'bias_out:', bias_out)

#### Grid Search and BO #####
threshold_ph = threshold_ph
n_in  = 0           #Number of Initial random points

spec_in     = 0.1    #range for hyperparameters (spectral radius and input scaling)
spec_end    = 1.0 
in_scal_in  = np.log10(0.1)
in_scal_end = np.log10(5.0)

# In case we want to start from a grid_search, the first n_grid_x*n_grid_y points are from grid search
n_grid_x = grid_x
n_grid_y = grid_y
n_bo     = added_points  #number of points to be acquired through BO after grid search
n_tot    = n_grid_x*n_grid_y + n_bo #Total Number of Function Evaluatuions

# computing the points in the grid
if n_grid_x > 0:
    x1    = [[None] * 2 for i in range(n_grid_x*n_grid_y)]
    k     = 0
    for i in range(n_grid_x):
        for j in range(n_grid_y):
            x1[k] = [spec_in + (spec_end - spec_in)/(n_grid_x-1)*i,
                     in_scal_in + (in_scal_end - in_scal_in)/(n_grid_y-1)*j]
            k   += 1

# range for hyperparameters
search_space = [Real(spec_in, spec_end, name='spectral_radius'),
                Real(in_scal_in, in_scal_end, name='input_scaling')]

# ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0))*\
                  Matern(length_scale=[0.2,0.2], nu=2.5, length_scale_bounds=(5e-2, 1e1)) 

print('initial grid', x1)

#Hyperparameter Optimization using Grid Search plus Bayesian Optimization
def g(val):
    
    #Gaussian Process reconstruction
    b_e = GPR(kernel = kernell,
            normalize_y = True, #if true mean assumed to be equal to the average of the obj function data, otherwise =0
            n_restarts_optimizer = 3,  #number of random starts to find the gaussian process hyperparameters
            noise = 1e-10, # only for numerical stability
            random_state = 10) # seed
    
    
    #Bayesian Optimization
    res = skopt.gp_minimize(val,                         # the function to minimize
                      search_space,                      # the bounds on each dimension of x
                      base_estimator       = b_e,        # GP kernel
                      acq_func             = "gp_hedge",       # the acquisition function
                      n_calls              = n_tot,      # total number of evaluations of f
                      x0                   = x1,         # Initial grid search points to be evaluated at
                      n_random_starts      = n_in,       # the number of additional random initialization points
                      n_restarts_optimizer = 3,          # number of tries for each acquisition
                      random_state         = 10,         # seed
                           )   
    return res

print(search_space)

#Number of Networks in the ensemble
ensemble = ens

data_dir = '/Run_n_units{0:}_ensemble{1:}_normalisation{2:}_washout{3:}_config{4:}/'.format(N_units, ensemble, normalisation, washout_len, config_number)
output_path = output_path+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')
    
data_dir = '/GP{0:}_{1:}/'.format(n_grid_x*n_grid_y, n_bo)
output_path = output_path+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

# Which validation strategy (implemented in Val_Functions.ipynb)
val      = eval(val)
alpha = alpha
N_fw     = n_forward*N_lyap
#N_fo     = (N_train-N_val-N_washout_val)//N_fw + 1 
N_fo     = 50                     # number of validation intervals
N_in     = N_washout                 # timesteps before the first validation interval (can't be 0 due to implementation)
#N_fw     = (N_train-N_val-N_washout)//(N_fo-1) # how many steps forward the validation interval is shifted (in this way they are evenly spaced)
N_splits = 4                         # reduce memory requirement by increasing N_splits
print('Number of folds', N_fo)
print('how many steps forward validation interval moves', N_fw)
print('how many LTs forward validation interval moves', N_fw//N_lyap)

#Quantities to be saved
par      = np.zeros((ensemble, 4))      # GP parameters
x_iters  = np.zeros((ensemble,n_tot,2)) # coordinates in hp space where f has been evaluated
f_iters  = np.zeros((ensemble,n_tot))   # values of f at those coordinates
minimum  = np.zeros((ensemble, 4))      # minima found per each member of the ensemble

# to store optimal hyperparameters and matrices
tikh_opt = np.zeros(n_tot)
Woutt    = np.zeros(((ensemble, N_units+1,dim)))
Winn     = [] #save as list to keep single elements sparse
Ws       = []
Xa1_states =  np.zeros(((ensemble, N_train-1, N_units+1))) ####added  

# save the final gp reconstruction for each network
gps        = [None]*ensemble

# to print performance of every set of hyperparameters
print_flag = False

hyperparams_path = output_path+'/hyperparams/'
if not os.path.exists(hyperparams_path):
    os.makedirs(hyperparams_path)
    print('made directory')

# optimize ensemble networks (to account for the random initialization of the input and state matrices)
for i in range(ensemble):

    print('Realization    :',i+1)

    k   = 0

    # Win and W generation
    seed= i+1
    rnd = np.random.RandomState(seed)

    #sparse syntax for the input and state matrices
    Win  = lil_matrix((N_units,dim+1))
    for j in range(N_units):
        Win[j,rnd.randint(0, dim+1)] = rnd.uniform(-1, 1) #only one element different from zero
    Win = Win.tocsr()

    W = csr_matrix( #on average only connectivity elements different from zero
        rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1-sparseness)))

    spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
    W = (1/spectral_radius)*W #scaled to have unitary spec radius

    # Bayesian Optimization
    tt       = time.time()
    res      = g(val)
    print('Total time for the network:', time.time() - tt)


    #Saving Quantities for post_processing
    gps[i]     = res.models[-1]
    gp         = gps[i]
    x_iters[i] = np.array(res.x_iters)
    f_iters[i] = np.array(res.func_vals)
    minimum[i] = np.append(res.x,[tikh_opt[np.argmin(f_iters[i])],res.fun])
    params     = gp.kernel_.get_params()
    key        = sorted(params)
    par[i]     = np.array([params[key[2]],params[key[5]][0], params[key[5]][1], gp.noise_])

    #saving matrices
    train      = train_save_n(U_washout, U_tv, Y_tv,
                              minimum[i,2],10**minimum[i,1], minimum[i,0], minimum[i,3]) ###changed
    Woutt[i]   = train[0] ###changed
    Winn    += [Win]
    Ws      += [W]
    
    Xa1_states[i]  = train[1] ###changed


    #Plotting Optimization Convergence for each network
    print('Best Results: x', minimum[i,0], 10**minimum[i,1], minimum[i,2],
          'f', -minimum[i,-1])
    # Full path for saving the file
    hyp_file = '_ESN_hyperparams_ens%i.json' % i

    output_path_hyp = os.path.join(hyperparams_path, hyp_file)

    hyps = {
    "test": i,
    "no. modes": n_components,
    "spec rad": minimum[i,0],
    "input scaling": 10**minimum[i,1],
    "tikh": minimum[i,2],
    "min f": minimum[i,-1],
    }

    with open(output_path_hyp, "w") as file:
        json.dump(hyps, file, indent=4)
    
    fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
    plot_convergence(res)
    fig.savefig(hyperparams_path+'/convergence_realisation%i.png' % i)
    plt.close()

##### visualise grid search #####
# Plot Gaussian Process reconstruction for each network in the ensemble after n_tot evaluations
# The GP reconstruction is based on the n_tot function evaluations decided in the search

# points to evaluate the GP at
n_length    = 100
xx, yy      = np.meshgrid(np.linspace(spec_in, spec_end,n_length), np.linspace(in_scal_in, in_scal_end,n_length))
x_x         = np.column_stack((xx.flatten(),yy.flatten()))
x_gp        = res.space.transform(x_x.tolist())     ##gp prediction needs this normalized format
y_pred      = np.zeros((ensemble,n_length,n_length))


for i in range(ensemble):
    # retrieve the gp reconstruction
    gp         = gps[i]

    pred, pred_std = gp.predict(x_gp, return_std=True)

    fig, ax  = plt.subplots(1, figsize=(12,10), tight_layout=True)

    amin = np.amin([10,f_iters.max()])

    y_pred[i] = np.clip(-pred, a_min=-amin,
                        a_max=-f_iters.min()).reshape(n_length,n_length)
                        # Final GP reconstruction for each realization at the evaluation points

    ax.set_title('Mean GP of realization \#'+ str(i+1))

    #Plot GP Mean
    ax.set_xlabel('Spectral Radius')
    ax.set_ylabel('$\log_{10}$Input Scaling')
    CS      = ax.contourf(xx, yy, y_pred[i],levels=10,cmap='Blues')
    cbar = fig.colorbar(CS, ax=ax)
    cbar.set_label('-$\log_{10}$(MSE)',labelpad=15)
    CSa     = ax.contour(xx, yy, y_pred[i],levels=10,colors='black',
                          linewidths=1, linestyles='solid')

    #   Plot the n_tot search points
    ax.scatter(x_iters[i,:n_grid_x*n_grid_y,0],
                x_iters[i,:n_grid_x*n_grid_y,1], c='r', marker='^') #grid points
    ax.scatter(x_iters[i,n_grid_x*n_grid_y:,0],
                x_iters[i,n_grid_x*n_grid_y:,1], c='lime', marker='s') #bayesian opt points

    fig.savefig(output_path + '/GP_%i.png' % i)
    plt.close()
np.save(output_path + '/f_iters.npy', f_iters)

#Save the details and results of the search for post-process
opt_specs = [spec_in,spec_end,in_scal_in,in_scal_end]

fln = output_path + '/ESN_matrices' + '.mat'
with open(fln,'wb') as f:  # need 'wb' in Python3
    savemat(f, {"norm": norm})
    savemat(f, {"fix_hyp": np.array([bias_in, N_washout],dtype='float64')})
    savemat(f, {'opt_hyp': np.column_stack((minimum[:,0], 10**minimum[:,1]))})
    savemat(f, {"Win": Winn})
    savemat(f, {'W': Ws})
    savemat(f, {"Wout": Woutt})

ESN_params = 'ESN_params.json' 

output_ESN_params = os.path.join(output_path, ESN_params)
with open(output_ESN_params, "w") as f:
    json.dump(hyperparams, f, indent=4) 

validation_interval = True
test_interval       = True

if validation_interval:
    print('VALIDATION (TEST)')
    N_test   = N_fo                    #number of intervals in the test set
    if reduce_domain:
        N_tstart = N_washout
    elif reduce_domain2:
        N_tstart = N_washout
    else:
        N_tstart = N_washout                    #where the first test interval starts
    N_intt   = test_len*N_lyap            #length of each test set interval
    N_gap    = N_intt

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    ensemble_test = ens

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))
    ens_nrmse_ch    = np.zeros((ensemble_test))
    ens_nrmse_ch_pl = np.zeros((ensemble_test))
    ens_pl_acc      = np.zeros((ensemble_test))

    metrics_val_path = output_path+'/validation_metrics/'
    if not os.path.exists(metrics_val_path):
        os.makedirs(metrics_val_path)
        print('made directory')

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = minimum[j,0].copy()
        sigma_in = 10**minimum[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = 3
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print('index:', N_tstart + i*N_intt)
            print('start_time:', time_vals[N_tstart + i*N_intt])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt].copy()
            Y_t       = U[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt].copy()

            if reduce_domain:
                fig,ax = plt.subplots(1, figsize=(12,3))
                c1=ax.contourf(time_vals[N_tstart + i*N_intt : N_tstart + i*N_intt + N_intt], x, data_reconstructed[N_tstart + i*N_intt : N_tstart + i*N_intt + N_intt,:,32,0].T)
                fig.colorbar(c1, ax=ax)
                fig.savefig(output_path+'/validation_slice_w.png')

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t,xa,Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
            print(np.shape(Yh_t))
            if i == 0:
                ens_pred[:, :, j] = Yh_t
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]
            nrmse_error[i, :] = Y_err

            ##### reconstructions ####
            if w_threshold:
                _, reconstructed_truth       = inverse_POD(Y_t[:,:n_components], pca_)
                _, reconstructed_predictions = inverse_POD(Yh_t[:,:n_components], pca_)
                region_encoding_truth        = Y_t[:,n_components:]
                region_encoding_predictions  = Yh_t[:,n_components:]

            
            if Data == 'RB_plume':
                _, reconstructed_truth       = inverse_POD(Y_t[:,:n_components], pca_)
                _, reconstructed_predictions = inverse_POD(Yh_t[:,:n_components], pca_)
                plume_features_truth         = Y_t[:,n_components:]
                plume_features_predictions   = Yh_t[:,n_components:]
            else:
                _, reconstructed_truth       = inverse_POD(Y_t, pca_)
                _, reconstructed_predictions = inverse_POD(Yh_t, pca_)

            # rescale
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            # metrics
            nrmse = NRMSE(reconstructed_truth, reconstructed_predictions)
            mse   = MSE(reconstructed_truth, reconstructed_predictions)
            evr   = EVR_recon(reconstructed_truth, reconstructed_predictions)
            SSIM  = compute_ssim_for_4d(reconstructed_truth, reconstructed_predictions)
            nrmse_ch = NRMSE_per_channel(reconstructed_truth, reconstructed_predictions)

            if len(variables) == 4:
                active_array, active_array_reconstructed, mask, mask_expanded_recon = active_array_calc(reconstructed_truth, reconstructed_predictions, z)
                accuracy = np.mean(active_array == active_array_reconstructed)
                if np.any(mask):  # Check if plumes exist
                    masked_truth = reconstructed_truth[mask]
                    masked_pred = reconstructed_predictions[mask]
                    
                    print("Shape truth after mask:", masked_truth.shape)
                    print("Shape pred after mask:", masked_pred.shape)

                    # Compute NRMSE only if mask is not empty
                    nrmse_plume = NRMSE(masked_truth, masked_pred)

                    mask_original     = mask[..., 0]
                    nrmse_sep_plume   = NRMSE_per_channel_masked(reconstructed_truth, reconstructed_predictions, mask_original, global_stds) 

                else:
                    print("Mask is empty, no plumes detected.")
                    nrmse_plume = 0  # Simply add 0 to maintain shape
                    nrmse_sep_plume = 0
            else:
                nrmse_plume = np.inf
                nrmse_sep_plume = np.inf

            if Data == 'RB_plume':
                pred_counts_rounded = np.round(plume_features_predictions[:, 0]).astype(int)
                true_counts = plume_features_truth[:, 0].astype(int)

                if len(true_counts) > 0:
                    exact_match = (pred_counts_rounded == true_counts).sum()
                    plume_count_accuracy = exact_match / len(true_counts)
                else:
                    plume_count_accuracy = 0.0  # or -1 if you want to flag it
            else:
                plume_count_accuracy = 0.0

            print('NRMSE', nrmse)
            print('MSE', mse)
            print('EVR_recon', evr)
            print('SSIM', SSIM)
            print('NRMSE plume', nrmse_plume)
            print('no plume accuracy', plume_count_accuracy)

            # Full path for saving the file
            output_file = 'ESN_test_metrics_ens%i_test%i.json' % (j,i)

            output_path_met = os.path.join(metrics_val_path, output_file)

            metrics = {
            "test": i,
            "no. modes": n_components,
            "EVR": evr,
            "MSE": mse,
            "NRMSE": nrmse,
            "SSIM": SSIM,
            "NRMSE plume": nrmse_plume,
            "PH": PH[i],
            "NRMSE per channel": nrmse_ch,
            "NRMSE per channel in plume": nrmse_sep_plume,
            "plume_count_accuracy": plume_count_accuracy,
            }

            with open(output_path_met, "w") as file:
                json.dump(metrics, file, indent=4)

            ens_nrmse[j]       += nrmse
            ens_ssim[j]        += SSIM
            ens_nrmse_plume[j] += nrmse_plume
            ens_evr[j]         += evr
            ens_PH2[j]         += PH[i]
            ens_nrmse_ch[j]    += nrmse_ch
            nrmse_sep_plume     = np.nan_to_num(nrmse_sep_plume, nan=0.0)  # replace NaNs with zero
            ens_nrmse_ch_pl[j] += nrmse_sep_plume
            ens_pl_acc[j]      += plume_count_accuracy

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted

                images_val_path = output_path+'/validation_images/'
                if not os.path.exists(images_val_path):
                    os.makedirs(images_val_path)
                    print('made directory')

                if i<n_plot:
                    if j % 1 == 0:
                        
                        print('indexes_to_plot', indexes_to_plot)
                        print(np.shape(U_wash))
                        xx = np.arange(U_wash[:,0].shape[0])/N_lyap
                        plot_modes_washout(U_wash, Uh_wash, xx, i, j, indexes_to_plot, images_val_path+'/washout_validation', Modes=True)

                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        plot_modes_prediction(Y_t, Yh_t, xx, i, j, indexes_to_plot, images_val_path+'/prediction_validation', Modes=True)
                        plot_PH(Y_err, threshold_ph, xx, i, j, images_val_path+'/PH_validation')
                        
                        plot_reservoir_states_norm(Xa1, Xa2, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, N_units, images_val_path+'/resnorm_validation')
                        plot_input_states_norm(U_wash, Y_t, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, images_val_path+'/inputnorm_validation')

                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, 1*N_lyap, x, z, xx, names, images_val_path+'/ESN_validation_ens%i_test%i' %(j,i))

                        if len(variables) == 4:
                            plot_active_array(active_array, active_array_reconstructed, x, xx, i, j, variables, images_val_path+'/active_plumes_validation')

                        if Data == 'RB_plume':
                            plotting_number_of_plumes(true_counts, pred_counts_rounded, xx, i, j, images_val_path+f"/number_of_plumes")

        # accumulation for each ensemble member
        ens_nrmse[j]       = ens_nrmse[j] / N_test
        ens_nrmse_plume[j] = ens_nrmse_plume[j] / N_test
        ens_ssim[j]        = ens_ssim[j] / N_test
        ens_evr[j]         = ens_evr[j] / N_test
        ens_PH2[j]         = ens_PH2[j] / N_test  
        ens_nrmse_ch[j]    = ens_nrmse_ch[j] / N_test
        ens_nrmse_ch_pl[j] = ens_nrmse_ch_pl[j] / N_test
        ens_pl_acc[j]      = ens_pl_acc[j] / N_test
             
    # Full path for saving the file
    output_file_ALL = 'ESN_validation_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    flatten_PH = ens_PH.flatten()
    print('flat PH', flatten_PH)

    metrics_ens_ALL = {
    "mean PH": np.mean(ens_PH2),
    "lower PH": np.quantile(flatten_PH, 0.75),
    "uppper PH": np.quantile(flatten_PH, 0.25),
    "median PH": np.median(flatten_PH),
    "mean NRMSE": np.mean(ens_nrmse),
    "mean NRMSE plume": np.mean(ens_nrmse_plume),
    "mean EVR": np.mean(ens_evr),
    "mean ssim": np.mean(ens_ssim),
    "mean NRMSE per channel": np.mean(ens_nrmse_ch),
    "mean NRMSE per channel in plume": np.nanmean(ens_nrmse_ch_pl),
    "ens_pl_acc": np.mean(ens_pl_acc),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished validations')

if test_interval:
    ##### quick test #####
    print('TESTING')
    N_test   = n_tests                    #number of intervals in the test set
    if reduce_domain:
        N_tstart = N_train + N_washout_val
    elif reduce_domain2:
        N_tstart = N_train + N_washout_val
    else:
        N_tstart = N_train + N_washout_val #850    #where the first test interval starts
    N_intt   = test_len*N_lyap             #length of each test set interval
    N_gap    = N_intt
    
    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))

    ensemble_test = ensemble_test

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))

    metrics_test_path = output_path+'/test_metrics/'
    if not os.path.exists(metrics_test_path):
        os.makedirs(metrics_test_path)
        print('made directory')

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = minimum[j,0].copy()
        sigma_in = 10**minimum[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = 3
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print(N_tstart + i*N_intt)
            print('start_time:', time_vals[N_tstart + i*N_intt])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt].copy()
            Y_t       = U[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt].copy()

            if reduce_domain:
                fig,ax = plt.subplots(1, figsize=(12,3))
                c1=ax.contourf(time_vals[N_tstart + i*N_intt : N_tstart + i*N_intt + N_intt], x, data_reconstructed[N_tstart + i*N_intt : N_tstart + i*N_intt + N_intt,:,32,0].T)
                fig.colorbar(c1, ax=ax)
                fig.savefig(output_path+'/test_slice_w.png')

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t, xa, Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
            print(np.shape(Yh_t))
            if i == 0:
                ens_pred[:, :, j] = Yh_t
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]
            nrmse_error[i, :] = Y_err

            ##### reconstructions ####
            if w_threshold:
                _, reconstructed_truth       = inverse_POD(Y_t[:,:n_components], pca_)
                _, reconstructed_predictions = inverse_POD(Yh_t[:,:n_components], pca_)
                region_encoding_truth        = Y_t[:,n_components:]
                region_encoding_predictions  = Yh_t[:,n_components:]
            else:
                _, reconstructed_truth       = inverse_POD(Y_t, pca_)
                _, reconstructed_predictions = inverse_POD(Yh_t, pca_)


            # rescale
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            # metrics
            nrmse = NRMSE(reconstructed_truth, reconstructed_predictions)
            mse   = MSE(reconstructed_truth, reconstructed_predictions)
            evr   = EVR_recon(reconstructed_truth, reconstructed_predictions)
            SSIM  = compute_ssim_for_4d(reconstructed_truth, reconstructed_predictions)

            if len(variables) == 4:
                active_array, active_array_reconstructed, mask, mask_expanded_recon = active_array_calc(reconstructed_truth, reconstructed_predictions, z)
                accuracy = np.mean(active_array == active_array_reconstructed)
                if np.any(mask):  # Check if plumes exist
                    masked_truth = reconstructed_truth[mask]
                    masked_pred = reconstructed_predictions[mask]
                    
                    print("Shape truth after mask:", masked_truth.shape)
                    print("Shape pred after mask:", masked_pred.shape)

                    # Compute NRMSE only if mask is not empty
                    nrmse_plume = NRMSE(masked_truth, masked_pred)
                else:
                    print("Mask is empty, no plumes detected.")
                    nrmse_plume = 0  # Simply add 0 to maintain shape
            else:
                nrmse_plume = np.inf

            print('NRMSE', nrmse)
            print('MSE', mse)
            print('EVR_recon', evr)
            print('SSIM', SSIM)
            print('NRMSE plume', nrmse_plume)

            # Full path for saving the file
            output_file = 'ESN_test_metrics_ens%i_test%i.json' % (j,i)

            output_path_met = os.path.join(metrics_test_path, output_file)

            metrics = {
            "test": i,
            "no. modes": n_components,
            "EVR": evr,
            "MSE": mse,
            "NRMSE": nrmse,
            "SSIM": SSIM,
            "NRMSE plume": nrmse_plume,
            "PH": PH[i],
            }

            with open(output_path_met, "w") as file:
                json.dump(metrics, file, indent=4)

            ens_nrmse[j]       += nrmse
            ens_ssim[j]        += SSIM
            ens_nrmse_plume[j] += nrmse_plume
            ens_evr[j]         += evr
            ens_PH2[j]         += PH[i]

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted

                images_test_path = output_path+'/test_images/'
                if not os.path.exists(images_test_path):
                    os.makedirs(images_test_path)
                    print('made directory')
                
                if i<n_plot:
                    if j % 1 == 0:
                        
                        print('indexes_to_plot', indexes_to_plot)
                        print(np.shape(U_wash))
                        xx = np.arange(U_wash[:,0].shape[0])/N_lyap
                        plot_modes_washout(U_wash, Uh_wash, xx, i, j, indexes_to_plot, images_test_path+'/washout_test', Modes=False)

                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        plot_modes_prediction(Y_t, Yh_t, xx, i, j, indexes_to_plot, images_test_path+'/prediction_test', Modes=False)
                        plot_PH(Y_err, threshold_ph, xx, i, j, images_val_path+'/PH_validation')
                        
                        plot_reservoir_states_norm(Xa1, Xa2, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, N_units, images_test_path+'/resnorm_test')
                        plot_input_states_norm(U_wash, Y_t, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, images_test_path+'/inputnorm_test')

                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, 1*N_lyap, x, z, xx, names, images_test_path+'/ESN_validation_ens%i_test%i' %(j,i))

                        if len(variables) == 4:
                            plot_active_array(active_array, active_array_reconstructed, x, xx, i, j, variables, images_test_path+'/active_plumes_test')


        # accumulation for each ensemble member
        ens_nrmse[j]       = ens_nrmse[j] / N_test
        ens_nrmse_plume[j] = ens_nrmse_plume[j] / N_test
        ens_ssim[j]        = ens_ssim[j] / N_test
        ens_evr[j]         = ens_evr[j] / N_test
        ens_PH2[j]         = ens_PH2[j] / N_test  
             
    # Full path for saving the file
    output_file_ALL = 'ESN_test_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    flatten_PH = ens_PH.flatten()
    print('flat PH', flatten_PH)

    metrics_ens_ALL = {
    "mean PH": np.mean(ens_PH2),
    "lower PH": np.quantile(flatten_PH, 0.75),
    "uppper PH": np.quantile(flatten_PH, 0.25),
    "median PH": np.median(flatten_PH),
    "mean NRMSE": np.mean(ens_nrmse),
    "mean NRMSE plume": np.mean(ens_nrmse_plume),
    "mean EVR": np.mean(ens_evr),
    "mean ssim": np.mean(ens_ssim),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished testing')

