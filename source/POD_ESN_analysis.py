"""
python script for POD.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path> --modes=<modes> --hyperparam_file=<hyperparam_file> --config_number=<config_number> --number_of_tests=<number_of_tests> --reduce_domain2=<reduce_domain2> --Data=<Data> --plumetype=<plumetype>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --modes=<modes>                      number of modes for POD 
    --hyperparam_file=<hyperparam_file>  hyperparameters for ESN
    --config_number=<config_number>      config_number [default: 0]
    --reduce_domain2=<reduce_domain2>    reduce size of domain keep time period [default: False]
    --number_of_tests=<number_of_tests>  number of tests [default: 5]
    --Data=<Data>                        data type [default: RB]
    --plumetype=<plumetype>              plumetype [default: features]
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
from scipy.stats import wasserstein_distance
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import uniform_filter
from collections import Counter
import pickle
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
number_of_tests = int(args['--number_of_tests'])
data_type = args['--Data']
plumetype = args['--plumetype']

reduce_domain2 = args['--reduce_domain2']
if reduce_domain2 == 'False':
    reduce_domain2 = False
    print('domain not reduced', reduce_domain2)
elif reduce_domain2 == 'True':
    reduce_domain2 = True
    print('domain reduced', reduce_domain2)

model_path = output_path

output_path = output_path + '/further_analysis/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

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
    Win_method = hyperparams.get("Win_method", "LM")
    sigma_in_feats = hyperparams.get("sigma_in_feats", 0.1)
    sigma_in_mult = hyperparams.get("sigma_in_mult", 1)

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
Data = data_type
plumetype = plumetype
print('datatype:', Data)
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
    x = np.load(input_path+'/x.npy')
    z = np.load(input_path+'/z.npy')
    snapshots_load = 16000
    if plumetype == 'features':
        variables = ['q_all', 'w_all', 'u_all', 'b_all']
        variables_plus_act = ['q_all', 'w_all', 'u_all', 'b_all', 'plume_features']
        names = ['q_all', 'w_all', 'u_all', 'b_all']
        names_plus_act = ['q', 'w', 'u', 'b', 'active']
        data_set, time_vals, plume_features = load_data_set_RB_act(input_path+'/data_4var_5000_48000_plumes.h5', variables_plus_act, snapshots_load)
    elif plumetype == 'positions':
        variables = ['q_all', 'w_all', 'u_all', 'b_all']
        variables_plus_act = ['q_all', 'w_all', 'u_all', 'b_all', 'plume_positions']
        names = ['q_all', 'w_all', 'u_all', 'b_all']
        names_plus_act = ['q', 'w', 'u', 'b', 'active']
        data_set, time_vals, plume_features = load_data_set_RB_act(input_path+'/data_4var_5000_48000_positions.h5', variables_plus_act, snapshots_load)
    elif plumetype == 'sincospositions':
        variables = ['q_all', 'w_all', 'u_all', 'b_all']
        variables_plus_act = ['q_all', 'w_all', 'u_all', 'b_all', 'plume_positions']
        names = ['q_all', 'w_all', 'u_all', 'b_all']
        names_plus_act = ['q', 'w', 'u', 'b', 'active']
        data_set, time_vals, plume_features = load_data_set_RB_act(input_path+'/data_4var_5000_48000_cossinpositions.h5', variables_plus_act, snapshots_load)
    print('shape of dataset', np.shape(data_set))
    dt = 2
    print('shape of plume_features', np.shape(plume_features))


reduce_domain2 = reduce_domain2

def global_parameters(data):
    if data.ndim == 4:
        print("data is 4D.")
    else:
        print("wrong format needs to be 4D.")

    avg_q = np.mean(data[:,:,:,0], axis=(1,2))
    ke = 0.5*data[:,:,:,1]*data[:,:,:,1]
    avg_ke = np.mean(ke, axis=(1,2))
    global_params = np.zeros((data.shape[0],2))
    global_params[:,0] = avg_ke
    global_params[:,1] = avg_q

    return global_params

reduce_data_set = False
if reduce_data_set:
    data_set = data_set[:, 147:211, :, :]
    x = x[147:211]
    print('reduced domain shape', np.shape(data_set))
    print('reduced x domain', np.shape(x))
    print('reduced x domain', len(x))
    print(x[0], x[-1])

if reduce_domain2:
    data_set = data_set[:,128:160,:,:] # 10LTs washout, 200LTs train, 1000LTs test
    x = x[128:160]
    time_vals = time_vals[:]
    print('reduced domain shape', np.shape(data_set))
    print('reduced x domain', np.shape(x))
    print('reduced x domain', len(x))
    print(x[0], x[-1])

### global ###
if Data == 'ToyData':
    print('no global')
    truth_global = 0
else:
    truth_global = global_parameters(data_set)
    global_labels=['KE', 'q']

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
print('dimension for ESN', dim)

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

n_feat = U_data.shape[1] - n_components
if Data == 'RB_plume':
    u_mean_modes_only     = U_data[:,:n_components].mean(axis=0)
    m_modes_only          = U_data[:,:n_components].min(axis=0)
    M_modes_only          = U_data[:,:n_components].max(axis=0)
    norm_modes_only       = M_modes_only - m_modes_only
    mean_feats            = U_data[:, n_components:].mean(axis=0)
    std_feats             = U_data[:, n_components:].std(axis=0)
    std_feats[std_feats == 0] = 1.0
    m_feats               = U_data[:,n_components:].min(axis=0)
    M_feats               = U_data[:,n_components:].max(axis=0)
    norm_feats            = M_feats - m_feats
else:
    mean_feats = np.zeros(n_feat)
    std_feats  = np.ones(n_feat)
    norm_feats = np.ones(n_feat)

#normalisation across all data
u_min_all  = U_data.min()
u_max_all  = U_data.max()
u_norm_all = u_max_all-u_min_all

print('norm', norm)
print('u_mean', u_mean)
print('norm_std', norm_std)

# find max mins for active array calcs
_, _, RH, w, b_anom = active_array_truth(data_set[:], z)
RH_min     = RH.min()
RH_max     = RH.max()
w_max      = w.max()
w_min      = w.min()
b_anom_min = b_anom.min()
b_anom_max = b_anom.max()

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
elif normalisation == 'off_plusfeatures':
    u_pods = U_data[:, :n_components]
    u_feats = (U_data[:, n_components:] - mean_feats)/ std_feats 
    u_combined = np.hstack((u_pods, u_feats))
    bias_in = np.array([np.mean(np.abs(u_combined))])            
    sigma_in_feats = 0.01        
elif normalisation == 'range_plusfeatures':
    u_pods = (U_data[:, :n_components]-u_mean_modes_only)/norm_modes_only
    u_feats = (U_data[:, n_components:] - mean_feats)/ norm_feats 
    print('shape of u_pods', np.shape(u_pods))
    print('shape of u_feats', np.shape(u_feats))
    u_combined = np.hstack((u_pods, u_feats))
    bias_in = np.array([np.mean(np.abs(u_combined))])            
    sigma_in_feats = 0.1    

elif normalisation == 'range_plusfeatures_IS_same':
    u_pods = (U_data[:, :n_components]-u_mean_modes_only)/norm_modes_only
    u_feats = (U_data[:, n_components:] - mean_feats)/ std_feats 
    print('shape of u_pods', np.shape(u_pods))
    print('shape of u_feats', np.shape(u_feats))
    u_combined = np.hstack((u_pods, u_feats))
    bias_in = np.array([np.mean(np.abs(u_combined))])  

elif normalisation == 'range_plusfeatures_doubled':
    u_pods = (U_data[:, :n_components]-u_mean_modes_only)/norm_modes_only
    u_feats = (U_data[:, n_components:] - mean_feats)/ norm_feats #std_feats 
    print('shape of u_pods', np.shape(u_pods))
    print('shape of u_feats', np.shape(u_feats))
    u_combined = np.hstack((u_pods, u_feats))
    bias_in = np.array([np.mean(np.abs(u_combined))])  

elif normalisation == 'modeweight':
    bias_in   = np.array([np.mean(np.abs(U_data))])

    explained_variance_ratio = pca_.explained_variance_ratio_

    power = 0.9
    weights_raw = explained_variance_ratio ** power
    weights = weights_raw / np.max(weights_raw)

    min_weight = 0.1
    weights = np.maximum(weights, min_weight)
elif normalisation == 'modeweight_pluson':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm))])

    explained_variance_ratio = pca_.explained_variance_ratio_

    power = 0.9
    weights_raw = explained_variance_ratio ** power
    weights = weights_raw / np.max(weights_raw)

    min_weight = 0.1
    weights = np.maximum(weights, min_weight)
elif normalisation == 'across_range':
    bias_in   = np.array([np.mean(np.abs((U_data-u_min_all)/u_norm_all))]) #input bias (average absolute value of the inputs)

bias_out  = np.array([1.]) #output bias

N_units      = Nr #neurons
connectivity = 3
sparseness   = 1 - connectivity/(N_units-1)

# load in file
# parameters from matrix 
N_units  = Nr

fln = model_path+'/ESN_matrices.mat'
data = loadmat(fln)
print(data.keys())

Winn             = data['Win'][0] #gives Winn
fix_hyp          = data['fix_hyp']
bias_in_value    = data['fix_hyp'][:,0]
N_washout        = data['fix_hyp'][:,1][0]
opt_hyp          = data['opt_hyp']
Ws               = data['W'][0]
Woutt            = data['Wout']
norm             = data['norm'][0]
bias_in          = np.ones((1))*bias_in_value

print('fix_hyp shape:', np.shape(fix_hyp))
print('bias_in_value:', bias_in_value, 'N_washout:', N_washout)
print('shape of bias_in:', np.shape(bias_in))
print('opt_hyp shape:', np.shape(opt_hyp))
print('W shape:', np.shape(Ws))
print('Win shape:', np.shape(Winn))
print('Wout shape:', np.shape(Woutt))
print('norm:', norm)
print('u_mean:', u_mean)
print('shape of norm:', np.shape(norm))

test_interval = False
validation_interval = False
statistics_interval = False
initiation_interval = False
initiation_interval2 = False
initiation_score_interval = False
active_thresholds = True

train_data = data_set[:int(N_washout+N_train)]
global_stds = [np.std(train_data[..., c]) for c in range(train_data.shape[-1])]

if Data == 'RB_plume':
    U_tv_position = U_tv[:,n_components:]
    x_positions_training = extract_plume_positions(U_tv_position, x_domain=(0,20), max_plumes=3, threshold_predictions=False, KE_threshold=0.00005)
    np.save(output_path+'/x_positions_training.npy', x_positions_training)

# fig, ax = plt.subplots(1, figsize=(12,3))
# N_washput = int(N_washout)
# N_train   = int(N_train)
# time_vals_tv = time_vals[N_washout:N_washout+N_train-1] #inputs
# for i in range(len(indexes_to_plot)):
#     index_i = indexes_to_plot[i]
#     ax.plot(time_vals_tv, U_tv[:, index_i], label=f"Mode={index_i}")
#     fig.savefig(output_path+f"/training_data_modes.png")

if validation_interval:
    print('VALIDATION (TEST)')
    N_test   = 50                    #number of intervals in the test set
    if reduce_domain2:
        N_tstart = int(N_washout)
    else:
        N_tstart = int(N_washout)                 #where the first test interval starts
    N_intt   = test_len*N_lyap            #length of each test set interval
    N_gap    = int(n_forward*N_lyap)
    val_indexes = []

    # #prediction horizon normalization factor and threshold
    sigma_ph       = np.sqrt(np.mean(np.var(U,axis=1)))
    sigma_ph_modes = np.sqrt(np.mean(np.var(U[:,:n_components],axis=1)))
    threshold_ph = 0.2

    ensemble_test = ens

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH_modes    = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))
    ens_nrmse_ch    = np.zeros((ensemble_test))
    ens_nrmse_ch_pl = np.zeros((ensemble_test))
    ens_pl_acc      = np.zeros((ensemble_test))

    images_val_path = output_path+'/validation_images2/'
    if not os.path.exists(images_val_path):
        os.makedirs(images_val_path)
        print('made directory')
    metrics_val_path = output_path+'/validation_metrics2/'
    if not os.path.exists(metrics_val_path):
        os.makedirs(metrics_val_path)
        print('made directory')

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = 20
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print(N_tstart + i*N_gap)
            #if j == 0:
                #val_indexes.append(N_tstart + i*N_gap)
            print('start time of test', time_vals[N_tstart + i*N_gap])
            # data for washout and target in each interval
            p = N_tstart + i*N_gap
            U_wash    = U[           p : N_tstart + p].copy()
            Y_t       = U[N_tstart + p : N_tstart + p + N_intt].copy()

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t, _, Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
            print(np.shape(Yh_t))
            if i == 0:
                ens_pred[:, :, j] = Yh_t
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]
            nrmse_error[i, :] = Y_err

            if Data == 'RB_plume':
                Y_err_modes      = np.sqrt(np.mean((Y_t[:,:n_components]-Yh_t[:,:n_components])**2,axis=1))/sigma_ph_modes
                PH_modes_val = np.argmax(Y_err_modes>threshold_ph)/N_lyap
                if PH_modes_val == 0 and Y_err_modes[0]<threshold_ph: PH_modes_val = N_intt/N_lyap #(in case PH is larger than interval)
                ens_PH_modes[i,j] = PH_modes_val
            else:
                ens_PH_modes[i,j] = ens_PH[i,j]


            ##### reconstructions ####
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
            nrmse    = NRMSE(reconstructed_truth, reconstructed_predictions)
            mse      = MSE(reconstructed_truth, reconstructed_predictions)
            evr      = EVR_recon(reconstructed_truth, reconstructed_predictions)
            SSIM     = compute_ssim_for_4d(reconstructed_truth, reconstructed_predictions)
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
                if plumetype == 'features':
                    pred_counts_rounded = np.round(plume_features_predictions[:, 0]).astype(int)
                    true_counts = plume_features_truth[:, 0].astype(int)

                    if len(true_counts) > 0:
                        exact_match = (pred_counts_rounded == true_counts).sum()
                        plume_count_accuracy = exact_match / len(true_counts)
                    else:
                        plume_count_accuracy = 0.0  # or -1 if you want to flag it
                else:
                    plume_count_accuracy = 0.0
            else:
                plume_count_accuracy = 0.0

            print('NRMSE', nrmse)
            print('MSE', mse)
            print('EVR_recon', evr)
            print('SSIM', SSIM)
            print('NRMSE plume', nrmse_plume)
            print('NRMSE per channel', nrmse_ch)
            print('NRMSE per channel in plume', nrmse_sep_plume)

            # Full path for saving the file
            output_file = 'ESN_validation_metrics_ens%i_test%i.json' % (j,i)

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
                if i<n_plot:
                    if j % 5 == 0:
                        
                        print('indexes_to_plot', indexes_to_plot)
                        print(np.shape(U_wash))
                        xx = np.arange(U_wash[:,0].shape[0])/N_lyap
                        plot_modes_washout(U_wash, Uh_wash, xx, i, j, indexes_to_plot, images_val_path+'/washout_validation', Modes=False)

                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        plot_modes_prediction(Y_t, Yh_t, xx, i, j, indexes_to_plot, images_val_path+'/prediction_validation', Modes=False)
                        plot_PH(Y_err, threshold_ph, xx, i, j, images_val_path+'/PH_validation')
                        
                        plot_reservoir_states_norm(Xa1, Xa2, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, N_units, images_val_path+'/resnorm_validation')
                        plot_input_states_norm(U_wash, Y_t, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, images_val_path+'/inputnorm_validation')

                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, int(0.5*N_lyap), x, z, xx, names, images_val_path+'/ESN_validation_ens%i_test%i' %(j,i))

                        plot_active_array(active_array, active_array_reconstructed, x, xx, i, j, variables, images_val_path+'/active_plumes_validation')

                        if Data == 'RB_plume':
                            if plumetype == 'features':
                                plotting_number_of_plumes(true_counts, pred_counts_rounded, xx, i, j, images_val_path+f"/number_of_plumes")
                                hovmoller_plus_plumes(reconstructed_truth, reconstructed_predictions, plume_features_truth, plume_features_predictions, xx, x, 1, i, j, images_val_path+f"/hovmol_plumes")

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

    flatten_PH       = ens_PH.flatten()
    flatten_PH_modes = ens_PH_modes.flatten()
    print('flat PH', flatten_PH)

    metrics_ens_ALL = {
    "mean PH": np.mean(ens_PH2),
    "mean PH2": np.mean(flatten_PH),
    "lower PH": np.quantile(flatten_PH, 0.75),
    "upper PH": np.quantile(flatten_PH, 0.25),
    "median PH": np.median(flatten_PH),
    "mean NRMSE": np.mean(ens_nrmse),
    "mean NRMSE plume": np.mean(ens_nrmse_plume),
    "mean EVR": np.mean(ens_evr),
    "mean ssim": np.mean(ens_ssim),
    "mean NRMSE per channel": np.mean(ens_nrmse_ch),
    "mean NRMSE per channel in plume": np.nanmean(ens_nrmse_ch_pl),
    "ens_pl_acc": np.mean(ens_pl_acc),
    "mean PH2 (modes)":  np.mean(flatten_PH_modes),
    "lower PH (modes)": np.quantile(flatten_PH_modes, 0.75),
    "upper PH (modes)": np.quantile(flatten_PH_modes, 0.25),
    "median PH (modes)": np.median(flatten_PH_modes),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)

    np.save('Ra2e8/POD_ESN/val_indexes.npy', val_indexes)
    print('finished validations')

if test_interval:
    print('TESTING')
    N_washout = int(N_washout)
    N_test   = 40                  #number of intervals in the test set
    if reduce_domain2:
        N_tstart = int(N_washout + N_train)
    else:
        N_tstart = int(N_washout + N_train)  #where the first test interval starts
    N_intt   = 3*N_lyap             #length of each test set interval
    N_gap    = int(1*N_lyap)
    #N_washout_val = 4*N_lyap
    test_indexes = []

    # #prediction horizon normalization factor and threshold
    sigma_ph       = np.sqrt(np.mean(np.var(U,axis=1)))
    sigma_ph_modes = np.sqrt(np.mean(np.var(U[:,:n_components],axis=1)))
    threshold_ph = 0.2

    ensemble_test = ens

    plume_at_t0 = [6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39]
    no_plumes   = [0, 1, 2, 21, 22, 23]
    plume_init  = [3, 4, 5, 13, 14, 15, 24, 25, 26, 35]

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH_modes    = np.zeros((N_test, ensemble_test))
    ens_NRMSE       = np.zeros((N_test, ensemble_test))
    ens_pNRMSE      = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))
    ens_nrmse_ch    = np.zeros((ensemble_test))
    ens_nrmse_ch_pl = np.zeros((ensemble_test))
    ens_pl_acc      = np.zeros((ensemble_test))

    ens_nrmse_inPHregion = np.zeros((ensemble_test))

    images_test_path = output_path+'/test_images/'
    if not os.path.exists(images_test_path):
        os.makedirs(images_test_path)
        print('made directory')
    metrics_test_path = output_path+'/test_metrics/'
    if not os.path.exists(metrics_test_path):
        os.makedirs(metrics_test_path)
        print('made directory')

    ens_pred_global = np.zeros((N_intt, 2, N_test, ensemble_test))
    true_POD_global = np.zeros((N_intt, 2, N_test))
    true_global     = np.zeros((N_intt, 2, N_test))
    ens_nrmse_global= np.zeros((ensemble_test))
    ens_mse_global  = np.zeros((ensemble_test))

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = 18
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print(N_tstart + i*N_gap)
            #if j == 0:
                #test_indexes.append(N_tstart + i*N_gap)
            print('start time of test', time_vals[N_tstart + i*N_gap])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap].copy()
            Y_t       = U[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt].copy()
            data_set_Y_t =  data_set[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt]

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t,_,Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
            print(np.shape(Yh_t))
            if i == 0:
                ens_pred[:, :, j] = Yh_t
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]
            nrmse_error[i, :] = Y_err

            if Data == 'RB_plume':
                Y_err_modes      = np.sqrt(np.mean((Y_t[:,:n_components]-Yh_t[:,:n_components])**2,axis=1))/sigma_ph_modes
                PH_modes_val = np.argmax(Y_err_modes>threshold_ph)/N_lyap
                if PH_modes_val == 0 and Y_err_modes[0]<threshold_ph: PH_modes_val = N_intt/N_lyap #(in case PH is larger than interval)
                ens_PH_modes[i,j] = PH_modes_val
            else:
                ens_PH_modes[i,j] = ens_PH[i,j]

            ##### reconstructions ####
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
            PH_index = int(PH[i]*N_lyap)
            print('PH_index', PH_index)
            if PH_index == 0:
                nrmse_inPHreg = 0
            else:
                nrmse_inPHreg = NRMSE_per_channel(reconstructed_truth[:PH_index], reconstructed_predictions[:PH_index])

            if len(variables) == 4:
                active_array_true, mask_expanded_true, _,_,_ = active_array_truth(data_set_Y_t, z)
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
                if plumetype == 'features':
                    pred_counts_rounded = np.round(plume_features_predictions[:, 0]).astype(int)
                    true_counts = plume_features_truth[:, 0].astype(int)

                    if len(true_counts) > 0:
                        exact_match = (pred_counts_rounded == true_counts).sum()
                        plume_count_accuracy = exact_match / len(true_counts)
                    else:
                        plume_count_accuracy = 0.0  # or -1 if you want to flag it
                else:
                    plume_count_accuracy = 0.0
            else:
                plume_count_accuracy = 0.0

            # def check_arr(arr):
            #     v1, v2, v3 = arr[:,0], arr[:,1], arr[:,2]
            #     condition = (v1 != 0) & (v2 != 0) & (v3 <= 0)
            #     return not np.any(condition) 
            
            # is_true = check_arr(plume_features_predictions[:,0:3])

            print('NRMSE', nrmse)
            print('MSE', mse)
            print('EVR_recon', evr)
            print('SSIM', SSIM)
            print('NRMSE plume', nrmse_plume)
            print('NRMSE per channel', nrmse_ch)
            print('NRMSE per channel in plume', nrmse_sep_plume)
            print('no plume accuracy', plume_count_accuracy)
            # print('minm of features', np.min(plume_features_predictions))
            # print('maxm of features', np.max(plume_features_predictions))
            # print(' plumes have no location when KE <0', is_true)
            

            ## global parameters ##
            if Data == 'ToyData':
                print('no globals')
                nrmse_global = 0
                mse_global = 0
            else:
                PODtruth_global    = global_parameters(reconstructed_truth)
                predictions_global = global_parameters(reconstructed_predictions)
                ens_pred_global[:,:,i,j] = predictions_global
                true_POD_global[:,:,i] = PODtruth_global
                true_global[:,:,i] = truth_global[N_tstart + i*N_gap: N_tstart + i*N_gap + N_intt]
                # metrics
                nrmse_global = NRMSE(PODtruth_global, predictions_global)
                mse_global   = MSE(PODtruth_global, predictions_global)

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
            "NRMSE global": nrmse_global,
            "MSE global": mse_global,
            "NRMSE per channel": nrmse_ch,
            "NRMSE per channel in plume": nrmse_sep_plume,
            "plume_count_accuracy": plume_count_accuracy,
            "nrmse_inPHreg": float(nrmse_inPHreg),
            }

            with open(output_path_met, "w") as file:
                json.dump(metrics, file, indent=4)

            ens_nrmse[j]       += nrmse
            ens_ssim[j]        += SSIM
            ens_nrmse_plume[j] += nrmse_plume
            ens_evr[j]         += evr
            ens_PH2[j]         += PH[i]
            ens_nrmse_ch[j]    += nrmse_ch
            nrmse_sep_plume = np.nan_to_num(nrmse_sep_plume, nan=0.0)  # replace NaNs with zero
            ens_nrmse_ch_pl[j] += nrmse_sep_plume
            ens_pl_acc[j]      += plume_count_accuracy
            ens_nrmse_inPHregion[j] += nrmse_inPHreg

            ens_nrmse_global[j]+= nrmse_global
            ens_mse_global[j]  += mse_global

            ens_NRMSE[i, j] = nrmse_ch
            ens_pNRMSE[i, j] = nrmse_sep_plume

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                 if i<n_plot:
                    if j % 5 == 0:
                        
                        print('indexes_to_plot', indexes_to_plot)
                        print(np.shape(U_wash))
                        xx = np.arange(U_wash[:,0].shape[0])/N_lyap
                        plot_modes_washout(U_wash, Uh_wash, xx, i, j, indexes_to_plot, images_test_path+'/washout_test', Modes=True)

                        xx_washout = np.arange(U_wash[:,0].shape[0]) / N_lyap - (U_wash[:,0].shape[0]) / N_lyap
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        plot_modes_prediction(Y_t, Yh_t, xx, i, j, indexes_to_plot, images_test_path+'/prediction_test', Modes=True)
                        plot_PH(Y_err, threshold_ph, xx, i, j, images_test_path+'/PH_test')
                        plot_modes_washoutprediction(U_wash, Uh_wash, Y_t, Yh_t, xx_washout, xx, i, j, indexes_to_plot, images_test_path+'/washout_prediction_test', Modes=True)
                        plot_modes_prediction_FFT(Y_t, Yh_t, xx, i, j, indexes_to_plot, images_test_path+'/prediction_FFT_test', Modes=True)

                        plot_reservoir_states_norm(Xa1, Xa2, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, N_units, images_test_path+'/resnorm_test')
                        plot_input_states_norm(U_wash, Y_t, time_vals, N_tstart, N_washout_val, i, j, N_gap,  N_intt, images_test_path+'/inputnorm_test')

                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, int(1*N_lyap), x, z, xx, names, images_test_path+'/ESN_validation_ens%i_test%i' %(j,i), type='POD')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 48, int(2*N_lyap), x, z, xx, names, images_test_path+'/ESN_validation_alt_ens%i_test%i' %(j,i), type='POD')


                        if len(variables)==4:
                            plot_active_array(active_array, active_array_reconstructed, x, xx, i, j, variables, images_test_path+'/active_plumes_test')
                            plot_global_prediction_ts(PODtruth_global, predictions_global, xx, i, j, images_test_path+'/global_prediciton')
                            plot_active_array_with_true(active_array_true, active_array, active_array_reconstructed, x, xx, i, j, variables, images_test_path+'/active_plumes_wtrue_test')
                            #plot_global_prediction_ps(PODtruth_global, predictions_global, i, j, stats_path+'/global_prediciton')

                        if Data == 'RB_plume':
                            if plumetype == 'features':
                                plotting_number_of_plumes(true_counts, pred_counts_rounded, xx, i, j, images_test_path+f"/number_of_plumes")
                                hovmoller_plus_plumes(reconstructed_truth, reconstructed_predictions, plume_features_truth, plume_features_predictions, xx, x, 1, i, j, images_test_path+f"/hovmol_plumes")
                            elif plumetype == 'positions':
                                hovmoller_plus_plume_pos(reconstructed_truth, reconstructed_predictions, plume_features_truth, plume_features_predictions, xx, x, 1, i, j, images_test_path+f"/hovmol_plume_positions")
                            elif plumetype == 'sincospositions':
                                hovmoller_plus_plume_sincospos(reconstructed_truth, reconstructed_predictions, plume_features_truth, plume_features_predictions, xx, x, 1, i, j,  images_test_path+f"/hovmol_plume_sincospositions", x_domain=(0,20))
                                hovmoller_plus_plume_sincospos(reconstructed_truth, reconstructed_predictions, plume_features_truth, plume_features_predictions, xx, x, 1, i, j,  images_test_path+f"/hovmol_plume_sincospositions", x_domain=(0,20), threshold_predictions=True)

        # accumulation for each ensemble member
        ens_nrmse[j]       = ens_nrmse[j] / N_test
        ens_nrmse_plume[j] = ens_nrmse_plume[j] / N_test
        ens_ssim[j]        = ens_ssim[j] / N_test
        ens_evr[j]         = ens_evr[j] / N_test
        ens_PH2[j]         = ens_PH2[j] / N_test 
        ens_nrmse_ch[j]    = ens_nrmse_ch[j] / N_test
        ens_nrmse_ch_pl[j] = ens_nrmse_ch_pl[j] / N_test 
        ens_pl_acc[j]      = ens_pl_acc[j] / N_test
        ens_nrmse_inPHregion[j] = ens_nrmse_inPHregion[j] / N_test

                
    # Full path for saving the file
    output_file_ALL = 'ESN_test_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    np.save(output_path+'/ens_PH_all.npy', ens_PH)

    # Create separate arrays for each category
    ens_PH_plume_at_t0 = ens_PH[plume_at_t0, :]      # shape = (len(plume_at_t0), ensemble_test)
    ens_PH_no_plumes   = ens_PH[no_plumes, :]        # shape = (len(no_plumes), ensemble_test)
    ens_PH_plume_init  = ens_PH[plume_init, :]       # shape = (len(plume_init), ensemble_test)

    flatten_plume_at_t0 = ens_PH_plume_at_t0.flatten()
    flatten_no_plumes   = ens_PH_no_plumes.flatten()
    flatten_plume_init  = ens_PH_plume_init.flatten()

    ens_NRMSE_plume_at_t0 = ens_NRMSE[plume_at_t0, :]      # shape = (len(plume_at_t0), ensemble_test)
    ens_NRMSE_no_plumes   = ens_NRMSE[no_plumes, :]        # shape = (len(no_plumes), ensemble_test)
    ens_NRMSE_plume_init  = ens_NRMSE[plume_init, :]  

    NMRSE_flatten_plume_at_t0 = ens_NRMSE_plume_at_t0.flatten()
    NRMSE_flatten_no_plumes   = ens_NRMSE_no_plumes.flatten()
    NRMSE_flatten_plume_init  = ens_NRMSE_plume_init.flatten()

    ens_pNRMSE_plume_at_t0 = ens_pNRMSE[plume_at_t0, :]      # shape = (len(plume_at_t0), ensemble_test)
    ens_pNRMSE_no_plumes   = ens_pNRMSE[no_plumes, :]        # shape = (len(no_plumes), ensemble_test)
    ens_pNRMSE_plume_init  = ens_pNRMSE[plume_init, :]  

    pNMRSE_flatten_plume_at_t0 = ens_pNRMSE_plume_at_t0.flatten()
    pNRMSE_flatten_no_plumes   = ens_pNRMSE_no_plumes.flatten()
    pNRMSE_flatten_plume_init  = ens_pNRMSE_plume_init.flatten()

    flatten_PH       = ens_PH.flatten()
    flatten_PH_modes = ens_PH_modes.flatten()
    print('flat PH', flatten_PH)
    print('channel plume nrmse', ens_nrmse_ch_pl)

    metrics_ens_ALL = {
    "mean PH": np.mean(ens_PH2),
    "mean PH2": np.mean(flatten_PH),
    "lower PH": np.quantile(flatten_PH, 0.75),
    "uppper PH": np.quantile(flatten_PH, 0.25),
    "median PH": np.median(flatten_PH),
    "mean NRMSE": np.mean(ens_nrmse),
    "mean NRMSE plume": np.mean(ens_nrmse_plume),
    "mean EVR": np.mean(ens_evr),
    "mean ssim": np.mean(ens_ssim),
    "mean NRMSE per channel": np.mean(ens_nrmse_ch),
    "mean NRMSE per channel in plume": np.nanmean(ens_nrmse_ch_pl),
    "ens_nrmse_inPHregion": np.mean(ens_nrmse_inPHregion),
    "mean PH2 (modes)":  np.mean(flatten_PH_modes),
    "lower PH (modes)": np.quantile(flatten_PH_modes, 0.75),
    "upper PH (modes)": np.quantile(flatten_PH_modes, 0.25),
    "median PH (modes)": np.median(flatten_PH_modes),
    "median NRMSE per channel": np.median(ens_nrmse_ch),
    "upper NRMSE per channel": np.percentile(ens_nrmse_ch, 75),
    "lower NRMSE per channel": np.percentile(ens_nrmse_ch, 25),
    "mean_PH_plume_at_t0": np.mean(flatten_plume_at_t0),
    "mean_PH_no_plumes": np.mean(flatten_no_plumes),
    "mean_PH_plume_init": np.mean(flatten_plume_init),
    "mean_NRMSE_plume_at_t0": np.mean(NMRSE_flatten_plume_at_t0),
    "mean_NRMSE_no_plumes": np.mean(NRMSE_flatten_no_plumes),
    "mean_NRMSE_plume_init": np.mean(NRMSE_flatten_plume_init),
    "mean_pNRMSE_plume_at_t0": np.mean(pNMRSE_flatten_plume_at_t0),
    "mean_pNRMSE_no_plumes": np.mean(pNRMSE_flatten_no_plumes),
    "mean_pNRMSE_plume_init": np.mean(pNRMSE_flatten_plume_init)
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)

    np.save('Ra2e8/POD_ESN/test_indexes.npy', test_indexes)
    print('finished testing')


if statistics_interval:
    #### STATISTICS ####
    stats_path = output_path + '/statistics/35LTs'
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
        print('made directory')

    N_test   = 50                    #number of intervals in the test set
    N_tstart = int(N_washout)   #where the first test interval starts
    N_intt   = 35*N_lyap             #length of each test set interval
    N_washout = int(N_washout)
    N_gap = int(N_lyap)

    print('N_tstart:', N_tstart)
    print('N_intt:', N_intt)
    print('N_washout:', N_washout)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

    ensemble_test = ensemble_test

    ens_pred_global = np.zeros((N_intt, 2, N_test, ensemble_test))
    true_POD_global = np.zeros((N_intt, 2, N_test))
    true_global     = np.zeros((N_intt, 2, N_test))
    ens_PH          = np.zeros((N_intt, ensemble_test))
    ens_nrmse_global= np.zeros((ensemble_test))
    ens_mse_global  = np.zeros((ensemble_test))

    true_data  = U_data
    pred_data  = np.zeros((N_intt, dim, N_test, ensemble_test))

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
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
            print(N_tstart + i*N_gap)
            print('start time of test', time_vals[N_tstart + i*N_gap])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap].copy()
            Y_t       = U[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt].copy()

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
            pred_data[:, :, i, j] = Yh_t

            print(np.shape(Yh_t))

            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]

            ##### reconstructions ####
            _, reconstructed_truth       = inverse_POD(Y_t, pca_)
            _, reconstructed_predictions = inverse_POD(Yh_t, pca_)

            # rescale
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            ## global parameters ##
            if Data == 'ToyData':
                nrmse_global = 0
                mse_global   = 0
            else:
                PODtruth_global    = global_parameters(reconstructed_truth)
                predictions_global = global_parameters(reconstructed_predictions)
                ens_pred_global[:,:,i,j] = predictions_global
                true_POD_global[:,:,i] = PODtruth_global
                true_global[:,:,i] = truth_global[N_tstart + i*N_gap: N_tstart + i*N_gap + N_intt]

                # metrics
                nrmse_global = NRMSE(PODtruth_global, predictions_global)
                mse_global   = MSE(PODtruth_global, predictions_global)


            ens_nrmse_global[j]+= nrmse_global
            ens_mse_global[j]  += mse_global

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    if j % 1 == 0:
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ### global prediction ###
                        if Data == 'ToyData':
                            print('no globals')
                        else:
                            plot_global_prediction_ts(PODtruth_global, predictions_global, xx, i, j, stats_path+'/global_prediciton')
                            plot_global_prediction_ps(PODtruth_global, predictions_global, i, j, stats_path+'/global_prediciton_ts')
            if Data == 'ToyData':
                stats_pdf_modes(Y_t, Yh_t, indexes_to_plot, i, j, stats_path+'/stats_pdf_modes', Modes=True)
            else:
                stats_pdf_modes(Y_t, Yh_t, indexes_to_plot, i, j, stats_path+'/stats_pdf_modes', Modes=True)
                stats_pdf_global(PODtruth_global, predictions_global, i, j, stats_path+'/stats_pdf_global')

            fig, ax = plt.subplots(1, figsize=(8,6))
            ax.scatter(Y_t[:,0], Y_t[:,1], label='truth')
            ax.scatter(Yh_t[:,0], Yh_t[:,1], label='prediction')
            ax.grid()
            ax.set_xlabel('Mode 1')
            ax.set_ylabel('Mode 2')
            ax.legend()
            fig.savefig(stats_path+f"/trajectories_ens{j}_test{i}.png")

        # accumulation for each ensemble member
        ens_nrmse_global[j]= ens_nrmse_global[j]/N_test

    pred_data_flat = pred_data.reshape(pred_data.shape[0]*pred_data.shape[2]*pred_data.shape[3], pred_data.shape[1])
    wasserstein_per_mode = []

    scaler_modes = StandardScaler()
    true_data_ss = scaler_modes.fit_transform(true_data)
    pred_data_flat_ss = scaler_modes.transform(pred_data_flat)

    for m in range(dim):
        wd = wasserstein_distance(true_data_ss[:, m], pred_data_flat_ss[:, m])
        wasserstein_per_mode.append(wd)

    mean_wasserstein = np.mean(wasserstein_per_mode)

    # Full path for saving the file
    output_file_ALL = 'ESN_statistics_metrics_all.json' 

    output_path_met_ALL = os.path.join(stats_path, output_file_ALL)

    metrics_ens_ALL = {
    **{f"Wasserstein distance mode {m+1}": float(wasserstein_per_mode[m]) for m in range(dim)},
    "Mean Wasserstein distance": float(mean_wasserstein),
    "mean global NRMSE": np.mean(ens_nrmse_global),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished statistics')

if initiation_interval:
    #### INITIATION ####
    init_path = output_path + '/initation/'
    if not os.path.exists(init_path):
        os.makedirs(init_path)
        print('made directory')

    #test_indexes = [210, 420, 555]
    N_test   = 40                    #number of intervals in the test set
    N_tstart = int(N_washout + N_train)   #where the first test interval starts
    N_intt   = 3*N_lyap             #length of each test set interval
    N_washout = int(N_washout)
    N_gap = int(1*N_lyap)

    print('N_tstart:', N_tstart)
    print('N_intt:', N_intt)
    print('N_washout:', N_washout)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = threshold_ph

    ensemble_test = 5
    n_bins        = 3

    ens_pred             = np.zeros((N_intt, dim, ensemble_test))
    ens_PH               = np.zeros((N_test, ensemble_test))
    ens_prec             = np.zeros((N_test, ensemble_test))
    ens_recall           = np.zeros((N_test, ensemble_test))
    ens_f1               = np.zeros((N_test, ensemble_test))
    ens_acc              = np.zeros((N_test, ensemble_test))
    interval_true_counts = np.zeros((N_intt, N_test, ensemble_test))
    interval_pred_counts = np.zeros((N_intt, N_test, ensemble_test))
    # pre-allocate arrays for metrics: shape (N_test, ensemble_test, n_bins)
    spatial_precision = np.zeros((N_test, ensemble_test, n_bins))
    spatial_recall    = np.zeros((N_test, ensemble_test, n_bins))
    spatial_f1        = np.zeros((N_test, ensemble_test, n_bins))

    images_test_path = init_path+'/test_images/'
    if not os.path.exists(images_test_path):
        os.makedirs(images_test_path)
        print('made directory')
    metrics_test_path = init_path+'/test_metrics/'
    if not os.path.exists(metrics_test_path):
        os.makedirs(metrics_test_path)
        print('made directory')

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = N_test
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print('test', i+1)
            print(N_tstart + i*N_gap)
            print('start time of test', time_vals[N_tstart + i*N_gap])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap].copy()
            Y_t       = U[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt].copy()
            data_set_Y_t =  data_set[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt]

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t,_,Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
            print(np.shape(Yh_t))
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]

            ##### reconstructions ####
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

            ### active array based on score points
            plume_score_threshold = 0.17
            active_array, mask_expanded, _,_,_ = active_array_truth(data_set_Y_t, z)
            active_array_POD, active_array_reconstructed, mask, mask_expanded_recon = active_array_calc(reconstructed_truth, reconstructed_predictions, z)
            active_array_POD_score, active_array_reconstructed_score, mask_score, mask_reconstructed_score = active_array_calc_prob(reconstructed_truth, reconstructed_predictions, z, RH_min, RH_max, w_min, w_max, b_anom_min, b_anom_max, plume_score_threshold)

            flatten_POD_active_array         = active_array_POD.flatten().astype(int)
            flatten_recon_active_array       = active_array_reconstructed.flatten().astype(int)
            flatten_POD_active_array_score   = active_array_POD_score.flatten().astype(int)
            flatten_recon_active_array_score = active_array_reconstructed_score.flatten().astype(int)

            precision = precision_score(flatten_POD_active_array_score, flatten_recon_active_array_score)
            recall    = recall_score(flatten_POD_active_array_score, flatten_recon_active_array_score)
            f1        = f1_score(flatten_POD_active_array_score, flatten_recon_active_array_score)
            acc_sco   = accuracy_score(flatten_POD_active_array_score, flatten_recon_active_array_score)

            print(f"for threshold score: {plume_score_threshold}; precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {acc_sco}")

            ens_prec[i, j]   = precision
            ens_recall[i, j] = recall
            ens_f1[i, j]     = f1
            ens_acc[i, j]    = acc_sco

            if Data == 'RB_plume':
                if plumetype == 'sincospositions':
                    ### direct errror on plumes cos, sin, KE
                    plume_MSE_pp = plume_mse_per_plume(plume_features_truth, plume_features_predictions) # (time, MSE1, MSE2, MSE3)
                    plume_MSE    = plume_mse_overall(plume_features_truth, plume_features_predictions) # (MSE) over all each plume,feature and time
                    ang_error_pp = angular_error_per_plume(plume_features_truth, plume_features_predictions) # (time, ang_1, ang_2, ang_3)

                    ### timing only ###
                    strength_threshold = 0
                    true_counts = np.sum(plume_features_truth[:, 2::3] > strength_threshold, axis=1)
                    pred_counts = np.sum(plume_features_predictions[:, 2::3] > strength_threshold, axis=1)
                    
                    interval_true_counts[:, i, j] = true_counts
                    interval_pred_counts[:, i, j] = pred_counts

                    ### timing and spatial ###
                    x_min = 0
                    x_max = 20
                    delta_x = 1.0  # tolerance for a hit
                    n_bins  = 3

                    p, r ,f = plume_metrics(plume_features_truth, plume_features_predictions, num_plumes=3, delta_x=1.0, 
                         x_min=0, x_max=20, n_bins=3) 
                    
                    spatial_precision[i, j, :] = p
                    spatial_recall[i, j, :]    = r
                    spatial_f1[i, j, :]        = f


            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                 if i<n_plot:
                    if j % 5 == 0:
                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        #plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, int(0.5*N_lyap), x, z, xx, names, images_test_path+'/ESN_validation_ens%i_test%i' %(j,i))
                        
                        cmap_b = ListedColormap(['white', 'red'])
                        bounds_b = [-0.5, 0.5, 1.5]
                        norm_b = BoundaryNorm(bounds_b, cmap_b.N)

                        fig, ax = plt.subplots(3, figsize=(12,12), tight_layout=True)
                        c1 = ax[0].pcolormesh(xx, x, active_array[:, :, 32].T, cmap=cmap_b, norm=norm_b)
                        fig.colorbar(c1, ax=ax[0])
                        ax[0].set_title('True Active Points')
                        c2 = ax[1].pcolormesh(xx, x, active_array_POD_score[:,:, 32].T, cmap=cmap_b, norm=norm_b)
                        fig.colorbar(c2, ax=ax[1])
                        ax[1].set_title('POD Reconstruction Scored Points')
                        c3 = ax[2].pcolormesh(xx, x, active_array_reconstructed_score[:,:, 32].T, cmap=cmap_b, norm=norm_b)
                        fig.colorbar(c3, ax=ax[2])
                        ax[2].set_title('ESN Reconstruction Scored Points')
                        for v in range(3):
                            ax[v].set_xlabel('Time [LTs]')
                            ax[v].set_ylabel('x')
                        fig.savefig(images_test_path+f"/active_plumes_truevscorebinary{plume_score_threshold}_ens{j}_test{i}.png")
                        plt.close()

                        if Data == 'RB_plume':
                            if plumetype == 'sincospositions':
                                fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
                                for v in range(3):
                                    ax.plot(xx, plume_MSE_pp[:, v], label=f"Plume {v+1}")
                                ax.set_xlabel('Time [LTs]')
                                ax.set_ylabel('MSE')
                                ax.legend()
                                ax.grid()
                                fig.savefig(images_test_path+f"/MSE_plume_features_ens_{j}_test{i}.png")

                                fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
                                for v in range(3):
                                    ax.plot(xx, ang_error_pp[:, v], label=f"Plume {v+1}")
                                ax.set_xlabel('Time [LTs]')
                                ax.set_ylabel('Angular Error')
                                ax.legend()
                                ax.grid()
                                fig.savefig(images_test_path+f"/angerr_plume_features_ens_{j}_test{i}.png")


                                fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
                                ax.plot(xx, true_counts, label='True')
                                ax.plot(xx, pred_counts, label='ESN')
                                ax.set_xlabel('Time [LTs]')
                                ax.set_ylabel('Number of plumes')
                                ax.legend()
                                ax.grid()
                                fig.savefig(images_test_path+f"/number_of_plumes_ens{j}_test{i}.png")

                                fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
                                fig.suptitle(f'Ensemble Member {j+1}: Metric statistics per bin')

                                precision_vals = spatial_precision[i, j, :]  # shape: (N_test, n_bins)
                                recall_vals    = spatial_recall[i, j, :]
                                f1_vals        = spatial_f1[i, j, :]


                                bar_width = 0.25
                                bins = [1,2,3]
                                axs[0].scatter(bins, precision_vals)
                                axs[1].scatter(bins, recall_vals)
                                axs[2].scatter(bins, f1_vals)

                                plt.tight_layout(rect=[0, 0, 1, 0.96])
                                axs[-1].set_xlabel('sub interval')
                                fig.savefig(images_test_path+f"/metrics_spatial_ens{j}_test{i}.png")

    def plot_barchart_errors(tests, median, mean, lower, upper, x_label, bar_width, fig, ax, color1='tab:blue', color2='black', marker2='o'):
        lower_error = median - lower
        upper_error = upper - median
        yerr = np.vstack([lower_error, upper_error])

        ax.bar(tests, mean, width=bar_width, align='center', label='Mean', color=color1, capsize=5, zorder=1) #align='center'
        ax.errorbar(tests, median, yerr=yerr, fmt='o', ecolor=color2, markerfacecolor=color2, markeredgecolor=color2, capsize=5, label='Median with Q1-Q3')

        ax.grid()
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(r"$\overline{\mathrm{PH}}$", fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_ylim(0,1)

    def plot_metric(ax, vals, label, color):
        mean_vals = vals.mean(axis=0)        # mean across tests
        median_vals = np.median(vals, axis=0)
        LQ = np.percentile(vals, 25, axis=0)
        UQ = np.percentile(vals, 75, axis=0)

        ax.plot(bins, mean_vals, label=f'{label} mean', color=color, marker='o')
        ax.fill_between(bins, LQ, UQ, color=color, alpha=0.2)
        ax.plot(bins, median_vals, label=f'{label} median', color=color, linestyle='--')
        ax.set_xlabel('Sub-Interval Bin')
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()


    ### metrics from active array based on score ###
    metrics = {
        "precision": ens_prec,
        "recall": ens_recall,
        "f1": ens_f1,
        "accuracy": ens_acc
    }

    ### per test
    stats = {}
    for name, arr in metrics.items():
        mean   = np.mean(arr, axis=1)
        median = np.median(arr, axis=1)
        uq     = np.percentile(arr, 75, axis=1)
        lq     = np.percentile(arr, 25, axis=1)
        
        stats[name] = {
            "mean": mean,
            "median": median,
            "UQ": uq,
            "LQ": lq,
        }

    fig, axes = plt.subplots(4, figsize=(12,12), tight_layout=True)
    axes = axes.flatten()
    PTs = np.arange(1, 41, 1)
    #PTs = ['0']
    metrics_labels = ["precision", "recall", "f1", "accuracy"]
    for index, element in enumerate(metrics_labels):
        ax = axes[index]
        plot_barchart_errors(PTs, stats[element]["median"], stats[element]["mean"], stats[element]["LQ"], stats[element]["UQ"], 'Tests', 0.5, fig, ax, color1='tab:blue', color2='black', marker2='o')
        ax.set_ylabel(element)
    fig.savefig(init_path+f"/metrics_per_test_{plume_score_threshold}.png")

    ### per ensemble
    metrics = {
        "precision": ens_prec,
        "recall": ens_recall,
        "f1": ens_f1,
        "accuracy": ens_acc
    }

    # Compute stats per ensemble
    stats_per_ens = {}
    for name, arr in metrics.items():
        mean = np.mean(arr, axis=0)       # shape: (ensemble_test, n_bins)
        median = np.median(arr, axis=0)
        uq = np.percentile(arr, 75, axis=0)
        lq = np.percentile(arr, 25, axis=0)
        
        stats_per_ens[name] = {
            "mean": mean,
            "median": median,
            "UQ": uq,
            "LQ": lq,
        }

    # Compute overall stats (average across all ensembles and tests)
    stats_all = {}
    for name, arr in metrics.items():
        mean = np.mean(arr)
        median = np.median(arr)
        uq = np.percentile(arr, 75)
        lq = np.percentile(arr, 25)
        
        stats_all[name] = {
            "mean": mean,
            "median": median,
            "UQ": uq,
            "LQ": lq,
        }

    # Plotting
    labels = [f"Mem {i+1}" for i in range(ensemble_test)] + ["Avg"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), tight_layout=True)

    for idx, metric in enumerate(["precision", "recall", "f1", "accuracy"]):
        ax = axes[idx]
        
        # Ensemble members
        ax.bar(x[:-1], stats_per_ens[metric]["median"], yerr=[stats_per_ens[metric]["median"] - stats_per_ens[metric]["LQ"],
                                                            stats_per_ens[metric]["UQ"] - stats_per_ens[metric]["median"]],
            capsize=5, label="Ensemble Members")
        ax.scatter(x[:-1], stats_per_ens[metric]["mean"])
        
        # Overall average
        ax.bar(x[-1], stats_all[metric]["median"], yerr=[[stats_all[metric]["median"] - stats_all[metric]["LQ"]],
                                                    [stats_all[metric]["UQ"] - stats_all[metric]["median"]]],
            color='black', label="Avg")
        ax.scatter(x[-1], stats_all[metric]["mean"])
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.legend()
    fig.savefig(init_path+f"/metrics_per_ens_{plume_score_threshold}.png")

    ## metrics from cos sin positions plumes ##
    # Flatten everything: timesteps × test intervals × ensembles → 1D arrays
    true_flat = (interval_true_counts > 0).flatten().astype(int)   # consider a plume present if count > 0
    pred_flat = (interval_pred_counts > 0).flatten().astype(int)

    # Compute metrics
    precision_cs = precision_score(true_flat, pred_flat)
    recall_cs    = recall_score(true_flat, pred_flat)
    f1_cs        = f1_score(true_flat, pred_flat)
    accuracy_cs  = accuracy_score(true_flat, pred_flat)

    # Full path for saving the file
    output_file_ALL = 'ESN_init_metrics_all.json' 

    output_path_met_ALL = os.path.join(init_path, output_file_ALL)

    metrics_ens_ALL = {
    "plume_MSE_features": plume_MSE,
    "precision_cs": precision_cs,
    "recall_cs": recall_cs,
    "f1_cs": f1_cs,
    "accuracy_cs": accuracy_cs,
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)

    ## cos sin spatial positions plumes ##
    # spatial arrays shape: (N_test, ensemble_test, n_bins)
    mean_precision_per_bin = spatial_precision.mean(axis=(0,1))  # shape (n_bins,)
    mean_recall_per_bin    = spatial_recall.mean(axis=(0,1))
    mean_f1_per_bin        = spatial_f1.mean(axis=(0,1))

    print("Mean precision per bin:", mean_precision_per_bin)
    print("Mean recall per bin   :", mean_recall_per_bin)
    print("Mean F1 per bin       :", mean_f1_per_bin)

    bins = np.arange(1, n_bins+1)

    #(vals, xlabel, bar_width, fig, ax, color1='tab:blue', color2='black', marker2='o'):

    for e in range(ensemble_test):
        fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
        fig.suptitle(f'Ensemble Member {e+1}: Metric statistics per bin')

        precision_vals = spatial_precision[:, e, :]  # shape: (N_test, n_bins)
        recall_vals    = spatial_recall[:, e, :]
        f1_vals        = spatial_f1[:, e, :]


        bar_width = 0.25
        print(len(bins), np.shape(precision_vals))
        plot_barchart_errors2(bins, precision_vals, 'Precision', bar_width, fig, axs[0])
        plot_barchart_errors2(bins, recall_vals, 'Recall', bar_width, fig, axs[1])
        plot_barchart_errors2(bins, f1_vals, 'F1', bar_width, fig, axs[2])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        axs[-1].set_xlabel('sub interval')
        fig.savefig(init_path+f"/metrics_spatial_ens{e}.png")

    # ----- Average across all ensembles -----
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle('Average Across All Ensemble Members')

    precision_vals_avg = spatial_precision.mean(axis=1)  # mean across ensembles
    recall_vals_avg    = spatial_recall.mean(axis=1)
    f1_vals_avg        = spatial_f1.mean(axis=1)

    bar_width = 0.25
    plot_barchart_errors2(bins, precision_vals_avg, 'Precision', bar_width, fig, axs[0])
    plot_barchart_errors2(bins, recall_vals_avg, 'Recall', bar_width, fig, axs[1])
    plot_barchart_errors2(bins, f1_vals_avg, 'F1', bar_width, fig, axs[2])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    axs[-1].set_xlabel('sub interval')
    fig.savefig(init_path+f"/metrics_spatial_allens.png")

    print('finished testing')

if initiation_score_interval:
    #### INITIATION ####
    init_path = output_path + '/initation_score/'
    if not os.path.exists(init_path):
        os.makedirs(init_path)
        print('made directory')

    #test_indexes = [210, 420, 555]
    N_test   = 40                    #number of intervals in the test set
    N_tstart = int(N_washout + N_train)   #where the first test interval starts
    N_intt   = 3*N_lyap             #length of each test set interval
    N_washout = int(N_washout)
    N_gap = int(1*N_lyap)

    print('N_tstart:', N_tstart)
    print('N_intt:', N_intt)
    print('N_washout:', N_washout)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = threshold_ph

    ensemble_test = 5

    ens_pred               = np.zeros((N_intt, dim, ensemble_test))
    ens_PH                 = np.zeros((N_test, ensemble_test))
    x_positions_truth_all  = np.zeros((N_intt, 3, N_test, ensemble_test))
    x_positions_pred_all   = np.zeros((N_intt, 3, N_test, ensemble_test))
    x_strength_truth_all   = np.zeros((N_intt, 3, N_test, ensemble_test))
    x_strength_pred_all    = np.zeros((N_intt, 3, N_test, ensemble_test))

    all_scores = {}
    all_scores_extended = {}
    all_probs = {}
    truths    = {}

    images_test_path = init_path+'/test_images/'
    if not os.path.exists(images_test_path):
        os.makedirs(images_test_path)
        print('made directory')
    metrics_test_path = init_path+'/test_metrics/'
    if not os.path.exists(metrics_test_path):
        os.makedirs(metrics_test_path)
        print('made directory')

    for j in range(ensemble_test):

        print('Realization    :',j+1)
        all_scores[j] = {}
        all_scores_extended[j] = {}
        all_probs[j] = {}
        truths[j]    = {}

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = N_test
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print('test', i+1)
            print(N_tstart + i*N_gap)
            print('start time of test', time_vals[N_tstart + i*N_gap])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap].copy()
            Y_t       = U[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt].copy()
            data_set_Y_t =  data_set[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt]

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t,_,Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
            print(np.shape(Yh_t))
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]

            ##### reconstructions ####
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

            if j ==0:
                if i == 15 or i == 26 or i == 34:
                    np.save(init_path+f"/reconstructed_truth#_ens{j}_test{i}.npy", reconstructed_truth)
                    np.save(init_path+f"/reconstructed_pred_ens{j}_test{i}.npy", reconstructed_predictions)

            if Data == 'RB_plume':
                if plumetype == 'sincospositions':
                    x_positions_truth, x_strength_truth = extract_plume_positions(plume_features_truth, x_domain=(0,20), max_plumes=3, threshold_predictions=False, KE_threshold=0.00005)
                    x_positions_pred, x_strength_pred  = extract_plume_positions(plume_features_predictions, x_domain=(0,20), max_plumes=3, threshold_predictions=True, KE_threshold=0.00005)

                    x_positions_truth_all[:, :, i, j] = x_positions_truth
                    x_positions_pred_all[:, :, i, j]  = x_positions_pred

                    x_strength_truth_all[:, :, i, j] = x_strength_truth
                    x_strength_pred_all[:, :, i, j]  = x_strength_pred

                    # scores = score_plumes(x_positions_truth, x_positions_pred, N_lyap,
                    #                         checkpoints=(0.5, 1.0, 1.5, 2.0, 2.5),
                    #                         thresholds=(1,3,10),  # very good, good, medium
                    #                         weights=None)
                    
                    # all_scores[j][i] = scores
                    # if j ==0:
                    #     print(scores)

                    scores_extended = score_plumes2(x_positions_truth, x_positions_pred, N_lyap,
                                            checkpoints=(0.5, 1.0, 1.5, 2.0, 2.5),
                                            thresholds=(1,3,5))
                    all_scores_extended[j][i] = scores_extended
                    if j == 0:
                        print(scores_extended)

                    # --- NEW: probability at fixed x positions ---
                    # checkpoints  = (0.5, 1.0, 1.5, 2.0, 2.5)   # in Lyapunov times
                    # x_queries    = np.arange(0, 20, 2)         # positions 0,2,...,18   
                    
                    # all_probs[j][i] = {}
                    # for ck in checkpoints:
                    #     ck_step = int(ck * N_lyap)  # convert to time index
                    #     prob, score = probability_at_location(
                    #         pred_pos      = x_positions_pred,
                    #         pred_strength = None,  # or pass strengths if you have them
                    #         ck_step       = ck_step,
                    #         x_query       = x_queries,
                    #         x_domain      = (0,20),
                    #         sigma_x       = 1.0,
                    #         temporal_radius = 2,   # or >0 to include neighbors
                    #         sigma_t       = 1.0,
                    #         use_strength  = False
                    #     )
                    #     all_probs[j][i][ck] = {"probs": prob, "scores": score}

                    # for ck in checkpoints:
                    #     ck_step = int(ck * N_lyap)  # convert to time index
                    #     truth_val = probability_at_location(x_positions_truth,                # (T, P) array of true plume x (np.nan if absent)
                    #                     ck_step,
                    #                     x_queries,
                    #                     x_domain=(0.0,20.0),
                    #                     label_radius=1.0,
                    #                     temporal_radius=0)
                    #     truths[j][i][ck] = truth_val

                    ## initiation socres
                        
    #         if plot:
    #             #left column has the washout (open-loop) and right column the prediction (closed-loop)
    #             # only first n_plot test set intervals are plotted
    #              if i<n_plot:
    #                 if j % 5 == 0:
    #                     categories = ['very good', 'good', 'medium', 'FN', 'FP', 'TN']
    #                     lts = sorted(scores.keys())  # [0.5, 1.0, 1.5, 2.0, 2.5]
    #                     print("lts:", lts)
    #                     lts_labels = np.array(lts)/N_lyap
    #                     lts_labels  = [f"{lt:.1f}" for lt in lts_labels]
    #                     # Extract counts for each category at each LT
    #                     counts_per_category = {cat: [] for cat in categories}
    #                     overalls            = []

    #                     for lt in lts:
    #                         counter = scores[lt]['counts']
    #                         for cat in categories:
    #                             counts_per_category[cat].append(counter.get(cat, 0))  # default 0 if missing
    #                         overalls.append(scores[lt]['overall_score'])

    #                     print(f"ens{j}: {overalls}")
    #                     # Plot grouped bar chart
    #                     x_range = range(len(lts))
    #                     width = 0.15
    #                     fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
    #                     for k, cat in enumerate(categories):
    #                         ax.bar([xi + k*width for xi in x_range], counts_per_category[cat], width=width, label=cat)
    #                     ax.plot([xi + width*2 for xi in x_range], overalls, color='black', marker='o', linestyle='-', label='Overall Score')
    #                     ax.set_xticks([xi + width*2 for xi in x_range])
    #                     ax.set_xticklabels(lts_labels)
    #                     ax.set_xlabel('Lead Times [LTs]')
    #                     ax.set_ylabel('Number')
    #                     ax.legend()
    #                     ax.grid()
    #                     fig.savefig(images_test_path+f"/plume_scores_ens{j}_test{i}.png")
    #                     plt.close()

    # ensemble_stats = {}

    # for j in range(ensemble_test):
    #     all_test_scores = []
    #     for i in range(N_test):
    #         scores_dict = all_scores[j][i]
    #         overall_scores = [v['overall_score'] for v in scores_dict.values()]
    #         all_test_scores.append(overall_scores)
    #     all_test_scores = np.array(all_test_scores)
    #     ensemble_stats[j] = {
    #         'mean_per_checkpoint': np.mean(all_test_scores, axis=0),
    #         'median_per_checkpoint': np.median(all_test_scores, axis=0),
    #         'std_per_checkpoint': np.std(all_test_scores, axis=0),
    #         'UQ_per_checkpoint': np.percentile(all_test_scores, 75, axis=0),
    #         'LQ_per_checkpoint': np.percentile(all_test_scores, 25, axis=0)
    #     }

    # checkpoints=(0.5, 1.0, 1.5, 2.0, 2.5)
    # for j in range(ensemble_test):
    #     stats = ensemble_stats[j]
    #     print(f"ens {j} stats: {stats['mean_per_checkpoint']}")
    #     fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
    #     plot_barchart_errors(checkpoints, stats['median_per_checkpoint'], stats['mean_per_checkpoint'], stats['LQ_per_checkpoint'], stats['UQ_per_checkpoint'], 'Lead Time', 0.4, fig, ax, color1='tab:blue', color2='black', marker2='o')
    #     ax.set_ylabel('Score')
    #     fig.savefig(init_path+f"/Score_ens{j}.png")
    #     plt.close()
    
    # all_means    = np.array([ensemble_stats[j]['mean_per_checkpoint'] for j in range(ensemble_test)])
    # all_medians  = np.array([ensemble_stats[j]['median_per_checkpoint'] for j in range(ensemble_test)])
    # all_UQs      = np.array([ensemble_stats[j]['UQ_per_checkpoint'] for j in range(ensemble_test)])
    # all_LQs      = np.array([ensemble_stats[j]['LQ_per_checkpoint'] for j in range(ensemble_test)])
    # avg_mean_across_ensembles   = np.nanmean(all_means, axis=0)
    # avg_median_across_ensembles = np.nanmean(all_medians, axis=0)
    # avg_UQ_across_ensembles = np.nanmean(all_UQs, axis=0)
    # avg_LQ_across_ensembles = np.nanmean(all_LQs, axis=0)

    # fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
    # plot_barchart_errors(checkpoints, avg_median_across_ensembles, avg_mean_across_ensembles, avg_LQ_across_ensembles, avg_UQ_across_ensembles, 'Lead Time', 0.4, fig, ax, color1='tab:blue', color2='black', marker2='o')
    # ax.set_ylabel('Score')
    # fig.savefig(init_path+f"/Score_avg_ens.png")

    # # Save the dictionary
    # with open(init_path+'/all_scores.pkl', 'wb') as f:
    #     pickle.dump(all_scores, f)

    # Save the dictionary
    with open(init_path+'/all_scores_extended.pkl', 'wb') as f:
        pickle.dump(all_scores_extended, f)

    # np.save(init_path+'/x_positions_truth_all.npy', x_positions_truth_all)
    # np.save(init_path+'/x_positions_pred_all.npy', x_positions_pred_all)

    np.save(init_path+'/x_strength_truth_all.npy', x_strength_truth_all)
    np.save(init_path+'/x_strength_pred_all.npy', x_strength_pred_all)

if initiation_interval2:
    #### INITIATION ####
    plume_score_threshold = 0.75
    testval = 0
    test_nos = [4,7,15,18,26]
    for tindex, tval in enumerate(test_nos):
        testval = tindex
        test_no  = test_nos[testval]
        init_path = output_path + f"/initation/test{test_no}/RunUpToTest/"
        if not os.path.exists(init_path):
            os.makedirs(init_path)
            print('made directory')

        #test_indexes = [10605, 10650, 10770, 10815, 10935]
        init_indexes = [10605+30,10650+8,10770+8,10815+3,10935+9]
        test_index = init_indexes[testval] -2*N_lyap #10935 #10650 #- 2*N_lyap
        N_test   = 9                    #number of intervals in the test set
        N_tstart = int(test_index) #int(N_washout + N_train)   #where the first test interval starts
        N_intt   = 3*N_lyap #3*N_lyap             #length of each test set interval
        N_washout = int(N_washout)
        N_gap = int(0.25*N_lyap)

        fig, ax =plt.subplots(1, figsize=(12,3))
        Y_t       = U[N_tstart :N_tstart + N_intt].copy()
        #Y_t       = U[N_tstart + 2*N_lyap :N_tstart + 2*N_lyap + 3*N_intt].copy()
        _, reconstructed_truth       = inverse_POD(Y_t, pca_)
        ax.pcolormesh(reconstructed_truth[:,:,32,1].T)
        fig.savefig(init_path+f"/testinit{test_no}_data.png")

        data_set_Y_t = data_set[N_tstart :N_tstart + N_intt]
        active_array, mask_expanded, _,_,_ = active_array_truth(data_set_Y_t, z)
        fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
        c1 = ax.contourf(active_array[:, :, 32].T, cmap='Reds')
        fig.colorbar(c1, ax=ax)
        fig.savefig(init_path+f"/testinit{test_no}_data_active.png")

        print('N_tstart:', N_tstart)
        print('N_intt:', N_intt)
        print('N_washout:', N_washout)

        # #prediction horizon normalization factor and threshold
        sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
        threshold_ph = threshold_ph

        ensemble_test = 5

        ens_pred        = np.zeros((N_intt, dim, ensemble_test))
        ens_PH          = np.zeros((N_test, ensemble_test))
        ens_prec        = np.zeros((N_test, ensemble_test))
        ens_recall      = np.zeros((N_test, ensemble_test))
        ens_f1          = np.zeros((N_test, ensemble_test))
        ens_acc         = np.zeros((N_test, ensemble_test))

        images_test_path = init_path+'/test_images/'
        if not os.path.exists(images_test_path):
            os.makedirs(images_test_path)
            print('made directory')
        metrics_test_path = init_path+'/test_metrics/'
        if not os.path.exists(metrics_test_path):
            os.makedirs(metrics_test_path)
            print('made directory')

        for j in range(ensemble_test):

            print('Realization    :',j+1)

            #load matrices and hyperparameters
            Wout     = Woutt[j].copy()
            Win      = Winn[j] #csr_matrix(Winn[j])
            W        = Ws[j]   #csr_matrix(Ws[j])
            rho      = opt_hyp[j,0].copy()
            sigma_in = opt_hyp[j,1].copy()
            print('Hyperparameters:',rho, sigma_in)

            # to store prediction horizon in the test set
            PH             = np.zeros(N_test)
            nrmse_error    = np.zeros((N_test, N_intt))

            # to plot results
            plot = True
            Plotting = True
            if plot:
                n_plot = N_test
                plt.rcParams["figure.figsize"] = (15,3*n_plot)
                plt.figure()
                plt.tight_layout()

            #run different test intervals
            for i in range(N_test):
                print('test', i+1)
                print(N_tstart + i*N_gap)
                print('start time of test', time_vals[N_tstart + i*N_gap])
                # data for washout and target in each interval
                U_wash    = U[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap].copy()
                Y_t       = U[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt].copy()
                data_set_Y_t =  data_set[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt]

                #washout for each interval
                Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
                Uh_wash = np.dot(Xa1, Wout)

                # Prediction Horizon
                Yh_t,_,Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
                print(np.shape(Yh_t))
                Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
                PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
                if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
                ens_PH[i,j] = PH[i]

                ##### reconstructions ####
                _, reconstructed_truth       = inverse_POD(Y_t, pca_)
                _, reconstructed_predictions = inverse_POD(Yh_t, pca_)

                # rescale
                reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
                reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

                active_array, mask_expanded, _,_,_ = active_array_truth(data_set_Y_t, z)
                active_array_POD_score, active_array_reconstructed_score, mask_score, mask_reconstructed_score = active_array_calc_prob(reconstructed_truth, reconstructed_predictions, z, RH_min, RH_max, w_min, w_max, b_anom_min, b_anom_max, plume_score_threshold)

                flatten_POD_active_array        = active_array_POD_score.flatten().astype(int)
                flatten_recon_active_array_score = active_array_reconstructed_score.flatten().astype(int)

                precision = precision_score(flatten_POD_active_array, flatten_recon_active_array_score)
                recall    = recall_score(flatten_POD_active_array, flatten_recon_active_array_score)
                f1        = f1_score(flatten_POD_active_array, flatten_recon_active_array_score)
                acc_sco   = accuracy_score(flatten_POD_active_array, flatten_recon_active_array_score)

                print(f"for threshold score: {plume_score_threshold}; precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {acc_sco}")

                ens_prec[i, j]   = precision
                ens_recall[i, j] = recall
                ens_f1[i, j]     = f1
                ens_acc[i, j]    = acc_sco

                if plot:
                    #left column has the washout (open-loop) and right column the prediction (closed-loop)
                    # only first n_plot test set intervals are plotted
                    if i<n_plot:
                        if j % 5 == 0:
                            
                            print('indexes_to_plot', indexes_to_plot)
                            print(np.shape(U_wash))
                            xx = np.arange(U_wash[:,0].shape[0])/N_lyap
                            #plot_modes_washout(U_wash, Uh_wash, xx, i, j, indexes_to_plot, images_test_path+'/washout_test', Modes=False)

                            xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                            plot_modes_prediction(Y_t, Yh_t, xx, i, j, indexes_to_plot, images_test_path+'/prediction_test', Modes=False)
                            plot_PH(Y_err, threshold_ph, xx, i, j, images_test_path+'/PH_test')
                            
                            #plot_reservoir_states_norm(Xa1, Xa2, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, N_units, images_test_path+'/resnorm_test')
                            #plot_input_states_norm(U_wash, Y_t, time_vals, N_tstart, N_washout_val, i, j, N_gap,  N_intt, images_test_path+'/inputnorm_test')

                            # reconstruction after scaling
                            print('reconstruction and error plot')
                            plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, int(0.5*N_lyap), x, z, xx, names, images_test_path+'/ESN_validation_ens%i_test%i' %(j,i))


                            fig, ax = plt.subplots(3, figsize=(12,12), tight_layout=True)
                            c1 = ax[0].contourf(xx, x, active_array[:, :, 32].T, cmap='Reds')
                            fig.colorbar(c1, ax=ax[0])
                            ax[0].set_title('True Active Points')
                            c2 = ax[1].contourf(xx, x, active_array_POD_score[:,:, 32].T, cmap='Reds')
                            fig.colorbar(c2, ax=ax[1])
                            ax[1].set_title('POD Reconstruction Scored Points')
                            c3 = ax[2].contourf(xx, x, active_array_reconstructed_score[:,:, 32].T, cmap='Reds')
                            fig.colorbar(c3, ax=ax[2])
                            ax[2].set_title('ESN Reconstruction Scored Points')
                            for v in range(3):
                                ax[v].set_xlabel('time')
                                ax[v].set_ylabel('x')
                            fig.savefig(images_test_path+f"/active_plumes_truevscore{plume_score_threshold}_ens{j}_test{i}.png")
                            plt.close()

                            #plot_active_array(active_array, active_array_reconstructed, x, xx, i, j, variables, images_test_path+'/active_plumes_test')
                            
                            #plot_global_prediction_ts(PODtruth_global, predictions_global, xx, i, j, images_test_path+'/global_prediciton')
                            #plot_global_prediction_ps(PODtruth_global, predictions_global, i, j, stats_path+'/global_prediciton')

        metrics = {
            "precision": ens_prec,
            "recall": ens_recall,
            "f1": ens_f1,
            "accuracy": ens_acc
        }

        stats = {}
        for name, arr in metrics.items():
            mean   = np.mean(arr, axis=1)
            median = np.median(arr, axis=1)
            uq     = np.percentile(arr, 75, axis=1)
            lq     = np.percentile(arr, 25, axis=1)
            
            stats[name] = {
                "mean": mean,
                "median": median,
                "UQ": uq,
                "LQ": lq,
            }
        
        def plot_barchart_errors(tests, median, mean, lower, upper, x_label, bar_width, fig, ax, color1='tab:blue', color2='black', marker2='o'):
            lower_error = median - lower
            upper_error = upper - median
            yerr = np.vstack([lower_error, upper_error])

            ax.bar(tests, mean, width=bar_width, align='center', label='Mean', color=color1, capsize=5, zorder=1) #align='center'
            ax.errorbar(tests, median, yerr=yerr, fmt='o', ecolor=color2, markerfacecolor=color2, markeredgecolor=color2, capsize=5, label='Median with Q1-Q3')

            ax.grid()
            ax.set_xlabel(x_label, fontsize=16)
            ax.set_ylabel(r"$\overline{\mathrm{PH}}$", fontsize=16)
            ax.tick_params(labelsize=12)
            ax.set_ylim(0,1)

        fig, axes = plt.subplots(2,2, figsize=(12,8), tight_layout=True)
        axes = axes.flatten()
        PTs = np.arange(-2, 0.25, 0.25)
        #PTs = ['0']
        metrics = ["precision", "recall", "f1", "accuracy"]
        for index, element in enumerate(metrics):
            ax = axes[index]
            plot_barchart_errors(PTs, stats[element]["median"], stats[element]["mean"], stats[element]["LQ"], stats[element]["UQ"], 'PTs', 0.15, fig, ax, color1='tab:blue', color2='black', marker2='o')
            ax.set_ylabel(element)
        fig.savefig(images_test_path+'/metrics.png')

        print('finished testing')

if active_thresholds:
    print('active thresholds .. ')
    N_washout = int(N_washout)
    N_test   = 40                  #number of intervals in the test set
    if reduce_domain2:
        N_tstart = int(N_washout + N_train)
    else:
        N_tstart = int(N_washout + N_train)  #where the first test interval starts
    N_intt   = 3*N_lyap             #length of each test set interval
    N_gap    = int(1*N_lyap)
    #N_washout_val = 4*N_lyap
    test_indexes = []

    ensemble_test = ens

    # Asymmetric windows for FSS
    windows = [(1,1), (3,1), (5,3), (9,3)]

    ensemble_results = []

    active_path = output_path+'/active_thresholds/'
    if not os.path.exists(active_path):
        os.makedirs(active_path)
        print('made directory')

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # To store results per test interval
        test_metrics_original = []
        test_metrics_softer = []

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = 18
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print(f'Starting test interval {i+1}/{N_test}')
            start_idx = N_tstart + i*N_gap
            end_idx   = start_idx + N_intt

            # --- Extract data ---
            U_wash = U[start_idx - N_washout_val : start_idx].copy()
            Y_t    = U[start_idx:end_idx].copy()
            data_set_Y_t = data_set[start_idx:end_idx]
            time_vals_Y_t = time_vals[start_idx:end_idx]

            # --- Washout ---
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # --- Prediction ---
            Yh_t, _, Xa2 = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)

            # --- Reconstruct ---
            if Data == 'RB_plume':
                _, reconstructed_truth       = inverse_POD(Y_t[:,:n_components], pca_)
                _, reconstructed_predictions = inverse_POD(Yh_t[:,:n_components], pca_)
                plume_features_truth         = Y_t[:,n_components:]
                plume_features_predictions   = Yh_t[:,n_components:]
            else:
                _, reconstructed_truth       = inverse_POD(Y_t, pca_)
                _, reconstructed_predictions = inverse_POD(Yh_t, pca_)

            reconstructed_truth       = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            if Data == 'RB':
                # --- ORIGINAL THRESHOLDS ---
                active_array, active_array_reconstructed, mask_expanded, mask_reconstructed = active_array_calc(reconstructed_truth, reconstructed_predictions, z)

                # Downsample 4x4
                nx_new, nz_new = len(x)//4, len(z)//4
                new_array_reconstructed = active_array_reconstructed.reshape(len(time_vals_Y_t), nx_new,4,nz_new,4).max(axis=(2,4))
                new_array = active_array.reshape(len(time_vals_Y_t), nx_new,4,nz_new,4).max(axis=(2,4))

                # Plume metrics
                metrics_original = plume_detection_metrics(new_array, new_array_reconstructed)

                # FSS
                fss_scores = fss_asymmetric(active_array, active_array_reconstructed, windows)
                metrics_original['fss'] = fss_scores
                test_metrics_original.append(metrics_original)

                # --- SOFTER THRESHOLDS ---
                RH_thresh, w_thresh, b_thresh = 0.92, 0.0, 0.0
                active_array, active_array_reconstructed, mask_expanded, mask_reconstructed = active_array_calc_softer(reconstructed_truth, reconstructed_predictions, z, RH_threshold=RH_thresh, w_threshold=w_thresh, b_threshold=b_thresh, both_soft=True)

                # Downsample again
                new_array_reconstructed = active_array_reconstructed.reshape(len(time_vals_Y_t), nx_new,4,nz_new,4).max(axis=(2,4))
                new_array = active_array.reshape(len(time_vals_Y_t), nx_new,4,nz_new,4).max(axis=(2,4))

                # Plume metrics
                metrics_softer = plume_detection_metrics(new_array, new_array_reconstructed)

                # FSS
                fss_scores = fss_asymmetric(active_array, active_array_reconstructed, windows)
                metrics_softer['fss'] = fss_scores
                test_metrics_softer.append(metrics_softer)

        # --- Average over test intervals ---
        def average_metrics(metrics_list):
            if not metrics_list:
                return {}
            keys = [k for k in metrics_list[0].keys() if k != 'fss']
            avg = {k: np.mean([m[k] for m in metrics_list]) for k in keys}
            std = {k+"_std": np.std([m[k] for m in metrics_list]) for k in keys}
            # Average FSS per window
            fss_keys = metrics_list[0]['fss'].keys()
            fss_avg = {f"fss_{w[0]}x{w[1]}": np.mean([m['fss'][w] for m in metrics_list]) for w in fss_keys}
            fss_std = {f"fss_{w[0]}x{w[1]}_std": np.std([m['fss'][w] for m in metrics_list]) for w in fss_keys}
            return {**avg, **std, **fss_avg, **fss_std}

        avg_original = average_metrics(test_metrics_original)
        avg_softer   = average_metrics(test_metrics_softer)
        avg_original['ensemble_id'] = j
        avg_softer['ensemble_id'] = j

        ensemble_results.append({'original': avg_original, 'softer': avg_softer})

    # --- Average across ensembles ---
    def ensemble_average(results_list):
        keys = results_list[0].keys()
        avg = {k: np.mean([r[k] for r in results_list]) for k in keys}
        std = {k+"_std": np.std([r[k] for r in results_list]) for k in keys}
        return {**avg, **std}

    ensemble_avg_original = ensemble_average([e['original'] for e in ensemble_results])
    ensemble_avg_softer   = ensemble_average([e['softer'] for e in ensemble_results])

    print("\n=== Mean metrics over all ensembles ===")
    print("Original thresholds:", ensemble_avg_original)
    print("Softer thresholds  :", ensemble_avg_softer)

    # --- Save JSON ---

    # --- Save JSON ---
    save_path = os.path.join(active_path, "plume_metrics_ensemble.json")
    with open(save_path, 'w') as f:
        json.dump(ensemble_results, f, indent=4)
    print(f"Saved metrics and FSS to {save_path}")

    save_path = os.path.join(active_path, "plume_metrics_original.json")
    with open(save_path, 'w') as f:
        json.dump(ensemble_avg_original, f, indent=4)
    print(f"Saved metrics and FSS to {save_path}")

    save_path = os.path.join(active_path, "plume_metrics_softer.json")
    with open(save_path, 'w') as f:
        json.dump(ensemble_avg_softer, f, indent=4)
    print(f"Saved metrics and FSS to {save_path}")