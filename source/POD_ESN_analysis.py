"""
python script for POD.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path> --modes=<modes> --hyperparam_file=<hyperparam_file> --config_number=<config_number> --number_of_tests=<number_of_tests> --reduce_domain2=<reduce_domain2> --Data=<Data>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --modes=<modes>                      number of modes for POD 
    --hyperparam_file=<hyperparam_file>  hyperparameters for ESN
    --config_number=<config_number>      config_number [default: 0]
    --reduce_domain2=<reduce_domain2>    reduce size of domain keep time period [default: False]
    --number_of_tests=<number_of_tests>  number of tests [default: 5]
    --Data=<Data>                        data type [default: RB]
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
    u_mean_modes_only = U_data[:,:n_components].mean(axis=0)
    m_modes_only = U_data[:,:n_components].min(axis=0)
    M_modes_only = U_data[:,:n_components].max(axis=0)
    norm_modes_only = M_modes_only - m_modes_only
    mean_feats = U_data[:, n_components:].mean(axis=0)
    std_feats  = U_data[:, n_components:].std(axis=0)
    std_feats[std_feats == 0] = 1.0
else:
    mean_feats = np.zeros(n_feat)
    std_feats  = np.ones(n_feat)

#normalisation across all data
u_min_all  = U_data.min()
u_max_all  = U_data.max()
u_norm_all = u_max_all-u_min_all

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
elif normalisation == 'off_plusfeatures':
    u_pods = U_data[:, :n_components]
    u_feats = (U_data[:, n_components:] - mean_feats)/ std_feats 
    u_combined = np.hstack((u_pods, u_feats))
    bias_in = np.array([np.mean(np.abs(u_combined))])            
    sigma_in_feats = 0.01        
elif normalisation == 'range_plusfeatures':
    u_pods = (U_data[:, :n_components]-u_mean_modes_only)/norm_modes_only
    u_feats = (U_data[:, n_components:] - mean_feats)/ std_feats 
    print('shape of u_pods', np.shape(u_pods))
    print('shape of u_feats', np.shape(u_feats))
    u_combined = np.hstack((u_pods, u_feats))
    bias_in = np.array([np.mean(np.abs(u_combined))])            
    sigma_in_feats = 0.1    
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

test_interval = True
validation_interval = False
statistics_interval = False
initiation_interval = False
initiation_interval2 = False

train_data = data_set[:int(N_washout+N_train)]
global_stds = [np.std(train_data[..., c]) for c in range(train_data.shape[-1])]

if validation_interval:
    print('VALIDATION (TEST)')
    N_test   = 50                    #number of intervals in the test set
    if reduce_domain2:
        N_tstart = int(N_washout)
    else:
        N_tstart = int(N_washout)                 #where the first test interval starts
    N_intt   = test_len*N_lyap            #length of each test set interval
    N_gap    = int(3*N_lyap)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

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

    images_val_path = output_path+'/validation_images/'
    if not os.path.exists(images_val_path):
        os.makedirs(images_val_path)
        print('made directory')
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
            print('start time of test', time_vals[N_tstart + i*N_gap])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap].copy()
            Y_t       = U[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt].copy()

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

    flatten_PH = ens_PH.flatten()
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
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
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

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

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
            n_plot = 20
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
            Yh_t,_,Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
            print(np.shape(Yh_t))
            if i == 0:
                ens_pred[:, :, j] = Yh_t
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]
            nrmse_error[i, :] = Y_err

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
            print('NRMSE per channel', nrmse_ch)
            print('NRMSE per channel in plume', nrmse_sep_plume)
            print('no plume accuracy', plume_count_accuracy)

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

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                 if i<n_plot:
                    if j % 5 == 0:
                        
                        print('indexes_to_plot', indexes_to_plot)
                        print(np.shape(U_wash))
                        xx = np.arange(U_wash[:,0].shape[0])/N_lyap
                        plot_modes_washout(U_wash, Uh_wash, xx, i, j, indexes_to_plot, images_test_path+'/washout_test', Modes=True)

                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        plot_modes_prediction(Y_t, Yh_t, xx, i, j, indexes_to_plot, images_test_path+'/prediction_test', Modes=True)
                        plot_PH(Y_err, threshold_ph, xx, i, j, images_test_path+'/PH_test')
                        
                        plot_reservoir_states_norm(Xa1, Xa2, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, N_units, images_test_path+'/resnorm_test')
                        plot_input_states_norm(U_wash, Y_t, time_vals, N_tstart, N_washout_val, i, j, N_gap,  N_intt, images_test_path+'/inputnorm_test')

                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, int(0.5*N_lyap), x, z, xx, names, images_test_path+'/ESN_validation_ens%i_test%i' %(j,i), type='POD')

                        if len(variables)==4:
                            plot_active_array(active_array, active_array_reconstructed, x, xx, i, j, variables, images_test_path+'/active_plumes_test')
                            plot_global_prediction_ts(PODtruth_global, predictions_global, xx, i, j, images_test_path+'/global_prediciton')
                            #plot_global_prediction_ps(PODtruth_global, predictions_global, i, j, stats_path+'/global_prediciton')

                        if Data == 'RB_plume':
                            plotting_number_of_plumes(true_counts, pred_counts_rounded, xx, i, j, images_test_path+f"/number_of_plumes")
                            hovmoller_plus_plumes(reconstructed_truth, reconstructed_predictions, plume_features_truth, plume_features_predictions, xx, x, 1, i, j, images_test_path+f"/hovmol_plumes")

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

    flatten_PH = ens_PH.flatten()
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
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
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
    threshold_ph = 0.2

    ensemble_test = 1

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))

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

            ##### reconstructions ####
            _, reconstructed_truth       = inverse_POD(Y_t, pca_)
            _, reconstructed_predictions = inverse_POD(Yh_t, pca_)

            # rescale
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

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

                        #plot_active_array(active_array, active_array_reconstructed, x, xx, i, j, variables, images_test_path+'/active_plumes_test')
                        
                        #plot_global_prediction_ts(PODtruth_global, predictions_global, xx, i, j, images_test_path+'/global_prediciton')
                        #plot_global_prediction_ps(PODtruth_global, predictions_global, i, j, stats_path+'/global_prediciton')

    print('finished testing')

if initiation_interval2:
    #### INITIATION ####
    init_path = output_path + '/initation/test18/'
    if not os.path.exists(init_path):
        os.makedirs(init_path)
        print('made directory')

    test_indexes = [210, 420, 555]
    N_test   = 10                    #number of intervals in the test set
    N_tstart = 10800 #int(N_washout + N_train)   #where the first test interval starts
    N_intt   = 3*N_lyap             #length of each test set interval
    N_washout = int(N_washout)
    N_gap = int(0.25*N_lyap)

    print('N_tstart:', N_tstart)
    print('N_intt:', N_intt)
    print('N_washout:', N_washout)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

    ensemble_test = 1

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))

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

            ##### reconstructions ####
            _, reconstructed_truth       = inverse_POD(Y_t, pca_)
            _, reconstructed_predictions = inverse_POD(Yh_t, pca_)

            # rescale
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

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

                        #plot_active_array(active_array, active_array_reconstructed, x, xx, i, j, variables, images_test_path+'/active_plumes_test')
                        
                        #plot_global_prediction_ts(PODtruth_global, predictions_global, xx, i, j, images_test_path+'/global_prediciton')
                        #plot_global_prediction_ps(PODtruth_global, predictions_global, i, j, stats_path+'/global_prediciton')

    print('finished testing')
