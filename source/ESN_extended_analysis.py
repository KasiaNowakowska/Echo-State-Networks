"""
python script for ESN.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --hyperparam_file=<hyperparam_file> --config_number=<config_number> --number_of_tests=<number_of_tests>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --hyperparam_file=<hyperparam_file>  hyperparameters for ESN
    --config_number=<config_number>      config number file
    --number_of_tests=<number_of_tests>  number of tests [default: 5]
"""

import os
os.environ["OMP_NUM_THREADS"] = '1' # imposes cores
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
from scipy.io import loadmat, savemat
import time as time
from skopt.plots import plot_convergence
from Eval_Functions import *
from sklearn.preprocessing import StandardScaler
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

from docopt import docopt
args = docopt(__doc__)

exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('run functions files')

print("Current time:", time.time())

input_path = args['--input_path']
output_path = args['--output_path']
hyperparam_file = args['--hyperparam_file']
config_number= args['--config_number']
number_of_tests = int(args['--number_of_tests'])

model_path = output_path

output_path = output_path + '/further_analysis/extended/'
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
    N_forward = hyperparams["N_forward"]
    alpha0 = hyperparams.get("alpha0", None)

if alpha0 == None:
    alpha0 = 1
print('alpha0', alpha0)

from sklearn.metrics import mean_squared_error
def NRMSE(original_data, reconstructed_data):
    if original_data.ndim == 3:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2])
    elif original_data.ndim == 4:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2]*original_data.shape[3])
    if reconstructed_data.ndim == 3:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2])
    elif reconstructed_data.ndim == 4:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2]*reconstructed_data.shape[3])

    # Check if both data arrays have the same dimensions and the dimension is 2
    if original_data.ndim == reconstructed_data.ndim == 2:
        print("Both data arrays have the same dimensions and are 2D.")
    else:
        print("The data arrays either have different dimensions or are not 2D.")
    rmse = np.sqrt(mean_squared_error(original_data, reconstructed_data))
    
    variance = np.var(original_data)
    std_dev  = np.sqrt(variance)
    
    nrmse = (rmse/std_dev)
    
    return nrmse

def MSE(original_data, reconstructed_data):
    if original_data.ndim == 3:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2])
    elif original_data.ndim == 4:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2]*original_data.shape[3])
    if reconstructed_data.ndim == 3:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2])
    elif reconstructed_data.ndim == 4:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2]*reconstructed_data.shape[3])

    # Check if both data arrays have the same dimensions and the dimension is 2
    if original_data.ndim == reconstructed_data.ndim == 2:
        print("Both data arrays have the same dimensions and are 2D.")
    else:
        print("The data arrays either have different dimensions or are not 2D.")
    mse = mean_squared_error(original_data, reconstructed_data)
    return mse

def DTW(original_data, reconstructed_data):
    # Check if both data arrays have the same dimensions and the dimension is 2
    assert original_data.ndim == 2 and reconstructed_data.ndim == 2, "Both data arrays must be 2D with the same dimensions."
    dtw_distance, _ = fastdtw(original_data, reconstructed_data, dist=euclidean)
    normalised_dtw_distance = dtw_distance/original_data.shape[0]
    return normalised_dtw_distance

def pearson_coef(original_data, reconstructed_data):
    assert original_data.ndim == 2 and reconstructed_data.ndim == 2, "Both data arrays must be 2D with the same dimensions."
    correlations = np.zeros((original_data.shape[1]))
    for d in range(original_data.shape[1]):
        original = original_data[:, d]
        recon    = reconstructed_data[:,d]
        corr, _ = pearsonr(original, recon)
        correlations[d] = corr
    
    return correlations

def onset_truth(Y_t, PT, N_lyap, threshold_e):
    result_truth = np.any(Y_t[PT:PT+N_lyap,0] > threshold_e)
    return result_truth

def onset_prediction(Yh_t, PT, N_lyap, threshold_e):
    result_prediction = np.any(Yh_t[PT:PT+N_lyap,0] > threshold_e)
    return result_prediction

def onset_ensemble(true_onset, pred_onset):
    if true_onset == True and pred_onset == True:
        flag = 'TP'
    elif true_onset == True and pred_onset == False:
        flag = 'FN'
    elif true_onset == False and pred_onset == True:
        flag = 'FP'
    elif true_onset == False and pred_onset == False:
        flag = 'TN'
    return flag

def ss_transform(data, scaler):
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
    if data_reshape.ndim == 2:
        print("data array is 2D.")
    else:
        print("data array is not 2D")
        
    data_scaled = scaler.transform(data_reshape)
    data_scaled = data_scaled.reshape(data.shape)
    
    if data_scaled.ndim == 4:
        print('scaled and reshaped to 4 dimensions')
    else:
        print('not scaled properly')
        
    return data_scaled
    
def ss_inverse_transform(data, scaler):
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
    if data_reshape.ndim == 2:
        print("data array is 2D.")
    else:
        print("data array is not 2D")
        
    print('shape before inverse scaling', np.shape(data_reshape))

    data_unscaled = scaler.inverse_transform(data_reshape)
    data_unscaled = data_unscaled.reshape(data.shape)
    
    if data_unscaled.ndim == 4:
        print('unscaled and reshaped to 4 dimensions')
    else:
        print('not unscaled properly')
        
    return data_unscaled


#### Load Data ####
q = np.load(input_path + 'q5000_30000.npy')
ke = np.load(input_path + 'KE5000_30000.npy')
total_time = np.load(input_path + 'total_time5000_30000.npy')
global_var = ['KE', 'q']

# Reshape the arrays into column vectors
ke_column = ke.reshape(len(ke), 1)
q_column = q.reshape(len(q), 1)
#evap_column = evap.reshape(len(evap), 1)

data = np.hstack((ke_column, q_column))

# Print the shape of the combined array
print(data.shape)

U = data

#### dataset generation ####
# number of time steps for washout, train, validation, test
t_lyap    = t_lyap
dt        = 1
N_lyap    = int(t_lyap//dt)
print('N_lyap', N_lyap)
N_washout = washout_len*N_lyap #75
N_train   = train_len*N_lyap #600
N_val     = val_len*N_lyap #45
N_test    = test_len*N_lyap #45
dim       = 2

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)

# standardisation 
norm_std = U_data.std(axis=0)
normalisation = normalisation #on, off, standard

print('norm', norm)
print('u_mean', u_mean)
print('norm_std', norm_std)

# washout
U_washout = U[:N_washout].copy()
# data to be used for training + validation
U_tv  = U[N_washout:N_washout+N_train-1].copy() #inputs
Y_tv  = U[N_washout+1:N_washout+N_train].copy() #data to match at next timestep

#### ESN hyperparameters #####
if normalisation == 'on':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm))]) #input bias (average absolute value of the inputs)
elif normalisation == 'standard':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm_std))]) #input bias (average absolute value of the inputs)
elif normalisation == 'off':
    bias_in   = np.array([np.mean(np.abs(U_data))]) #input bias (average absolute value of the inputs)
bias_out  = np.array([1.]) #output bias

N_units      = Nr #neurons
connectivity = 3

# load in file
# parameters from matrix 
N_units  = Nr

fln = model_path+'/RVC_Noise' + str(Nr) + '.mat'
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

testing = True
statistics = False

if testing:
    ##### quick test #####
    print('TESTING')
    N_test   = number_of_tests                    #number of intervals in the test set
    N_tstart = int(N_washout+N_train)   #where the first test interval starts
    N_intt   = test_len*N_lyap             #length of each test set interval
    N_washout = int(N_washout)
    N_gap = int(0.5*N_lyap)

    ensemble_test = ens

    print('N_tstart:', N_tstart)
    print('N_intt:', N_intt)
    print('N_washout:', N_washout)

    # Prediction Times
    threshold_KE = 0.00015
    PI           = int(0.5*N_lyap)
    PTs          = [0,0.5,1,1.5,2,2.5] 
    flag_pred = np.empty((len(PTs), N_test, ensemble_test), dtype=object)

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
            n_plot = 2
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print('test:', i)
            print('start index:', N_tstart + i*N_gap)
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout +i*N_gap: N_tstart + i*N_gap].copy()
            Y_t       = U[N_tstart + i*N_gap           : N_tstart + i*N_gap + N_intt].copy()

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
            print(np.shape(Yh_t))

            for p in range(len(PTs)):
                PT = int(PTs[p]*N_lyap)
                print('PT is', PT, 'to', PT+N_lyap)
                print('Prediction Time is from', PT/N_lyap, 'LTs for an interval of ', PI/N_lyap, 'LTs')
                true_onset = onset_truth(Y_t, PT, PI, threshold_KE)
                pred_onset = onset_prediction(Yh_t, PT, PI, threshold_KE)
                flag = onset_ensemble(true_onset, pred_onset)
                flag_pred[p,i,j] = flag
                if i==0:
                    if j==0:
                        print(true_onset, pred_onset, flag)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        fig,ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
                        ax.plot(xx, Y_t[:,0], color='tab:blue', label='truth')
                        ax.plot(xx, Yh_t[:,0], color='tab:orange', label='prediction')
                        ax.set_ylabel('$\overline{KE}$')
                        ax.set_xlabel('LT')
                        ax.grid()
                        ax.legend()
                        fig.savefig(output_path+'/plot%i.png')

            if plot:
                if j == 0:
                    if i == 0:
                        print(f"plotting test{i} ensemble{j}")
                        for p in range(len(PTs)):
                            PT = int(PTs[p]*N_lyap)
                            xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                            fig,ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
                            ax.plot(xx, Y_t[:,0], color='tab:blue', label='truth')
                            ax.plot(xx, Yh_t[:,0], color='tab:orange', label='prediciton')
                            ax.axhline(y=threshold_KE, linestyle='--', color='tab:red', label='threshold')
                            ax.fill_between((PTs[p],PTs[p]+(PI/N_lyap)), y1=0, y2=0.0003, alpha=0.2, color='tab:green', label='prediction interval')
                            ax.set_ylabel('$\overline{KE}$')
                            ax.set_xlabel('LT')
                            ax.grid()
                            ax.legend()
                            ax.text(PTs[p]+(PI/(2*N_lyap)), 0.00025, flag_pred[p, i, j], fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
                            fig.savefig(output_path+'/PTs%i.png' % p)

    TP_mask = flag_pred == 'TP'
    FP_mask = flag_pred == 'FP'
    FN_mask = flag_pred == 'FN'
    TN_mask = flag_pred == 'TN'

    TP_list = np.sum(TP_mask, axis=1)
    FP_list = np.sum(FP_mask, axis=1)
    FN_list = np.sum(FN_mask, axis=1)
    TN_list = np.sum(TN_mask, axis=1)

    # Compute precision, recall, and F1-score safely
    P = np.divide(TP_list, TP_list + FP_list, where=(TP_list + FP_list) != 0)
    R = np.divide(TP_list, TP_list + FN_list, where=(TP_list + FN_list) != 0)
    F = np.divide(2 * P * R, P + R, where=(P + R) != 0)

    print(np.shape(P))

    # Handle cases where P or R is zero (avoid NaNs)
    F[np.isnan(F)] = 0

    P_median = np.median(P, axis=1)
    R_median = np.median(R, axis=1)
    F_median = np.median(F, axis=1)

    # Compute 25th and 75th percentiles for error bars
    P_25, P_75 = np.percentile(P, [25, 75], axis=1)
    R_25, R_75 = np.percentile(R, [25, 75], axis=1)
    F_25, F_75 = np.percentile(F, [25, 75], axis=1)

    # Compute the error values (distance from median)
    P_err_lower = P_median - P_25
    P_err_upper = P_75 - P_median
    R_err_lower = R_median - R_25
    R_err_upper = R_75 - R_median
    F_err_lower = F_median - F_25
    F_err_upper = F_75 - F_median

    print(P, P_median, P_25, P_75)

    fig, ax = plt.subplots(1, 3, figsize=(12,3), constrained_layout=True)
    ax[0].errorbar(PTs, P_median, yerr=[P_err_lower, P_err_upper], fmt='o--', capsize=5, capthick=2, elinewidth=1.5)
    ax[1].errorbar(PTs, R_median, yerr=[R_err_lower, R_err_upper], fmt='o--', capsize=5, capthick=2, elinewidth=1.5)
    ax[2].errorbar(PTs, F_median, yerr=[F_err_lower, F_err_upper], fmt='o--', capsize=5, capthick=2, elinewidth=1.5)

    for v in range(3):
        ax[v].set_xlabel('PT')
        ax[v].grid()
        ax[v].set_ylim(0,1.05)
    ax[0].set_ylabel('Precision')
    ax[1].set_ylabel('Recall')
    ax[2].set_ylabel('F1')

    fig.savefig(output_path+f"/FPR_PI{PI}_threshold{threshold_KE}.png")
