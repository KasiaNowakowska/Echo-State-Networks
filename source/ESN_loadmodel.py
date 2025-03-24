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
q = np.load(input_path + 'q5000_33500.npy')
ke = np.load(input_path + 'KE5000_33500.npy')
total_time = np.load(input_path + 'total_time5000_33500.npy')
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

fln = model_path+'/' +str(val) + str(Nr) + '.mat'
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

testing = False
statistics = True
validation = False

if testing:
    ##### quick test #####
    print('TESTING')
    N_test   = number_of_tests                    #number of intervals in the test set
    N_tstart = int(N_washout+N_train)   #where the first test interval starts
    N_intt   = test_len*N_lyap             #length of each test set interval
    N_washout = int(N_washout)
    N_gap = int(0.5*N_lyap)

    print('N_tstart:', N_tstart)
    print('N_intt:', N_intt)
    print('N_washout:', N_washout)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.05

    ensemble_test = ens

    ens_pred = np.zeros((N_intt, dim, N_test, ensemble_test))
    true_data = np.zeros((N_intt, dim, N_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_MSE         = np.zeros((N_test, ensemble_test))
    ens_nrmse_global= np.zeros((ensemble_test))
    ens_dtw         = np.zeros((ensemble_test))
    ens_pearson     = np.zeros((ensemble_test))
    Mean            = np.zeros((ensemble_test))
    MSE_vals        = np.zeros((ensemble_test))

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
        plot = False
        Plotting = True
        if plot:
            n_plot = 3
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
            true_data[:,:,i] = Y_t

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
            print(np.shape(Yh_t))
            ens_pred[:,:,i,j] = Yh_t
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]
            Mean[j]     += np.log10(np.mean((Y_t-Yh_t)**2)) 
            ens_MSE[i,j] = np.log10(np.mean((Y_t-Yh_t)**2)) 
            nrmse_global = NRMSE(Y_t, Yh_t)
            DTW_val      = DTW(Y_t, Yh_t)
            print('DTW', DTW_val)
            pc           = pearson_coef(Y_t, Yh_t)
            mean_pc      = np.mean(pc)


            ens_nrmse_global[j] += nrmse_global
            ens_dtw[j]          += DTW_val
            ens_pearson[j]      += mean_pc

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    fig,ax =plt.subplots(2,sharex=True)
                    xx = np.arange(U_wash[:,-2].shape[0])/N_lyap
                    for v in range(dim):
                        ax[v].plot(xx,U_wash[:,v], color='tab:blue', label='True')
                        ax[v].plot(xx,Uh_wash[:-1,v], color='tab:orange', label='ESN')
                        ax[v].grid()
                    #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                    ax[1].set_xlabel('Time[Lyapunov Times]')
                    ax[0].set_ylabel('KE')
                    ax[1].set_ylabel('q')
                    if v==0:
                        ax[v].legend(ncol=2)
                    fig.savefig(output_path + '/washout_ens{:02d}_test{:02d}.png'.format(j, i))
                    plt.close()

                    fig,ax =plt.subplots(2,sharex=True)
                    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                    for v in range(dim):
                        ax[v].plot(xx,Y_t[:,v], color='tab:blue', label='True')
                        ax[v].plot(xx,Yh_t[:,v], color='tab:orange', label='ESN')
                        ax[v].grid()
                    #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                    ax[1].set_xlabel('Time [Lyapunov Times]')
                    ax[0].set_ylabel('KE')
                    ax[1].set_ylabel('q')
                    if v==0:
                        ax[v].legend(ncol=2)
                    fig.savefig(output_path + '/prediction_ens{:02d}_test{:02d}.png'.format(j, i))
                    plt.close()

                    fig,ax =plt.subplots(1,sharex=True)
                    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                    ax.scatter(Y_t[:,1],Y_t[:,0], color='tab:blue', label='True', marker = '.')
                    ax.scatter(Yh_t[:,1],Yh_t[:,0], color='tab:orange', label='ESN', marker='x')
                    ax.grid()
                    #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                    ax.set_ylabel('KE')
                    ax.set_xlabel('q')
                    ax.legend(ncol=2)
                    ax.set_xlim(0.265, 0.300)
                    ax.set_ylim(-0.00005, 0.0003)
                    fig.savefig(output_path + '/phasespace_ens{:02d}_test{:02d}.png'.format(j, i))
                    plt.close()

        # Percentiles of the prediction horizon
        print('PH quantiles [Lyapunov Times]:',
            np.quantile(ens_PH[i,:],.75), np.median(ens_PH[i,:]), np.quantile(ens_PH[i,:],.25))
        print('')

    # Full path for saving the file
    output_file_ALL = 'ESN_test_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    metrics_ens_ALL = {
    "threshold PH": threshold_ph,
    "mean PH": np.mean(ens_PH),
    "lower PH": np.round(np.quantile(ens_PH, 0.75), 5),
    "uppper PH": np.round(np.quantile(ens_PH, 0.25),5),
    "median PH": np.round(np.median(ens_PH),5),
    "mean nrmse global": np.sum(ens_nrmse_global)/(N_test*ensemble_test),
    "Pearson Coeff": np.sum(ens_pearson)/(N_test*ensemble_test),
    "DTW": np.sum(ens_dtw)/(N_test*ensemble_test),
    "mean MSE": np.sum(Mean)/(N_test*ensemble_test),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)

    for i in range(N_test):
        fig, ax =plt.subplots(2, figsize=(12,6), sharex=True)
        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
        mean_ens = np.mean(ens_pred[:,:,i,:], axis=-1)
        median_ens = np.percentile(ens_pred[:,:,i,:], 50, axis=-1)
        lower = np.percentile(ens_pred[:,:,i,:], 5, axis=-1)
        upper = np.percentile(ens_pred[:,:,i,:], 95, axis=-1)
        print('shape of mean:', np.shape(mean_ens))
        #print('shape of truth:', np.shape(true_data[:,v,i]))
        for v in range(2):
            ax[v].plot(xx, true_data[:,v,i], color='tab:blue', label='truth')
            ax[v].plot(xx, median_ens[:,v], color='tab:orange', label='ESN median prediction')
            ax[v].fill_between(xx, lower[:,v], upper[:,v], color='tab:orange', alpha=0.3, label='ESN 90% confidence interval')
            ax[v].grid()
            ax[v].legend()
        ax[1].set_xlabel('Lyapunov Time')
        ax[0].set_ylabel('KE')
        ax[1].set_ylabel('q')
        fig.savefig(output_path+'/ens_pred_median_test%i.png' % i)
        plt.close()

    avg_PH = np.mean(ens_PH, axis=0)
    np.save(output_path+'/avg_PH.npy', avg_PH)
    fig, ax =plt.subplots(1, figsize=(3,6), constrained_layout=True)
    #np.save(output_path+'/avg_PH{:.2f}.npy'.format(element), avg_PH[:,index])
    # Create a violin plot
    ax.violinplot(ens_PH[:,:].flatten(), showmeans=False, showmedians=True)
    ax.set_ylabel('PH')
    fig.savefig(output_path+'/violin_plot_flatten_test_{:.1f}.png'.format(N_test))

    np.save(output_path+'/ens_PH.npy', ens_PH)
    np.save(output_path+'/ens_MSE.npy', ens_MSE)
    med_PH   = np.median(ens_PH, axis=1)
    lower_PH = np.percentile(ens_PH, 25, axis=1) 
    upper_PH = np.percentile(ens_PH, 75, axis=1) 
    np.save(output_path+'/med_ph.npy', med_PH)
    np.save(output_path+'/lower_ph.npy', lower_PH)
    np.save(output_path+'/uppper_ph.npy', upper_PH)
    N_test_list = np.arange(1,N_test+1,1)
    fig, ax =plt.subplots(1, figsize=(12,3), tight_layout=True)
    ax.plot(N_test_list, med_PH, color='tab:orange')
    ax.plot(N_test_list, lower_PH, color='tab:orange', linestyle='--')
    ax.plot(N_test_list, upper_PH, color='tab:orange', linestyle='--')
    ax.grid()
    ax.set_ylabel('$\overline{PH}$')
    ax.set_xlabel('LT')    
    ax.fill_between(N_test_list, lower_PH, upper_PH, color='tab:orange', alpha=0.2)
    fig.savefig(output_path+'/stats_test_points.png')

if validation:
    ##### quick test #####
    print('VALIDATION')
    output_path = output_path + '/further_validation/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('made directory')
    N_fw      = N_forward*N_lyap
    N_test    = int((N_train-N_val-N_washout)//N_fw + 1)   #number of intervals in the test set
    N_tstart  = int(N_washout)   #where the first test interval starts
    N_intt    = test_len*N_lyap             #length of each test set interval
    N_washout = int(N_washout)
    N_gap     = int(N_lyap)

    print('N_tstart:', N_tstart)
    print('N_intt:', N_intt)
    print('N_washout:', N_washout)
    print('N_test', N_test)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.05

    ensemble_test = ens

    ens_pred = np.zeros((N_intt, dim, N_test, ensemble_test))
    true_data = np.zeros((N_intt, dim, N_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_MSE         = np.zeros((N_test, ensemble_test))
    ens_nrmse_global= np.zeros((ensemble_test))
    ens_dtw         = np.zeros((ensemble_test))
    ens_pearson     = np.zeros((ensemble_test))
    Mean            = np.zeros((ensemble_test))
    MSE_vals        = np.zeros((ensemble_test))

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
        plot = False
        Plotting = True
        if plot:
            n_plot = 3
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
            true_data[:,:,i] = Y_t

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
            print(np.shape(Yh_t))
            ens_pred[:,:,i,j] = Yh_t
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]
            Mean[j]     += np.log10(np.mean((Y_t-Yh_t)**2)) 
            ens_MSE[i,j] = np.log10(np.mean((Y_t-Yh_t)**2)) 
            nrmse_global = NRMSE(Y_t, Yh_t)
            DTW_val      = DTW(Y_t, Yh_t)
            print('DTW', DTW_val)
            pc           = pearson_coef(Y_t, Yh_t)
            mean_pc      = np.mean(pc)


            ens_nrmse_global[j] += nrmse_global
            ens_dtw[j]          += DTW_val
            ens_pearson[j]      += mean_pc

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    fig,ax =plt.subplots(2,sharex=True)
                    xx = np.arange(U_wash[:,-2].shape[0])/N_lyap
                    for v in range(dim):
                        ax[v].plot(xx,U_wash[:,v], color='tab:blue', label='True')
                        ax[v].plot(xx,Uh_wash[:-1,v], color='tab:orange', label='ESN')
                        ax[v].grid()
                    #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                    ax[1].set_xlabel('Time[Lyapunov Times]')
                    ax[0].set_ylabel('KE')
                    ax[1].set_ylabel('q')
                    if v==0:
                        ax[v].legend(ncol=2)
                    fig.savefig(output_path + '/washout_ens{:02d}_test{:02d}.png'.format(j, i))
                    plt.close()

                    fig,ax =plt.subplots(2,sharex=True)
                    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                    for v in range(dim):
                        ax[v].plot(xx,Y_t[:,v], color='tab:blue', label='True')
                        ax[v].plot(xx,Yh_t[:,v], color='tab:orange', label='ESN')
                        ax[v].grid()
                    #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                    ax[1].set_xlabel('Time [Lyapunov Times]')
                    ax[0].set_ylabel('KE')
                    ax[1].set_ylabel('q')
                    if v==0:
                        ax[v].legend(ncol=2)
                    fig.savefig(output_path + '/prediction_ens{:02d}_test{:02d}.png'.format(j, i))
                    plt.close()

                    fig,ax =plt.subplots(1,sharex=True)
                    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                    ax.scatter(Y_t[:,1],Y_t[:,0], color='tab:blue', label='True', marker = '.')
                    ax.scatter(Yh_t[:,1],Yh_t[:,0], color='tab:orange', label='ESN', marker='x')
                    ax.grid()
                    #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                    ax.set_ylabel('KE')
                    ax.set_xlabel('q')
                    ax.legend(ncol=2)
                    ax.set_xlim(0.265, 0.300)
                    ax.set_ylim(-0.00005, 0.0003)
                    fig.savefig(output_path + '/phasespace_ens{:02d}_test{:02d}.png'.format(j, i))
                    plt.close()

        # Percentiles of the prediction horizon
        print('PH quantiles [Lyapunov Times]:',
            np.quantile(ens_PH[i,:],.75), np.median(ens_PH[i,:]), np.quantile(ens_PH[i,:],.25))
        print('')

    # Full path for saving the file
    output_file_ALL = 'ESN_test_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    metrics_ens_ALL = {
    "threshold PH": threshold_ph,
    "mean PH": np.mean(ens_PH),
    "lower PH": np.round(np.quantile(ens_PH, 0.75), 5),
    "uppper PH": np.round(np.quantile(ens_PH, 0.25),5),
    "median PH": np.round(np.median(ens_PH),5),
    "mean nrmse global": np.sum(ens_nrmse_global)/(N_test*ensemble_test),
    "Pearson Coeff": np.sum(ens_pearson)/(N_test*ensemble_test),
    "DTW": np.sum(ens_dtw)/(N_test*ensemble_test),
    "mean MSE": np.sum(Mean)/(N_test*ensemble_test),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)

    for i in range(N_test):
        fig, ax =plt.subplots(2, figsize=(12,6), sharex=True)
        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
        mean_ens = np.mean(ens_pred[:,:,i,:], axis=-1)
        median_ens = np.percentile(ens_pred[:,:,i,:], 50, axis=-1)
        lower = np.percentile(ens_pred[:,:,i,:], 5, axis=-1)
        upper = np.percentile(ens_pred[:,:,i,:], 95, axis=-1)
        print('shape of mean:', np.shape(mean_ens))
        #print('shape of truth:', np.shape(true_data[:,v,i]))
        for v in range(2):
            ax[v].plot(xx, true_data[:,v,i], color='tab:blue', label='truth')
            ax[v].plot(xx, median_ens[:,v], color='tab:orange', label='ESN median prediction')
            ax[v].fill_between(xx, lower[:,v], upper[:,v], color='tab:orange', alpha=0.3, label='ESN 90% confidence interval')
            ax[v].grid()
            ax[v].legend()
        ax[1].set_xlabel('Lyapunov Time')
        ax[0].set_ylabel('KE')
        ax[1].set_ylabel('q')
        fig.savefig(output_path+'/ens_pred_median_test%i.png' % i)
        plt.close()

    avg_PH = np.mean(ens_PH, axis=0)
    np.save(output_path+'/avg_PH.npy', avg_PH)
    fig, ax =plt.subplots(1, figsize=(3,6), constrained_layout=True)
    #np.save(output_path+'/avg_PH{:.2f}.npy'.format(element), avg_PH[:,index])
    # Create a violin plot
    ax.violinplot(ens_PH[:,:].flatten(), showmeans=False, showmedians=True)
    ax.set_ylabel('PH')
    fig.savefig(output_path+'/violin_plot_flatten_test_{:.1f}.png'.format(N_test))

    np.save(output_path+'/ens_PH.npy', ens_PH)
    np.save(output_path+'/ens_MSE.npy', ens_MSE)
    med_PH   = np.median(ens_PH, axis=1)
    lower_PH = np.percentile(ens_PH, 25, axis=1) 
    upper_PH = np.percentile(ens_PH, 75, axis=1) 
    np.save(output_path+'/med_ph.npy', med_PH)
    np.save(output_path+'/lower_ph.npy', lower_PH)
    np.save(output_path+'/uppper_ph.npy', upper_PH)
    N_test_list = np.arange(1,N_test+1,1)
    fig, ax =plt.subplots(1, figsize=(12,3), tight_layout=True)
    ax.plot(N_test_list, med_PH, color='tab:orange')
    ax.plot(N_test_list, lower_PH, color='tab:orange', linestyle='--')
    ax.plot(N_test_list, upper_PH, color='tab:orange', linestyle='--')
    ax.grid()
    ax.set_ylabel('$\overline{PH}$')
    ax.set_xlabel('LT')    
    ax.fill_between(N_test_list, lower_PH, upper_PH, color='tab:orange', alpha=0.2)
    fig.savefig(output_path+'/stats_test_points.png')


if statistics:
    ##### statistics  #####
    print('STATISTICS')
    output_path = output_path + '/statistics/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('made directory')

    N_test   = 37                    #number of intervals in the test set
    N_tstart = int(N_washout)   #where the first test interval starts
    N_intt   = 3*N_lyap             #length of each test set interval
    N_washout = int(N_washout)
    N_gap = int(1*N_lyap)

    print('N_tstart:', N_tstart)
    print('N_intt:', N_intt)
    print('N_washout:', N_washout)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.05

    ensemble_test = ens

    ens_pred = np.zeros((N_intt, dim, N_test, ensemble_test))
    true_data = np.zeros((N_intt, dim, N_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_nrmse_global= np.zeros((ensemble_test))
    ens_dtw         = np.zeros((ensemble_test))
    ens_pearson     = np.zeros((ensemble_test))
    Mean            = np.zeros((ensemble_test))
    MSE_vals        = np.zeros((ensemble_test))

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
            print('test:', i)
            print('start index:', N_tstart + i*N_gap)
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout +i*N_gap: N_tstart + i*N_gap].copy()
            Y_t       = U[N_tstart + i*N_gap           : N_tstart + i*N_gap + N_intt].copy()
            true_data[:,:,i] = Y_t

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
            print(np.shape(Yh_t))
            ens_pred[:,:,i,j] = Yh_t
            Y_err        = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]        = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j]  = PH[i]
            nrmse_global = NRMSE(Y_t, Yh_t)  
            DTW_val      = DTW(Y_t, Yh_t)
            print('DTW', DTW_val)
            pc           = pearson_coef(Y_t, Yh_t)
            mean_pc      = np.mean(pc)


            ens_nrmse_global[j] += nrmse_global
            ens_dtw[j]          += DTW_val
            ens_pearson[j]      += mean_pc  
            
            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    fig,ax =plt.subplots(2,sharex=True)
                    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                    for v in range(dim):
                        ax[v].plot(xx,Y_t[:,v], color='tab:blue', label='True')
                        ax[v].plot(xx,Yh_t[:,v], color='tab:orange', label='ESN')
                        ax[v].grid()
                    #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                    ax[1].set_xlabel('Time [Lyapunov Times]')
                    ax[0].set_ylabel('KE')
                    ax[1].set_ylabel('q')
                    if v==0:
                        ax[v].legend(ncol=2)
                    fig.savefig(output_path + '/prediction_ens{:02d}_test{:02d}.png'.format(j, i))
                    plt.close()

                    fig,ax =plt.subplots(1,sharex=True)
                    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                    ax.scatter(Y_t[:,1],Y_t[:,0], color='tab:blue', label='True', marker = '.')
                    ax.scatter(Yh_t[:,1],Yh_t[:,0], color='tab:orange', label='ESN', marker='x')
                    ax.grid()
                    #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                    ax.set_ylabel('KE')
                    ax.set_xlabel('q')
                    ax.legend(ncol=2)
                    fig.savefig(output_path + '/phasespace_ens{:02d}_test{:02d}.png'.format(j, i))
                    plt.close()

        # Percentiles of the prediction horizon
        print('PH quantiles [Lyapunov Times]:',
            np.quantile(ens_PH[i,:],.75), np.median(ens_PH[i,:]), np.quantile(ens_PH[i,:],.25))
        print('')
    
    # Full path for saving the file
    output_file_ALL = 'ESN_test_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    metrics_ens_ALL = {
    "threshold PH": threshold_ph,
    "mean PH": np.mean(ens_PH),
    "lower PH": np.round(np.quantile(ens_PH, 0.75), 5),
    "uppper PH": np.round(np.quantile(ens_PH, 0.25),5),
    "median PH": np.round(np.median(ens_PH),5),
    "mean nrmse global": np.sum(ens_nrmse_global)/(N_test*ensemble_test),
    "Pearson Coeff": np.sum(ens_pearson)/(N_test*ensemble_test),
    "DTW": np.sum(ens_dtw)/(N_test*ensemble_test),
    "mean MSE": np.sum(Mean)/(N_test*ensemble_test),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)

    from scipy.stats import gaussian_kde
    for j in range(ensemble_test):
        fig, ax =plt.subplots(1,2, figsize=(12,6))
        for v in range(2):
            ### pred ###
            ens_pred_flat = ens_pred[:,v,:,j].flatten()
            kde = gaussian_kde(ens_pred_flat)
            var_vals = np.linspace(min(ens_pred_flat), max(ens_pred_flat), 1000)  # X range
            pdf_vals = kde(var_vals)

            ### true ###
            true_flat = true_data[:,v,:].flatten()
            kde_true = gaussian_kde(true_flat)
            var_vals_true = np.linspace(min(true_flat), max(true_flat), 1000)  # X range
            pdf_vals_true = kde_true(var_vals_true)

            ax[v].plot(var_vals, pdf_vals, label="prediction")
            ax[v].plot(var_vals_true, pdf_vals_true, label="true")
            ax[v].grid()
            ax[v].legend()
            ax[v].set_ylabel(f"Density")
            ax[v].set_xlabel(f"Value of {global_var[v]}")
        fig.savefig(output_path+f"/pdfs_{j}.png")


    fig, ax =plt.subplots(1,2, figsize=(12,6))
    for v in range(2):
        ### pred ###
        ens_pred_flat = ens_pred[:,v,:,:].flatten()
        kde = gaussian_kde(ens_pred_flat)
        var_vals = np.linspace(min(ens_pred_flat), max(ens_pred_flat), 1000)  # X range
        pdf_vals = kde(var_vals)

        ### true ###
        true_flat = true_data[:,v,:].flatten()
        kde_true = gaussian_kde(true_flat)
        var_vals_true = np.linspace(min(true_flat), max(true_flat), 1000)  # X range
        pdf_vals_true = kde_true(var_vals_true)

        ax[v].plot(var_vals, pdf_vals, label="prediction")
        ax[v].plot(var_vals_true, pdf_vals_true, label="true")
        ax[v].grid()
        ax[v].legend()
        ax[v].set_ylabel(f"Density")
        ax[v].set_xlabel(f"Value of {global_var[v]}")
    fig.savefig(output_path+f"/pdfs_all.png")
