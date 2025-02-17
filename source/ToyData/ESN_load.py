import os
import time
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
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import time as time

exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('run functions files')

input_path='./ToyData/'
output_path = './ToyData/Run_n_units200_ensemble3_normalisationstandard/GP36_4/' 
model_path = output_path

output_path = output_path + '/testing'
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

def load_data(file, name):
    with h5py.File(file, 'r') as hf:
        print(name)
        print(hf[name])
        data = np.array(hf[name])

        x = hf['x'][:]  # 1D x-axis
        z = hf['z'][:]  # 1D z-axis
        time_vals = hf['time'][:]  # 1D time vector

    return data, x, z, time_vals

def POD(data, c,  file_str, Plotting=True):
    data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    print('shape of data for POD:', np.shape(data_matrix))
    pca = PCA(n_components=c, svd_solver='randomized', random_state=42)
    pca.fit(data_matrix)
    data_reduced = pca.transform(data_matrix)
    print('shape of reduced_data:', np.shape(data_reduced))
    data_reconstructed_reshaped = pca.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data.shape[0], data.shape[1], data.shape[2])
    print('shape of data reconstructed:', np.shape(data_reconstructed))

    components = pca.components_
    print(np.shape(components))
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print('cumulative explained variance for', c, 'components is', cumulative_explained_variance[-1])

    if Plotting:
        # Plot cumulative explained variance
        fig, ax = plt.subplots(1, figsize=(12,3))
        ax.plot(cumulative_explained_variance*100, marker='o', linestyle='-', color='b')
        ax2 = ax.twinx()
        ax2.semilogy(explained_variance_ratio*100,  marker='o', linestyle='-', color='r')
        ax.set_xlabel('Number of Modes')
        ax.set_ylabel('Cumulative Energy (%)', color='blue')
        #plt.axvline(x=truncated_modes, color='r', linestyle='--', label=f'Truncated Modes: {truncated_modes}')
        ax2.set_ylabel('Inidvidual Energy (%)', color='red')
        fig.savefig(output_path+file_str+'_EVRmodes.png')
        #plt.show()
        plt.close()

        # Plot the time coefficients and mode structures
        indexes_to_plot = np.array([1, 2, 3, 4] ) -1
        indexes_to_plot = indexes_to_plot[indexes_to_plot <= (c-1)]
        print('plotting for modes', indexes_to_plot)
        print('number of modes', len(indexes_to_plot))

        #time coefficients
        fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
        for index, element in enumerate(indexes_to_plot):
            #ax.plot(index - data_reduced[:, element], label='S=%i' % (element+1))
            ax.plot(data_reduced[:, element], label='S=%i' % (element+1))
        ax.grid()
        ax.legend()
        #ax.set_yticks([])
        ax.set_xlabel('Time Step $\Delta t$')
        ax.set_ylabel('Time Coefficients')
        fig.savefig(output_path+file_str+'_coef.png')
        #plt.show()
        plt.close()

        # Visualize the modes
        minm = components.min()
        maxm = components.max()
        fig, ax =plt.subplots(len(indexes_to_plot), figsize=(12,6), tight_layout=True, sharex=True)
        for i in range(len(indexes_to_plot)):
            mode = components[indexes_to_plot[i]].reshape(data.shape[1], data.shape[2])  # Reshape to original dimensions for visualization
            c1 = ax[i].pcolormesh(x, z, mode[:, :].T, cmap='viridis', vmin=minm, vmax=maxm)  # Visualizing the first variable
            #ax[i].axis('off')
            ax[i].set_title('mode % i' % (indexes_to_plot[i]+1))
            fig.colorbar(c1, ax=ax[i])
            ax[i].set_ylabel('z')
        ax[-1].set_xlabel('x')
        fig.savefig(output_path+file_str+'_modes.png')
        #plt.show()
        plt.close()

    return data_reduced, data_reconstructed_reshaped, data_reconstructed, pca

def inverse_POD(data_reduced, pca_):
    data_reconstructed_reshaped = pca_.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data_reconstructed_reshaped.shape[0], len(x), len(z))
    print('shape of data reconstructed:', np.shape(data_reconstructed))
    return data_reconstructed_reshaped, data_reconstructed 

def plot_reconstruction(original, reconstruction, z_value, t_value, time_vals, file_str):
    fig, ax = plt.subplots(2, figsize=(12,6), tight_layout=True, sharex=True)
    minm = min(np.min(original[t_value, :, :]), np.min(reconstruction[t_value, :, :]))
    maxm = max(np.max(original[t_value, :, :]), np.max(reconstruction[t_value, :, :]))
    c1 = ax[0].pcolormesh(x, z, original[t_value,:,:].T, vmin=minm, vmax=maxm)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('true')
    c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:].T, vmin=minm, vmax=maxm)
    fig.colorbar(c1, ax=ax[1])
    ax[1].set_title('reconstruction')
    for v in range(2):
        ax[v].set_ylabel('z')
    ax[-1].set_xlabel('x')
    fig.savefig(output_path+'/snapshot_recon'+file_str+'.png')

    fig, ax = plt.subplots(2, figsize=(12,6), tight_layout=True, sharex=True)
    minm = min(np.min(original[:, :, z_value]), np.min(reconstruction[:, :, z_value]))
    maxm = max(np.max(original[:, :, z_value]), np.max(reconstruction[:, :, z_value]))
    print(np.max(original[:, :, z_value]))
    print(minm, maxm)
    c1 = ax[0].pcolormesh(time_vals, x, original[:, :, z_value].T)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('true')
    c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value].T)
    fig.colorbar(c1, ax=ax[1])
    ax[1].set_title('reconstruction')
    for v in range(2):
        ax[v].set_ylabel('x')
    ax[-1].set_xlabel('time')
    fig.savefig(output_path+'/hovmoller_recon'+file_str+'.png')

#### Metrics ####
from sklearn.metrics import mean_squared_error
def NRMSE(original_data, reconstructed_data):
    if original_data.ndim == 3:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2])
    if reconstructed_data.ndim == 3:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2])

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

from skimage.metrics import structural_similarity as ssim
def compute_ssim_for_4d(original, decoded):
    """
    Compute the average SSIM across all timesteps and channels for 4D arrays.
    """
    if original.ndim == 3:
        original = original.reshape(original.shape[0], original.shape[1], original.shape[2], 1)
    if decoded.ndim == 3:
        decoded = decoded.reshape(decoded.shape[0], decoded.shape[1], decoded.shape[2], 1)
    
    # Check if both data arrays have the same dimensions and the dimension is 4
    if original.ndim == decoded.ndim == 4:
        print("Both data arrays have the same dimensions and are 4D.")
    else:
        print("The data arrays either have different dimensions or are not 4D.")

    # Initialize SSIM accumulator
    total_ssim = 0
    timesteps = original.shape[0]
    channels = original.shape[-1]
  
    for t in range(timesteps):
        for c in range(channels):
            # Extract the 2D slice for each timestep and channel
            orig_slice = original[t, :, :, c]
            dec_slice = decoded[t, :, :, c]
            
            # Compute SSIM for the current slice
            batch_ssim = ssim(orig_slice, dec_slice, data_range=orig_slice.max() - orig_slice.min(), win_size=3)
            total_ssim += batch_ssim
  
    # Compute the average SSIM across all timesteps and channels
    avg_ssim = total_ssim / (timesteps * channels)
    return avg_ssim

def EVR_recon(original_data, reconstructed_data):
    if original_data.ndim == 3:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2])
    if reconstructed_data.ndim == 3:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2])

    # Check if both data arrays have the same dimensions and the dimension is 2
    if original_data.ndim == reconstructed_data.ndim == 2:
        print("Both data arrays have the same dimensions and are 2D.")
    else:
        print("The data arrays either have different dimensions or are not 2D.")

    numerator = np.sum((original_data - reconstructed_data) **2)
    denominator = np.sum(original_data ** 2)
    evr_reconstruction = 1 - (numerator/denominator)
    print('num', numerator, 'den', denominator)
    return evr_reconstruction

def MSE(original_data, reconstructed_data):
    if original_data.ndim == 3:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2])
    if reconstructed_data.ndim == 3:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2])

    # Check if both data arrays have the same dimensions and the dimension is 2
    if original_data.ndim == reconstructed_data.ndim == 2:
        print("Both data arrays have the same dimensions and are 2D.")
    else:
        print("The data arrays either have different dimensions or are not 2D.")
    mse = mean_squared_error(original_data, reconstructed_data)
    return mse

#### LOAD DATA AND POD ####
name='combined'
n_components = 3
data_set, x, z, time_vals = load_data(input_path+'/plume_wave_dataset.h5', name)
data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_ = POD(data_set, n_components, name)

U = data_reduced 
print('shape of data for ESN', np.shape(U))

# number of time steps for washout, train, validation, test
N_lyap    = int((2/3) // 0.05 )
print('N_lyap', N_lyap)
N_washout = 5*N_lyap #65
N_train   = 40*N_lyap #520
N_val     = 3*N_lyap #39
N_test    = 2*N_lyap
dim       = n_components

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)

# standardisation 
norm_std = U_data.std(axis=0)
normalisation = 'standard' #on, off, standard

print('norm', norm)
print('u_mean', u_mean)
print('norm_std', norm_std)

# washout
U_washout = U[:N_washout].copy()
# data to be used for training + validation
U_tv  = U[N_washout:N_washout+N_train-1].copy() #inputs
Y_tv  = U[N_washout+1:N_washout+N_train].copy() #data to match at next timestep

indexes_to_plot = np.array([1, 2, 3] ) -1

# adding noise to training set inputs with sigma_n the noise of the data
# improves performance and regularizes the error as a function of the hyperparameters
seed = 42   #to be able to recreate the data, set also seed for initial condition u0
rnd1  = np.random.RandomState(seed)
noisy = True
if noisy:
    data_std = np.std(U,axis=0)
    sigma_n = 1e-3     #change this to increase/decrease noise in training inputs (up to 1e-1)
    for i in range(n_components):
        U_tv[:,i] = U_tv[:,i] \
                        + rnd1.normal(0, sigma_n*data_std[i], N_train-1)

# load in file
# parameters from matrix 
val      = RVC_Noise
N_units  = 200
bias_out  = np.array([1.]) #output bias
print('bias_out:', bias_out)

fln = model_path+ '/ESN_matrices' + '.mat'
data = loadmat(fln)
print(data.keys())

Winn             = data['Win'][0] #gives Winn
fix_hyp          = data['fix_hyp']
bias_in_value    = data['fix_hyp'][:,0]
N_washout        = data['fix_hyp'][:,1]
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

##### Testing #####
N_test   = 10                    #number of intervals in the test set
N_tstart = N_washout + N_train   #where the first test interval starts
N_intt   = 2*N_lyap             #length of each test set interval
N_gap    = 1*N_lyap


# #prediction horizon normalization factor and threshold
sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
threshold_ph = 0.2

ensemble_test = 3

ens_pred = np.zeros((N_intt, dim, ensemble_test))
ens_PH = np.zeros((N_test, ensemble_test))

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
    PH       = np.zeros(N_test)

    # to plot results
    plot = True
    if plot:
        n_plot = 10
        plt.rcParams["figure.figsize"] = (15,3*n_plot)
        plt.figure()
        plt.tight_layout()

    #run different test intervals
    for i in range(N_test):
        print(N_tstart + i*N_intt)
        # data for washout and target in each interval
        U_wash    = U[N_tstart - N_washout +i*N_intt : N_tstart + i*N_intt].copy()
        Y_t       = U[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt].copy()

        #washout for each interval
        Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
        Uh_wash = np.dot(Xa1, Wout)

        # Prediction Horizon
        Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
        print(np.shape(Yh_t))
        if i == 0:
            ens_pred[:, :, j] = Yh_t
        Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
        PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
        if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
        ens_PH[i, j] = PH[i]

        if plot:
            #left column has the washout (open-loop) and right column the prediction (closed-loop)
            # only first n_plot test set intervals are plotted
            washout_plot = False
            prediction_plot = False
            error_plot = False
            
            if i<n_plot:
                if ensemble_test % 1 == 0:
                    #### modes prediction ####
                    if washout_plot:
                        fig,ax =plt.subplots(len(indexes_to_plot),sharex=True, tight_layout=True)
                        xx = np.arange(U_wash[:,-2].shape[0])/N_lyap
                        print(np.shape(xx), xx[0], xx[-1])
                        for v in range(len(indexes_to_plot)):
                            index = indexes_to_plot[v]
                            ax[v].plot(xx,U_wash[:,index], 'b', label='True')
                            ax[v].plot(xx,Uh_wash[:-1,index], '--r', label='ESN')
                            ax[v].grid()
                            ax[v].set_ylabel('mode %i' % (index+1))
                        ax[-1].set_xlabel('Time[Lyapunov Times]')
                        if i==0:
                            ax[0].legend(ncol=2)
                        fig.suptitle('washout_ens%i_test%i' % (j,i))
                        fig.savefig(output_path+'/washout_ens%i_test%i.png' % (j,i))
                        plt.close()

                    if prediction_plot:
                        fig,ax =plt.subplots(len(indexes_to_plot),sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        for v in range(len(indexes_to_plot)):
                            index = indexes_to_plot[v]
                            ax[v].plot(xx,Y_t[:,index], 'b')
                            ax[v].plot(xx,Yh_t[:,index], '--r')
                            ax[v].grid()
                            ax[v].set_ylabel('mode %i' % (index+1))
                        ax[-1].set_xlabel('Time [Lyapunov Times]')
                        fig.savefig(output_path+'/prediction_ens%i_test%i.png' % (j,i))
                        plt.close()
                    
                    if error_plot:
                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(xx,Y_err, 'b')
                        ax.axhline(y=threshold_ph, xmin=xx[0], xmax=xx[-1])
                        ax.grid()
                        ax.set_ylabel('PH')
                        ax.set_xlabel('Time')
                        fig.savefig(output_path+'/PH_ens%i_test%i.png' % (j,i))
                        plt.close()

                    ##### reconstructions ####
                    _, reconstructed_truth       = inverse_POD(Y_t, pca_)
                    _, reconstructed_predictions = inverse_POD(Yh_t, pca_)
                    plot_reconstruction(reconstructed_truth, reconstructed_predictions, 32, 20, xx, 'ens%i_test%i' %(j,i))
                    nrmse = NRMSE(reconstructed_truth, reconstructed_predictions)
                    mse   = MSE(reconstructed_truth, reconstructed_predictions)
                    evr   = EVR_recon(reconstructed_truth, reconstructed_predictions)
                    SSIM  = compute_ssim_for_4d(reconstructed_truth, reconstructed_predictions)

                    print('NRMSE', nrmse)
                    print('MSE', mse)
                    print('EVR_recon', evr)
                    print('SSIM', SSIM)

                    # Full path for saving the file
                    output_file = name + '_ESN_test_metrics_ens%i_test%i.json' % (j,i)

                    output_path_met = os.path.join(output_path, output_file)

                    metrics = {
                    "test": i,
                    "no. modes": n_components,
                    "EVR": evr,
                    "MSE": mse,
                    "NRMSE": nrmse,
                    "SSIM": SSIM,
                    }

                    with open(output_path_met, "w") as file:
                        json.dump(metrics, file, indent=4)
