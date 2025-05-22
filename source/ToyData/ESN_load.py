"""
python script for POD.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path> --modes=<modes> --hyperparam_file=<hyperparam_file> --config_number=<config_number>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --modes=<modes>                      number of modes for POD 
    --hyperparam_file=<hyperparam_file>  hyperparameters for ESN
    --config_number=<config_number>      config_number [default: 0]
"""

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

from docopt import docopt
args = docopt(__doc__)

input_path = args['--input_path']
output_path = args['--output_path']
modes = int(args['--modes'])
hyperparam_file = args['--hyperparam_file']
config_number = int(args['--config_number'])

model_path = output_path

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

output_path = output_path + '/testing/'
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

def load_data_set(file, names, snapshots):
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

def POD(data, c,  file_str, Plotting=True):
    data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    print('shape of data for POD:', np.shape(data_matrix))
    pca = PCA(n_components=c, svd_solver='randomized', random_state=42)
    pca.fit(data_matrix)
    data_reduced = pca.transform(data_matrix)
    print('shape of reduced_data:', np.shape(data_reduced))
    data_reconstructed_reshaped = pca.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])
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
        indexes_to_plot = np.array([1, 2, 3, 4, 5] ) -1
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

    return data_reduced, data_reconstructed_reshaped, data_reconstructed, pca, cumulative_explained_variance[-1]

def inverse_POD(data_reduced, pca_):
    data_reconstructed_reshaped = pca_.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data_reconstructed_reshaped.shape[0], len(x), len(z), len(names))
    print('shape of data reconstructed:', np.shape(data_reconstructed))
    return data_reconstructed_reshaped, data_reconstructed 

def ss_inverse_transform(data, scaler):
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
    else:
        print('data needs to be 4 dimensions')
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

def plot_reconstruction_and_error(original, reconstruction, z_value, t_value, time_vals, file_str):
    abs_error = np.abs(original-reconstruction)
    residual  = original - reconstruction
    vmax_res = np.max(np.abs(residual))  # Get maximum absolute value
    vmin_res = -vmax_res
    if original.ndim == 3: #len(time_vals), len(x), len(z)

        fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
        minm = min(np.min(original[t_value, :, :]), np.min(reconstruction[t_value, :, :]))
        maxm = max(np.max(original[t_value, :, :]), np.max(reconstruction[t_value, :, :]))
        c1 = ax[0].pcolormesh(x, z, original[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('true')
        c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c1, ax=ax[1])
        ax[1].set_title('reconstruction')
        c3 = ax[2].pcolormesh(x, z, abs_error[t_value,:,:].T, cmap='Reds')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error')
        for v in range(3):
            ax[v].set_ylabel('z')
        ax[-1].set_xlabel('x')
        fig.savefig(output_path+file_str+'_hovmoller_recon_error.png')

        fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
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
        c3 = ax[2].pcolormesh(time_vals, x, abs_error[:, :, z_value].T, cmap='Reds')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error')
        for v in range(2):
            ax[v].set_ylabel('x')
        ax[-1].set_xlabel('time')
        fig.savefig(output_path+file_str+'_snapshot_recon_error.png')
    
    elif original.ndim == 4: #len(time_vals), len(x), len(z), var
        for i in range(original.shape[3]):
            name = names[i]
            print(name)
            fig, ax = plt.subplots(3, figsize=(12,6), tight_layout=True, sharex=True)
            minm = min(np.min(original[t_value, :, :, i]), np.min(reconstruction[t_value, :, :, i]))
            maxm = max(np.max(original[t_value, :, :, i]), np.max(reconstruction[t_value, :, :, i]))
            c1 = ax[0].pcolormesh(x, z, original[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(x, z, abs_error[t_value,:,:, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('z')
            ax[-1].set_xlabel('x')
            fig.savefig(output_path+file_str+name+'_snapshot_recon_error.png')
            plt.close()

            fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
            minm = min(np.min(original[:, :, z_value,i]), np.min(reconstruction[:, :, z_value,i]))
            maxm = max(np.max(original[:, :, z_value,i]), np.max(reconstruction[:, :, z_value,i]))
            print(np.max(original[:, :, z_value,i]))
            print(minm, maxm)
            print("time shape:", np.shape(time_vals))
            print("x shape:", np.shape(x))
            print("original[:, :, z_value] shape:", original[:, :, z_value,i].T.shape)
            c1 = ax[0].pcolormesh(time_vals, x, original[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(time_vals, x,  abs_error[:,:,z_value, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('x')
            ax[-1].set_xlabel('time')
            fig.savefig(output_path+file_str+name+'_hovmoller_recon_error.png')
            plt.close()

            fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
            minm = min(np.min(original[:, :, z_value,i]), np.min(reconstruction[:, :, z_value,i]))
            maxm = max(np.max(original[:, :, z_value,i]), np.max(reconstruction[:, :, z_value,i]))
            print(np.max(original[:, :, z_value,i]))
            print(minm, maxm)
            print("time shape:", np.shape(time_vals))
            print("x shape:", np.shape(x))
            print("original[:, :, z_value] shape:", original[:, :, z_value,i].T.shape)
            c1 = ax[0].pcolormesh(time_vals, x, original[:, :, z_value, i].T)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value, i].T)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(time_vals, x,  abs_error[:,:,z_value, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('x')
            ax[-1].set_xlabel('time')
            fig.savefig(output_path+file_str+name+'_hovmoller_recon_diffbar_error.png')
            plt.close()


            fig, ax = plt.subplots(3, figsize=(12,6), tight_layout=True, sharex=True)
            minm = min(np.min(original[t_value, :, :, i]), np.min(reconstruction[t_value, :, :, i]))
            maxm = max(np.max(original[t_value, :, :, i]), np.max(reconstruction[t_value, :, :, i]))
            c1 = ax[0].pcolormesh(x, z, original[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(x, z, residual[t_value,:,:, i].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('z')
                ax[v].tick_params(axis='both', labelsize=12)
            ax[-1].set_xlabel('x')
            fig.savefig(output_path+file_str+name+'_hovmoller_recon_residual.png')

            fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
            minm = min(np.min(original[:, :, z_value,i]), np.min(reconstruction[:, :, z_value,i]))
            maxm = max(np.max(original[:, :, z_value,i]), np.max(reconstruction[:, :, z_value,i]))
            print(np.max(original[:, :, z_value,i]))
            print(minm, maxm)
            print("time shape:", np.shape(time_vals))
            print("x shape:", np.shape(x))
            print("original[:, :, z_value] shape:", original[:, :, z_value,i].T.shape)
            c1 = ax[0].pcolormesh(time_vals, x, original[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(time_vals, x,  residual[:,:,z_value, i].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('x')
                ax[v].tick_params(axis='both', labelsize=12)
            ax[-1].set_xlabel('time')
            fig.savefig(output_path+file_str+name+'_snapshot_recon_residual.png')




def global_parameters(data):
    if data.ndim == 4:
        print("data is 4D.")
    else:
        print("wrong format needs to be 4D.")

    avg_data = np.mean(data[:,:,:,0], axis=(1,2))

    return avg_data

#### Metrics ####
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

    numerator = np.sum((original_data - reconstructed_data) **2)
    denominator = np.sum(original_data ** 2)
    evr_reconstruction = 1 - (numerator/denominator)
    return evr_reconstruction

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

#### LOAD DATA AND POD ####
name=['combined']
names = ['combined']
n_components = 3
# name=['upgraded']
# names = ['upgraded']
# n_components = 5

num_variables = 1
snapshots = 1000
data_set, x, z, time_vals = load_data_set(input_path+'/plume_wave_dataset.h5', name, snapshots)
#data_set, x, z, time_vals = load_data_set(input_path+'/upgraded_dataset.h5', name, snapshots)
print(np.shape(data_set))

def add_noise(data, noise_level=0.01, seed=42):
    """
    Add Gaussian noise to a dataset of shape (time, x, z, channels).
    
    Parameters:
        data (np.ndarray): Input data of shape (T, X, Z, C).
        noise_level (float): Standard deviation of Gaussian noise.
    
    Returns:
        noisy_data (np.ndarray): Noisy version of the input data.
    """
    np.random.seed(seed)
    noise = noise_level * np.random.randn(*data.shape)
    noisy_data = data + noise
    return noisy_data

noise_level = 0.5
data_set = add_noise(data_set, noise_level=noise_level)
fig, ax =plt.subplots(1)
ax.contourf(data_set[:,:,32,0].T)
fig.savefig(output_path+f"/combined_data_noise{noise_level}.png")

data_reshape = data_set.reshape(-1, data_set.shape[-1])
print('shape of data reshaped', np.shape(data_reshape))

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

n_components = modes
data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_, cev  = POD(data_scaled, n_components, f"modes{n_components}")
print('cumulative equiv ratio', cev)

U = data_reduced 
print('shape of data for ESN', np.shape(U))

# number of time steps for washout, train, validation, test
# number of time steps for washout, train, validation, test
t_lyap    = 2/3
dt        = 0.05
N_lyap    = int(t_lyap//dt)
print('N_lyap', N_lyap)
N_washout = int(washout_len*N_lyap) #75
N_washout_val = int(washout_len_val*N_lyap)
N_train   = train_len*N_lyap #600
N_val     = val_len*N_lyap #45
N_test    = test_len*N_lyap #45
dim       = U.shape[1]

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

indexes_to_plot = np.array([1, 2, 3, 4, 5] ) -1
indexes_to_plot = indexes_to_plot[indexes_to_plot <= (n_components-1)]

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
validation_interval = True
statistics_interval = True

if validation_interval:
    ##### quick test #####
    print('VALIDATION (TEST)')
    print(N_washout_val)
    N_test   = 8                    #number of intervals in the test set
    N_tstart = int(N_washout_val)                    #where the first test interval starts
    N_intt   = int(3*N_lyap)            #length of each test set interval

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

    ensemble_test = ensemble_test

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))
    print('shape of ens_PH', np.shape(ens_PH))

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
        PH       = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        if plot:
            n_plot = 3
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print(N_tstart + i*N_intt)
            print('start time:', time_vals[N_tstart + i*N_intt])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt].copy()
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
            nrmse_error[i, :] = Y_err

            ##### reconstructions ####
            _, reconstructed_truth       = inverse_POD(Y_t, pca_)
            _, reconstructed_predictions = inverse_POD(Yh_t, pca_)
            
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            # metrics
            nrmse = NRMSE(reconstructed_truth, reconstructed_predictions)
            mse   = MSE(reconstructed_truth, reconstructed_predictions)
            evr   = EVR_recon(reconstructed_truth, reconstructed_predictions)
            SSIM  = compute_ssim_for_4d(reconstructed_truth, reconstructed_predictions)

            print('NRMSE', nrmse)
            print('MSE', mse)
            print('EVR_recon', evr)
            print('SSIM', SSIM)

            # Full path for saving the file
            output_file = '_ESN_validation_metrics_ens%i_test%i.json' % (j,i)

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

            ens_nrmse[j]       += nrmse
            ens_ssim[j]        += SSIM
            ens_evr[j]         += evr
            ens_PH2[j]         += PH[i]
            ens_PH[i,j]         = PH[i]
                
            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    if ensemble_test % 1 == 0:
                    
                        #### modes prediction ####
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
                        fig.savefig(output_path+'/washout_validation_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(len(indexes_to_plot),sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        for v in range(len(indexes_to_plot)):
                            index = indexes_to_plot[v]
                            ax[v].plot(xx,Y_t[:,index], 'b')
                            ax[v].plot(xx,Yh_t[:,index], '--r')
                            ax[v].grid()
                            ax[v].set_ylabel('mode %i' % (index+1))
                        ax[-1].set_xlabel('Time [Lyapunov Times]')
                        fig.savefig(output_path+'/prediction_validation_ens%i_test%i.png' % (j,i))
                        plt.close()
                        
                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(xx,Y_err, 'b')
                        ax.axhline(y=threshold_ph, xmin=xx[0], xmax=xx[-1])
                        ax.grid()
                        ax.set_ylabel('PH')
                        ax.set_xlabel('Time')
                        fig.savefig(output_path+'/PH_validation_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(np.linalg.norm(Xa1[:, :N_units], axis=1))
                        ax.grid()
                        ax.set_ylabel('res_states')
                        fig.savefig(output_path+'/res_states_validation_washout_ens%i_test%i.png' % (j,i))
                        plt.close()

                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, 1*N_lyap, xx, 'ESN_validation_ens%i_test%i' %(j,i))

        # accumulation for each ensemble member
        ens_nrmse[j]       = ens_nrmse[j] / N_test
        ens_ssim[j]        = ens_ssim[j] / N_test
        ens_evr[j]         = ens_evr[j] / N_test
        ens_PH2[j]         = ens_PH2[j] / N_test

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
    "mean EVR": np.mean(ens_evr),
    "mean ssim": np.mean(ens_ssim),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished validations')

if test_interval:
    ##### quick test #####
    print('TESTING')
    print(N_washout_val)
    N_test   = 8                    #number of intervals in the test set
    N_tstart = int(N_train + N_washout_val)                    #where the first test interval starts
    N_intt   = int(3*N_lyap)            #length of each test set interval

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

    ensemble_test = ensemble_test

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))
    print('shape of ens_PH', np.shape(ens_PH))

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
        PH       = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        if plot:
            n_plot = 3
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print(N_tstart + i*N_intt)
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt].copy()
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
            nrmse_error[i, :] = Y_err

            ##### reconstructions ####
            _, reconstructed_truth       = inverse_POD(Y_t, pca_)
            _, reconstructed_predictions = inverse_POD(Yh_t, pca_)
            
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            # metrics
            nrmse = NRMSE(reconstructed_truth, reconstructed_predictions)
            mse   = MSE(reconstructed_truth, reconstructed_predictions)
            evr   = EVR_recon(reconstructed_truth, reconstructed_predictions)
            SSIM  = compute_ssim_for_4d(reconstructed_truth, reconstructed_predictions)

            print('NRMSE', nrmse)
            print('MSE', mse)
            print('EVR_recon', evr)
            print('SSIM', SSIM)

            # Full path for saving the file
            output_file = '_ESN_test_metrics_ens%i_test%i.json' % (j,i)

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
                
            ens_nrmse[j]       += nrmse
            ens_ssim[j]        += SSIM
            ens_evr[j]         += evr
            ens_PH2[j]         += PH[i]
            ens_PH[i,j]         = PH[i]

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    if ensemble_test % 1 == 0:
                    
                        #### modes prediction ####
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
                        fig.savefig(output_path+'/washout_test_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(len(indexes_to_plot),sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        for v in range(len(indexes_to_plot)):
                            index = indexes_to_plot[v]
                            ax[v].plot(xx,Y_t[:,index], 'b')
                            ax[v].plot(xx,Yh_t[:,index], '--r')
                            ax[v].grid()
                            ax[v].set_ylabel('mode %i' % (index+1))
                        ax[-1].set_xlabel('Time [Lyapunov Times]')
                        fig.savefig(output_path+'/prediction_test_ens%i_test%i.png' % (j,i))
                        plt.close()
                        
                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(xx,Y_err, 'b')
                        ax.axhline(y=threshold_ph, xmin=xx[0], xmax=xx[-1])
                        ax.grid()
                        ax.set_ylabel('PH')
                        ax.set_xlabel('Time')
                        fig.savefig(output_path+'/PH_test_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(np.linalg.norm(Xa1[:, :N_units], axis=1))
                        ax.grid()
                        ax.set_ylabel('res_states')
                        fig.savefig(output_path+'/res_states_test_washout_ens%i_test%i.png' % (j,i))
                        plt.close()

                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, 1*N_lyap, xx, 'ESN_test_ens%i_test%i' %(j,i))
        
        # accumulation for each ensemble member
        ens_nrmse[j]       = ens_nrmse[j] / N_test
        ens_ssim[j]        = ens_ssim[j] / N_test
        ens_evr[j]         = ens_evr[j] / N_test
        ens_PH2[j]         = ens_PH2[j] / N_test

    # Full path for saving the file
    output_file_ALL = 'ESN_test_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    flatten_PH = ens_PH.flatten()
    print('flat PH', flatten_PH)

    print('UQ', np.quantile(flatten_PH, 0.75))

    metrics_ens_ALL = {
    "mean PH": np.mean(ens_PH2),
    "lower PH": np.quantile(flatten_PH, 0.75),
    "uppper PH": np.quantile(flatten_PH, 0.25),
    "median PH": np.median(flatten_PH),
    "mean NRMSE": np.mean(ens_nrmse),
    "mean EVR": np.mean(ens_evr),
    "mean ssim": np.mean(ens_ssim),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished testing')



if statistics_interval:
    print('STATISTICS')
    print(N_washout_val)
    N_test   = 2                    #number of intervals in the test set
    N_tstart = int(N_washout_val)                    #where the first test interval starts
    N_intt   = int(24*N_lyap)            #length of each test set interval
    N_gap    = int(3*N_lyap)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

    ensemble_test = ensemble_test

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
        PH       = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        if plot:
            n_plot = 3
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print(N_tstart + i*N_gap)
            print('start time:', time_vals[N_tstart + i*N_gap])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap].copy()
            Y_t       = U[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt].copy()

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
            print(np.shape(Yh_t))
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)

            ##### reconstructions ####
            _, reconstructed_truth       = inverse_POD(Y_t, pca_)
            _, reconstructed_predictions = inverse_POD(Yh_t, pca_)
            
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            global_var_truth = global_parameters(reconstructed_truth)
            global_var_predictions = global_parameters(reconstructed_predictions)

            nrmse_global = NRMSE(global_var_truth, global_var_predictions)

            ens_nrmse_global[j]+= nrmse_global
                
            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    if ensemble_test % 1 == 0:

                        ### global prediction ###
                        fig, ax = plt.subplots(1, figsize=(12,3), sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(xx, global_var_truth, label='POD')
                        ax.plot(xx, global_var_predictions, label='ESN')
                        ax.grid()
                        ax.legend()
                        ax.set_ylabel('global')
                        ax.set_xlabel('LT')
                        fig.savefig(output_path+f"/global_prediciton_ens{j}_test{i}.png")
                        plt.close()

            from scipy.stats import gaussian_kde
            kde_true  = gaussian_kde(global_var_truth)
            kde_pred  = gaussian_kde(global_var_predictions)

            var_vals_true      = np.linspace(min(global_var_truth), max(global_var_truth), 1000)  # X range
            pdf_vals_true      = kde_true(var_vals_true)
            var_vals_pred      = np.linspace(min(global_var_predictions), max(global_var_predictions), 1000)  # X range
            pdf_vals_pred      = kde_pred(var_vals_pred)

            fig, ax = plt.subplots(1, figsize=(8,6))
            ax.plot(var_vals_true, pdf_vals_true, label="truth")
            ax.plot(var_vals_pred, pdf_vals_pred, label="prediction")
            ax.grid()
            ax.set_ylabel('Denisty')
            ax.set_xlabel('Value')
            ax.legend()
            fig.savefig(output_path+f"/stats_pdf_global_ens{j}_test{i}.png")

            fig, ax = plt.subplots(1, figsize=(8,6))
            ax.scatter(Y_t[:,0], Y_t[:,1], label='truth')
            ax.scatter(Yh_t[:,0], Yh_t[:,1], label='prediction')
            ax.grid()
            ax.set_xlabel('mode 1')
            ax.set_ylabel('mode 2')
            ax.legend()
            fig.savefig(output_path+f"/trajectories_ens{j}_test{i}.png")

            fig, ax = plt.subplots(1,3, figsize=(12, 6))
            for v in range(3):
                kde_true  = gaussian_kde(Y_t[:,v])
                kde_pred  = gaussian_kde(Yh_t[:,v])

                var_vals_true      = np.linspace(min(Y_t[:,v]), max(Y_t[:,v]), 1000)  # X range
                pdf_vals_true      = kde_true(var_vals_true)
                var_vals_pred      = np.linspace(min(Yh_t[:,v]), max(Yh_t[:,v]), 1000)  # X range
                pdf_vals_pred      = kde_pred(var_vals_pred)

                ax[v].plot(var_vals_true, pdf_vals_true, label="truth")
                ax[v].plot(var_vals_pred, pdf_vals_pred, label="prediction")
                ax[v].grid()
                ax[v].set_ylabel('Denisty')
                ax[v].set_xlabel(f"Mode {v}")
                ax[v].legend()
            fig.savefig(output_path+f"/stats_pdf_modes_ens{j}_test{i}.png")

        # accumulation for each ensemble member
        ens_nrmse_global[j]= ens_nrmse_global[j]/N_test

    # Full path for saving the file
    output_file_ALL = 'ESN_statistics_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    metrics_ens_ALL = {
    "mean global NRMSE": np.mean(ens_nrmse_global),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished statistics')

