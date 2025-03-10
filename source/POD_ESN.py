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
    alpha = hyperparams["alpha"]

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

def POD(data, c,  file_str, Plotting=True):
    if data.ndim == 3: #len(time_vals), len(x), len(z)
            data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    elif data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
            data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2]*data.shape[3])
    print('shape of data for POD:', np.shape(data_matrix))
    pca = PCA(n_components=c, svd_solver='randomized', random_state=42)
    pca.fit(data_matrix)
    data_reduced = pca.transform(data_matrix)
    print('shape of reduced_data:', np.shape(data_reduced))
    data_reconstructed_reshaped = pca.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data.shape)
    print('shape of data reconstructed:', np.shape(data_reconstructed))
    np.save(output_path+file_str+'_data_reduced.npy', data_reduced)

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
        indexes_to_plot = np.array([1, 2, 10, 50, 100] ) -1
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
        if data.ndim == 3:
            fig, ax =plt.subplots(len(indexes_to_plot), figsize=(12,6), tight_layout=True, sharex=True)
            for i in range(len(indexes_to_plot)):
                if len(indexes_to_plot) == 1:
                    mode = components[indexes_to_plot[i]].reshape(data.shape[1], data.shape[2])  # Reshape to original dimensions for visualization
                    c1 = ax.pcolormesh(x, z, mode[:, :].T, cmap='viridis', vmin=minm, vmax=maxm)  # Visualizing the first variable
                    ax.set_title('mode % i' % (indexes_to_plot[i]+1))
                    fig.colorbar(c1, ax=ax)
                    ax.set_ylabel('z')
                    ax.set_xlabel('x')
                else:
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
        elif data.ndim == 4:
            for v in range(len(variables)):
                fig, ax =plt.subplots(len(indexes_to_plot), figsize=(12,6), tight_layout=True, sharex=True)
                for i in range(len(indexes_to_plot)):
                    if len(indexes_to_plot) == 1:
                        mode = components[indexes_to_plot[i]].reshape(data.shape[1], data.shape[2], data.shape[3])  # Reshape to original dimensions for visualization
                        c1 = ax.pcolormesh(x, z, mode[:, :, v].T, cmap='viridis')  # Visualizing the first variable
                        ax.set_title('mode % i' % (indexes_to_plot[i]+1))
                        fig.colorbar(c1, ax=ax)
                        ax.set_ylabel('z')
                        ax.set_xlabel('x')
                    else:
                        mode = components[indexes_to_plot[i]].reshape(data.shape[1], data.shape[2], data.shape[3])  # Reshape to original dimensions for visualization              
                        c1 = ax[i].pcolormesh(x, z, mode[:, :, v].T, cmap='viridis')  # Visualizing the first variable
                        #ax[i].axis('off')
                        ax[i].set_title('mode % i' % (indexes_to_plot[i]+1))
                        fig.colorbar(c1, ax=ax[i])
                        ax[i].set_ylabel('z')
                        ax[-1].set_xlabel('x')
                fig.savefig(output_path+file_str+names[v]+'_modes.png')
                #plt.show()
                plt.close()

    return data_reduced, data_reconstructed_reshaped, data_reconstructed, pca, cumulative_explained_variance[-1]

def inverse_POD(data_reduced, pca_):
    data_reconstructed_reshaped = pca_.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data_reconstructed_reshaped.shape[0], len(x), len(z), len(variables))
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

def plot_reconstruction_and_error(original, reconstruction, z_value, t_value, time_vals, file_str):
    abs_error = np.abs(original-reconstruction)
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
            fig.colorbar(c1, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(x, z, abs_error[t_value,:,:, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('z')
            ax[-1].set_xlabel('x')
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
            c1 = ax[0].pcolormesh(time_vals, x, original[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(time_vals, x,  abs_error[:,:,z_value, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('x')
            ax[-1].set_xlabel('time')
            fig.savefig(output_path+file_str+name+'_snapshot_recon_error.png')
            plt.close()


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

def active_array_calc(original_data, reconstructed_data, z):
    beta = 1.201
    alpha = 3.0
    T = original_data[:,:,:,3] - beta*z
    T_reconstructed = reconstructed_data[:,:,:,3] - beta*z
    q_s = np.exp(alpha*T)
    q_s_reconstructed = np.exp(alpha*T)
    rh = original_data[:,:,:,0]/q_s
    rh_reconstructed = reconstructed_data[:,:,:,0]/q_s_reconstructed
    mean_b = np.mean(original_data[:,:,:,3], axis=1, keepdims=True)
    mean_b_reconstructed= np.mean(reconstructed_data[:,:,:,3], axis=1, keepdims=True)
    b_anom = original_data[:,:,:,3] - mean_b
    b_anom_reconstructed = reconstructed_data[:,:,:,3] - mean_b_reconstructed
    w = original_data[:,:,:,1]
    w_reconstructed = reconstructed_data[:,:,:,1]
    
    mask = (rh[:, :, :] >= 1) & (w[:, :, :] > 0) & (b_anom[:, :, :] > 0)
    mask_reconstructed = (rh_reconstructed[:, :, :] >= 1) & (w_reconstructed[:, :, :] > 0) & (b_anom_reconstructed[:, :, :] > 0)
    
    active_array = np.zeros((original_data.shape[0], len(x), len(z)))
    active_array[mask] = 1
    active_array_reconstructed = np.zeros((original_data.shape[0], len(x), len(z)))
    active_array_reconstructed[mask_reconstructed] = 1
    
    # Expand the mask to cover all features (optional, depending on use case)
    mask_expanded       =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
    mask_expanded_recon =  np.repeat(mask_reconstructed[:, :, :, np.newaxis], 4, axis=-1) # Shape: (256, 64, 1)
    
    return active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon

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

#### LOAD DATA AND POD ####
variables = ['q_all', 'w_all', 'u_all', 'b_all']
names = ['q', 'w', 'u', 'b']
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
snapshots = 2500
data_set, time_vals = load_data_set(input_path+'/data_4var_5000_30000.h5', variables, snapshots)
print(np.shape(data_set))

reduce_data_set = False
if reduce_data_set:
    data_set = data_set[:, 147:211, :, :]
    x = x[147:211]
    print('reduced domain shape', np.shape(data_set))
    print('reduced x domain', np.shape(x))
    print('reduced x domain', len(x))
    print(x[0], x[-1])

data_reshape = data_set.reshape(-1, data_set.shape[-1])
print('shape of data reshaped', data_reshape)

# fit the scaler
scaler = StandardScaler()
scaler.fit(data_reshape)
print('means', scaler.mean_)

print('shape of data before scaling', np.shape(data_reshape))
data_scaled_reshape = scaler.transform(data_reshape)
#reshape 
data_scaled = data_scaled_reshape.reshape(data_set.shape)

n_components = modes
data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_, cev = POD(data_scaled, n_components, f"modes{n_components}")

U = data_reduced 
print('shape of data for ESN', np.shape(U))

# number of time steps for washout, train, validation, test
t_lyap    = t_lyap
dt        = 2
N_lyap    = int(t_lyap//dt)
print('N_lyap', N_lyap)
N_washout = washout_len*N_lyap #75
N_train   = train_len*N_lyap #600
N_val     = val_len*N_lyap #45
N_test    = test_len*N_lyap #45
dim       = n_components

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

indexes_to_plot = np.array([1, 2, 10, 50, 100] ) -1

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
bias_out  = np.array([1.]) #output bias

N_units      = Nr #neurons
connectivity = 3
sparseness   = 1 - connectivity/(N_units-1)

tikh = np.array([1e-6,1e-9,1e-12])  # Tikhonov factor (optimize among the values in this list)

print('tikh:', tikh)
print('N_r:', N_units, 'sparsity:', sparseness)
print('bias_in:', bias_in, 'bias_out:', bias_out)

#### Grid Search and BO #####
n_in  = 0           #Number of Initial random points

spec_in     = .1    #range for hyperparameters (spectral radius and input scaling)
spec_end    = 1.   
in_scal_in  = np.log10(0.05)
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
N_fw     = 1*N_lyap
N_fo     = (N_train-N_val-N_washout)//N_fw + 1 
#N_fo     = 33                     # number of validation intervals
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

    output_path_hyp = os.path.join(output_path, hyp_file)

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
    fig.savefig(output_path+'/convergence_realisation%i.png' % i)
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

##### quick test #####
print('TESTING')
N_test   = n_tests                    #number of intervals in the test set
N_tstart = N_washout + N_train   #where the first test interval starts
N_intt   = test_len*N_lyap             #length of each test set interval

# #prediction horizon normalization factor and threshold
sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
threshold_ph = 0.2

ensemble_test = ensemble_test

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
    PH             = np.zeros(N_test)
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
        ens_PH[i,j] = PH[i]
        nrmse_error[i, :] = Y_err

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
                    fig.savefig(output_path+'/washout_ens%i_test%i.png' % (j,i))
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
                    fig.savefig(output_path+'/prediction_ens%i_test%i.png' % (j,i))
                    plt.close()
                    
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

                    # rescale
                    reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
                    reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

                    # metrics
                    nrmse = NRMSE(reconstructed_truth, reconstructed_predictions)
                    mse   = MSE(reconstructed_truth, reconstructed_predictions)
                    evr   = EVR_recon(reconstructed_truth, reconstructed_predictions)
                    SSIM  = compute_ssim_for_4d(reconstructed_truth, reconstructed_predictions)

                    # reconstruction after scaling
                    print('reconstruction and error plot')
                    plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, 2*N_lyap, xx, 'ESN_ens%i_test%i' %(j,i))

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

                        fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
                        c1 = ax[0].contourf(xx, x, active_array[:,:, 32].T, cmap='Reds')
                        fig.colorbar(c1, ax=ax[0])
                        ax[0].set_title('true')
                        c2 = ax[1].contourf(xx, x, active_array_reconstructed[:,:, 32].T, cmap='Reds')
                        fig.colorbar(c1, ax=ax[1])
                        ax[1].set_title('reconstruction')
                        for v in range(2):
                            ax[v].set_xlabel('time')
                            ax[v].set_ylabel('x')
                        fig.savefig(output_path+f"/active_plumes_ens{j}_test{i}.png")
                        plt.close()
                    else:
                        nrmse_plume = np.inf

                    print('NRMSE', nrmse)
                    print('MSE', mse)
                    print('EVR_recon', evr)
                    print('SSIM', SSIM)
                    print('NRMSE plume', nrmse_plume)

                    # Full path for saving the file
                    output_file = 'ESN_test_metrics_ens%i_test%i.json' % (j,i)

                    output_path_met = os.path.join(output_path, output_file)

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

        # Full path for saving the file
        output_file_all = 'ESN_test_metrics_ens%i_all.json' % j

        output_path_met_all = os.path.join(output_path, output_file_all)

        metrics_ens = {
        "ensemble": j,
        "mean PH": np.mean(PH),
        "lower PH": np.quantile(PH, 0.75),
        "uppper PH": np.quantile(PH, 0.25),
        "median PH": np.median(PH),
        }

        with open(output_path_met_all, "w") as file:
            json.dump(metrics_ens, file, indent=4)

        mean_nrmse_error = np.mean(nrmse_error, axis=0)
        fig, ax = plt.subplots(1, figsize=(8,6))
        ax.plot(xx, mean_nrmse_error)
        ax.grid()
        ax.set_xlabel('LT')
        ax.set_ylabel('mean nrmse error for ensemble')
        fig.savefig(output_path+f"meanerror_ensemble{j}.png")

                        
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

# Full path for saving the file
output_file_ALL = 'ESN_test_metrics_all.json' 

output_path_met_ALL = os.path.join(output_path, output_file_ALL)

metrics_ens_ALL = {
"mean PH": np.mean(ens_PH),
"lower PH": np.quantile(ens_PH, 0.75),
"uppper PH": np.quantile(ens_PH, 0.25),
"median PH": np.median(ens_PH),
}

with open(output_path_met_ALL, "w") as file:
    json.dump(metrics_ens_ALL, file, indent=4)


ESN_params = 'ESN_params.json' 

output_ESN_params = os.path.join(output_path, ESN_params)
with open(output_ESN_params, "w") as f:
    json.dump(hyperparams, f, indent=4) 
