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
import json

import sys
sys.stdout.reconfigure(line_buffering=True)

input_path='./ToyData/'
output_path='./ToyData/moderate/'

projection = False

def load_data(file, name):
    with h5py.File(file, 'r') as hf:
        print(name)
        print(hf[name])
        data = np.array(hf[name])

        x = hf['x'][:]  # 1D x-axis
        z = hf['z'][:]  # 1D z-axis
        time = hf['time'][:]  # 1D time vector

    return data, x, z, time

def load_data_2e7(file, name):
    with h5py.File(file, 'r') as hf:
        print(name)
        print(hf[name])
        data = np.array(hf[name])

        x = hf['x'][:]  # 1D x-axis
        z = hf['z'][:]  # 1D z-axis
        time = hf['total_time'][:]  # 1D time vector

    return data, x, z, time

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
        indexes_to_plot = np.array([1, 2, 3, 4, 8, 10] ) -1
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

    return data_reduced, data_reconstructed_reshaped, data_reconstructed, pca, cumulative_explained_variance[-1]

def inverse_POD(data_reduced, pca_):
    data_reconstructed_reshaped = pca_.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data.shape[0], data.shape[1], data.shape[2])
    print('shape of data reconstructed:', np.shape(data_reconstructed))
    return data_reconstructed_reshaped, data_reconstructed 

def plot_reconstruction(original, reconstruction, z_value, t_value, file_str):
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
    fig.savefig(output_path+file_str+'_hovmoller_recon.png')

    fig, ax = plt.subplots(2, figsize=(12,6), tight_layout=True, sharex=True)
    minm = min(np.min(original[:, :, z_value]), np.min(reconstruction[:, :, z_value]))
    maxm = max(np.max(original[:, :, z_value]), np.max(reconstruction[:, :, z_value]))
    print(np.max(original[:, :, z_value]))
    print(minm, maxm)
    c1 = ax[0].pcolormesh(time, x, original[:, :, z_value].T)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('true')
    c2 = ax[1].pcolormesh(time, x, reconstruction[:, :, z_value].T)
    fig.colorbar(c1, ax=ax[1])
    ax[1].set_title('reconstruction')
    for v in range(2):
        ax[v].set_ylabel('x')
    ax[-1].set_xlabel('time')
    fig.savefig(output_path+file_str+'_snapshot_recon.png')

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

def ss_inverse_transform(data, scaler):
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
    elif data.ndim == 3:
        data_reshape = data_set.reshape(-1, 1)
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

#### plume ###
data_names = ['moderate']#['combined']
n_modes = [4]
for index, name in enumerate(data_names):
    #data_set, x, z, time = load_data(input_path+'/plume_wave_dataset.h5', name)
    data_set, x, z, time = load_data(input_path+'/moderate_dataset.h5', name)
    print('shape of dataset', np.shape(data_set))

    noise_level = 0.5
    data_set = add_noise(data_set, noise_level=noise_level)
    fig, ax =plt.subplots(1)
    ax.contourf(data_set[:,:,32].T)
    fig.savefig(output_path+f"/combined_data_noise{noise_level}.png")

    # fit the scaler
    scaling = 'SS'
    if scaling == 'SS':
        print('applying standard scaler')
        data_reshape = data_set.reshape(-1, 1)
        print('shape of data reshaped', np.shape(data_reshape))
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
    
    
    data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_, cev = POD(data_scaled, n_modes[index], name)
    data_reconstructed = ss_inverse_transform(data_reconstructed, scaler)
    plot_reconstruction(data_set, data_reconstructed, 32, 20, name+'_scaled')
    nrmse = NRMSE(data_set, data_reconstructed)
    mse   = MSE(data_set, data_reconstructed)
    evr   = EVR_recon(data_set, data_reconstructed)
    SSIM  = compute_ssim_for_4d(data_set, data_reconstructed)

    print('NRMSE', nrmse)
    print('MSE', mse)
    print('EVR_recon', evr)
    print('SSIM', SSIM)

    # Full path for saving the file
    output_file = name + '_scaled_metrics.json' 

    output_path_met = os.path.join(output_path, output_file)

    metrics = {
    "no. modes": n_modes[index],
    "EVR": evr,
    "MSE": mse,
    "NRMSE": nrmse,
    "SSIM": SSIM,
    "EV from POD": cev,
    }

    with open(output_path_met, "w") as file:
        json.dump(metrics, file, indent=4)

'''
name = 'combined'
data_set, x, z, time_vals = load_data(input_path+'/plume_wave_dataset.h5', name)

projection = projection
if projection:
    print('starting projection since projectiion', projection)
    data_proj = data_set[500:1000, :, :]
    data_set = data_set[:500, :, :]
    time_vals_proj = time_vals[500:1000]
    time_vals = time_vals[:500]
    print('reduced dataset', np.shape(data_set))
    print('reduced time', np.shape(time))
    print('proejction dataset', np.shape(data_proj))
    print(x[0], x[-1])

data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_ = POD(data_set, 64, name)
plot_reconstruction(data_set, data_reconstructed, 32, 20, name)
nrmse = NRMSE(data_set, data_reconstructed)
mse   = MSE(data_set, data_reconstructed)
evr   = EVR_recon(data_set, data_reconstructed)
SSIM  = compute_ssim_for_4d(data_set, data_reconstructed)

print('NRMSE', nrmse)
print('MSE', mse)
print('EVR_recon', evr)
print('SSIM', SSIM)

# Full path for saving the file
output_file = name + '_metrics.json' 

output_path_met = os.path.join(output_path, output_file)

metrics = {
"no. modes": n_modes[index],
"EVR": evr,
"MSE": mse,
"NRMSE": nrmse,
"SSIM": SSIM,
"EV from POD": cumulative_explained_variance[-1],
}

with open(output_path_met, "w") as file:
    json.dump(metrics, file, indent=4)

if projection:
    print('starting projection')
    data_reduced_proj          = transform_POD(data_proj, pca_)
    _, data_reconstructed_proj = inverse_POD(data_reduced_proj, pca_)

    plot_reconstruction_and_error(data_proj, data_reconstructed_proj, 32, 20, time_vals_proj, 'proj_'+name)
    nrmse_proj = NRMSE(data_proj, data_reconstructed_proj)
    mse_proj   = MSE(data_proj, data_reconstructed_proj)
    evr_proj   = EVR_recon(data_proj, data_reconstructed_proj)
    SSIM_proj  = compute_ssim_for_4d(data_proj, data_reconstructed_proj)

    # Full path for saving the file
    output_file = name + '_proj_metrics.json' 

    output_path_met = os.path.join(output_path, output_file)

    metrics = {
    "no. modes": n_modes[index],
    "EVR": evr_proj,
    "MSE": mse_proj,
    "NRMSE": nrmse_proj,
    "SSIM": SSIM_proj,
    "EV from POD": cumulative_explained_variance[-1],
    }

    with open(output_path_met, "w") as file:
        json.dump(metrics, file, indent=4)
'''