import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import time
from scipy.ndimage import label, center_of_mass

### plotting ###
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


def plot_reconstruction_and_error(original, reconstruction, z_value, t_value, x, z, time_vals, names, file_str, type='Recon'):
    abs_error = np.abs(original-reconstruction)
    residual  = original - reconstruction
    if original.ndim == 3: #len(time_vals), len(x), len(z)
        fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
        minm = min(np.min(original[t_value, :, :]), np.min(reconstruction[t_value, :, :]))
        maxm = max(np.max(original[t_value, :, :]), np.max(reconstruction[t_value, :, :]))
        c1 = ax[0].pcolormesh(x, z, original[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('true')
        c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c2, ax=ax[1])
        ax[1].set_title('reconstruction')
        c3 = ax[2].pcolormesh(x, z, abs_error[t_value,:,:].T, cmap='Reds')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error')
        for v in range(3):
            ax[v].set_ylabel('z')
        ax[-1].set_xlabel('x')
        fig.savefig(file_str+'_hovmoller_recon_error.png')
        plt.close()

        fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
        minm = min(np.min(original[:, :, z_value]), np.min(reconstruction[:, :, z_value]))
        maxm = max(np.max(original[:, :, z_value]), np.max(reconstruction[:, :, z_value]))
        print(np.max(original[:, :, z_value]))
        print(minm, maxm)
        c1 = ax[0].pcolormesh(time_vals, x, original[:, :, z_value].T)
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('true')
        c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value].T)
        fig.colorbar(c2, ax=ax[1])
        ax[1].set_title('reconstruction')
        c3 = ax[2].pcolormesh(time_vals, x, abs_error[:, :, z_value].T, cmap='Reds')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error')
        for v in range(2):
            ax[v].set_ylabel('x')
        ax[-1].set_xlabel('time')
        fig.savefig(file_str+'_snapshot_recon_error.png')
        plt.close()

        fig, ax = plt.subplots(3, figsize=(12,6), tight_layout=True, sharex=True)
        minm = min(np.min(original[t_value, :, :]), np.min(reconstruction[t_value, :, :]))
        maxm = max(np.max(original[t_value, :, :]), np.max(reconstruction[t_value, :, :]))
        vmax_res = np.max(np.abs(residual[t_value,:,:]))  # Get maximum absolute value
        vmin_res = -vmax_res
        c1 = ax[0].pcolormesh(x, z, original[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('true', fontsize=18)
        c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c2, ax=ax[1])
        ax[1].set_title('reconstruction', fontsize=18)
        c3 = ax[2].pcolormesh(x, z, residual[t_value,:,:].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error', fontsize=18)
        for v in range(3):
            ax[v].set_ylabel('z', fontsize=16)
            ax[v].tick_params(axis='both', labelsize=12)
        ax[-1].set_xlabel('x', fontsize=16)
        fig.savefig(file_str+'_snapshot_recon_residual.png')
        plt.close()

        fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
        minm = min(np.min(original[:, :, z_value]), np.min(reconstruction[:, :, z_value]))
        maxm = max(np.max(original[:, :, z_value]), np.max(reconstruction[:, :, z_value]))
        vmax_res = np.max(np.abs(residual[:,:,z_value]))  # Get maximum absolute value
        vmin_res = -vmax_res
        print(minm, maxm)
        print("time shape:", np.shape(time_vals))
        print("x shape:", np.shape(x))
        print("original[:, :, z_value] shape:", original[:, :, z_value].T.shape)
        c1 = ax[0].pcolormesh(time_vals, x, original[:, :, z_value].T, vmin=minm, vmax=maxm)
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('true', fontsize=18)
        c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value].T, vmin=minm, vmax=maxm)
        fig.colorbar(c2, ax=ax[1])
        ax[1].set_title('reconstruction', fontsize=18)
        c3 = ax[2].pcolormesh(time_vals, x,  residual[:,:,z_value].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error', fontsize=18)
        for v in range(3):
            ax[v].set_ylabel('x', fontsize=16)
            ax[v].tick_params(axis='both', labelsize=12)
        ax[-1].set_xlabel('time', fontsize=16)
        fig.savefig(file_str+'_hovmoller_recon_residual.png')
        plt.close()
    
    elif original.ndim == 4: #len(time_vals), len(x), len(z), var
        for i in range(original.shape[3]):
            name = names[i]
            print(name)
            fig, ax = plt.subplots(3, figsize=(12,6), tight_layout=True, sharex=True)
            minm = min(np.min(original[t_value, :, :, i]), np.min(reconstruction[t_value, :, :, i]))
            maxm = max(np.max(original[t_value, :, :, i]), np.max(reconstruction[t_value, :, :, i]))
            c1 = ax[0].pcolormesh(x, z, original[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            if type == 'Recon':
                ax[0].set_title('True')
            elif type == 'CAE':
                ax[0].set_title('CAE Reconstruction (True)')
            elif type == 'POD':
                ax[0].set_title('POD Reconstruction (True)')
            c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            if type == 'Recon':
                ax[1].set_title('Reconstruction')
            elif type == 'CAE':
                ax[1].set_title('CAE Reconstruction (ESN)')
            elif type == 'POD':
                ax[1].set_title('POD Reconstruction (ESN)')
            c3 = ax[2].pcolormesh(x, z, abs_error[t_value,:,:, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('Error')
            for v in range(2):
                ax[v].set_ylabel('z')
            ax[-1].set_xlabel('x')
            fig.savefig(file_str+name+'_snapshot_recon_error.png')
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
            if type == 'Recon':
                ax[0].set_title('Reconstruction')
            elif type == 'CAE':
                ax[0].set_title('CAE Reconstruction (ESN)')
            elif type == 'POD':
                ax[0].set_title('POD Reconstruction (ESN)')
            c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            if type == 'Recon':
                ax[1].set_title('Reconstruction')
            elif type == 'CAE':
                ax[1].set_title('CAE Reconstruction (ESN)')
            elif type == 'POD':
                ax[1].set_title('POD Reconstruction (ESN)')
            c3 = ax[2].pcolormesh(time_vals, x,  abs_error[:,:,z_value, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('Error')
            for v in range(2):
                ax[v].set_ylabel('x')
            ax[-1].set_xlabel('time')
            fig.savefig(file_str+name+'_hovmoller_recon_error.png')
            plt.close()

            fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
            minm = min(np.min(original[t_value, :, :, i]), np.min(reconstruction[t_value, :, :, i]))
            maxm = max(np.max(original[t_value, :, :, i]), np.max(reconstruction[t_value, :, :, i]))
            vmax_res = np.max(np.abs(residual[t_value,:,:,i]))  # Get maximum absolute value
            vmin_res = -vmax_res
            c1 = ax[0].pcolormesh(x, z, original[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            if type == 'Recon':
                ax[0].set_title('Reconstruction', fontsize=18)
            elif type == 'CAE':
                ax[0].set_title('CAE Reconstruction (ESN)', fontsize=18)
            elif type == 'POD':
                ax[0].set_title('POD Reconstruction (ESN)', fontsize=18)
            c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            if type == 'Recon':
                ax[1].set_title('Reconstruction', fontsize=18)
            elif type == 'CAE':
                ax[1].set_title('CAE Reconstruction (ESN)', fontsize=18)
            elif type == 'POD':
                ax[1].set_title('POD Reconstruction (ESN)', fontsize=18)
            c3 = ax[2].pcolormesh(x, z, residual[t_value,:,:, i].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('Error', fontsize=18)
            for v in range(3):
                ax[v].set_ylabel('z', fontsize=16)
                ax[v].tick_params(axis='both', labelsize=12)
            ax[-1].set_xlabel('x', fontsize=16)
            fig.savefig(file_str+name+'_snapshot_recon_residual.png')
            plt.close()

            fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
            minm = min(np.min(original[:, :, z_value,i]), np.min(reconstruction[:, :, z_value,i]))
            maxm = max(np.max(original[:, :, z_value,i]), np.max(reconstruction[:, :, z_value,i]))
            vmax_res = np.max(np.abs(residual[:,:,z_value,i]))  # Get maximum absolute value
            vmin_res = -vmax_res
            print(np.max(original[:, :, z_value,i]))
            print(minm, maxm)
            print("time shape:", np.shape(time_vals))
            print("x shape:", np.shape(x))
            print("original[:, :, z_value] shape:", original[:, :, z_value,i].T.shape)
            c1 = ax[0].pcolormesh(time_vals, x, original[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            if type == 'Recon':
                ax[0].set_title('Reconstruction', fontsize=18)
            elif type == 'CAE':
                ax[0].set_title('CAE Reconstruction (ESN)', fontsize=18)
            elif type == 'POD':
                ax[0].set_title('POD Reconstruction (ESN)', fontsize=18)
            c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            if type == 'Recon':
                ax[1].set_title('Reconstruction', fontsize=18)
            elif type == 'CAE':
                ax[1].set_title('CAE Reconstruction (ESN)', fontsize=18)
            elif type == 'POD':
                ax[1].set_title('POD Reconstruction (ESN)', fontsize=18)
            c3 = ax[2].pcolormesh(time_vals, x,  residual[:,:,z_value, i].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('Error', fontsize=18)
            for v in range(3):
                ax[v].set_ylabel('x', fontsize=16)
                ax[v].tick_params(axis='both', labelsize=12)
            ax[-1].set_xlabel('Time', fontsize=16)
            fig.savefig(file_str+name+'_hovmoller_recon_residual.png')
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

def NRMSE_per_channel(true, pred, reduction='mean'):
    """
    Compute NRMSE per channel and optionally average them.
    
    Parameters:
        true (ndarray): Ground truth data of shape (time, width, height, channels)
        pred (ndarray): Predicted data of same shape
        reduction (str): 'mean', 'sum', or 'none' — how to aggregate channel NRMSEs
    
    Returns:
        float or list: Aggregated NRMSE or list of NRMSEs per channel
    """
    assert true.shape == pred.shape, "Input shapes must match"
    assert true.ndim == 4, "Expected input shape (time, width, height, channels)"
    
    n_channels = true.shape[-1]
    nrmse_per_channel = []

    for c in range(n_channels):
        true_c = true[..., c].ravel()
        pred_c = pred[..., c].ravel()
        
        rmse = np.sqrt(mean_squared_error(true_c, pred_c))
        std = np.std(true_c)
        nrmse = rmse / std if std != 0 else np.nan  # Avoid divide by zero
        nrmse_per_channel.append(nrmse)
    
    if reduction == 'mean':
        return np.nanmean(nrmse_per_channel)
    elif reduction == 'sum':
        return np.nansum(nrmse_per_channel)
    elif reduction == 'none':
        return nrmse_per_channel
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

def NRMSE_per_channel_masked(true, pred, mask, global_stds, reduction='mean'):
    """
    Compute NRMSE per channel using only masked points, normalized by global stds.
    
    Parameters:
        true (ndarray): Ground truth data, shape (time, width, height, channels)
        pred (ndarray): Predicted data, same shape as true
        mask (ndarray): Boolean mask, shape (time, width, height)
        global_stds (list or array): Global std for each channel
        reduction (str): 'mean', 'sum', or 'none' to aggregate channel NRMSEs

    Returns:
        float or list: Aggregated NRMSE or list of NRMSEs per channel
    """
    assert true.shape == pred.shape, "Input shapes must match"
    assert true.ndim == 4, "Expected input shape (time, width, height, channels)"
    assert mask.shape == true.shape[:3], "Mask shape must match (time, width, height)"
    assert len(global_stds) == true.shape[-1], "One std per channel required"

    n_channels = true.shape[-1]
    nrmse_per_channel = []

    for c in range(n_channels):
        true_c = true[..., c]
        pred_c = pred[..., c]
        
        # Select only masked points
        true_masked = true_c[mask]
        pred_masked = pred_c[mask]
        
        if true_masked.size == 0:
            nrmse_per_channel.append(np.nan)
            continue

        rmse = np.sqrt(np.mean((true_masked - pred_masked) ** 2))
        std = global_stds[c]
        nrmse = rmse / (std + 1e-6)  # small epsilon to avoid div by zero
        nrmse_per_channel.append(nrmse)

    if reduction == 'mean':
        return np.nanmean(nrmse_per_channel)
    elif reduction == 'sum':
        return np.nansum(nrmse_per_channel)
    elif reduction == 'none':
        return nrmse_per_channel
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    
def compute_nrmse_per_timestep_variable(original_data, reconstructed_data, normalize_by="std"):
    """
    Compute NRMSE for each timestep and variable.

    Args:
        original_data: np.array, shape (T, X, Z, V)
        reconstructed_data: np.array, shape (T, X, Z, V)
        normalize_by: "range" or "std" for normalization

    Returns:
        nrmse: np.array, shape (T, V)
    """
    if original_data.ndim != 4 or reconstructed_data.ndim != 4:
        print(f"Error: Expected 4D arrays, got shapes {original_data.shape} and {reconstructed_data.shape}.")
    elif original_data.shape != reconstructed_data.shape:
        print(f"Warning: Arrays are 4D but shapes do not match: {original_data.shape} vs {reconstructed_data.shape}.")
    else:
        print(f"Input check passed: both arrays are 4D and have shape {original_data.shape}.")
    
    T, X, Z, V = original_data.shape
    nrmse = np.zeros((T, V))

    for t in range(T):
        for v in range(V):
            true_tv = original_data[t, :, :, v]
            pred_tv = reconstructed_data[t, :, :, v]
            error = true_tv - pred_tv
            rmse = np.sqrt(np.mean(error**2))

            if normalize_by == "range":
                norm = np.max(true_tv) - np.min(true_tv)
            elif normalize_by == "std":
                norm = np.std(true_tv)
            else:
                raise ValueError("normalize_by must be 'range' or 'std'")

            nrmse[t, v] = rmse / (norm + 1e-8)  # avoid div by zero

    return nrmse

def find_prediction_horizon(nrmse, threshold):
    """
    Find the first timestep where any variable's NRMSE exceeds the threshold.

    Args:
        nrmse: np.array, shape (T, V)
        threshold: float

    Returns:
        horizon: int, first timestep where any variable exceeds threshold
    """
    T, V = nrmse.shape
    for t in range(T):
        if np.any(nrmse[t, :] > threshold):
            return t
    return T  # if no exceedance, return full horizon

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
    
    active_array = np.zeros((original_data.shape[0], original_data.shape[1], len(z)))
    active_array[mask] = 1
    active_array_reconstructed = np.zeros((original_data.shape[0], original_data.shape[1], len(z)))
    active_array_reconstructed[mask_reconstructed] = 1
    
    # Expand the mask to cover all features (optional, depending on use case)
    mask_expanded       =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
    mask_expanded_recon =  np.repeat(mask_reconstructed[:, :, :, np.newaxis], 4, axis=-1) # Shape: (256, 64, 1)
    
    return active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon

def ss_transform(data, scaler):
    print('implementing standard scaler')
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
    else:
        print('data needs to be 4 dimensions')
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
    print('inversing standard scaler')
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


#### Analysis ####
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

def extract_plume_features(new_array, z_range=(5, 10)):
    time_steps, nx, nz = new_array.shape

    # Storage for features
    features = np.zeros((time_steps, 4))  # num, mean_size, mean_x, x_spread

    # Use 8-connectivity for labeling
    structure = np.ones((3, 3))

    for t in range(time_steps):
        # Slice relevant z-range
        z_start, z_end = z_range
        subgrid = new_array[t, :, z_start:z_end]  # shape (64, 5)

        # Make binary
        binary_mask = (subgrid > 0).astype(int)

        # Connected component labeling
        labeled, num_plumes = label(binary_mask, structure=structure)

        # If no plumes, fill with zeros
        if num_plumes == 0:
            features[t] = [0, 0, 0, 0]
            continue

        # Plume sizes
        sizes = [(labeled == i).sum() for i in range(1, num_plumes + 1)]

        # Centroids: returns (x, z) per plume
        centroids = center_of_mass(binary_mask, labeled, range(1, num_plumes + 1))
        x_centroids = [c[0] for c in centroids]

        # Compute final features
        mean_size = np.mean(sizes)
        mean_x = np.mean(x_centroids)
        x_spread = np.std(x_centroids)

        # Store
        features[t] = [num_plumes, mean_size, mean_x, x_spread]

    return features  # shape (time, 4)