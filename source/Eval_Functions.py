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
                ax[0].set_title('CAE Reconstruction (True)')
            elif type == 'POD':
                ax[0].set_title('POD Reconstruction (True)')
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
            c1 = ax[0].pcolormesh(x, z, original[t_value,:,:,i].T, vmin=minm, vmax=maxm, rasterized=True, zorder=0, edgecolors='none')
            fig.colorbar(c1, ax=ax[0], label=names[v])
            if type == 'Recon':
                ax[0].set_title('Reconstruction', fontsize=18)
            elif type == 'CAE':
                ax[0].set_title('CAE Reconstruction (True)', fontsize=18)
            elif type == 'POD':
                ax[0].set_title('POD Reconstruction (True)', fontsize=18)
            c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:,i].T, vmin=minm, vmax=maxm, rasterized=True, zorder=0, edgecolors='none')
            fig.colorbar(c2, ax=ax[1], label=names[v])
            if type == 'Recon':
                ax[1].set_title('Reconstruction', fontsize=18)
            elif type == 'CAE':
                ax[1].set_title('CAE Reconstruction (ESN)', fontsize=18)
            elif type == 'POD':
                ax[1].set_title('POD Reconstruction (ESN)', fontsize=18)
            c3 = ax[2].pcolormesh(x, z, residual[t_value,:,:, i].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res, rasterized=True, zorder=0, edgecolors='none')
            fig.colorbar(c3, ax=ax[2], label=names[v])
            ax[2].set_title('Error', fontsize=18)
            for v in range(3):
                ax[v].set_ylabel('z', fontsize=16)
                ax[v].tick_params(axis='both', labelsize=12)
            ax[-1].set_xlabel('x', fontsize=16)
            fig.savefig(file_str+name+'_snapshot_recon_residual.png')
            fig.savefig(file_str+name+'_snapshot_recon_residual.eps', format='eps')
            fig.savefig(file_str+name+'_snapshot_recon_residual.pdf', format='pdf')
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
            c1 = ax[0].pcolormesh(time_vals, x, original[:, :, z_value, i].T, vmin=minm, vmax=maxm, rasterized=True, zorder=0, edgecolors='none')
            fig.colorbar(c1, ax=ax[0], label=names[v])
            if type == 'Recon':
                ax[0].set_title('Reconstruction', fontsize=18)
            elif type == 'CAE':
                ax[0].set_title('CAE Reconstruction (True)', fontsize=18)
            elif type == 'POD':
                ax[0].set_title('POD Reconstruction (True)', fontsize=18)
            c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value, i].T, vmin=minm, vmax=maxm, rasterized=True, zorder=0, edgecolors='none')
            fig.colorbar(c2, ax=ax[1], label=names[v])
            if type == 'Recon':
                ax[1].set_title('Reconstruction', fontsize=18)
            elif type == 'CAE':
                ax[1].set_title('CAE Reconstruction (ESN)', fontsize=18)
            elif type == 'POD':
                ax[1].set_title('POD Reconstruction (ESN)', fontsize=18)
            c3 = ax[2].pcolormesh(time_vals, x,  residual[:,:,z_value, i].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res, rasterized=True, zorder=0, edgecolors='none')
            fig.colorbar(c3, ax=ax[2], label=names[v])
            ax[2].set_title('Error', fontsize=18)
            for v in range(3):
                ax[v].set_ylabel('x', fontsize=16)
                ax[v].tick_params(axis='both', labelsize=12)
            ax[-1].set_xlabel('Time [Lyapunov Times]', fontsize=16)
            fig.savefig(file_str+name+'_hovmoller_recon_residual.png')
            fig.savefig(file_str+name+'_hovmoller_recon_residual.eps', format='eps')
            fig.savefig(file_str+name+'_hovmoller_recon_residual.pdf', format='pdf')
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

def active_array_truth(original_data, z):
    beta = 1.201
    alpha = 3.0
    T = original_data[:,:,:,3] - beta*z
    q_s = np.exp(alpha*T)
    rh = original_data[:,:,:,0]/q_s
    mean_b = np.mean(original_data[:,:,:,3], axis=1, keepdims=True)
    b_anom = original_data[:,:,:,3] - mean_b
    w = original_data[:,:,:,1]
    
    mask = (rh[:, :, :] >= 1) & (w[:, :, :] > 0) & (b_anom[:, :, :] > 0)

    active_array = np.zeros((original_data.shape[0], original_data.shape[1], len(z)))
    active_array[mask] = 1
    
    # Expand the mask to cover all features (optional, depending on use case)
    mask_expanded       =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
    
    return active_array, mask_expanded, rh, w, b_anom


def active_array_calc(original_data, reconstructed_data, z):
    beta = 1.201
    alpha = 3.0
    T = original_data[:,:,:,3] - beta*z
    T_reconstructed = reconstructed_data[:,:,:,3] - beta*z
    q_s = np.exp(alpha*T)
    q_s_reconstructed = np.exp(alpha*T_reconstructed)
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

def active_array_calc_softer(original_data, reconstructed_data, z, RH_threshold=0.8, w_threshold=0, b_threshold=0):
    beta = 1.201
    alpha = 3.0
    T = original_data[:,:,:,3] - beta*z
    T_reconstructed = reconstructed_data[:,:,:,3] - beta*z
    q_s = np.exp(alpha*T)
    q_s_reconstructed = np.exp(alpha*T_reconstructed)
    rh = original_data[:,:,:,0]/q_s
    rh_reconstructed = reconstructed_data[:,:,:,0]/q_s_reconstructed
    mean_b = np.mean(original_data[:,:,:,3], axis=1, keepdims=True)
    mean_b_reconstructed= np.mean(reconstructed_data[:,:,:,3], axis=1, keepdims=True)
    b_anom = original_data[:,:,:,3] - mean_b
    b_anom_reconstructed = reconstructed_data[:,:,:,3] - mean_b_reconstructed
    w = original_data[:,:,:,1]
    w_reconstructed = reconstructed_data[:,:,:,1]
    
    mask = (rh[:, :, :] >= 1) & (w[:, :, :] > 0) & (b_anom[:, :, :] > 0)
    mask_reconstructed = (rh_reconstructed[:, :, :] >= RH_threshold) & (w_reconstructed[:, :, :] > w_threshold) & (b_anom_reconstructed[:, :, :] > b_threshold)
    
    active_array = np.zeros((original_data.shape[0], original_data.shape[1], len(z)))
    active_array[mask] = 1
    active_array_reconstructed = np.zeros((original_data.shape[0], original_data.shape[1], len(z)))
    active_array_reconstructed[mask_reconstructed] = 1
    
    # Expand the mask to cover all features (optional, depending on use case)
    mask_expanded       =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
    mask_expanded_recon =  np.repeat(mask_reconstructed[:, :, :, np.newaxis], 4, axis=-1) # Shape: (256, 64, 1)
    
    return active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon


def active_array_calc_prob(original_data, reconstructed_data, z, rh_min, rh_max, w_min, w_max, b_anom_min, b_anom_max, plume_score_threshold):
    beta = 1.201
    alpha = 3.0
    T = original_data[:,:,:,3] - beta*z
    T_reconstructed = reconstructed_data[:,:,:,3] - beta*z
    q_s = np.exp(alpha*T)
    q_s_reconstructed = np.exp(alpha*T_reconstructed)
    rh = original_data[:,:,:,0]/q_s
    rh_reconstructed = reconstructed_data[:,:,:,0]/q_s_reconstructed
    mean_b = np.mean(original_data[:,:,:,3], axis=1, keepdims=True)
    mean_b_reconstructed= np.mean(reconstructed_data[:,:,:,3], axis=1, keepdims=True)
    b_anom = original_data[:,:,:,3] - mean_b
    b_anom_reconstructed = reconstructed_data[:,:,:,3] - mean_b_reconstructed
    w = original_data[:,:,:,1]
    w_reconstructed = reconstructed_data[:,:,:,1]

    del T, T_reconstructed
    
    RH_scaled = (rh - rh_min)/(rh_max - rh_min)
    w_scaled  = (w - w_min)/(w_max - w_min)
    b_scaled  = (b_anom - b_anom_min)/(b_anom_max - b_anom_min)

    RH_scaled_reconstructed = (rh_reconstructed - rh_min)/(rh_max - rh_min)
    w_scaled_reconstructed  = (w_reconstructed - w_min)/(w_max - w_min)
    b_scaled_reconstructed  = (b_anom_reconstructed - b_anom_min)/(b_anom_max - b_anom_min)

    RH_threshold = 0.80 # just below 25th %#0.389 # minm
    RH_threshold_scaled = (RH_threshold - rh_min)/(rh_max - rh_min)
    w_threshold = 0 #-0.036 #minm
    w_threshold_scaled = (w_threshold - w_min) / (w_max - w_min)
    b_threshold = 0 #-0.014 #minm
    b_threshold_scaled = (b_threshold - b_anom_min) / (b_anom_max - b_anom_min)
    
    print(f"RH min: {rh_min}, max: {rh_max}")
    print(f"RH threshold scaled: {RH_threshold_scaled}")

    RH_clip = np.clip(RH_scaled - RH_threshold_scaled, 0, None)
    w_clip = np.clip(w_scaled - w_threshold_scaled, 0, None) #w_clip = np.clip(w_scaled, 0, None)  # set negative w to 0
    b_clip = np.clip(b_scaled, 0, None)  # set negative b to 0

    RH_clip_reconstructed = np.clip(RH_scaled_reconstructed - RH_threshold_scaled, 0, None)
    w_clip_reconstructed = np.clip(w_scaled_reconstructed, 0, None)  # set negative w to 0
    b_clip_reconstructed = np.clip(b_scaled_reconstructed, 0, None)  # set negative b to 0

    plume_score = RH_clip + w_clip + b_clip
    plume_score_reconstructed = RH_clip_reconstructed + w_clip_reconstructed + b_clip_reconstructed
    plume_score /= 3
    plume_score_reconstructed /= 3

    mask = plume_score > plume_score_threshold
    mask_reconstructed = plume_score_reconstructed > plume_score_threshold #0.6 way too much masked

    active_array = np.zeros((original_data.shape[0], original_data.shape[1], len(z)))
    active_array[mask] = 1
    active_array_reconstructed = np.zeros((original_data.shape[0], original_data.shape[1], len(z)))
    active_array_reconstructed[mask_reconstructed] = 1
    
    # Expand the mask to cover all features (optional, depending on use case)
    mask_expanded       =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
    mask_expanded_recon =  np.repeat(mask_reconstructed[:, :, :, np.newaxis], 4, axis=-1) # Shape: (256, 64, 1)
    
    print(f"Plume score min/max original: {plume_score.min()}, {plume_score.max()}")
    print(f"Plume score min/max reconstructed: {plume_score_reconstructed.min()}, {plume_score_reconstructed.max()}")

    # Flatten the plume score array (use the original or reconstructed version)
    flat_scores = plume_score_reconstructed.flatten()

    # Sort the values
    sorted_scores = np.sort(flat_scores)

    # Compute cumulative count
    cumulative = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)

    # Plot
    fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
    ax.plot(sorted_scores, cumulative, label='Cumulative Distribution')
    ax.set_xlabel('Plume Score')
    ax.set_ylabel('Cumulative Fraction')
    ax.grid(True)
    fig.savefig('Ra2e8/POD/Thesis/scaler/snapshots11200/modes64/plume_score_dist.png')

    return active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon

def plume_positions_from_active(active_array, x_full, x_domain=(0,20), max_plumes=3, z_range=(5,10)):
    x_min, x_max = x_domain
    # Assume active_array.shape = (time, nx, nz)
    time_steps, nx, nz = active_array.shape[1], active_array.shape[2]

    # Storage for features
    positions = np.zeros((time_steps, max_plumes), dtype=float)  # x-centroids of up to 6 plumes

    # Use 8-connectivity for labeling
    structure = np.ones((3, 3))

    # Trim to multiples of 4
    nx_trim = (nx // 4) * 4
    nz_trim = (nz // 4) * 4
    active_trimmed = active_array[:, :nx_trim, :nz_trim]

    # Reshape into 4x4 blocks
    blocks = active_trimmed.reshape(len(time_steps), nx_trim//4, 4, nz_trim//4, 4)

    # Take max (equivalent to np.any) over the 4x4 subgrid
    new_array = blocks.max(axis=(2, 4))

    x_downsample = x_full[:nx_trim:4]

    for t in range(time_steps):
        # Slice relevant z-range
        z_start, z_end = z_range
        subgrid = new_array[t, :, z_start:z_end]
        
        # Make binary
        binary_mask = (subgrid > 0).astype(int)

        # Connected component labeling
        labeled, num_plumes = label(binary_mask, structure=structure)
        if num_plumes == 0:
            continue

        
        #  Centroids in downsampled index space
        centroids = center_of_mass(binary_mask, labeled, range(1, num_plumes + 1))
        x_centroids_idx = np.array([c[0] for c in centroids])

        # Map downsampled centroid index to physical x value
        # Assume evenly spaced downsampled x_full for new_array
        x_spacing = x_full[1] - x_full[0]  # spacing of full grid
        x_centroids_val = x_full[(x_centroids_idx * (len(x_full)/nx)).astype(int)]

        positions[t, :] = x_centroids_val

    return positions

def ss_transform(data, scaler):
    print('implementing standard scaler')
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
    elif data.ndim == 2:
        print('data already 2D')
        data_reshape = data
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

def _as_plumes(feats, num_plumes=3):
    """
    feats: (T, 3*num_plumes) laid out as [cos, sin, KE, cos, sin, KE, ...]
    returns: (T, num_plumes, 3) -> last axis = [cos, sin, KE]
    """
    cos = feats[:, 0::3]
    sin = feats[:, 1::3]
    ke  = feats[:, 2::3]
    return np.stack([cos, sin, ke], axis=2)

def plume_mse_per_plume(true_feats, pred_feats):
    """
    MSE per plume at each time.
    returns: (T, 3)
    """
    t3 = _as_plumes(true_feats)   # (T,3,3)
    p3 = _as_plumes(pred_feats)   # (T,3,3)
    return np.mean((t3 - p3)**2, axis=2)

def plume_mse_overall(true_feats, pred_feats):
    """
    Single scalar MSE averaged over time, plumes, and features.
    """
    t3 = _as_plumes(true_feats)
    p3 = _as_plumes(pred_feats)
    return np.mean((t3 - p3)**2)

def angular_error_per_plume(true_feats, pred_feats):
    """
    Angular difference (radians) between true/predicted positions per plume/time.
    returns: (T, 3)
    """
    t3 = _as_plumes(true_feats)
    p3 = _as_plumes(pred_feats)
    dot = t3[:,:,0]*p3[:,:,0] + t3[:,:,1]*p3[:,:,1]   # cos_t*cos_p + sin_t*sin_p
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def to_xpos(feats, num_plumes=3, x_min=0, x_max=20):
    """
    feats: (T, 3*num_plumes) laid out as [cos, sin, KE, cos, sin, KE, ...]
    returns: x positions (T, num_plumes), KE (T, num_plumes)
    """
    
    plume_feats_pp = _as_plumes(feats, num_plumes=3) # (T, num_plumes, 3) -> last axis = [cos, sin, KE]
    
    cos = plume_feats_pp[:,:,0]
    sin = plume_feats_pp[:,:,1]
    ke  = plume_feats_pp[:,:,2]

    angles = np.arctan2(sin, cos)  # [-π, π]
    angles[angles < 0] += 2*np.pi   # wrap to [0, 2π]

    xpos = x_min + (x_max - x_min) * angles / (2*np.pi)
    return xpos, ke

def bin_intervals(xpos, ke, N_int):
    """
    xpos, ke: (T, num_plumes)
    returns: list of arrays, one per interval
    """
    T = xpos.shape[0]
    n_bins = T // N_int
    bins = []
    for b in range(n_bins):
        start, end = b*N_int, (b+1)*N_int
        bins.append((xpos[start:end], ke[start:end]))
    return bins

def plume_metrics(true_feats, pred_feats, num_plumes=3, delta_x=1.0, 
                         x_min=0, x_max=20, n_bins=10):
    """
    true_feats, pred_feats: (T, 3*num_plumes) arrays
    returns:
        precision: (n_bins,)
        recall:    (n_bins,)
        f1:        (n_bins,)
    """

    T         = true_feats.shape[0]
    bin_edges = np.linspace(0, T, n_bins+1, dtype=int) 

    t3 = _as_plumes(true_feats)
    p3 = _as_plumes(pred_feats)

    x_true, ke_true = to_xpos(t3)
    x_pred, ke_pred = to_xpos(p3)

    precisions, recalls, f1s = [], [], []

    for b in range(n_bins):
        start, end = bin_edges[b], bin_edges[b+1]
        print(f"starting bin {b} from {start} to {end}")

        # gather plumes across the interval
        xt = x_true[start:end].reshape(-1)
        xp = x_pred[start:end].reshape(-1)
        kt = ke_true[start:end].reshape(-1)
        kp = ke_pred[start:end].reshape(-1)

        # keep only "active" plumes
        xt = xt[kt > 0]
        xp = xp[kp > 0]

        hits = 0
        false_pos = 0

        # for each true plume, see if pred is nearby
        for v in xt:
            if np.any(np.abs(xp - v) <= delta_x):
                hits += 1

        # false positives = preds without nearby true
        for v in xp:
            if not np.any(np.abs(xt - v) <= delta_x):
                false_pos += 1

        tp = hits
        fp = false_pos
        fn = len(xt) - hits

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return np.array(precisions), np.array(recalls), np.array(f1s) # (n_bins, n_bins, n_bins)


from Plotting_Functions import clean_strengths
def extract_plume_positions(features, x_domain=(0,20), max_plumes=3, threshold_predictions=False, KE_threshold=0.00005):
    """
    Extract plume x-positions over a full interval.

    Parameters
    ----------
    features : array, shape (timesteps, 3*P)
        Features for one interval, (cos, sin, KE) triplets for P plume slots.
    x_domain : tuple (x_min, x_max)
        Domain of plume positions.
    KE_threshold : float
        Minimum KE to count as active plume.
    max_plumes : int
        Maximum number of plume slots to return per timestep.

    Returns
    -------
    positions : array, shape (timesteps, max_plumes)
        X-positions of active plumes. If fewer than max_plumes
        plumes exist at a timestep, fill with NaN.
    """
    x_min, x_max = x_domain
    T = features.shape[0]

    feats = features.copy()
    # Only clean predictions if requested
    if threshold_predictions:
        feats = clean_strengths(feats, KE_threshold, KE_threshold)

    x_positions = np.full((T, 3), np.nan)

    for v in range(3):
        cos_vals = feats[:, v*3]      # cos
        sin_vals = feats[:, v*3 + 1]  # sin
        strength = feats[:, v*3 + 2]  # KE

        # Recover downsampled x position
        angles = np.arctan2(sin_vals, cos_vals)  # [-π, π]
        angles[angles < 0] += 2*np.pi  # wrap negative angles
        x_vals = x_min + (x_max - x_min) * angles / (2*np.pi)

        valid_mask = strength > 0

        x_positions[valid_mask,v] = x_vals[valid_mask]

    return x_positions

from collections import Counter
def score_plumes(truth_positions, pred_positions, N_lyap,
                 checkpoints=(0.5, 1.0, 1.5, 2.0, 3.0),
                 thresholds=(1,3,10),  # very good, good, medium
                 x_domain=(0.0,20.0),
                 weights=None):
    """
    Score predicted vs true plume positions at chosen checkpoints.

    Parameters
    ----------
    truth_positions : array, shape (T, max_plumes)
        Extracted plume x-positions (NaN if no plume).
    pred_positions : array, shape (T, max_plumes)
        Predicted plume x-positions (NaN if no plume).
    N_lyap : int
        Number of timesteps per LT.
    checkpoints : list of floats
        Lead times (in LTs) to evaluate (e.g. [0.5, 1.0, 2.0]).
    thresholds : tuple
        (very_good, good, medium) in number of indexes

    Returns
    -------
    all_scores : dict
        {checkpoint_time (in steps): list of scores for each plume}
    """

    if weights is None:
        weights = {
            "very good": 3,
            "good": 2,
            "medium": 1,
            "FN": -1,
            "FP": -1,   
            "TN": 0
        }

    x_min, x_max = x_domain
    L = x_max - x_min  # domain length

    def circ_dist(a, b):
        """Compute minimum distance in periodic domain."""
        d = np.abs(a - b)
        return np.minimum(d, L - d)

    T = truth_positions.shape[0]

    def match_scores(true_slots, pred_slots):
        labels = []
        matched_true_indices = set()

        # 1. Match predicted plumes to closest true plume
        for pp in pred_slots:
            if np.isnan(pp):
                continue  # skip empty prediction slot

            # compute distances to all true plumes
            valid_indices = [i for i, tp in enumerate(true_slots) if not np.isnan(tp)]
            if len(valid_indices) == 0:
                labels.append("FP")
                continue

            dists = np.array([circ_dist(pp, true_slots[i]) for i in valid_indices])
            min_idx = valid_indices[np.argmin(dists)]
            min_dist = dists.min()

            # assign category based on thresholds
            if min_dist <= thresholds[0]:
                labels.append("very good")
            elif min_dist <= thresholds[1]:
                labels.append("good")
            elif min_dist <= thresholds[2]:
                labels.append("medium")
            else:
                labels.append("FP")

            # mark as matched only if within medium threshold
            if min_dist <= thresholds[2]:
                matched_true_indices.add(min_idx)

        # 2. Any true plume not matched within medium threshold → FN
        for i, tp in enumerate(true_slots):
            if np.isnan(tp):
                continue
            if i not in matched_true_indices:
                labels.append("FN")

        # 3. TN: slots where both truth and prediction are empty
        for tp, pp in zip(true_slots, pred_slots):
            if np.isnan(tp) and np.isnan(pp):
                labels.append("TN")

        return labels

    result_dict = {}
    for lt in checkpoints:
        ck = int(round(lt * N_lyap))
        if ck >= T:
            continue

        true_slots = truth_positions[ck, :]
        pred_slots = pred_positions[ck, :]

        labels = match_scores(true_slots, pred_slots)
        counts = Counter(labels)
        overall_score = sum([weights[l] for l in labels])

        result_dict[ck] = {"counts": counts, "overall_score": overall_score}

    return result_dict

from collections import Counter
def score_plumes2(truth_positions, pred_positions, N_lyap,
                 checkpoints=(0.5, 1.0, 1.5, 2.0, 3.0),
                 thresholds=(1,2,3,4,5,6,7,8,9,10),  # very good, good, medium
                 x_domain=(0.0,20.0)):
    """
    Score predicted vs true plume positions at chosen checkpoints.

    Parameters
    ----------
    truth_positions : array, shape (T, max_plumes)
        Extracted plume x-positions (NaN if no plume).
    pred_positions : array, shape (T, max_plumes)
        Predicted plume x-positions (NaN if no plume).
    N_lyap : int
        Number of timesteps per LT.
    checkpoints : list of floats
        Lead times (in LTs) to evaluate (e.g. [0.5, 1.0, 2.0]).
    thresholds : tuple
        (very_good, good, medium) in number of indexes

    Returns
    -------
    all_scores : dict
        {checkpoint_time (in steps): list of scores for each plume}
    """

    x_min, x_max = x_domain
    L = x_max - x_min  # domain length

    def circ_dist(a, b):
        """Compute minimum distance in periodic domain."""
        d = np.abs(a - b)
        return np.minimum(d, L - d)

    T = truth_positions.shape[0]

    def match_scores(true_slots, pred_slots):
        labels = []
        matched_true_indices = set()

        # 1. Match predicted plumes to closest true plume
        for pp in pred_slots:
            if np.isnan(pp):
                continue  # skip empty prediction slot

            # compute distances to all true plumes
            valid_indices = [i for i, tp in enumerate(true_slots) if not np.isnan(tp)]
            if len(valid_indices) == 0:
                labels.append("FP")
                continue

            dists = np.array([circ_dist(pp, true_slots[i]) for i in valid_indices])
            min_idx = valid_indices[np.argmin(dists)]
            min_dist = dists.min()

            # assign label based on thresholds
            assigned_label = None
            for th in thresholds:  # thresholds = [1,2,3,...,10]
                if min_dist <= th:
                    assigned_label = th
                    break
            if assigned_label is None:
                labels.append("FP")
            else:
                labels.append(assigned_label)
                matched_true_indices.add(min_idx)  # mark as matched

        # 2. Any true plume not matched within medium threshold → FN
        for i, tp in enumerate(true_slots):
            if np.isnan(tp):
                continue
            if i not in matched_true_indices:
                labels.append("FN")

        # 3. TN: slots where both truth and prediction are empty
        for tp, pp in zip(true_slots, pred_slots):
            if np.isnan(tp) and np.isnan(pp):
                labels.append("TN")

        return labels

    result_dict = {}
    for lt in checkpoints:
        ck = int(round(lt * N_lyap))
        if ck >= T:
            continue

        true_slots = truth_positions[ck, :]
        pred_slots = pred_positions[ck, :]

        labels = match_scores(true_slots, pred_slots)
        counts = Counter(labels)

        result_dict[ck] = {"counts": counts}

    return result_dict



def circ_dist_scalar_array(xq, xval, L):
    """
    Minimum circular distance between scalar xval and array xq (or vice versa).
    xq: array_like (grid or queries)
    xval: scalar
    returns: array of same shape as xq
    """
    d = np.abs(np.asarray(xq) - float(xval))
    return np.minimum(d, L - d)

# --- main function: probability at one or several x locations ----------
def probability_at_location(pred_pos, pred_strength,      # arrays shape (T, P)
                            ck_step,                     # integer time index (checkpoint)
                            x_query,                     # scalar or array of x positions to query
                            x_domain=(0.0, 20.0),
                            sigma_x=1.0,                 # spatial kernel width (same units as x)
                            temporal_radius=0,           # include ± timesteps around ck_step
                            sigma_t=1.0,                 # temporal decay scale; <=0 disables temporal weighting
                            use_strength=True,
                            strength_power=1.0,
                            score_to_prob_scale=1.0):
    """
    Compute probability (and raw score) that a plume is present at x_query,
    using predicted plume positions & strengths for a single test & ensemble.

    Parameters
    ----------
    pred_pos : ndarray (T, P)
        Predicted plume positions (x) per timestep and slot; np.nan if empty slot.
    pred_strength : ndarray (T, P) or None
        Predicted strengths (sKE); can be None.
    ck_step : int
        Checkpoint timestep index.
    x_query : float or 1D array
        Position(s) to evaluate probability at.
    x_domain : (xmin, xmax)
        Periodic domain.
    sigma_x : float
        Gaussian kernel width (stddev) in x-units.
    temporal_radius : int
        ± timesteps around ck_step to include. 0 uses only ck_step.
    sigma_t : float
        Temporal decay scale. If <=0, only ck_step is used.
    use_strength : bool
    strength_power : float
    score_to_prob_scale : float
        S parameter for mapping score -> prob: p = 1 - exp(-score / S)

    Returns
    -------
    prob : ndarray (same shape as x_query flattened)
        Probability(s) at x_query.
    score : ndarray
        Raw score(s) before mapping.
    """

    xq = np.atleast_1d(x_query).astype(float)
    x_min, x_max = x_domain
    L = x_max - x_min

    # temporal weights
    if temporal_radius <= 0 or sigma_t <= 0:
        temporal_weights = {0: 1.0}
    else:
        temporal_weights = {dt: np.exp(-abs(dt) / sigma_t) for dt in range(-temporal_radius, temporal_radius+1)}

    two_sigx2 = 2.0 * (sigma_x**2)
    T, P = pred_pos.shape

    # accumulate score for each query point
    scores = np.zeros_like(xq, dtype=float)

    for dt, tw in temporal_weights.items():
        t = ck_step + dt
        if t < 0 or t >= T:
            continue
        for p in range(P):
            pp = pred_pos[t, p]
            if np.isnan(pp):
                continue
            # compute circular distances between this plume and all query points
            dists = circ_dist_scalar_array(xq, pp, L)  # array len(xq)
            spatial_kernel = np.exp(-(dists**2) / two_sigx2)   # peak=1 at distance 0
            w = 1.0
            if use_strength and (pred_strength is not None):
                sval = pred_strength[t, p]
                if not np.isnan(sval):
                    w = sval ** strength_power
            scores += tw * w * spatial_kernel

    # map raw score -> probability
    S = score_to_prob_scale if score_to_prob_scale > 0 else 1.0
    probs = 1.0 - np.exp(-scores / S)

    # return same shape as input x_query
    if np.isscalar(x_query):
        return float(probs[0]), float(scores[0])
    else:
        return probs, scores
    
def truth_at_location(true_pos,                # (T, P) array of true plume x (np.nan if absent)
                      ck_step,
                      x_query,
                      x_domain=(0.0,20.0),
                      label_radius=1.0,
                      temporal_radius=0):
    """
    Return binary truth (1/0) whether any true plume is within label_radius
    of x_query at checkpoint ck_step (optionally consider ±temporal_radius).
    """
    xq = np.atleast_1d(x_query).astype(float)
    x_min, x_max = x_domain
    L = x_max - x_min
    T, P = true_pos.shape

    # start with zeros
    truth = np.zeros_like(xq, dtype=int)

    for dt in range(-temporal_radius, temporal_radius+1):
        t = ck_step + dt
        if t < 0 or t >= T:
            continue
        row = true_pos[t, :]
        valid = row[~np.isnan(row)]
        if valid.size == 0:
            continue
        for pp in valid:
            d = np.minimum(np.abs(xq - pp), L - np.abs(xq - pp))
            truth[d <= label_radius] = 1

    if np.isscalar(x_query):
        return int(truth[0])
    return truth