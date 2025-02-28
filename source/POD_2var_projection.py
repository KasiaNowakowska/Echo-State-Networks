print('load packages')
import os
os.environ["OMP_NUM_THREADS"] = '1' # imposes cores
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sys
sys.stdout.reconfigure(line_buffering=True)

input_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/input_data/'
output_path1 = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/POD_Analysis/2variables_projection/analysis'

print(output_path1)
if not os.path.exists(output_path1):
    os.makedirs(output_path1)
    print('made directory')

from skimage.metrics import structural_similarity as ssim
def compute_ssim_for_4d(original, decoded):
    """
    Compute the average SSIM across all timesteps and channels for 4D arrays.
    """
  
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
    
def calculate_global_q_KE(data):
    avgq = np.mean(data[:,:,:,0], axis=(1,2))
    ke = 0.5*data[:,:,:,1]*data[:,:,:,1]
    avgke = np.mean(ke, axis=(1,2))
    globalvar = np.zeros((data.shape[0], 2))
    globalvar[:,0] = avgke
    globalvar[:,1] = avgq
    return globalvar
    
def NRMSE(original_data, reconstructed_data):
    rmse = np.sqrt(mean_squared_error(original_data, reconstructed_data))
    
    variance = np.var(original_data)
    std_dev  = np.sqrt(variance)
    
    nrmse = 1 - (rmse/std_dev)
    
    return nrmse

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
    mask_expanded       = np.expand_dims(mask, axis=-1)  # Shape: (256, 64, 1)
    mask_expanded       = np.repeat(mask_expanded, original_data.shape[-1], axis=-1)  # Shape: (256, 64, 4)
    mask_expanded_recon = np.expand_dims(mask_reconstructed, axis=-1)  # Shape: (256, 64, 1)
    mask_expanded_recon = np.repeat(mask_expanded_recon, reconstructed_data.shape[-1], axis=-1)  # Shape: (256, 64, 4)
    
    return active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon

#### load Data ####
#### larger data 5000-30000 hf ####
num_snapshots = [5001]
POD_snapshots = 5000
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
variables = num_variables = 2
variable_names = ['q', 'w']
scaling = 'scaled'
POD = 'together'

for snap in num_snapshots:
    ### generate directory to save images ###
    snapshots_path = '/CAE_comparison%i_128modes' % POD_snapshots
    output_path = output_path1+snapshots_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('made directory')

    #### larger data 5000-30000 hf ####
    total_num_snapshots = snap #snap
    x = np.load(input_path+'/x.npy')
    z = np.load(input_path+'/z.npy')
    variables = num_variables = 2
    variable_names = ['q', 'w']
    
    with h5py.File(input_path+'/data_4var_5000_30000.h5', 'r') as df:
        time_vals = np.array(df['total_time_all'][:total_num_snapshots])
        q = np.array(df['q_all'][:total_num_snapshots])
        w = np.array(df['w_all'][:total_num_snapshots])
    
        q = np.squeeze(q, axis=2)
        w = np.squeeze(w, axis=2)
    
        print(np.shape(q))
    
    print('shape of time_vals', np.shape(time_vals))
    
    # Reshape the arrays into column vectors
    q_array = q.reshape(len(time_vals), len(x), len(z), 1)
    w_array = w.reshape(len(time_vals), len(x), len(z), 1)
    
    del q
    del w
    
    data_all         = np.concatenate((q_array, w_array), axis=-1)
    data_all_reshape = data_all.reshape(len(time_vals), len(x) * len(z) * variables)
    
    data_for_POD = data_all[:POD_snapshots]
    data_for_POD_reshape = data_for_POD.reshape(POD_snapshots, len(x) * len(z) * variables)
    time_vals_for_POD = time_vals[:POD_snapshots]
    new_data_snapshots = total_num_snapshots-POD_snapshots
    new_data = data_all[POD_snapshots:]
    new_data_reshape = new_data.reshape(new_data_snapshots, len(x) * len(z) * variables)
    time_vals_new = time_vals[POD_snapshots:]
    
    ### scale data ###
    ss = StandardScaler()
    data_for_POD_scaled_reshape = ss.fit_transform(data_for_POD_reshape)
    data_for_POD_scaled = data_for_POD_scaled_reshape.reshape(POD_snapshots, len(x), len(z), variables)
    new_data_scaled_reshape = ss.transform(new_data_reshape)
    new_data_scaled = new_data_reshape.reshape(new_data_snapshots, len(x), len(z), variables)
    
    # Print the shape of the combined array
    print('shape of all data and POD data after scaling:', data_all.shape, data_for_POD_scaled.shape)

    #### global ####
    #groundtruth_global = calculate_global_q_KE(data_all)

    ### reshape data and run POD ###
    data_matrix = data_for_POD_scaled.reshape(POD_snapshots, -1) ##data_scaled originally
    print('shape of matirx for POD', np.shape(data_matrix)) 
    data_matrix_new = new_data_scaled.reshape(new_data_snapshots, -1) #

    from sklearn.decomposition import PCA
    n_components=[128]
    cumulative_explained_variance_values   = np.zeros(len(n_components))
    evr_reconstruction_POD                 = np.zeros(len(n_components))
    MSE_reconstruction_POD                 = np.zeros(len(n_components))
    NRMSE_reconstruction_POD               = np.zeros(len(n_components))
    accuracy_POD                           = np.zeros(len(n_components))
    MSE_plume_POD                          = np.zeros(len(n_components))
    SSIM_POD                               = np.zeros(len(n_components))
    
    evr_reconstruction_new                     = np.zeros(len(n_components))
    MSE_reconstruction_new                     = np.zeros(len(n_components))
    NRMSE_reconstruction_new                   = np.zeros(len(n_components))
    accuracy_new                               = np.zeros(len(n_components))
    MSE_plume_new                              = np.zeros(len(n_components))
    SSIM_new                                   = np.zeros(len(n_components))
    
    Plotting = True
    Projection = True

    c_index = 0
    for c in n_components:
        if POD == 'together':
            print('togther')
            pca = PCA(n_components=c, svd_solver='randomized', random_state=42)
            pca.fit(data_matrix)
            data_reduced = pca.transform(data_matrix)  # (5000, n_modes)
            data_reconstructed_reshaped = pca.inverse_transform(data_reduced)  # (5000, 256 * 2)
            data_reconstructed = data_reconstructed_reshaped.reshape(POD_snapshots, 256, 64, num_variables)  # (5000, 256, 1, 2)
            if scaling == 'scaled':
                 data_reconstructed_reshaped_unscaled = ss.inverse_transform(data_reconstructed_reshaped)
                 data_reconstructed_unscaled = data_reconstructed_reshaped_unscaled.reshape(POD_snapshots, 256, 64, num_variables)
            components = pca.components_
            
            # Get the explained variance ratio
            explained_variance_ratio = pca.explained_variance_ratio_
            # Calculate cumulative explained variance
            cumulative_explained_variance = np.cumsum(explained_variance_ratio)
            # add to list
            cumulative_explained_variance_values[c_index] = cumulative_explained_variance[-1]
            print('cumulative explained variance for', c, 'components is', cumulative_explained_variance[-1])
            
            
            if Plotting:
                # Plot cumulative explained variance 
                fig, ax = plt.subplots(1, figsize=(12,6))
                ax.plot(cumulative_explained_variance*100, marker='o', linestyle='-', color='b')
                ax2 = ax.twinx()
                ax2.semilogy(explained_variance_ratio*100,  marker='o', linestyle='-', color='r')
                ax.set_xlabel('Number of Modes')
                ax.set_ylabel('Cumulative Energy (%)', color='blue')
                #plt.axvline(x=truncated_modes, color='r', linestyle='--', label=f'Truncated Modes: {truncated_modes}')
                ax2.set_ylabel('Inidvidual Energy (%)', color='red')
                fig.savefig(output_path+'/cum_energy_modes%i_snapshots%i.png' % (c, snap))
                plt.close()
            
                # Plot the time coefficients and mode structures
                indexes_to_plot = np.array([1, 2, 10, 50, 100] ) -1
                indexes_to_plot = indexes_to_plot[indexes_to_plot <= c]
                
                #time coefficients
                fig, ax = plt.subplots(1, figsize=(8,3), tight_layout=True)
                for index, element in enumerate(indexes_to_plot):
                    #ax.plot(index - data_reduced[:, element], label='S=%i' % (element+1))
                    ax.plot(data_reduced[:, element], label='S=%i' % (element+1))
                ax.grid()
                ax.legend()
                #ax.set_yticks([])
                ax.set_xlabel('Time Step $\Delta t$')
                ax.set_ylabel('Time Coefficients')
                fig.savefig(output_path+'/modes_time_coef_modes%i_snapshots%i.png' % (c, snap))
                plt.close()
                
                # Visualize the modes
                minm = components.min()
                maxm = components.max()
                for v in range(variables):
                    fig, ax =plt.subplots(len(indexes_to_plot), figsize=(6,12), tight_layout=True, sharex=True)
                    for i in range(len(indexes_to_plot)):
                        mode = components[indexes_to_plot[i]].reshape(256, 64, variables)  # Reshape to original dimensions for visualization
                        c1 = ax[i].pcolormesh(x, z, mode[:, :, v].T, cmap='viridis', vmin=minm, vmax=maxm)  # Visualizing the first variable
                        #ax[i].axis('off')
                        ax[i].set_title('mode % i' % (indexes_to_plot[i]+1))
                        fig.colorbar(c1, ax=ax[i])
                        ax[i].set_ylabel('z')
                    ax[-1].set_xlabel('x')
                    fig.savefig(output_path+'/modes_structure%i_snapshots%i_var%i.png' % (c, snap, v))
                    plt.close()
        
        #### visualise reconstruction ####
        # EVR for reconstruction of POD data
        numerator = np.sum((data_matrix - data_reconstructed_reshaped) **2)
        denominator = np.sum(data_matrix ** 2)
        evr_reconstruction = 1 - (numerator/denominator)
        evr_reconstruction_POD[c_index] = evr_reconstruction
        
        #MSE for reconstruction of POD data
        mse = mean_squared_error(data_matrix, data_reconstructed_reshaped)
        MSE_reconstruction_POD[c_index] = mse
        
        #NRMSE
        nrmse = NRMSE(data_matrix, data_reconstructed_reshaped)
        NRMSE_reconstruction_POD[c_index] = nrmse
        
        if Plotting:
            for var in range(num_variables):
                fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
                minm = min(np.min(data_for_POD[:, :, 32, var]), np.min(data_reconstructed_unscaled[:, :, 32, var]))
                maxm = max(np.max(data_for_POD[:, :, 32, var]), np.max(data_reconstructed_unscaled[:, :, 32, var]))
                c1 = ax[0].pcolormesh(time_vals_for_POD, x, data_for_POD[:,:, 32, var].T, vmin=minm, vmax=maxm)
                fig.colorbar(c1, ax=ax[0])
                ax[0].set_title('true')
                c2 = ax[1].pcolormesh(time_vals_for_POD, x, data_reconstructed_unscaled[:,:,32,var].T, vmin=minm, vmax=maxm)
                fig.colorbar(c1, ax=ax[1])
                ax[1].set_title('reconstruction')
                for v in range(2):
                    ax[v].set_xlabel('time')
                    ax[v].set_ylabel('x')
                    ax[v].set_xlim(time_vals_for_POD[0], time_vals_for_POD[499])
                fig.savefig(output_path+'/reconstruction_modes_POD_set_c%i_snapshots%i_var%i.png' % (c, snap, var))
                plt.close()
      
        #SSIM
        ssim_value = compute_ssim_for_4d(data_for_POD, data_reconstructed_unscaled)
        SSIM_POD[c_index] = ssim_value
     
        import json

        # Full path for saving the file
        output_file = "metrics_POD_set_snapshots%i_components%i.json" % (POD_snapshots, c)
    
        output_path_met = os.path.join(output_path, output_file)
    
        metrics = {
        "EVR": evr_reconstruction,
        "mse": mse,
        "nrmse": nrmse,
        "test_ssim": ssim_value,
        }
    
        with open(output_path_met, "w") as file:
            json.dump(metrics, file, indent=4)
         
        if Projection:
            #### PROJECTION ####
            data_reduced_projection             = pca.transform(data_matrix_new)
            new_reconstructed_reshaped          = pca.inverse_transform(data_reduced_projection)
            new_reconstructed                   = new_reconstructed_reshaped.reshape(new_data_snapshots, 256, 64, num_variables)  # (5000, 256, 1, 2)
            
            #### visualise reconstruction ####
            # EVR for reconstruction of POD data
            numerator = np.sum((data_matrix_new - new_reconstructed_reshaped) **2)
            denominator = np.sum(data_matrix_new ** 2)
            evr_reconstruction = 1 - (numerator/denominator)
            evr_reconstruction_new[c_index] = evr_reconstruction
            
            #MSE for reconstruction of POD data
            mse = mean_squared_error(data_matrix_new, new_reconstructed_reshaped)
            MSE_reconstruction_new[c_index] = mse
            
            #NRMSE
            nrmse = NRMSE(data_matrix_new, new_reconstructed_reshaped)
            NRMSE_reconstruction_new[c_index] = nrmse
            
            if Plotting:
                for var in range(num_variables):
                    fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
                    minm = min(np.min(new_data[:, :, 32, var]), np.min(new_reconstructed[:, :, 32, var]))
                    maxm = max(np.max(new_data[:, :, 32, var]), np.max(new_reconstructed[:, :, 32, var]))
                    c1 = ax[0].pcolormesh(time_vals_new, x, new_data[:,:, 32, var].T, vmin=np.min(new_data[:, :, 32, var]), vmax=np.max(new_data[:, :, 32, var]))
                    fig.colorbar(c1, ax=ax[0])
                    ax[0].set_title('true')
                    c2 = ax[1].pcolormesh(time_vals_new, x, new_reconstructed[:,:,32,var].T, vmin=np.min(new_data[:, :, 32, var]), vmax=np.max(new_data[:, :, 32, var]))
                    fig.colorbar(c1, ax=ax[1])
                    ax[1].set_title('reconstruction')
                    for v in range(2):
                        ax[v].set_xlabel('time')
                        ax[v].set_ylabel('x')
                        ax[v].set_xlim(time_vals_new[0], time_vals_new[500])
                    fig.savefig(output_path+'/reconstruction_modes_new_set_c%i_snapshots%i_var%i.png' % (c, snap, var))
                    plt.close()
            
            #SSIM
            ssim_value = compute_ssim_for_4d(new_data, new_reconstructed)
            SSIM_new[c_index] = ssim_value
            
            # Full path for saving the file
            output_file = "metrics_new_set_snapshots%i_components%i.json" % (POD_snapshots, c)
        
            output_path_met = os.path.join(output_path, output_file)
        
            metrics = {
            "EVR": evr_reconstruction,
            "mse": mse,
            "nrmse": nrmse,
            "test_ssim": ssim_value,
            }
        
            with open(output_path_met, "w") as file:
                json.dump(metrics, file, indent=4)
            
        c_index += 1
