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

input_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/input_data/'
output_path1 = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/POD_Analysis/'

print(output_path1)
if not os.path.exists(output_path1):
    os.makedirs(output_path1)
    print('made directory')

def calculate_combined_avg_nrmse(images_true, images_reconstructed):
    """
    Calculate the average NRMSE between true and reconstructed images across all timesteps,
    treating all variables together for each image.

    Parameters:
    - images_true: np.array of shape (num_timesteps, height, width, 2) 
                   Original images with multiple variables.
    - images_reconstructed: np.array of shape (num_timesteps, height, width, 2)
                   Reconstructed images with multiple variables.

    Returns:
    - average_nrmse: float, combined average NRMSE across all timesteps and variables.
    """
    # Ensure input arrays are the same shape
    assert images_true.shape == images_reconstructed.shape, "Shape mismatch between input arrays."

    # Compute NRMSE for each timestep across all variables
    nrmse_per_timestep = []
    variance = np.var(images_true)
    for t in range(images_true.shape[0]):
        # Flatten the height, width, and variable dimensions to treat them as a single image
        mse = np.mean((images_true[t] - images_reconstructed[t]) ** 2)
        
        # Calculate combined NRMSE for this timestep
        if variance > 0:
            nrmse = np.sqrt(mse / variance)
        else:
            nrmse = 0  # or np.nan if preferred
        
        nrmse_per_timestep.append(nrmse)
    
    # Average NRMSE across all timesteps
    average_nrmse = np.mean(nrmse_per_timestep)

    return average_nrmse


def calculate_avg_nrmse_per_variable(images_true, images_reconstructed):
    """
    Calculate the average Normalized Root Mean Square Error (NRMSE) between
    true and reconstructed images across multiple timesteps, for each variable separately.

    Parameters:
    - images_true: np.array of shape (num_timesteps, height, width, 2) 
                   Original (true) images with multiple variables.
    - images_reconstructed: np.array of shape (num_timesteps, height, width, 2)
                   Reconstructed images with multiple variables.

    Returns:
    - average_nrmse_per_variable: list of floats, [NRMSE for variable 1, NRMSE for variable 2]
    """
    # Ensure input arrays are the same shape
    assert images_true.shape == images_reconstructed.shape, "Shape mismatch between input arrays."
    
    # Initialize list to store NRMSE values per variable
    average_nrmse_per_variable = []

    # Loop over each variable in the last dimension (2 in this case)
    for var in range(images_true.shape[-1]):
        # Calculate NRMSE for each timestep for the current variable
        variance = np.var(images_true[:, :, :, var])
        nrmse_per_timestep = []
        for t in range(images_true.shape[0]):
            # Calculate MSE and variance for the true images for the current variable and timestep
            mse = np.mean((images_true[t, :, :, var] - images_reconstructed[t, :, :, var]) ** 2)
            
            # Compute NRMSE for this timestep and variable, handle zero variance
            if variance > 0:
                nrmse = np.sqrt(mse / variance)
            else:
                nrmse = 0  # or np.nan if preferred
            
            nrmse_per_timestep.append(nrmse)
        
        # Average NRMSE for the current variable across all timesteps
        average_nrmse_per_variable.append(np.mean(nrmse_per_timestep))

    return average_nrmse_per_variable

def calculate_mse_over_time(true_values, reconstructed_values):
    # Check if both arrays have the same shape (snapshots, height, width)
    if true_values.shape != reconstructed_values.shape:
        raise ValueError("True values and reconstructed values must have the same shape.")
    
    # Calculate MSE for each snapshot (along height and width)
    mse_per_snapshot = np.mean((true_values - reconstructed_values) ** 2, axis=(1, 2))
    
    # Average the MSE across all snapshots (time dimension)
    mse_avg_time = np.mean(mse_per_snapshot)
    
    return mse_avg_time

#### load Data ####
#### larger data 5000-30000 hf ####
num_snapshots = [500,1000, 5000]
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
variables = num_variables = 2
variable_names = ['q', 'w']
scaling = 'minus_mean'
POD = 'seperate'

for snap in num_snapshots:
    ### generate directory to save images ###
    snapshots_path = '/%i' % snap
    output_path = output_path1+snapshots_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('made directory')

    #### save full data set ####
    with h5py.File(input_path+'/data_5000_30000.h5', 'r') as df:
        time_vals = np.array(df['total_time_all'][:snap])
        q = np.array(df['q_all'][:snap])
        w = np.array(df['w_all'][:snap])
    
        q = np.squeeze(q, axis=2)
        w = np.squeeze(w, axis=2)
        
        # Reshape the arrays into column vectors
        q_all = q.reshape(len(time_vals), len(x), len(z), 1)
        w_all = w.reshape(len(time_vals), len(x), len(z), 1)
        
        del q
        del w
        data_all = np.concatenate((q_all, w_all), axis=-1)
        # Print the shape of the combined array
        print(data_all.shape)

    if scaling == 'scaled':
        with h5py.File(input_path+'/data_5000_30000.h5', 'r') as df:
            time_vals = np.array(df['total_time_all'][:snap])
            q = np.array(df['q_all'][:snap])
            w = np.array(df['w_all'][:snap])
        
            q = np.squeeze(q, axis=2)
            w = np.squeeze(w, axis=2)
        
            q_mean = np.mean(q, axis=0)
            w_mean = np.mean(w, axis=0)
            q_std = np.std(q, axis=0)
            w_std = np.std(w, axis=0)
        
            q_array = (q - q_mean) / q_std
            w_array = (w - w_mean) / w_std
            print(np.shape(q_array))
            
    elif scaling == 'minus_mean':
        with h5py.File(input_path+'/data_5000_30000.h5', 'r') as df:
            time_vals = np.array(df['total_time_all'][:snap])
            q = np.array(df['q_all'][:snap])
            w = np.array(df['w_all'][:snap])
        
            q = np.squeeze(q, axis=2)
            w = np.squeeze(w, axis=2)
        
            q_mean = np.mean(q, axis=0)
            w_mean = np.mean(w, axis=0)
        
            q_array = (q - q_mean) 
            w_array = (w - w_mean) 
            print(np.shape(q_array))
    
    print('shape of time_vals', np.shape(time_vals))
    print(time_vals[0], time_vals[-1])

    # Reshape the arrays into column vectors
    q_array = q_array.reshape(len(time_vals), len(x), len(z), 1)
    w_array = w_array.reshape(len(time_vals), len(x), len(z), 1)
    
    del q
    del w
    data_full = np.concatenate((q_array, w_array), axis=-1)
    # Print the shape of the combined array
    print(data_full.shape)

    #### global ####
    groundtruth_avgq = np.mean(data_all[:,:,:,0]+q_mean, axis=(1,2))
    groundtruth_ke = 0.5*data_all[:,:,:,1]*data_all[:,:,:,1]
    groundtruth_avgke = np.mean(groundtruth_ke, axis=(1,2))
    groundtruth_global = np.zeros((snap,2))
    groundtruth_global[:,0] = groundtruth_avgke
    groundtruth_global[:,1] = groundtruth_avgq

    ### reshape data and run POD ###
    data_matrix = data_full.reshape(snap, -1)
    q_matrix = q_array.reshape(snap, -1)
    w_matrix = w_array.reshape(snap, -1)
    print(np.shape(data_matrix))

    from sklearn.decomposition import PCA
    cum_var_no_modes = []
    nrmse_no_modes = []
    n_components=np.arange(25,201,25)
    nrmse_sep = np.zeros((len(n_components), 2))
    MSE_sep = np.zeros((len(n_components), 2))

    c_index = 0
    for c in n_components:
        
    
        if POD == 'together':
            print('togther')
            pca = PCA(n_components=c, svd_solver='randomized', random_state=42)
            data_reduced = pca.fit_transform(data_matrix)  # (5000, n_modes)
            data_reconstructed_reshaped = pca.inverse_transform(data_reduced)  # (5000, 256 * 2)
            #data_reconstructed_reshaped = ss.inverse_transform(data_reconstructed_reshaped)
            data_reconstructed = data_reconstructed_reshaped.reshape(snap, 256, 64, num_variables)  # (5000, 256, 1, 2)
            if scaling == 'scaled':
                data_reconstructed[:,:,:,0] = data_reconstructed[:,:,:,0] * q_std + q_mean
                data_reconstructed[:,:,:,1] = data_reconstructed[:,:,:,1] * w_std + w_mean
            elif scaling == 'minus_mean':
                data_reconstructed[:,:,:,0] = data_reconstructed[:,:,:,0] + q_mean
                data_reconstructed[:,:,:,1] = data_reconstructed[:,:,:,1] + w_mean
            components = pca.components_
            
            # Get the explained variance ratio
            explained_variance_ratio = pca.explained_variance_ratio_
            # Calculate cumulative explained variance
            cumulative_explained_variance = np.cumsum(explained_variance_ratio)
            fig, ax = plt.subplots(1, figsize=(12,6))
            ax.plot(cumulative_explained_variance*100, marker='o', linestyle='-', color='b')
            ax2 = ax.twinx()
            ax2.semilogy(explained_variance_ratio*100,  marker='o', linestyle='-', color='r')
            ax.set_xlabel('Number of Modes')
            ax.set_ylabel('Cumulative Energy (%)', color='blue')
            #plt.axvline(x=truncated_modes, color='r', linestyle='--', label=f'Truncated Modes: {truncated_modes}')
            ax2.set_ylabel('Inidvidual Energy (%)', color='red')
            #fig.savefig(output_path+'/cum_energy_modes%i_snapshots%i.png' % (c, snap))
            plt.close()
            cum_var_no_modes.append(cumulative_explained_variance[-1])
            print(cumulative_explained_variance[-1])
            
            indexes_to_plot = np.array([1, 2, 10, 50, 100] ) -1
            indexes_to_plot = indexes_to_plot[indexes_to_plot <= c]
            print(c, snap)
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
                #fig.savefig(output_path+'/modes_structure%i_snapshots%i_var%i.png' % (c, snap, v))
                plt.close()
            
            
        elif POD == 'seperate':
            print('seperate')
            data_reconstructed = np.zeros((snap, len(x), len(z), variables))
            for v in range(variables):
                data_var = data_full[:,:,:, v]
                data_var_matrix = data_var.reshape(snap, -1)
                pca = PCA(n_components=c, svd_solver='randomized', random_state=42)
                var_reduced = pca.fit_transform(data_var_matrix)
                var_reconstructed_reshaped = pca.inverse_transform(var_reduced)  # (5000, 256 * 2)
                var_reconstructed = var_reconstructed_reshaped.reshape(snap, 256, 64)  # (5000, 256, 1, 2)
                data_reconstructed[:,:,:,v] = var_reconstructed
                components = pca.components_
                
                # Get the explained variance ratio
                explained_variance_ratio = pca.explained_variance_ratio_
                # Calculate cumulative explained variance
                cumulative_explained_variance = np.cumsum(explained_variance_ratio)
                fig, ax = plt.subplots(1, figsize=(12,6))
                ax.plot(cumulative_explained_variance*100, marker='o', linestyle='-', color='b')
                ax2 = ax.twinx()
                ax2.semilogy(explained_variance_ratio*100,  marker='o', linestyle='-', color='r')
                ax.set_xlabel('Number of Modes')
                ax.set_ylabel('Cumulative Energy (%)', color='blue')
                #plt.axvline(x=truncated_modes, color='r', linestyle='--', label=f'Truncated Modes: {truncated_modes}')
                ax2.set_ylabel('Inidvidual Energy (%)', color='red')
                fig.savefig(output_path+'/cum_energy_modes%i_snapshots%i_var%i.png' % (c, snap, v))
                plt.close()
                cum_var_no_modes.append(cumulative_explained_variance[-1])
                print(cumulative_explained_variance[-1])
    
                indexes_to_plot = np.array([1, 2, 10, 50, 100] ) -1
                indexes_to_plot = indexes_to_plot[indexes_to_plot <= c]
                print(c, snap)
                fig, ax = plt.subplots(1, figsize=(8,3), tight_layout=True)
                for index, element in enumerate(indexes_to_plot):
                    #ax.plot(index - data_reduced[:, element], label='S=%i' % (element+1))
                    ax.plot(var_reduced[:, element], label='S=%i' % (element+1))
                ax.grid()
                ax.legend()
                #ax.set_yticks([])
                ax.set_xlabel('Time Step $\Delta t$')
                ax.set_ylabel('Time Coefficients')
                fig.savefig(output_path+'/modes_time_coef_modes%i_snapshots%i_var%i.png' % (c, snap, v))
                plt.close()
                # Visualize the modes
                fig, ax =plt.subplots(len(indexes_to_plot), figsize=(6,12), tight_layout=True, sharex=True)
                minm = components.min()
                maxm = components.max()
                for i in range(len(indexes_to_plot)):
                    mode = components[indexes_to_plot[i]].reshape(256, 64)  # Reshape to original dimensions for visualization
                    c1 = ax[i].pcolormesh(x, z, mode[:, :].T, cmap='viridis', vmin=minm, vmax=maxm)  # Visualizing the first variable
                    #ax[i].axis('off')
                    ax[i].set_title('mode % i' % (indexes_to_plot[i]+1))
                    fig.colorbar(c1, ax=ax[i])
                    ax[i].set_ylabel('z')
                ax[-1].set_xlabel('x')
                fig.savefig(output_path+'/modes_structure%i_snapshots%i_var%i.png' % (c, snap, v))
                plt.close()
                
            if scaling == 'scaled':
                data_reconstructed[:,:,:,0] = data_reconstructed[:,:,:,0] * q_std + q_mean
                data_reconstructed[:,:,:,1] = data_reconstructed[:,:,:,1] * w_std + w_mean
            elif scaling == 'minus_mean':
                data_reconstructed[:,:,:,0] = data_reconstructed[:,:,:,0] + q_mean
                data_reconstructed[:,:,:,1] = data_reconstructed[:,:,:,1] + w_mean
                
            data_reconstructed_reshaped = data_reconstructed.reshape(snap, -1)
        
        #### visualise reconstruction ####
        for var in range(2):
            fig, ax = plt.subplots(2, figsize=(12,6), tight_layout=True)
            minm = min(np.min(data_all[:, :, 32, var]), np.min(data_reconstructed[:, :, 32, var]))
            maxm = max(np.max(data_all[:, :, 32, var]), np.max(data_reconstructed[:, :, 32, var]))
            c1 = ax[0].contourf(time_vals, x, data_all[:,:, 32, var].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].contourf(time_vals, x, data_reconstructed[:,:,32,var].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[1])
            ax[1].set_title('reconstruction')
            for v in range(variables):
                ax[v].set_xlabel('time')
                ax[v].set_ylabel('x')
            ax[0].set_xlim(5000,6000)
            ax[1].set_xlim(5000,6000)
            fig.savefig(output_path+'/reconstruction_modes_short%i_snapshots%i_var%i.png' % (c, snap, var))
            plt.close()
        
        NRMSE = calculate_combined_avg_nrmse(data_reconstructed[250:], data_all[250:])
        NRMSE_sep = calculate_avg_nrmse_per_variable(data_reconstructed[250:], data_all[250:])
        print(np.shape(NRMSE_sep))
        nrmse_no_modes.append(NRMSE)
        nrmse_sep[c_index,:] = NRMSE_sep
        
        MSE = np.zeros((2))
        for v in range(2):
            MSE[v] = calculate_mse_over_time(data_reconstructed[250:,:,:,v], data_all[250:,:,:,v])
        MSE_sep[c_index,:] = MSE
        
        
        # reconstuct global
        reconstructed_groundtruth_avgq = np.mean(data_reconstructed[:,:,:,0]+q_mean, axis=(1,2))
        reconstructed_groundtruth_ke = 0.5*data_reconstructed[:,:,:,1]*data_reconstructed[:,:,:,1]
        reconstructed_groundtruth_avgke = np.mean(reconstructed_groundtruth_ke, axis=(1,2))
        reconstructed_groundtruth_global = np.zeros((snap,2))
        reconstructed_groundtruth_global[:,0] = reconstructed_groundtruth_avgke
        reconstructed_groundtruth_global[:,1] = reconstructed_groundtruth_avgq
        print(np.shape(reconstructed_groundtruth_global))
        fig, ax = plt.subplots(2,1, figsize=(12,6), tight_layout=True, sharex=True)
        for i in range(2):
            ax[i].plot(time_vals, groundtruth_global[:,i], color='tab:blue', label='truth')
            ax[i].plot(time_vals, reconstructed_groundtruth_global[:,i], color='tab:orange', label='reconstruction')
            ax[i].grid()
        ax[0].set_ylabel('KE')
        ax[1].set_ylabel('q')
        ax[1].set_xlabel('time')
        ax[1].legend()
        ax[0].set_xlim(5000,6000)
        ax[1].set_xlim(5000,6000)
        fig.savefig(output_path+'/global_reconstruction_short%i_snapshots%i.png' % (c, snap))
        c_index += 1
       
    np.save(output_path+'/cum_var_no_modes_snapshots%i.npy' % snap, cum_var_no_modes)
    np.save(output_path+'/nrmse_no_modes_snapshots%i.npy' % snap, nrmse_no_modes)
    np.save(output_path+'/nrmse_sep_snapshots%i.npy' % snap, nrmse_sep)
    np.save(output_path+'/mse_sep_snapshots%i.npy' % snap, MSE_sep)
    print(MSE_sep)
'''
num_snapshots = [500,1000,5000]
n_components=np.arange(25,201,25)

'''
fig, ax =plt.subplots(1, figsize=(12,10), tight_layout=True)
for snap in num_snapshots:
    cum_var = np.load(output_path1+'/%i' % snap +'/cum_var_no_modes_snapshots%i.npy' % snap)
    ax.plot(n_components, cum_var, label='snapshot length = %i' % snap)
ax.grid()
ax.legend()
ax.set_xlabel('number of modes')
ax.set_ylabel('energy')
fig.savefig(output_path1+'/cum_energy.png')
plt.close()


fig, ax =plt.subplots(1, figsize=(12,10), tight_layout=True)
for snap in num_snapshots:
    NRMSE_vals = np.load(output_path1+'/%i' % snap +'/nrmse_no_modes_snapshots%i.npy' %  snap)
    ax.plot(n_components, NRMSE_vals, label='snapshot length = %i' % snap)
ax.grid()
ax.legend()
ax.set_xlabel('number of modes')
ax.set_ylabel('avg NRMSE')
fig.savefig(output_path1+'/nrmse.png')
plt.close()

fig, ax =plt.subplots(1, figsize=(12,10), tight_layout=True)
for snap in num_snapshots:
    NRMSE_vals = np.load(output_path1+'/%i' % snap +'/nrmse_no_modes_snapshots%i.npy' %  snap)
    ax.plot(n_components, 1-(NRMSE_vals)**2, label='snapshot length = %i' % snap)
ax.grid()
ax.legend()
ax.set_xlabel('number of modes')
ax.set_ylabel('1 - (avg_NRMSE)**2')
fig.savefig(output_path1+'/1_nrmse.png')
plt.close()

fig, ax =plt.subplots(1, figsize=(12,10), tight_layout=True)
for snap in num_snapshots:
    NRMSE_sep_vals = np.load(output_path1+'/%i' % snap +'/nrmse_sep_snapshots%i.npy' %  snap)
    q_vals = NRMSE_sep_vals[:,0]
    w_vals = NRMSE_sep_vals[:,1]
    ax.plot(n_components, q_vals, label='snapshot length = %i' % snap, linestyle='-')
    ax.plot(n_components, w_vals, label='snapshot length = %i' % snap, linestyle='--')
ax.grid()
ax.legend()
ax.set_xlabel('number of modes')
ax.set_ylabel('avg_NRMSE')
fig.savefig(output_path1+'/nrmse_sep.png')
plt.close()

fig, ax =plt.subplots(1, figsize=(12,10), tight_layout=True)
for snap in num_snapshots:
    MSE_sep_vals = np.load(output_path1+'/%i' % snap +'/mse_sep_snapshots%i.npy' %  snap)
    q_vals = MSE_sep_vals[:,0]
    w_vals = MSE_sep_vals[:,1]
    ax.plot(n_components, q_vals, label='snapshot length = %i' % snap, linestyle='-')
    ax.plot(n_components, w_vals, label='snapshot length = %i' % snap, linestyle='--')
ax.grid()
ax.legend()
ax.set_xlabel('number of modes')
ax.set_ylabel('avg_MSE')
fig.savefig(output_path1+'/nrmse_sep.png')
plt.close()


###### compare scalings ######
num_snapshots = [500, 1000, 5000]
n_components=np.arange(25,201,25)

fig, ax =plt.subplots(1, figsize=(12,10), tight_layout=True)
cum_var_mm = np.load(output_path1+'/minus_mean/500/cum_var_no_modes_snapshots500.npy')
ax.plot(n_components, cum_var_mm, label='minus mean')
cum_var_ss = np.load(output_path1+'/scaled/together/500/cum_var_no_modes_snapshots500.npy')
ax.plot(n_components, cum_var_ss, label='standard scaler')
ax.grid()
ax.legend()
ax.set_xlabel('number of modes')
ax.set_ylabel('cumulative variance/energy')
ax.set_title('N=500')
fig.savefig(output_path1+'/cumvar_mmss_500.png')



fig, ax =plt.subplots(1, figsize=(12,10), tight_layout=True)
cum_var_mm = np.load(output_path1+'/minus_mean/500/nrmse_no_modes_snapshots500.npy')
ax.plot(n_components, cum_var_mm, label='minus mean')
cum_var_ss = np.load(output_path1+'/scaled/together/500/nrmse_no_modes_snapshots500.npy')
ax.plot(n_components, cum_var_ss, label='standard scaler')
cum_var_ss_s = np.load(output_path1+'/scaled/seperate/500/nrmse_no_modes_snapshots500.npy')
ax.plot(n_components, cum_var_ss_s, label='standard scaler seperate POD')
cum_var_ss_s = np.load(output_path1+'/minus_mean/seperate/500/nrmse_no_modes_snapshots500.npy')
ax.plot(n_components, cum_var_ss_s, label='minus mean seperate POD')
ax.grid()
ax.legend()
ax.set_xlabel('number of modes')
ax.set_ylabel('avg nrmse')
ax.set_title('N=500')
fig.savefig(output_path1+'/nrmse_mmss_smm_500.png')

n_components=np.arange(25,201,25)

fig, ax =plt.subplots(2, figsize=(12,10), tight_layout=True, sharex=True)
NRMSE_sep_vals = np.load(output_path1+'/minus_mean/500/nrmse_sep_snapshots500.npy')
print(NRMSE_sep_vals)
q_vals = NRMSE_sep_vals[:,0]
w_vals = NRMSE_sep_vals[:,1]
ax[0].plot(n_components, q_vals, label='minus mean', linestyle='-', color='tab:blue')
ax[1].plot(n_components, w_vals, label='minus mean', linestyle='--', color='tab:blue')
NRMSE_sep_vals = np.load(output_path1+'/scaled/together/500/nrmse_sep_snapshots500.npy')
q_vals = NRMSE_sep_vals[:,0]
w_vals = NRMSE_sep_vals[:,1]
print(q_vals, w_vals)
ax[0].plot(n_components, q_vals, label='scaled', linestyle='-', color='tab:orange')
ax[1].plot(n_components, w_vals, label='scaled', linestyle='--', color='tab:orange')
NRMSE_sep_vals = np.load(output_path1+'/scaled/seperate/500/nrmse_sep_snapshots500.npy')
q_vals = NRMSE_sep_vals[:,0]
w_vals = NRMSE_sep_vals[:,1]
ax[0].plot(n_components, q_vals, label='scaled seperate POD', linestyle='-', color='tab:green')
ax[1].plot(n_components, w_vals, label='scaled seperate POD', linestyle='--', color='tab:green')
NRMSE_sep_vals = np.load(output_path1+'/minus_mean/seperate/500/nrmse_sep_snapshots500.npy')
q_vals = NRMSE_sep_vals[:,0]
w_vals = NRMSE_sep_vals[:,1]
ax[0].plot(n_components, q_vals, label='minus mean seperate POD', linestyle='-', color='tab:green')
ax[1].plot(n_components, w_vals, label='minus mean seperate POD', linestyle='--', color='tab:green')
ax[0].grid()
ax[0].legend()
ax[1].grid()
ax[1].legend()
ax[0].set_title('q')
ax[1].set_title('w')
ax[0].set_xlabel('number of modes')
ax[0].set_ylabel('avg nrmse')
ax[1].set_ylabel('avg nrmse')
#ax.set_title('N=500')
#ax.set_ylim(0,0.2)
#ax.set_yscale('log')
fig.savefig(output_path1+'/nrmse_mmss_smm_sep_500_2.png')


fig, ax =plt.subplots(2, figsize=(12,10), tight_layout=True, sharex=True)
MSE_sep_vals = np.load(output_path1+'/minus_mean/5000/mse_sep_snapshots500.npy')
print(MSE_sep_vals)
q_vals = MSE_sep_vals[:,0]
w_vals = MSE_sep_vals[:,1]
ax[0].plot(n_components, q_vals, label='minus mean', linestyle='-', color='tab:blue')
ax[1].plot(n_components, w_vals, label='minus mean', linestyle='--', color='tab:blue')
MSE_sep_vals = np.load(output_path1+'/scaled/together/5000/mse_sep_snapshots500.npy')
q_vals = MSE_sep_vals[:,0]
w_vals = MSE_sep_vals[:,1]
print(q_vals, w_vals)
ax[0].plot(n_components, q_vals, label='scaled', linestyle='-', color='tab:orange')
ax[1].plot(n_components, w_vals, label='scaled', linestyle='--', color='tab:orange')
ax[0].grid()
ax[0].legend()
ax[1].grid()
ax[1].legend()
ax[0].set_title('q')
ax[1].set_title('w')
ax[0].set_xlabel('number of modes')
ax[0].set_ylabel('avg mse')
ax[1].set_ylabel('avg mse')
#ax.set_title('N=500')
#ax.set_ylim(0,0.2)
#ax.set_yscale('log')
fig.savefig(output_path1+'/mse_mmss_sep_5000.png')

