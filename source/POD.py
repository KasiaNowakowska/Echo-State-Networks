"""
python script for POD.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path> --snapshots=<snapshots> --projection=<projection> --modes_no=<modes_no>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
    --snapshots=<snapshots>            number of snapshots 
    --projection=<projection>          projection of POD [default: False]
    --modes_no=<modes_no>                    number of modes
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
import json
from Eval_Functions import *

import sys
sys.stdout.reconfigure(line_buffering=True)

from docopt import docopt
args = docopt(__doc__)

input_path = args['--input_path']
output_path = args['--output_path']
snapshots = int(args['--snapshots'])
projection = args['--projection']
modes_no = int(args['--modes_no'])
print(projection)
if projection == 'False':
    projection = False
elif projection == 'True':
    projection = True

output_path = output_path+f"/snapshots{snapshots}/"
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

def load_data(file, name):
    with h5py.File(file, 'r') as hf:
        print(hf.keys())
        print(name)
        print(hf[name])
        data = np.array(hf[name])

        x = hf['x'][:]  # 1D x-axis
        z = hf['z'][:]  # 1D z-axis
        time = hf['total_time'][:]  # 1D time vector

    return data, x, z, time

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

def load_data_set_Ra2e7(file, names, snapshots):
    with h5py.File(file, 'r') as hf:
        print(hf.keys())
        time_vals = np.array(hf['total_time'][:snapshots])
        
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
    np.save(output_path+file_str+'_components.npy', components)
    np.save(output_path+file_str+'_cumEV.npy', cumulative_explained_variance)

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
            for v in range(data.shape[-1]):
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

def transform_POD(data, pca_):
    if data.ndim == 3: #len(time_vals), len(x), len(z)
            data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    elif data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
            data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2]*data.shape[3])
    data_reduced = pca_.transform(data_matrix)
    print('shape of reduced_data:', np.shape(data_reduced))
    return data_reduced

'''
#### plume ###
data_names = ['plume', 'wave', 'combined']
n_modes = [1, 2, 3]
for index, name in enumerate(data_names):
    data_set, x, z, time = load_data(input_path+'/plume_wave_dataset.h5', name)
    data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_ = POD(data_set, n_modes[index], name)
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
    }

    with open(output_path_met, "w") as file:
        json.dump(metrics, file, indent=4)
'''

variables = ['q_all', 'w_all', 'u_all', 'b_all']
names = ['q', 'w', 'u', 'b']
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
snapshots = 11200 #snapshots
data_set, time_vals = load_data_set(input_path+'/data_4var_5000_48000.h5', variables, snapshots)
#variables = ['q_vertical', 'w_vertical', 'u_vertical', 'b_vertical']
#data_set, time_vals = load_data_set_Ra2e7(input_path+'/data_all.h5', variables, snapshots)
print('shape of dataset', np.shape(data_set))

#### change chape of dataset/add projecyion ####
reduce_data_set = False
if reduce_data_set:
    data_set = data_set[200:392,60:80,:,:]
    x = x[60:80]
    time_vals = time_vals[200:392]
    print('reduced domain shape', np.shape(data_set))
    print('reduced x domain', np.shape(x))
    print('reduced x domain', len(x))
    print(x[0], x[-1])

reduce_data_set2 = False
if reduce_data_set2:
    data_set = data_set[:4650,128:160,:,:] # 10LTs washout, 200LTs train, 1000LTs test
    x = x[128:160]
    time_vals = time_vals[:4650]
    print('reduced domain shape', np.shape(data_set))
    print('reduced x domain', np.shape(x))
    print('reduced x domain', len(x))
    print(x[0], x[-1])


projection = projection
if projection:
    print('starting projection since projection', projection)
    data_proj = data_set[16000:20000, :, :, :] #16000:20000
    data_set = data_set[:11200, :, :, :] #:11200
    time_vals_proj = time_vals[16000:20000]
    time_vals = time_vals[:11200]
    print('reduced dataset', np.shape(data_set))
    print('reduced time', np.shape(time))
    print('proejction dataset', np.shape(data_proj))
    print('time of projection', time_vals_proj[0], time_vals_proj[1])
    print(x[0], x[-1])

    # data_proj = data_set[500:, :, :, :] #16000:20000
    # data_set = data_set[:500, :, :, :] #:11200
    # time_vals_proj = time_vals[500:]
    # time_vals = time_vals[:500]
    # print('reduced dataset', np.shape(data_set))
    # print('reduced time', np.shape(time))
    # print('proejction dataset', np.shape(data_proj))
    # print(x[0], x[-1])

#### plot dataset ####
fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
c1 = ax.pcolormesh(time_vals, x, data_set[:,:,32,0].T)
fig.colorbar(c1, ax=ax)
fig.savefig(input_path+'/combined_hovmoller_small_domain.png')

fig, ax = plt.subplots(len(variables), figsize=(12, 3*len(variables)), tight_layout=True, sharex=True)
for i in range(len(variables)):
    if len(variables) == 1:
        ax.pcolormesh(time_vals, x, data_set[:,:,32,i].T)
        ax.set_ylabel('x')
        ax.set_xlabel('time')
        ax.set_title(names[i])
    else:
        ax[i].pcolormesh(time_vals, x, data_set[:,:,32,i].T)
        ax[i].set_ylabel('x')
        ax[i].set_title(names[i])
        ax[-1].set_xlabel('time')
fig.savefig(output_path+'/hovmoller.png')

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

n_modes_list = [modes_no]#[16,32,64,100,128,256] #[10, 16, 32, 64, 100]
#n_modes_list = np.arange(16, 192+16, 16)
c_names = [f'Ra2e8_c{n}' for n in n_modes_list]
#n_modes_list = [4, 8, 16, 25, 32, 64]
#c_names = ['Ra2e8_c4', 'Ra2e8_c8', 'Ra2e8_c16', 'Ra2e8_c32', 'Ra2e8_c64', 'Ra2e8_c100','Ra2e8_c128'] #['Ra2e8_c10', 'Ra2e8_c16', 'Ra2e8_c32', 'Ra2e8_c64', 'Ra2e8_c100']
index=0

nrmse_list, evr_list, ssim_list, cumEV_list, nrmse_plume_list = [], [], [], [], []
pnrmse_list, pevr_list, pssim_list, pnrmse_plume_list = [], [], [], []

POD_type = 'together'
if POD_type == 'together':
    for n_modes in n_modes_list:
        data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_, cev = POD(data_scaled, n_modes, c_names[index])
        if scaling == 'SS':
            data_reconstructed = ss_inverse_transform(data_reconstructed, scaler)
        #plot_reconstruction(data_set, data_reconstructed, 32, 20, 'Ra2e7')
        plot_reconstruction_and_error(data_set, data_reconstructed, 32, 75, x, z, time_vals, names, c_names[index])
        nrmse = NRMSE(data_set, data_reconstructed)
        mse   = MSE(data_set, data_reconstructed)
        evr   = EVR_recon(data_set, data_reconstructed)
        SSIM  = compute_ssim_for_4d(data_set, data_reconstructed)

        if len(variables) == 4:
            active_array, active_array_reconstructed, mask, mask_reconstructed = active_array_calc(data_set, data_reconstructed, z)
            print(np.shape(active_array))
            print(np.shape(mask))
            nrmse_plume             = NRMSE(data_set[:,:,:,:][mask], data_reconstructed[:,:,:,:][mask])

            fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
            c1 = ax[0].contourf(time_vals, x, active_array[:,:, 32].T, cmap='Reds')
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].contourf(time_vals, x, active_array_reconstructed[:,:, 32].T, cmap='Reds')
            fig.colorbar(c1, ax=ax[1])
            ax[1].set_title('reconstruction')
            for v in range(2):
                ax[v].set_xlabel('time')
                ax[v].set_ylabel('x')
            fig.savefig(output_path+f"/active_plumes_{c_names[index]}.png")
            plt.close()
        else:
            nrmse_plume = np.inf

        ### plt part of domain ###
        plot_reconstruction_and_error(data_set[:500], data_reconstructed[:500], 32, 75, x, z, time_vals[:500], names, c_names[index]+'_500')


        if projection:
            print('starting projection')
            data_scaled_proj           = ss_transform(data_proj, scaler)
            data_reduced_proj          = transform_POD(data_scaled_proj, pca_)
            _, data_reconstructed_proj = inverse_POD(data_reduced_proj, pca_)
            if scaling == 'SS':
                data_reconstructed_proj     = ss_inverse_transform(data_reconstructed_proj, scaler)

            plot_reconstruction_and_error(data_proj, data_reconstructed_proj, 32, 20, time_vals_proj, 'proj_'+c_names[index])
            nrmse_proj = NRMSE(data_proj, data_reconstructed_proj)
            mse_proj   = MSE(data_proj, data_reconstructed_proj)
            evr_proj   = EVR_recon(data_proj, data_reconstructed_proj)
            SSIM_proj  = compute_ssim_for_4d(data_proj, data_reconstructed_proj)

            if len(variables) == 4:
                active_array, active_array_reconstructed, mask, mask_reconstructed = active_array_calc(data_proj, data_reconstructed_proj, z)
                print(np.shape(active_array))
                print(np.shape(mask))
                nrmse_plume_proj             = NRMSE(data_proj[:,:,:,:][mask], data_reconstructed_proj[:,:,:,:][mask])

                fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
                c1 = ax[0].contourf(time_vals_proj, x, active_array[:,:, 32].T, cmap='Reds')
                fig.colorbar(c1, ax=ax[0])
                ax[0].set_title('true')
                c2 = ax[1].contourf(time_vals_proj, x, active_array_reconstructed[:,:, 32].T, cmap='Reds')
                fig.colorbar(c1, ax=ax[1])
                ax[1].set_title('reconstruction')
                for v in range(2):
                    ax[v].set_xlabel('time')
                    ax[v].set_ylabel('x')
                fig.savefig(output_path+f"/proj_active_plumes_{c_names[index]}.png")
                plt.close()
            else:
                nrmse_plume_proj = np.inf



        print('NRMSE', nrmse)
        print('MSE', mse)
        print('EVR_recon', evr)
        print('SSIM', SSIM)
        print('NRMSE plume', nrmse_plume)

        # Full path for saving the file
        output_file = c_names[index] + '_metrics.json' 

        output_path_met = os.path.join(output_path, output_file)

        metrics = {
        "no. modes": n_modes,
        "EVR": evr,
        "MSE": mse,
        "NRMSE": nrmse,
        "SSIM": SSIM,
        "cumEV from POD": cev,
        "NRMSE plume": nrmse_plume,
        }

        with open(output_path_met, "w") as file:
            json.dump(metrics, file, indent=4)

        nrmse_list.append(nrmse)
        ssim_list.append(SSIM)
        evr_list.append(evr)
        cumEV_list.append(cev)
        nrmse_plume_list.append(nrmse_plume)

        if projection:
            # Full path for saving the file
            output_file = c_names[index] + '_proj_metrics.json' 

            output_path_met = os.path.join(output_path, output_file)

            metrics = {
            "no. modes": n_modes,
            "start time of projection": time_vals_proj[0], 
            "end time of projection": time_vals_proj[1],
            "EVR": evr_proj,
            "MSE": mse_proj,
            "NRMSE": nrmse_proj,
            "SSIM": SSIM_proj,
            "NRMSE plume": nrmse_plume_proj,
            }

            with open(output_path_met, "w") as file:
                json.dump(metrics, file, indent=4)

            pnrmse_list.append(nrmse_proj)
            pssim_list.append(SSIM_proj)
            pevr_list.append(evr_proj)
            pnrmse_plume_list.append(nrmse_plume_proj)

        index +=1

if POD_type == 'seperate':
    for n_modes in n_modes_list:
        recon_per_variable = []
        data_true_per_variable = []

        for V in range(len(variables)):
            data_scaled_sep = data_scaled[:, :, :, V:V+1]
            print('data scaled seperate shape', np.shape(data_scaled_sep))

            data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_, cev = POD(data_scaled_sep, n_modes, c_names[index])
            print('shape of data_reconstructed from POD', np.shape(data_reconstructed))
            #plot_reconstruction(data_set, data_reconstructed, 32, 20, 'Ra2e7')
            recon_per_variable.append(data_reconstructed)
            
            
        data_reconstructed = np.concatenate(recon_per_variable, axis=-1)
        print('shape of data reconstructed after stiching varibales together', np.shape(data_reconstructed))
        if scaling == 'SS':
            print('applying inverse scaling')
            data_reconstructed = ss_inverse_transform(data_reconstructed, scaler)
        
        plot_reconstruction_and_error(data_set, data_reconstructed, 32, 20, time_vals, c_names[index])
        nrmse = NRMSE(data_set, data_reconstructed)
        mse   = MSE(data_set, data_reconstructed)
        evr   = EVR_recon(data_set, data_reconstructed)
        SSIM  = compute_ssim_for_4d(data_set, data_reconstructed)

        if len(variables) == 4:
            active_array, active_array_reconstructed, mask, mask_reconstructed = active_array_calc(data_set, data_reconstructed, z)
            print(np.shape(active_array))
            print(np.shape(mask))
            nrmse_plume             = NRMSE(data_set[:,:,:,:][mask], data_reconstructed[:,:,:,:][mask])

            fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
            c1 = ax[0].contourf(time_vals, x, active_array[:,:, 32].T, cmap='Reds')
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].contourf(time_vals, x, active_array_reconstructed[:,:, 32].T, cmap='Reds')
            fig.colorbar(c1, ax=ax[1])
            ax[1].set_title('reconstruction')
            for v in range(2):
                ax[v].set_xlabel('time')
                ax[v].set_ylabel('x')
            fig.savefig(output_path+f"/active_plumes_{c_names[index]}.png")
            plt.close()
        else:
            nrmse_plume = np.inf

        if projection:
            print('starting projection')
            data_reduced_proj          = transform_POD(data_proj, pca_)
            _, data_reconstructed_proj = inverse_POD(data_reduced_proj, pca_)
            if scaling == 'SS':
                data_reconstructed_proj     = ss_inverse_transform(data_reconstructed_proj, scaler)

            plot_reconstruction_and_error(data_proj[:1000], data_reconstructed_proj[:1000], 32, 20, time_vals_proj, 'proj_partial_'+c_names[index])
            nrmse_proj = NRMSE(data_proj, data_reconstructed_proj)
            mse_proj   = MSE(data_proj, data_reconstructed_proj)
            evr_proj   = EVR_recon(data_proj, data_reconstructed_proj)
            SSIM_proj  = compute_ssim_for_4d(data_proj, data_reconstructed_proj)

            if len(variables) == 4:
                active_array, active_array_reconstructed, mask, mask_reconstructed = active_array_calc(data_proj, data_reconstructed_proj, z)
                print(np.shape(active_array))
                print(np.shape(mask))
                nrmse_plume_proj             = NRMSE(data_proj[:,:,:,:][mask], data_reconstructed_proj[:,:,:,:][mask])

                fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
                c1 = ax[0].contourf(time_vals_proj, x, active_array[:,:, 32].T, cmap='Reds')
                fig.colorbar(c1, ax=ax[0])
                ax[0].set_title('true')
                c2 = ax[1].contourf(time_vals_proj, x, active_array_reconstructed[:,:, 32].T, cmap='Reds')
                fig.colorbar(c1, ax=ax[1])
                ax[1].set_title('reconstruction')
                for v in range(2):
                    ax[v].set_xlabel('time')
                    ax[v].set_ylabel('x')

                fig.savefig(output_path+f"/proj_active_plumes_{c_names[index]}.png")
                plt.close()
            else:
                nrmse_plume_proj = np.inf



        print('NRMSE', nrmse)
        print('MSE', mse)
        print('EVR_recon', evr)
        print('SSIM', SSIM)
        print('NRMSE plume', nrmse_plume)

        # Full path for saving the file
        output_file = c_names[index] + '_metrics.json' 

        output_path_met = os.path.join(output_path, output_file)

        metrics = {
        "no. modes": n_modes,
        "EVR": evr,
        "MSE": mse,
        "NRMSE": nrmse,
        "SSIM": SSIM,
        "cumEV from POD": cev,
        "NRMSE plume": nrmse_plume,
        }

        with open(output_path_met, "w") as file:
            json.dump(metrics, file, indent=4)

        nrmse_list.append(nrmse)
        ssim_list.append(SSIM)
        evr_list.append(evr)
        cumEV_list.append(cev)
        nrmse_plume_list.append(nrmse_plume)


np.save(output_path+'/ssim_list.npy', ssim_list)
np.save(output_path+'/evr_list.npy', evr_list)
np.save(output_path+'/nrmse_list.npy', nrmse_list)
np.save(output_path+'/cumEV_list.npy', cumEV_list)
np.save(output_path+'/nrmse_plume_list.npy', nrmse_plume_list)

fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
ax.plot(n_modes_list, ssim_list)
ax.set_xlabel('modes')
ax.set_ylabel('SSIM')
ax.grid()
fig.savefig(output_path+'/SSIM_list.png')
plt.close()

fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
ax.plot(n_modes_list, nrmse_list)
ax.set_xlabel('modes')
ax.set_ylabel('NRMSE')
ax.grid()
fig.savefig(output_path+'/NRMSE_list.png')
plt.close()

fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
ax.plot(n_modes_list, evr_list)
ax.set_xlabel('modes')
ax.set_ylabel('EVR')
ax.grid()
fig.savefig(output_path+'/EVR_list.png')
plt.close()

fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
ax.plot(n_modes_list, cumEV_list)
ax.set_xlabel('modes')
ax.set_ylabel('cumulative equivalence ratio')
ax.grid()
fig.savefig(output_path+'/cumEV_list.png')
plt.close()

fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
ax.plot(n_modes_list, nrmse_plume_list)
ax.set_xlabel('modes')
ax.set_ylabel('NRMSE in plume')
ax.grid()
fig.savefig(output_path+'/nrmse_plume_list.png')
plt.close()

if projection:
    np.save(output_path+'/proj_ssim_list.npy', pssim_list)
    np.save(output_path+'/proj_evr_list.npy', pevr_list)
    np.save(output_path+'/proj_nrmse_list.npy', pnrmse_list)
    np.save(output_path+'/proj_nrmse_plume_list.npy', pnrmse_plume_list)

    fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
    ax.plot(n_modes_list, pssim_list)
    ax.set_xlabel('modes')
    ax.set_ylabel('SSIM')
    ax.grid()
    fig.savefig(output_path+'/pSSIM_list.png')
    plt.close()

    fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
    ax.plot(n_modes_list, pnrmse_list)
    ax.set_xlabel('modes')
    ax.set_ylabel('NRMSE')
    ax.grid()
    fig.savefig(output_path+'/pNRMSE_list.png')
    plt.close()

    fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
    ax.plot(n_modes_list, pevr_list)
    ax.set_xlabel('modes')
    ax.set_ylabel('EVR')
    ax.grid()
    fig.savefig(output_path+'/pEVR_list.png')
    plt.close()

    fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
    ax.plot(n_modes_list, pnrmse_plume_list)
    ax.set_xlabel('modes')
    ax.set_ylabel('NRMSE in plume')
    ax.grid()
    fig.savefig(output_path+'/pnrmse_plume_list.png')
    plt.close()

