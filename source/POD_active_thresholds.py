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
import csv
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import percentileofscore
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import json
import gc
from Eval_Functions import *
from Plotting_Functions import *
from POD_functions import *

import sys
sys.stdout.reconfigure(line_buffering=True)

from docopt import docopt
args = docopt(__doc__)

input_path = args['--input_path']
output_path = args['--output_path']
snapshots_POD = int(args['--snapshots'])
projection = args['--projection']
modes_no = int(args['--modes_no'])
print(projection)
if projection == 'False':
    projection = False
elif projection == 'True':
    projection = True

output_path = output_path+f"/snapshots{snapshots_POD}/"
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

def load_data_set_TD(file, names, snapshots):
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

def load_data_set_RB(file, names, snapshots):
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

def load_data_set_RB_act(file, names, snapshots):
    with h5py.File(file, 'r') as hf:
        print(hf.keys())
        time_vals = np.array(hf['total_time_all'][:snapshots])
        
        data = np.zeros((len(time_vals), len(x), len(z), len(names)))
        
        index=0
        for name in names:
            print(name)
            print(hf[name])
            Var = np.array(hf[name])
            print(np.shape(Var))
            if index == 4:
                data[:,:,:,index] = Var[:snapshots,:,:]
            else:
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

#### LOAD DATA AND POD ####
Data = 'RB'
if Data == 'ToyData':
    name = names = variables = ['combined']
    n_components = 3
    num_variables = 1
    snapshots = 25000
    data_set, x, z, time_vals = load_data_set_TD(input_path+'/plume_wave_dataset_smallergrid_longertime.h5', name, snapshots)
    print('shape of dataset', np.shape(data_set))
    dt = 0.05

elif Data == 'RB':
    variables = ['q_all', 'w_all', 'u_all', 'b_all']
    names = ['q', 'w', 'u', 'b']
    x = np.load(input_path+'/x.npy')
    z = np.load(input_path+'/z.npy')
    snapshots_load = 16000
    data_set, time_vals = load_data_set_RB(input_path+'/data_4var_5000_48000.h5', variables, snapshots_load)
    print('shape of dataset', np.shape(data_set))
    dt = 2

elif Data =='RBplusActive':
    variables = ['q_all', 'w_all', 'u_all', 'b_all', 'active_array']
    names = ['q', 'w', 'u', 'b', 'active']
    x = np.load(input_path+'/x.npy')
    z = np.load(input_path+'/z.npy')
    snapshots_load = 16000
    data_set, time_vals = load_data_set_RB_act(input_path+'/data_4var_5000_48000_act.h5', variables, snapshots_load)
    print('shape of dataset', np.shape(data_set))
    dt = 2

#### plot dataset ####
if Data == 'ToyData':
    fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
    c1 = ax.pcolormesh(time_vals[:500], x, data_set[:500,:,32,0].T)
    fig.colorbar(c1, ax=ax, label=r"$h$")
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel(r"$x$", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    fig.savefig(input_path+'/combined_hovmoller_small_domain.png')
if Data == 'RB':
    fig, ax = plt.subplots(len(variables), figsize=(12, 3*len(variables)), tight_layout=True, sharex=True)
    for i in range(len(variables)):
        if len(variables) == 1:
            ax.pcolormesh(time_vals, x, data_set[:,:,32,i].T)
            ax.set_ylabel('x')
            ax.set_xlabel('time')
            ax.set_title(names[i])
        else:
            ax[i].pcolormesh(time_vals[:500], x, data_set[:500,:,32,i].T)
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

snapshots_POD  = snapshots_POD
print('snapshots for POD', snapshots_POD)
data_set       = data_set[:snapshots_POD]
data_scaled    = data_scaled[:snapshots_POD]
time_vals      = time_vals[:snapshots_POD]

global_stds = [np.std(data_set[..., c]) for c in range(data_set.shape[-1])]

### ---- find true active array ----
active_array, mask_expanded, rh, w, b_anom = active_array_truth(data_set, z)

new_array = np.zeros((len(time_vals), len(x) // 4, len(z) // 4))

# Iterate over the time dimension
for t in range(len(time_vals)):
    # Iterate over the 64 subgrids along the z-axis (along the x-dimension)
    for i in range(0, len(x), 4):
        # Iterate over the 16 subgrids along the x-axis (along the z-dimension)
        for j in range(0, len(z), 4):
            # Check if any value in the 4x4 subgrid is 1
            if np.any(active_array[t, i:i+4, j:j+4] == 1):
                new_array[t, i // 4, j // 4] = 1

x_downsample = x[::4]
z_downsample = z[::4]

### ----- POD ------
n_modes = 64
c_names = f"Ra2e8_c{n_modes}"
index=0

output_path = output_path+f"/modes{n_modes}/ActiveThresholdsNew/"
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

preload_data = True

if preload_data:
    print('loading in data ...')
    data_reconstructed = np.load(output_path+'data_reconstructed.npy')
else:
    data_reduced, data_reconstructed_reshaped, data_reconstructed, pca_, cev = POD(data_scaled, n_modes, x, z, variables, output_path+c_names[index], Plotting=True)
    np.save(output_path+'data_reconstructed.npy', data_reconstructed)

if scaling == 'SS':
    data_reconstructed = ss_inverse_transform(data_reconstructed, scaler)

# if Data == 'RBplusActive':
#     active_array               = data_set[...,4]
#     active_array_reconstructed = data_reconstructed[...,4]
#     mask                       = (active_array == 1)
#     mask_reconstructed         = (active_array_reconstructed == 1)

#     # Expand the mask to cover all features (optional, depending on use case)
#     mask               =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
#     mask_reconstructed =  np.repeat(mask_reconstructed[:, :, :, np.newaxis], 4, axis=-1) # Shape: (256, 64, 1)
    
#     nrmse_plume            = NRMSE(data_set[:,:,:,:4][mask], data_reconstructed[:,:,:,:4][mask])

#     mask_original     = mask[..., 0]
#     nrmse_sep_plume   = NRMSE_per_channel_masked(data_set, data_reconstructed, mask_original, global_stds) 

# elif Data == 'RB':
#     _, active_array_reconstructed, _, mask_reconstructed = active_array_calc(data_set, data_reconstructed, z)
#     mask_original     = mask_expanded[..., 0]

#     new_array_reconstructed = np.zeros((len(time_vals), len(x) // 4, len(z) // 4))

#     # Iterate over the time dimension
#     for t in range(len(time_vals)):
#         # Iterate over the 64 subgrids along the z-axis (along the x-dimension)
#         for i in range(0, len(x), 4):
#             # Iterate over the 16 subgrids along the x-axis (along the z-dimension)
#             for j in range(0, len(z), 4):
#                 # Check if any value in the 4x4 subgrid is 1
#                 if np.any(active_array_reconstructed[t, i:i+4, j:j+4] == 1):
#                     new_array_reconstructed[t, i // 4, j // 4] = 1

### ------- Precision, Recall, F1 ------
def plume_detection_metrics(new_array, new_array_reconstructed):
    TP = np.sum((new_array == 1) & (new_array_reconstructed == 1))
    FP = np.sum((new_array == 0) & (new_array_reconstructed == 1))
    FN = np.sum((new_array == 1) & (new_array_reconstructed == 0))
    TN = np.sum((new_array == 0) & (new_array_reconstructed == 0))

    precision = TP / (TP + FP + 1e-8)   # of predicted plumes, how many are real
    recall    = TP / (TP + FN + 1e-8)   # of real plumes, how many are captured
    f1_score  = 2 * precision * recall / (precision + recall + 1e-8)

    csi = TP / (TP + FP + FN + 1e-8)

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'csi': csi
    }

# metrics = plume_detection_metrics(new_array, new_array_reconstructed)
# print("For original active plume criteria:")
# print(f"Precision = {metrics['precision']}")
# print(f"Recall = {metrics['recall']}")
# print(f"F1 = {metrics['f1']}")
# print(f"CSI = {metrics['csi']}")

### ---- plot figures -----
cmap = ListedColormap(['white', 'tab:orange'])
bounds = [0, 0.5, 1]  # boundaries between colors
norm = BoundaryNorm(bounds, cmap.N)

cmap2 = ListedColormap(['white', 'tab:blue'])
bounds2 = [0, 0.5, 1]  # boundaries between colors
norm2 = BoundaryNorm(bounds2, cmap2.N)

# fig, ax = plt.subplots(2, figsize=(12,6), constrained_layout=True)
# index = 0
# print(f"Figure at {z_downsample[8]}")
# ax[0].contourf(time_vals[index:index+500], x_downsample, new_array[index:index+500, :, 8].T, cmap=cmap, norm=norm)
# ax[1].contourf(time_vals[index:index+500], x_downsample, new_array_reconstructed[index:index+500, :, 8].T, cmap=cmap, norm=norm)
# fig.savefig(output_path+f"/actual_criteria_{int(time_vals[index])}.png")

# fig, ax = plt.subplots(2, figsize=(12,6), constrained_layout=True)
# index = 5000
# print(f"Figure at {z_downsample[8]}")
# ax[0].contourf(time_vals[index:index+500], x_downsample, new_array[index:index+500, :, 8].T, cmap=cmap, norm=norm)
# ax[1].contourf(time_vals[index:index+500], x_downsample, new_array_reconstructed[index:index+500, :, 8].T, cmap=cmap, norm=norm)
# fig.savefig(output_path+f"/actual_criteria_{int(time_vals[index])}.png")

index = 5000
data_set = data_set[index:index+500]
data_reconstructed = data_reconstructed[index:index+500]
time_vals = time_vals[index:index+500]

_, active_array_reconstructed, _, mask_reconstructed = active_array_calc(data_set, data_reconstructed, z)
mask_original     = mask_expanded[..., 0]

new_array_reconstructed = np.zeros((len(time_vals), len(x) // 4, len(z) // 4))

# Iterate over the time dimension
for t in range(len(time_vals)):
    # Iterate over the 64 subgrids along the z-axis (along the x-dimension)
    for i in range(0, len(x), 4):
        # Iterate over the 16 subgrids along the x-axis (along the z-dimension)
        for j in range(0, len(z), 4):
            # Check if any value in the 4x4 subgrid is 1
            if np.any(active_array_reconstructed[t, i:i+4, j:j+4] == 1):
                new_array_reconstructed[t, i // 4, j // 4] = 1

fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
print(f"Figure at {z_downsample[8]}")
cf = ax.contourf(time_vals, x_downsample, new_array[index:index+500, :, 8].T, cmap=cmap2, norm=norm, alpha=0.75)
cf = ax.contourf(time_vals, x_downsample, new_array_reconstructed[:, :, 8].T, cmap=cmap, norm=norm,alpha=0.5)
# cbar = fig.colorbar(cf, ax=ax, ticks=[0, 1])
# cbar.set_label('Active', fontsize=14)  
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('x', fontsize=14)
fig.savefig(output_path+f"/original_true_{int(time_vals[0])}.png")


# fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
# print(f"Figure at {z_downsample[8]}")
# cf = ax.contourf(time_vals, x_downsample, new_array_reconstructed[:, :, 8].T, cmap=cmap, norm=norm)
# cbar = fig.colorbar(cf, ax=ax, ticks=[0, 1])
# cbar.set_label('Active', fontsize=14)  
# ax.set_xlabel('Time', fontsize=14)
# ax.set_ylabel('x', fontsize=14)
# fig.savefig(output_path+f"/original_{int(time_vals[0])}.png")

### ---- change the thresholds ----
# from itertools import product

# RH_thresholds = [0.91, 0.92, 0.93, 0.94]
# w_thresholds  = [-0.02, -0.01, 0]
# b_thresholds  = [-0.02, -0.01, 0]

# threshold_grid = list(product(RH_thresholds, w_thresholds, b_thresholds))
# total_tests = len(RH_thresholds) * len(w_thresholds) * len(b_thresholds)

# results = []

# if Data == 'RBplusActive':
#     active_array               = data_set[...,4]
#     active_array_reconstructed = data_reconstructed[...,4]
#     mask                       = (active_array == 1)
#     mask_reconstructed         = (active_array_reconstructed == 1)

#     # Expand the mask to cover all features (optional, depending on use case)
#     mask               =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
#     mask_reconstructed =  np.repeat(mask_reconstructed[:, :, :, np.newaxis], 4, axis=-1) # Shape: (256, 64, 1)
    
#     nrmse_plume            = NRMSE(data_set[:,:,:,:4][mask], data_reconstructed[:,:,:,:4][mask])

#     mask_original     = mask[..., 0]
#     nrmse_sep_plume   = NRMSE_per_channel_masked(data_set, data_reconstructed, mask_original, global_stds) 

# elif Data == 'RB':
#     test_index = 1
#     T_total = data_scaled.shape[0]
#     batch_size = 500  # adjust based on memory
#     block_size_x, block_size_z = 4, 4
#     nx, nz = 256, 64
#     nx_new, nz_new = nx // block_size_x, nz // block_size_z

#     for RH_thresh, w_thresh, b_thresh in threshold_grid:
#         print(f"Test {test_index} of {total_tests}")

#         # Temporary storage for all batches
#         all_downsampled = []

#         # Loop over batches to compute active array and downsample
#         for start in range(0, T_total, batch_size):
#             end = min(start + batch_size, T_total)
#             batch_set = data_set[start:end]
#             batch_reconstructed = data_reconstructed[start:end]

#             # Compute active array for this batch & threshold
#             _, active_array_batch, _, _ = active_array_calc_softer(
#                 batch_set, batch_reconstructed, z,
#                 RH_threshold=RH_thresh, w_threshold=w_thresh, b_threshold=b_thresh
#             )
#             active_array_batch = active_array_batch.astype(np.bool_)  # save memory

#             # Vectorized 4x4 downsampling
#             T_batch = active_array_batch.shape[0]
#             downsampled_batch = active_array_batch.reshape(
#                 T_batch, nx_new, block_size_x, nz_new, block_size_z
#             ).max(axis=(2, 4))  # shape (T_batch, 64, 16)

#             all_downsampled.append(downsampled_batch)

#         # Concatenate all batches for this threshold
#         full_downsampled = np.concatenate(all_downsampled, axis=0)

#         # Compute metrics once for the full dataset
#         metrics = plume_detection_metrics(new_array, full_downsampled)
#         metrics.update({'RH_thresh': RH_thresh, 'w_thresh': w_thresh, 'b_thresh': b_thresh})
#         results.append(metrics)

#         test_index += 1

#     # Make sure results is not empty
#     if len(results) == 0:
#         raise ValueError("No results to save!")

#     # Field names (columns) inferred from keys of the first dict
#     fieldnames = list(results[0].keys())

#     # Save to CSV
#     with open(output_path+'plume_metrics.csv', 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()       # write column headers
#         writer.writerows(results)  # write all rows

#     print("Saved metrics to plume_metrics.csv")

### --- plot with softer threshold ---

index = 5000
data_set = data_set[index:index+500]
data_reconstructed = data_reconstructed[index:index+500]
time_vals = time_vals[index:index+500]

RH_thresh = 0.92
w_thresh  = 0
b_thresh  = 0

_, active_array_reconstructed, _, mask_reconstructed = active_array_calc_softer(data_set, data_reconstructed, z, RH_threshold=RH_thresh, w_threshold=w_thresh, b_threshold=b_thresh)
mask_original     = mask_expanded[..., 0]

new_array_reconstructed = np.zeros((len(time_vals), len(x) // 4, len(z) // 4))

# Iterate over the time dimension
for t in range(len(time_vals)):
    # Iterate over the 64 subgrids along the z-axis (along the x-dimension)
    for i in range(0, len(x), 4):
        # Iterate over the 16 subgrids along the x-axis (along the z-dimension)
        for j in range(0, len(z), 4):
            # Check if any value in the 4x4 subgrid is 1
            if np.any(active_array_reconstructed[t, i:i+4, j:j+4] == 1):
                new_array_reconstructed[t, i // 4, j // 4] = 1

# Binary colormap: white=0, red=1
cmap = ListedColormap(['white', 'tab:orange'])
bounds = [0, 0.5, 1]  # boundaries between colors
norm = BoundaryNorm(bounds, cmap.N)

cmap2 = ListedColormap(['white', 'tab:blue'])
bounds2 = [0, 0.5, 1]  # boundaries between colors
norm2 = BoundaryNorm(bounds2, cmap2.N)

# fig, ax = plt.subplots(2, figsize=(12,6), constrained_layout=True)
# print(f"Figure at {z_downsample[8]}")
# ax[0].contourf(time_vals, x_downsample, new_array[index:index+500, :, 8].T, cmap=cmap, norm=norm)
# ax[1].contourf(time_vals, x_downsample, new_array_reconstructed[:, :, 8].T, cmap=cmap, norm=norm)
# fig.savefig(output_path+f"/soft_criteria_{int(time_vals[0])}.png")


# fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
# print(f"Figure at {z_downsample[8]}")
# cf = ax.contourf(time_vals, x_downsample, new_array[index:index+500, :, 8].T, cmap=cmap, norm=norm)
# cbar = fig.colorbar(cf, ax=ax, ticks=[0, 1])
# cbar.set_label('Active', fontsize=14)  
# ax.set_xlabel('Time', fontsize=14)
# ax.set_ylabel('x', fontsize=14)
# fig.savefig(output_path+f"/true_{int(time_vals[0])}.png")

# fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
# print(f"Figure at {z_downsample[8]}")
# cf = ax.contourf(time_vals, x_downsample, new_array_reconstructed[:, :, 8].T, cmap=cmap, norm=norm)
# cbar = fig.colorbar(cf, ax=ax, ticks=[0, 1])
# cbar.set_label('Active', fontsize=14)  
# ax.set_xlabel('Time', fontsize=14)
# ax.set_ylabel('x', fontsize=14)
# fig.savefig(output_path+f"/soft_{int(time_vals[0])}.png")

fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
print(f"Figure at {z_downsample[8]}")
cf = ax.contourf(time_vals, x_downsample, new_array[index:index+500, :, 8].T, cmap=cmap2, norm=norm, alpha=0.75)
cf = ax.contourf(time_vals, x_downsample, new_array_reconstructed[:, :, 8].T, cmap=cmap, norm=norm,alpha=0.5)
# cbar = fig.colorbar(cf, ax=ax, ticks=[0, 1])
# cbar.set_label('Active', fontsize=14)  
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('x', fontsize=14)
fig.savefig(output_path+f"/soft_true_{int(time_vals[0])}.png")
