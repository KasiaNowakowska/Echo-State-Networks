"""
python script for POD.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
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

import sys
sys.stdout.reconfigure(line_buffering=True)

from docopt import docopt
args = docopt(__doc__)

input_path = args['--input_path']
output_path = args['--output_path']

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

'''
snapshots = [500, 2500, 5000]
modes = [10, 16, 32, 64, 100]

NRMSE_vals = np.zeros((len(modes), len(snapshots)))
cumEV_vals = np.zeros((len(modes), len(snapshots)))
ssim_vals  = np.zeros((len(modes), len(snapshots)))
NRMSE_plume_vals = np.zeros((len(modes), len(snapshots)))

for s in range(len(snapshots)):
    no_snaps         = snapshots[s]
    input_path_snaps = input_path+f"/snapshots{no_snaps}/"

    NRMSE_vals[:, s] = np.load(input_path_snaps + '/nrmse_list.npy')
    cumEV_vals[:, s] = np.load(input_path_snaps + '/cumEV_list.npy')
    ssim_vals[:, s]  = np.load(input_path_snaps + '/ssim_list.npy')
    NRMSE_plume_vals[:, s]  = np.load(input_path_snaps + '/nrmse_plume_list.npy')

fig, ax = plt.subplots(1, figsize=(8,6))
for s in range(len(snapshots)):
    ax.plot(modes, NRMSE_vals[:, s], label=f"snapshots={snapshots[s]}")
ax.set_xlabel('modes')
ax.set_ylabel('NRMSE')
ax.legend()
ax.grid()
fig.savefig(output_path+'/NRMSE.png')

fig, ax = plt.subplots(1, figsize=(8,6))
for s in range(len(snapshots)):
    ax.plot(modes, NRMSE_plume_vals[:, s], label=f"snapshots={snapshots[s]}")
ax.set_xlabel('modes')
ax.set_ylabel('NRMSE in plume')
ax.legend()
ax.grid()
fig.savefig(output_path+'/NRMSE_plume.png')

fig, ax = plt.subplots(1, figsize=(8,6))
for s in range(len(snapshots)):
    ax.plot(modes, ssim_vals[:, s],label=f"snapshots={snapshots[s]}")
ax.set_xlabel('modes')
ax.set_ylabel('SSIM')
ax.legend()
ax.grid()
fig.savefig(output_path+'/SSIM.png')

fig, ax = plt.subplots(1, figsize=(8,6))
for s in range(len(snapshots)):
    ax.plot(modes, cumEV_vals[:, s],label=f"snapshots={snapshots[s]}")
ax.set_xlabel('modes')
ax.set_ylabel('cumulative equivalence ratio')
ax.legend()
ax.grid()
fig.savefig(output_path+'/cumEV.png')
'''

snapshots = 1000
labels = ['POD', 'Projection']
modes = [10, 16, 32, 64, 100]

NRMSE_vals = np.zeros((len(modes), len(labels)))
ssim_vals  = np.zeros((len(modes), len(labels)))
NRMSE_plume_vals = np.zeros((len(modes), len(labels)))

for s in range(len(labels)):
    no_snaps         = snapshots
    if s == 0:
        input_path_snaps = input_path+f"/snapshots{no_snaps}/"

        NRMSE_vals[:, s] = np.load(input_path_snaps + '/nrmse_list.npy')
        ssim_vals[:, s]  = np.load(input_path_snaps + '/ssim_list.npy')
        NRMSE_plume_vals[:, s]  = np.load(input_path_snaps + '/nrmse_plume_list.npy')
    if s == 1:
        input_path_snaps = input_path+f"/snapshots{no_snaps}/"
        
        NRMSE_vals[:, s] = np.load(input_path_snaps + '/proj_nrmse_list.npy')
        ssim_vals[:, s]  = np.load(input_path_snaps + '/proj_ssim_list.npy')
        NRMSE_plume_vals[:, s]  = np.load(input_path_snaps + '/proj_nrmse_plume_list.npy')

fig, ax = plt.subplots(1, figsize=(8,6))
for s in range(len(labels)):
    ax.plot(modes, NRMSE_vals[:, s], label=f"{labels[s]}")
ax.set_xlabel('modes')
ax.set_ylabel('NRMSE')
ax.legend()
ax.grid()
fig.savefig(output_path+'/NRMSE_proj.png')

fig, ax = plt.subplots(1, figsize=(8,6))
for s in range(len(labels)):
    ax.plot(modes, NRMSE_plume_vals[:, s], label=f"{labels[s]}")
ax.set_xlabel('modes')
ax.set_ylabel('NRMSE in plume')
ax.legend()
ax.grid()
fig.savefig(output_path+'/NRMSE_plume_proj.png')

fig, ax = plt.subplots(1, figsize=(8,6))
for s in range(len(labels)):
    ax.plot(modes, ssim_vals[:, s],label=f"{labels[s]}")
ax.set_xlabel('modes')
ax.set_ylabel('SSIM')
ax.legend()
ax.grid()
fig.savefig(output_path+'/SSIM_proj.png')