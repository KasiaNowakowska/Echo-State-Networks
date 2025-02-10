import os
import time
import sys
sys.path.append(os.getcwd())
import matplotlib
#matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
sys.stdout.reconfigure(line_buffering=True)

output_path='./ToyData/'

# Parameters
nx, nz = 128, 64  # Grid resolution in x and z
nt = 500         # Number of snapshots
time = np.linspace(0, 20, nt, endpoint=False)  # Time vector

print(np.shape(time), time[0], time[-1])
dt = time[1]-time[0]
print(dt)

x = np.linspace(0, 10, nx)
z = np.linspace(0, 10, nz)
x_grid, z_grid = np.meshgrid(x, z)

# Plume parameters
A = 2.0             # Amplitude of Gaussian plume
sigma_x = 0.25         # Standard deviation of Gaussian plume in x
sigma_z = 3      # Standard deviation of Gaussian plume in z (for narrow plume in height) #10
x0 = 7.0            # Static x-center of plume
z0 = 7.0            # Static z-center of plume

# Wave parameters
k = 2 * np.pi / 2  # Wavenumber
omega = 2 * np.pi / 2  # Angular frequency
c = omega / k       # Wave speed
print(k, omega, c)

# Initialize datasets
plume_data = np.zeros((nt, nx, nz))
wave_data = np.zeros((nt, nx, nz))
combined_data = np.zeros((nt, nx, nz))

# Generate data
for t_idx, t in enumerate(time):
    if t_idx<100 or 200<t_idx<300 or 400<t_idx<500:
        x0 = 7.0  # Plume position (7,7)
    else:
        x0 = 2.0  # Plume position (2,7)

    # Gaussian plume
    plume = A * np.exp(-((x_grid - x0)**2 / (2 * sigma_x**2) + (z_grid - z0)**2 / (2 * sigma_z**2)))

    # Traveling wave
    wave = np.sin(k * x_grid - omega * t)

    # Store data
    plume_data[t_idx] = plume.T
    wave_data[t_idx] = wave.T
    combined_data[t_idx] = plume.T + wave.T

with h5py.File(output_path+'plume_wave_dataset.h5', 'w') as hf:
    hf.create_dataset('plume', data=plume_data)
    hf.create_dataset('wave', data=wave_data)
    hf.create_dataset('combined', data=combined_data)

    # Save grid information
    hf.create_dataset('x', data=x)  # 1D x-axis
    hf.create_dataset('z', data=z)  # 1D z-axis
    hf.create_dataset('time', data=time)  # 1D time vector
    hf.create_dataset('x_grid', data=x_grid)  # 2D meshgrid for x
    hf.create_dataset('z_grid', data=z_grid)  # 2D meshgrid for z

    hf.attrs['description'] = "Dataset with Gaussian plume, sine waves, and their combination"
    hf.attrs['grid'] = f"64x64 (x,z) with {nt} time steps"
    hf.attrs['plume_params'] = f"A={A}, sigma_x={sigma_x}, sigma_z={sigma_z}"
    hf.attrs['wave_params'] = f"k={k}, omega={omega}, c={c}"
    hf.attrs['num_snapshots'] = nt
    hf.attrs['dt'] = dt  # Time step
print("Dataset saved as plume_wave_dataset.h5")


#### Visualise ####
Plotting = True

def plot_hovmoller(data, z_value, filename):
    fig, ax =plt.subplots(1, figsize=(12,3), constrained_layout=True)
    c=ax.contourf(time, x, data[:, :, z_value].T)
    fig.colorbar(c,label='Amplitude', ax=ax)
    ax.set_xlabel('time')
    ax.set_ylabel('x')
    fig.savefig(output_path+filename)

def plot_snapshot(data, t_idx, filename):
    fig, ax =plt.subplots(1, figsize=(12,3), constrained_layout=True)
    c=ax.contourf(x, z, data[t_idx,:,:].T)
    fig.colorbar(c,label='Amplitude', ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    fig.savefig(output_path+filename)

if Plotting:
    plot_hovmoller(plume_data, 32, 'plume_hov.png')
    plot_hovmoller(wave_data, 32, 'wave_hov.png')
    plot_hovmoller(combined_data, 32, 'combined_hov.png')

    snapshot_idx = 450  # Example snapshot index
    plot_snapshot(plume_data, snapshot_idx, 'plume_snap.png')
    plot_snapshot(wave_data, snapshot_idx, 'wave_snap.png')
    plot_snapshot(combined_data, snapshot_idx, 'wave_snap.png')


