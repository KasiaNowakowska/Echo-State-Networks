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
import json
import time as time

import pywt

input_path='./ToyData/'
output_path='./ToyData/'

def load_data(file, name):
    with h5py.File(file, 'r') as hf:
        print(name)
        print(hf[name])
        data = np.array(hf[name])

        x = hf['x'][:]  # 1D x-axis
        z = hf['z'][:]  # 1D z-axis
        time_vals = hf['time'][:]  # 1D time vector

    return data, x, z, time_vals

def wavelet_transform(data, time_value, dt, name, z_value=32):
    signal = data[time_value,:,z_value]
    print(np.shape(signal))
    print('time = ', time_vals[time_value])

    fig,ax = plt.subplots(1, figsize=(12,3), sharex=True, tight_layout=True)
    c=ax.pcolormesh(time_vals, x, data[:,:,z_value].T)
    plt.colorbar(c, ax=ax, label='w')
    ax.axvline(x=time_vals[time_value], color='tab:red', linestyle='--')
    ax.set_xlabel('time')
    ax.set_ylabel('x')
    fig.savefig(output_path+name+'_signal_hovmol.png')


    fig, ax = plt.subplots(1)
    ax.plot(x, signal)
    ax.set_xlabel('time')
    ax.set_ylabel('signal')
    ax.grid()
    fig.savefig(output_path+name+'_signal.png')

    wavelet     = 'morl'
    # set the wavelet scales
    scales = np.arange(1, 100)
    coef, freqs = pywt.cwt(signal,scales,wavelet)

    fig, ax = plt.subplots(1)
    cwt_abs = np.abs(coef)
    c1 = ax.contourf(x, freqs, cwt_abs, levels=10)
    ax.set_xlabel('x')
    ax.set_ylabel('freqs')
    fig.colorbar(c1, ax=ax, label='Magnitude')
    ax.set_title(time_vals[time_value])
    fig.savefig(output_path+name+'_cwt.png')

    '''
    fig, ax = plt.subplots(1)
    c1 = ax.contourf(x, freqs, np.log(np.abs(coef)), levels=10)
    ax.set_xlabel('x')
    ax.set_ylabel('freqs')
    fig.colorbar(c1, ax=ax, label='log Magnitude')
    plt.show()
    '''

    print('shape of cwt_abs', np.shape(cwt_abs))
    print('shape of scales', np.shape(scales))

    wavelet_function, time_values = pywt.ContinuousWavelet('morl').wavefun() # finds the function and time_values associated with the wavelet
    y_0 = wavelet_function[np.argmin(np.abs(time_values))] # this find he amplitude of the wavelet at the time step closest to time=0

    #r_sum = np.transpose(np.sum(np.transpose(coef)/ scales ** 0.5, axis=-1))
    r_sum = np.sum(coef/ (scales[:, np.newaxis] ** 1.5), axis=0)
    fully_reconstructed_signal = r_sum * (1 / y_0)

    coef_new = np.zeros_like(coef)
    argmax = np.argmax(coef)
    row, col = np.unravel_index(argmax, coef.shape)  # Convert to row, column indices
    # Print the maximum value and its location
    print(f"Maximum value: {coef[row, col]}")
    print(f"Location (row, col): ({row}, {col})")
    coef_new[row, col] = coef[row, col]

    #r_sum = np.transpose(np.sum(np.transpose(coef_new)/ scales ** 0.5, axis=-1))
    r_sum = np.sum(coef_new/ (scales[:, np.newaxis] ** 1.5), axis=0)
    partly_reconstructed_signal = r_sum * (1 / y_0)

    fig, ax = plt.subplots(1)
    ax.plot(x, signal, label='original')
    ax.plot(x, fully_reconstructed_signal, label='fully reconstructed')
    ax.plot(x, partly_reconstructed_signal, label='partly reconstrcuted')
    ax.set_xlabel('x')
    ax.set_ylabel('signal')
    ax.grid()
    plt.legend()
    fig.savefig(output_path+name+'_sigal_recon.png')

    plume_location_original = x[np.argmax(signal)]
    plume_location_fully = x[np.argmax(fully_reconstructed_signal)]
    plume_location_partly = x[np.argmax(partly_reconstructed_signal)]
    print('plume location from original = ', plume_location_original)
    print('plume location from fully reconstructed signal = ', plume_location_fully)
    print('plume location from partly reconstructed signal = ', plume_location_partly)
