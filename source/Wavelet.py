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
import pycwt as wavelet


import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import time as time

import pywt

input_path='./input_data/' #'./ToyData/'
output_path='./Ra2e8/Wavelet_Transform/newcwt/'

def load_data(file, name):
    with h5py.File(file, 'r') as hf:
        print(name)
        print(hf[name])
        data = np.array(hf[name][:,:,0,:])

        x = hf['x'][:]  # 1D x-axis
        z = hf['z'][:]  # 1D z-axis
        time_vals = hf['total_time'][:]  # 1D time vector

    return data, x, z, time_vals

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

def fft(variable, x):
    variable = variable - np.mean(variable)
    fft = np.fft.fft(variable)
    fft = np.fft.fftshift(fft)
    end = x[-1]
    start = x[0]
    m = np.fft.fftfreq(len(x), d=(end-start)/len(x))
    m = np.fft.fftshift(m)
    m = 2*np.pi*m/(1)
    magnitude_w = np.abs(fft)
    psd = magnitude_w
    return psd, m

def wavelet_transform(data, time_value, dt, name, z_value=32):
    signal = data[time_value,:,z_value,0]
    print(np.shape(signal))
    print('time = ', time_vals[time_value])

    fig,ax = plt.subplots(1, figsize=(12,3), sharex=True, tight_layout=True)
    c=ax.pcolormesh(time_vals, x, data[:,:,z_value,0].T)
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
    fig.savefig(output_path+name+'_signal_recon.png')

    plume_location_original = x[np.argmax(signal)]
    plume_location_fully = x[np.argmax(fully_reconstructed_signal)]
    plume_location_partly = x[np.argmax(partly_reconstructed_signal)]
    print('plume location from original = ', plume_location_original)
    print('plume location from fully reconstructed signal = ', plume_location_fully)
    print('plume location from partly reconstructed signal = ', plume_location_partly)

def dis_wavelet_transform(signal, level, wavelet = 'db4', z_value=32):
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='symmetric')

    fig, ax = plt.subplots(level+1, figsize=(12,6))
    for i in range(level+1):
        ax[i].plot(coeffs[i])
        ax[i].grid()
    fig.savefig(output_path+'/discrete_wavelet_coeffs.png')

    plume_coeffs = [coeffs[0]] + [np.zeros(c.shape) for c in coeffs[1:]]
    plume_reconstructed = pywt.waverec(plume_coeffs, wavelet)
    fig, ax = plt.subplots(1, figsize=(12,3))
    ax.plot(x, signal, label='true signal')
    ax.plot(x, plume_reconstructed, label='reconstruction')
    ax.set_title('reconstruction from approximation coefficient')
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('magnitude')
    ax.legend()
    fig.savefig(output_path+'/approximation_reconstruction.png')

    wave_coeffs = [np.zeros_like(coeffs[0])] + [coeffs[1]] + [coeffs[2]]
    wave_reconstructed = pywt.waverec(wave_coeffs, wavelet)
    print(np.shape(wave_reconstructed))
    print(np.shape(x))
    fig, ax = plt.subplots(1, figsize=(12,3))
    ax.plot(x, signal, label='true signal')
    ax.plot(x, wave_reconstructed, label='reconstruction')
    ax.set_title('reconstruction from detail coefficients')
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('magnitude')
    ax.legend()
    fig.savefig(output_path+'/detail_reconstruction.png')

def dis_wavelet_transform_time(signal, level, wavelet = 'db4', z_value=32):
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='symmetric')

    plume_coeffs = [coeffs[0]] + [np.zeros(c.shape) for c in coeffs[1:]]
    plume_reconstructed = pywt.waverec(plume_coeffs, wavelet)


    wave_coeffs = [np.zeros_like(coeffs[0])] + [coeffs[1]] + [coeffs[2]]
    wave_reconstructed = pywt.waverec(wave_coeffs, wavelet)

    return plume_reconstructed, wave_reconstructed

names = ['q', 'w', 'u', 'b']
variables = ['q_all']
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
snapshots =1000
data_set, time_vals = load_data_set(input_path+'/data_4var_5000_30000.h5', variables, 1000)
time_value = 200
z_value=32
dt = time_vals[1]-time_vals[0]
dx = x[1]-x[0]
signal = data_set[time_value,:,z_value,0]

psd, m = fft(signal, x)
fig, ax =plt.subplots(1, figsize=(6,8), constrained_layout=True)
ax.plot(m[len(m)//2:], psd[len(m)//2:])
ax.set_xlabel('freq')
ax.set_ylabel('magnitude')
ax.grid()
fig.savefig(output_path+'/Ra2e8_fft.png')

fig,ax = plt.subplots(1, figsize=(12,3), sharex=True, tight_layout=True)
c=ax.pcolormesh(time_vals, x, data_set[:,:,z_value,0].T)
plt.colorbar(c, ax=ax, label='w')
ax.axvline(x=time_vals[time_value], color='tab:red', linestyle='--')
ax.set_xlabel('time')
ax.set_ylabel('x')
fig.savefig(output_path+'/signal_hovmol.png')

#wavelet_transform(data_set, time_value, dt, 'Ra2e7', z_value=32)

#dis_wavelet_transform(signal, 2, wavelet='db4', z_value=32)

'''
total_time = 1000
plumes_only = np.zeros((total_time, 256))
waves_only = np.zeros((total_time, 256))
for t_v in range(total_time):
    signal = data_set[t_v,:,z_value,0]

    plume_reconstructed, wave_reconstructed = dis_wavelet_transform_time(signal, level=2, wavelet = 'db4', z_value=32)

    plumes_only[t_v] = plume_reconstructed
    waves_only[t_v] = wave_reconstructed

fig, ax =plt.subplots(1, figsize=(12,3), constrained_layout=True)
c=ax.contourf(plumes_only.T)
fig.colorbar(c,label='Amplitude', ax=ax)
ax.set_xlabel('time')
ax.set_ylabel('x')
fig.savefig(output_path+'/plumes_only.png')

fig, ax =plt.subplots(1, figsize=(12,3), constrained_layout=True)
c=ax.contourf(waves_only.T)
fig.colorbar(c,label='Amplitude', ax=ax)
ax.set_xlabel('time')
ax.set_ylabel('x')
fig.savefig(output_path+'/waves_only.png')
'''

# normalise signal
mean = np.mean(signal)
std = np.std(signal)
var = std**2
signal_norm = (signal-mean)/std
N = signal.size

mother = wavelet.Morlet(6)
s0 = -1 #2 * dx  # Starting scale, in this case 2 * 0.25 years = 6 months
dj = 1 / 12  # Twelve sub-octaves per octaves
J = -1 #7 / dj  # Seven powers of two with dj sub-octaves

def estimate_alpha(signal_norm):
    try:
        alpha, _, _ = wavelet.ar1(signal_norm)  # Lag-1 autocorrelation for red noise
        return alpha
    except Exception as e:
        print(f"AR1 failed: {e}")
    try:
        alpha = np.corrcoef(signal_norm[:-1], signal_norm[1:])[0,1]
        return alpha
    except Exception as e:
        print(f"ACF failed: {e}")

    # if all methods fail return default
    print("returning default value alpha=0.5")
    alpha=0.5
    return alpha

alpha = estimate_alpha(signal_norm)

wave, scales, freqs, coi, fft_val, fft_freqs = wavelet.cwt(signal_norm, dx, dj, s0, J, mother)
iwave = wavelet.icwt(wave, scales, dx, dj, mother)

power = (np.abs(wave)) ** 2
fft_power = np.abs(fft_val) ** 2
period = 1 / freqs
power /= scales[:, None]

signif95, fft_theor95 = wavelet.significance(1.0, dx, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
sig95 = np.ones([1, N]) * signif95[:, None]
sig95 = power / sig95


fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
c1 = ax.contourf(x, np.log2(period), np.log2(power), np.log2(levels),
            extend='both')
fig.colorbar(c1, ax=ax, label='log2(level)')
extent = [x.min(), x.max(), 0, max(period)]
ax.contour(x, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
           extent=extent)
#ax.fill(np.concatenate([x, x[-1:] + dx, x[-1:] + dx, x[:1] - dx, x[:1] - dx]),
                #np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]),
                #'k', alpha=0.3, hatch='x')
ax.set_ylabel('log2(period)')
ax.set_xlabel('x')
fig.savefig(output_path+'/cwt_with_sig.png')
plt.close()

fig, ax = plt.subplots(1)
ax.plot(x, signal, label='signal')
ax.plot(x, (iwave*std)+mean, label='inverse wavelet')
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('amplitude')
fig.savefig(output_path+'/full_reconstruction.png')
plt.close()

# Mask power where sig95 < 1
masked_wave = np.where(sig95 >= 1, wave, 0)
masked_power = np.where(sig95 >= 1, power, 0)
iwave_sig95 = wavelet.icwt(masked_wave, scales, dx, dj, mother)

# Mask power where sig95 < 1
masked_wave_less = np.where(sig95 < 1, wave, 0) 
masked_power_less = np.where(sig95 < 1, power, 0)
iwave_sig_less = wavelet.icwt(masked_wave_less, scales, dx, dj, mother)


fig, ax =plt.subplots(1, figsize=(8,6), tight_layout=True)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
c1 = ax.contourf(x, np.log2(period), np.log2(masked_power), np.log2(levels),
            extend='both')
fig.colorbar(c1, ax=ax, label='log2(level)')
extent = [x.min(), x.max(), 0, max(period)]
ax.contour(x, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
           extent=extent)
#ax.fill(np.concatenate([x, x[-1:] + dx, x[-1:] + dx, x[:1] - dx, x[:1] - dx]),
                #np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]),
                #'k', alpha=0.3, hatch='x')
ax.set_ylabel('period')
ax.set_xlabel('x')
fig.savefig(output_path+'/cwt_masked95.png')
plt.close()

fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
ax.plot(x, signal, label='signal')
ax.plot(x, (iwave_sig95*std)+mean, label='inverse wavelet')
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('amplitude')
fig.savefig(output_path+'/95_reconstruction.png')
plt.close()

fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
ax.plot(x, signal, label='signal')
ax.plot(x, (iwave_sig_less*std)+mean, label='inverse wavelet')
ax.grid()
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('amplitude')
fig.savefig(output_path+'/less95_reconstruction.png')
plt.close()

print('finished example now running through time steps ...')

time_total = total_time = 100
wave_reconstruction=np.zeros((len(time_vals[:time_total]), len(x)))
plume_reconstruction=np.zeros((len(time_vals[:time_total]), len(x)))
for t_idx, t in enumerate(time_vals[:time_total]):
    signal = data_set[t_idx,:,z_value]

    # normalise signal
    mean = np.mean(signal)
    std = np.std(signal)
    var = std**2
    signal_norm = (signal-mean)/std
    N = signal.size

    mother = wavelet.Morlet(6)
    s0 = -1 #2 * dx  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = -1 #7 / dj  # Seven powers of two with dj sub-octaves
    alpha = estimate_alpha(signal_norm)

    wave, scales, freqs, coi, fft_val, fft_freqs = wavelet.cwt(signal_norm, dx, dj, s0, J, mother)

    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft_val) ** 2
    period = 1 / freqs
    power /= scales[:, None]

    signif95, fft_theor95 = wavelet.significance(1.0, dx, scales, 0, alpha,
                                            significance_level=0.95,
                                            wavelet=mother)
    sig95 = np.ones([1, N]) * signif95[:, None]
    sig95 = power / sig95

    # Mask power where sig95 > 1
    masked_wave = np.where(sig95 >= 1, wave, 0)
    masked_power = np.where(sig95 >= 1, power, 0)
    iwave_sig95 = wavelet.icwt(masked_wave, scales, dx, dj, mother)

    # Mask power where sig95 < 1
    masked_wave_less = np.where(sig95 < 1, wave, 0) 
    masked_power_less = np.where(sig95 < 1, power, 0)
    iwave_sig_less = wavelet.icwt(masked_wave_less, scales, dx, dj, mother)

    plume_reconstruction[t_idx, :] = iwave_sig95*std + mean
    wave_reconstruction[t_idx, :] = iwave_sig_less*std + mean

fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True)
plume_error = np.abs(data_set[:total_time, :, z_value] - plume_reconstruction)
print(np.min(data_set[:total_time, :, z_value]))
minm = min(np.min(data_set[:total_time, :, z_value]), np.min(plume_reconstruction))
maxm = max(np.max(data_set[:total_time, :, z_value]), np.max(plume_reconstruction))
c1=ax[0].pcolormesh(time_vals[:total_time], x, data_set[:total_time, :, z_value].T, vmin=minm, vmax=maxm)
fig.colorbar(c1, ax=ax[0])
ax[0].set_title('True')
c2=ax[1].pcolormesh(time_vals[:total_time], x, plume_reconstruction.T, vmin=minm, vmax=maxm)
fig.colorbar(c2, ax=ax[1])
ax[1].set_title('Plume only Reconstruction')
c3=ax[2].pcolormesh(time_vals[:total_time], x, plume_error.T, cmap='Reds')
fig.colorbar(c3, ax=ax[2])
ax[2].set_title('Error')
ax[-1].set_xlabel('time')
for v in range(3):
    ax[v].set_ylabel('x')
fig.savefig(output_path+'/truthvsplume.png')

fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True)
wave_error = np.abs(data_set[:total_time, :, z_value] - wave_reconstruction)
minm = min(np.min(data_set[:total_time, :, z_value]), np.min(wave_reconstruction))
maxm = max(np.max(data_set[:total_time, :, z_value]), np.max(wave_reconstruction))
c1=ax[0].pcolormesh(time_vals[:total_time], x, data_set[:total_time, :, z_value].T, vmin=minm, vmax=maxm)
fig.colorbar(c1, ax=ax[0])
ax[0].set_title('True')
c2=ax[1].pcolormesh(time_vals[:total_time], x, wave_reconstruction.T, vmin=minm, vmax=maxm)
fig.colorbar(c2, ax=ax[1])
ax[1].set_title('wave only Reconstruction')
c3=ax[2].pcolormesh(time_vals[:total_time], x, wave_error.T, cmap='Reds')
fig.colorbar(c3, ax=ax[2])
ax[2].set_title('Error')
ax[-1].set_xlabel('time')
for v in range(3):
    ax[v].set_ylabel('x')
fig.savefig(output_path+'/truthvswave.png')