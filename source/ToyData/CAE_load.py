"""
python script for convolutional autoencoder.

Usage: CAE.py [--input_path=<input_path> --output_path=<output_path> --model_path=<model_path> --hyperparam_file=<hyperparam_file> --job_id=<job_id> --sweep_id=<sweep_id>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --model_path=<model_path>            file path to location of job 
    --hyperparam_file=<hyperparam_file>  file with hyperparmas
    --job_id=<job_id>                    job_id
    --sweep_id=<sweep_id>                sweep_id
"""

# import packages
import time

import sys
sys.path.append('/nobackup/mm17ktn/ENS/skesn/skesn/')

import time

import os
sys.path.append(os.getcwd())
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import h5py
import yaml

from docopt import docopt
args = docopt(__doc__)

os.environ["OMP_NUM_THREADS"] = "1" #set cores for numpy
import tensorflow as tf
import json
tf.get_logger().setLevel('ERROR') #no info and warnings are printed
tf.config.threading.set_inter_op_parallelism_threads(1) #set cores for TF
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU') #runs the code without GPU
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import wandb
#wandb.login()

input_path = args['--input_path']
output_path = args['--output_path']
model_path = args['--model_path']
hyperparam_file = args['--hyperparam_file']
job_id = args['--job_id']
sweep_id = args['--sweep_id']

### make output_directories ###
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

output_path = output_path+f"/sweep_{sweep_id}"
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

output_path = output_path+f"/job_{job_id}"
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

def load_data(file, name):
    with h5py.File(file, 'r') as hf:
        print(name)
        print(hf[name])
        data = np.array(hf[name])

        x = hf['x'][:]  # 1D x-axis
        z = hf['z'][:]  # 1D z-axis
        time = hf['time'][:]  # 1D time vector

    return data, x, z, time

def add_noise(data, noise_level=0.01, seed=42):
    """
    Add Gaussian noise to a dataset of shape (time, x, z, channels).
    
    Parameters:
        data (np.ndarray): Input data of shape (T, X, Z, C).
        noise_level (float): Standard deviation of Gaussian noise.
    
    Returns:
        noisy_data (np.ndarray): Noisy version of the input data.
    """
    np.random.seed(seed)
    noise = noise_level * np.random.randn(*data.shape)
    noisy_data = data + noise
    return noisy_data

#### LOAD DATA ####
name='combined'
data_set, x, z, time_vals = load_data(input_path+'/plume_wave_dataset.h5', name)
print('shape of data', np.shape(data_set))

noise_level = 0
data_set = add_noise(data_set, noise_level=noise_level)
fig, ax =plt.subplots(1, figsize=(12,3), tight_layout=True)
c=ax.contourf(time_vals, x, data_set[:,:,32].T)
fig.colorbar(c, ax=ax)
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('x', fontsize=14)
fig.savefig(output_path+f"/combined_data_noise{noise_level}.png")

data_set = data_set.reshape(len(time_vals), len(x), len(z), 1)
print('shape of reshaped data set', np.shape(data_set))

U = data_set
dt = time_vals[1]-time_vals[0]
print('dt:', dt)
num_variables = 1
variables = names = ['A']

# Load hyperparameters from a YAML file
with open(hyperparam_file, 'r') as file:
    hyperparameters = yaml.safe_load(file)

lat_dep = hyperparameters.get("lat_dep", {}).get("value")
n_epochs = hyperparameters.get("n_epochs", {}).get("value")
l_rate = hyperparameters.get("l_rate", {}).get("value")
b_size = hyperparameters.get("b_size", {}).get("value")
lrate_mult =  hyperparameters.get("lrate_mult", {}).get("value")
N_lr = hyperparameters.get("N_lr", {}).get("value")
N_parallel = hyperparameters.get("N_parallel", {}).get("value")
N_layers =  hyperparameters.get("N_layers", {}).get("value")
kernel_choice = hyperparameters.get("kernel_choice", {}).get("value")

print(f"Building Model with learning_rate: {l_rate}, batch_size: {b_size}, N_parallel: {N_parallel}, latent_depth: {lat_dep}, kernel_choice: {kernel_choice}")

def split_data(U, b_size, n_batches):

    '''
    Splits the data in batches. Each batch is created by sampling the signal with interval
    equal to n_batches
    '''
    data   = np.zeros((n_batches, b_size, U.shape[1], U.shape[2], U.shape[3]))
    for j in range(n_batches):
        data[j] = U[::skip][j::n_batches]

    return data

@tf.function #this creates the tf graph
def model(inputs, enc_mods, dec_mods, is_train=False):

    '''
    Multiscale autoencoder, taken from Hasegawa 2020. The contribution of the CNNs at different
    scales are simply summed.
    '''

    # sum of the contributions of the different CNNs
    encoded = 0
    for enc_mod in enc_mods:
        encoded += enc_mod(inputs, training=is_train)

    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(encoded, training=is_train)

    return encoded, decoded

@tf.function #this creates the tf graph
def train_step(inputs, enc_mods, dec_mods, train=True):

    """
    Trains the model by minimizing the loss between input and output
    """

    # autoencoded field
    decoded  = model(inputs, enc_mods, dec_mods, is_train=train)[-1]

    # loss with respect to the data
    loss     = Loss_Mse(inputs, decoded)

    # compute and apply gradients inside tf.function environment for computational efficiency
    if train:
        # create a variable with all the weights to perform gradient descent on
        # appending lists is done by plus sign
        varss    = [] #+ Dense.trainable_weights
        for enc_mod in enc_mods:
            varss  += enc_mod.trainable_weights
        for dec_mod in dec_mods:
            varss +=  dec_mod.trainable_weights

        grads   = tf.gradients(loss, varss)
        optimizer.apply_gradients(zip(grads, varss))

    return loss, decoded

class PerPad2D(tf.keras.layers.Layer):
    """
    Periodic Padding layer
    """
    def __init__(self, padding=1, asym=False, **kwargs):
        self.padding = padding
        self.asym    = asym
        super(PerPad2D, self).__init__(**kwargs)

    def get_config(self): #needed to be able to save and load the model with this layer
        config = super(PerPad2D, self).get_config()
        config.update({
            'padding': self.padding,
            'asym': self.asym,
        })
        return config

    def call(self, x):
        return periodic_padding(x, self.padding, self.asym)

def periodic_padding(image, padding=1, asym=False):
    '''
    Create a periodic padding (same of np.pad('wrap')) around the image,
    to mimic periodic boundary conditions.
    '''
    '''
    # Get the shape of the input tensor
    shape = tf.shape(image)
    print(shape)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channel = shape[3]
    print(batch_size, height, width, channel)
    '''

    if asym:
        right_pad = image[:,:,:padding+1]
    else:
        right_pad = image[:,:,:padding]

    if padding != 0:
        left_pad = image[:,:,-padding:]
        partial_image = tf.concat([left_pad, image, right_pad], axis=2)
    else:
        partial_image = tf.concat([image, right_pad], axis=2)
    #print(tf.shape(partial_image))

    shape = tf.shape(partial_image)
    #print(shape)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channel = shape[3]
    #print(batch_size, height, width, channel)

    if asym:
        bottom_pad = tf.zeros([batch_size,padding+1,width,channel], dtype=image.dtype)
    else:
        bottom_pad = tf.zeros([batch_size,padding,width,channel], dtype=image.dtype)
    if padding != 0 :
        top_pad = tf.zeros([batch_size,padding,width,channel], dtype=image.dtype)
        padded_image = tf.concat([top_pad, partial_image, bottom_pad], axis=1)
    else:
        padded_image = tf.concat([partial_image, bottom_pad], axis=1)
    #print("shape of padded image: ", padded_image.shape)
    return padded_image

def plot_reconstruction(original, reconstruction, z_value, t_value, file_str):
    if original.ndim == 4:
        original = original.reshape(original.shape[0], original.shape[1], original.shape[2])
    if reconstruction.ndim == 4:
        reconstruction = reconstruction.reshape(reconstruction.shape[0], reconstruction.shape[1], reconstruction.shape[2])
        
    # Check if both data arrays have the same dimensions and the dimension is 2
    if original.ndim == reconstruction.ndim == 3:
        print("Both data arrays have the same dimensions and are 3D.")
    else:
        print("The data arrays either have different dimensions or are not 3D.")
    
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
    fig.savefig(output_path+file_str+'_snapshot_recon_t%i.png' % t_value)

    fig, ax = plt.subplots(2, figsize=(12,6), tight_layout=True, sharex=True)
    minm = min(np.min(original[:, :, z_value]), np.min(reconstruction[:, :, z_value]))
    maxm = max(np.max(original[:, :, z_value]), np.max(reconstruction[:, :, z_value]))
    print(np.max(original[:, :, z_value]))
    print(minm, maxm)
    time_zone = np.linspace(0, original.shape[0],  original.shape[0])
    c1 = ax[0].pcolormesh(time_zone, x, original[:, :, z_value].T)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('true')
    c2 = ax[1].pcolormesh(time_zone, x, reconstruction[:, :, z_value].T)
    fig.colorbar(c1, ax=ax[1])
    ax[1].set_title('reconstruction')
    for v in range(2):
        ax[v].set_ylabel('x')
    ax[-1].set_xlabel('time')
    fig.savefig(output_path+file_str+'_hovmoller_recon.png')

def plot_reconstruction_and_error(original, reconstruction, z_value, t_value, time_vals, file_str):
    time_zone = np.linspace(0, original.shape[0],  original.shape[0])
    abs_error = np.abs(original-reconstruction)
    residual  = original - reconstruction
    vmax_res = np.max(np.abs(residual))  # Get maximum absolute value
    vmin_res = -vmax_res
    if original.ndim == 3: #len(time_vals), len(x), len(z)

        fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
        minm = min(np.min(original[t_value, :, :]), np.min(reconstruction[t_value, :, :]))
        maxm = max(np.max(original[t_value, :, :]), np.max(reconstruction[t_value, :, :]))
        c1 = ax[0].pcolormesh(x, z, original[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('true')
        c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c1, ax=ax[1])
        ax[1].set_title('reconstruction')
        c3 = ax[2].pcolormesh(x, z, abs_error[t_value,:,:].T, cmap='Reds')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error')
        for v in range(3):
            ax[v].set_ylabel('z')
        ax[-1].set_xlabel('x')
        fig.savefig(output_path+file_str+'_hovmoller_recon_error.png')

        fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
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
        c3 = ax[2].pcolormesh(time_vals, x, abs_error[:, :, z_value].T, cmap='Reds')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error')
        for v in range(2):
            ax[v].set_ylabel('x')
        ax[-1].set_xlabel('time')
        fig.savefig(output_path+file_str+'_snapshot_recon_error.png')

        fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
        minm = min(np.min(original[t_value, :, :]), np.min(reconstruction[t_value, :, :]))
        maxm = max(np.max(original[t_value, :, :]), np.max(reconstruction[t_value, :, :]))
        c1 = ax[0].pcolormesh(x, z, original[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('true')
        c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:].T, vmin=minm, vmax=maxm)
        fig.colorbar(c1, ax=ax[1])
        ax[1].set_title('reconstruction')
        c3 = ax[2].pcolormesh(x, z, abs_error[t_value,:,:].T, cmap='Reds')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error')
        for v in range(3):
            ax[v].set_ylabel('z')
        ax[-1].set_xlabel('x')
        fig.savefig(output_path+file_str+'_hovmoller_recon_error.png')

        fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
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
        c3 = ax[2].pcolormesh(time_vals, x, abs_error[:, :, z_value].T, cmap='Reds')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error')
        for v in range(2):
            ax[v].set_ylabel('x')
        ax[-1].set_xlabel('time')
        fig.savefig(output_path+file_str+'_snapshot_recon_error.png')
    

    elif original.ndim == 4: #len(time_vals), len(x), len(z), var
        for i in range(original.shape[3]):
            name = names[i]
            print(name)
            fig, ax = plt.subplots(3, figsize=(12,6), tight_layout=True, sharex=True)
            minm = min(np.min(original[t_value, :, :, i]), np.min(reconstruction[t_value, :, :, i]))
            maxm = max(np.max(original[t_value, :, :, i]), np.max(reconstruction[t_value, :, :, i]))
            c1 = ax[0].pcolormesh(x, z, original[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(x, z, abs_error[t_value,:,:, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('z')
            ax[-1].set_xlabel('x')
            fig.savefig(output_path+file_str+name+'_snapshot_recon_error.png')
            plt.close()

            fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
            minm = min(np.min(original[:, :, z_value,i]), np.min(reconstruction[:, :, z_value,i]))
            maxm = max(np.max(original[:, :, z_value,i]), np.max(reconstruction[:, :, z_value,i]))
            print(np.max(original[:, :, z_value,i]))
            print(minm, maxm)
            print("time shape:", np.shape(time_zone))
            print("x shape:", np.shape(x))
            print("original[:, :, z_value] shape:", original[:, :, z_value,i].T.shape)
            c1 = ax[0].pcolormesh(time_zone, x, original[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(time_zone, x, reconstruction[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(time_zone, x,  abs_error[:,:,z_value, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('x')
            ax[-1].set_xlabel('time')
            fig.savefig(output_path+file_str+name+'_hovmoller_recon_error.png')
            plt.close()

            fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
            minm = min(np.min(original[:, :, z_value,i]), np.min(reconstruction[:, :, z_value,i]))
            maxm = max(np.max(original[:, :, z_value,i]), np.max(reconstruction[:, :, z_value,i]))
            print(np.max(original[:, :, z_value,i]))
            print(minm, maxm)
            print("time shape:", np.shape(time_zone))
            print("x shape:", np.shape(x))
            print("original[:, :, z_value] shape:", original[:, :, z_value,i].T.shape)
            c1 = ax[0].pcolormesh(time_zone, x, original[:, :, z_value, i].T)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(time_zone, x, reconstruction[:, :, z_value, i].T)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(time_zone, x,  abs_error[:,:,z_value, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('x')
            ax[-1].set_xlabel('time')
            fig.savefig(output_path+file_str+name+'_hovmoller_recon_diffbar_error.png')
            plt.close()


            fig, ax = plt.subplots(3, figsize=(12,6), tight_layout=True, sharex=True)
            minm = min(np.min(original[t_value, :, :, i]), np.min(reconstruction[t_value, :, :, i]))
            maxm = max(np.max(original[t_value, :, :, i]), np.max(reconstruction[t_value, :, :, i]))
            c1 = ax[0].pcolormesh(x, z, original[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(x, z, residual[t_value,:,:, i].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('z')
                ax[v].tick_params(axis='both', labelsize=12)
            ax[-1].set_xlabel('x')
            fig.savefig(output_path+file_str+name+'_hovmoller_recon_residual.png')

            fig, ax = plt.subplots(3, figsize=(12,9), tight_layout=True, sharex=True)
            minm = min(np.min(original[:, :, z_value,i]), np.min(reconstruction[:, :, z_value,i]))
            maxm = max(np.max(original[:, :, z_value,i]), np.max(reconstruction[:, :, z_value,i]))
            print(np.max(original[:, :, z_value,i]))
            print(minm, maxm)
            print("time shape:", np.shape(time_zone))
            print("x shape:", np.shape(x))
            print("original[:, :, z_value] shape:", original[:, :, z_value,i].T.shape)
            c1 = ax[0].pcolormesh(time_zone, x, original[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[0])
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(time_zone, x, reconstruction[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c2, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(time_zone, x,  residual[:,:,z_value, i].T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('x')
                ax[v].tick_params(axis='both', labelsize=12)
            ax[-1].set_xlabel('time')
            fig.savefig(output_path+file_str+name+'_snapshot_recon_residual.png')



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
    
    #print('shape of data:', np.shape(original))
    # Initialize SSIM accumulator
    total_ssim = 0
    #print('total_ssim', total_ssim)
    timesteps = original.shape[0]
    channels = original.shape[-1]
    #print(timesteps, channels)
  
    for t in range(timesteps):
        for c in range(channels):
            #print(t, c)
            # Extract the 2D slice for each timestep and channel
            orig_slice = original[t, :, :, c]
            dec_slice = decoded[t, :, :, c]
            #print(orig_slice, dec_slice)
            
            # Compute SSIM for the current slice
            batch_ssim = ssim(orig_slice, dec_slice, data_range=orig_slice.max() - orig_slice.min(), win_size=3)
            #print(batch_ssim)
            total_ssim += batch_ssim
            #print(total_ssim)
  
    # Compute the average SSIM across all timesteps and channels
    avg_ssim = total_ssim / (timesteps * channels)
    print('avg_ssim', avg_ssim)
    return avg_ssim

def EVR_recon(original_data, reconstructed_data):
    print(original_data.ndim)
    print(reconstructed_data.ndim)
    if original_data.ndim == 3:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2])
    elif original_data.ndim == 4:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2]*original_data.shape[3])
    if reconstructed_data.ndim == 3:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2])
    elif reconstructed_data.ndim == 4:
        print('reshape')
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
    
def ss_transform(data, scaler):
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
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
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
    if data_reshape.ndim == 2:
        print("data array is 2D.")
    else:
        print("data array is not 2D")
        
    data_unscaled = scaler.inverse_transform(data_reshape)
    data_unscaled = data_unscaled.reshape(data.shape)
    
    if data_unscaled.ndim == 4:
        print('unscaled and reshaped to 4 dimensions')
    else:
        print('not unscaled properly')
        
    return data_unscaled

#### load in data ###
#b_size      = wandb.config.b_size   #batch_size
n_batches   = int((U.shape[0]/b_size) *0.7)  #number of batches #20
val_batches = int((U.shape[0]/b_size) *0.2)    #int(n_batches*0.2) # validation set size is 0.2 the size of the training set #2
test_batches = int((U.shape[0]/b_size) *0.1)
skip        = 1
print(n_batches, val_batches, test_batches)

#print(b_size*n_batches*skip*dt*upsample)

print('Train Data%  :',b_size*n_batches*skip/U.shape[0]) #how much of the data we are using for training
print('Val   Data%  :',b_size*val_batches*skip/U.shape[0])
print('Test   Data%  :',b_size*test_batches*skip/U.shape[0])

# training data
U_tt        = np.array(U[:b_size*n_batches*skip])            #to be used for random batches
# validation data
U_vv        = np.array(U[b_size*n_batches*skip:
                         b_size*n_batches*skip+b_size*val_batches*skip])
# test data
U_tv        = np.array(U[b_size*n_batches*skip+b_size*val_batches*skip:
                         b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip])

U_tt_reshape = U_tt.reshape(-1, U_tt.shape[-1])
U_vv_reshape = U_vv.reshape(-1, U_vv.shape[-1])
U_tv_reshape = U_tv.reshape(-1, U_tv.shape[-1])

# fit the scaler
scaler = StandardScaler()
scaler.fit(U_tt_reshape)
print('means', scaler.mean_)

#transform training, val and test sets
U_tt_scaled = scaler.transform(U_tt_reshape)
U_vv_scaled = scaler.transform(U_vv_reshape)
U_tv_scaled = scaler.transform(U_tv_reshape)

#reshape 
U_tt_scaled = U_tt_scaled.reshape(U_tt.shape)
U_vv_scaled = U_vv_scaled.reshape(U_vv.shape)
U_tv_scaled = U_tv_scaled.reshape(U_tv.shape)

test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                         b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 

U_train     = split_data(U_tt_scaled, b_size, n_batches).astype('float32') #to be used for randomly shuffled batches
U_val       = split_data(U_vv_scaled, b_size, val_batches).astype('float32')

del U_vv, U_tt, U_tt_scaled

# define the model
# we do not have pooling and upsampling, instead we use stride=2

# define the model
# we do not have pooling and upsampling, instead we use stride=2
#global lat_dep                 #latent space depth
#global N_parallel                       #number of parallel CNNs for multiscale
#global kernel_choice
ker_size      = [(3,3), (5,5), (7,7)]      #kernel sizes
if N_parallel == 1:
    ker_size  = [ker_size[kernel_choice]]
#global N_layers    #number of layers in every CNN
if N_layers == 4:
    n_fil         = [6,12,24,lat_dep]          #number of filters ecnoder
    n_dec         = [24,12,6,3]                #number of filters decoder
elif N_layers == 5:
    n_fil         = [6,12,24,48,lat_dep]          #number of filters ecnoder
    n_dec         = [48,24,12,6,3]                #number of filters decoder
act           = 'tanh'                     #activation function

pad_enc       = 'valid'         #no padding in the conv layer
pad_dec       = 'valid'
p_size        = [0,1,2]         #stride = 2 periodic padding size
if N_parallel == 1:
    p_size    = [p_size[kernel_choice]]
p_fin         = [1,2,3]         #stride = 1 periodic padding size
if N_parallel == 1:
    p_fin     = [p_fin[kernel_choice]]
p_dec         = 1               #padding in the first decoder layer
p_crop        = U.shape[1], U.shape[2]      #crop size of the output equal to input size


#initialize the encoders and decoders with different kernel sizes
enc_mods      = [None]*(N_parallel)
dec_mods      = [None]*(N_parallel)
for i in range(N_parallel):
    enc_mods[i] = tf.keras.Sequential(name='Enc_' + str(i))
    dec_mods[i] = tf.keras.Sequential(name='Dec_' + str(i))

#generate encoder layers
for j in range(N_parallel):
    for i in range(N_layers):

        if i == N_layers-1:
            #stride=2 padding and conv
            enc_mods[j].add(PerPad2D(padding=p_size[j], asym=True,
                                            name='Enc_' + str(j)+'_PerPad_'+str(i)))
            enc_mods[j].add(tf.keras.layers.Conv2D(filters = n_fil[i], kernel_size=ker_size[j],
                                        activation=act, padding=pad_enc, strides=4,
                            name='Enc_' + str(j)+'_ConvLayer_'+str(i)))
        else:
            #stride=2 padding and conv
            enc_mods[j].add(PerPad2D(padding=p_size[j], asym=True,
                                            name='Enc_' + str(j)+'_PerPad_'+str(i)))
            enc_mods[j].add(tf.keras.layers.Conv2D(filters = n_fil[i], kernel_size=ker_size[j],
                                        activation=act, padding=pad_enc, strides=2,
                            name='Enc_' + str(j)+'_ConvLayer_'+str(i)))
        

        #stride=1 padding and conv
        if i<N_layers-1:
            enc_mods[j].add(PerPad2D(padding=p_fin[j], asym=False,
                                                        name='Enc_'+str(j)+'_Add_PerPad1_'+str(i)))
            enc_mods[j].add(tf.keras.layers.Conv2D(filters=n_fil[i],
                                                    kernel_size=ker_size[j],
                                                activation=act,padding=pad_dec,strides=1,
                                                    name='Enc_'+str(j)+'_Add_Layer1_'+str(i)))

# Obtain the shape of the latent space
output = enc_mods[-1](U_train[0])
N_1 = output.shape
print('shape of latent space', N_1)
N_latent = N_1[-3] * N_1[-2] * N_1[-1]
print("Latent space dimensions:", N_latent)


#generate decoder layers            
for j in range(N_parallel):

    for i in range(N_layers):

        #initial padding of latent space
        if i==0: 
            dec_mods[j].add(PerPad2D(padding=p_dec, asym=False,
                                            name='Dec_' + str(j)+'_PerPad_'+str(i))) 
            dec_mods[j].add(tf.keras.layers.Conv2DTranspose(filters = n_dec[i],
                            output_padding=None,kernel_size=ker_size[j],
                            activation=act, padding=pad_dec, strides=4,
                name='Dec_' + str(j)+'_ConvLayer_'+str(i)))
        else:
            #Transpose convolution with stride = 2 
            dec_mods[j].add(tf.keras.layers.Conv2DTranspose(filters = n_dec[i],
                                        output_padding=None,kernel_size=ker_size[j],
                                        activation=act, padding=pad_dec, strides=2,
                                name='Dec_' + str(j)+'_ConvLayer_'+str(i)))
        
        #Convolution with stride=1
        if  i<N_layers-1:       
            dec_mods[j].add(tf.keras.layers.Conv2D(filters=n_dec[i],
                                        kernel_size=ker_size[j], 
                                        activation=act,padding=pad_dec,strides=1,
                                        name='Dec_' + str(j)+'_ConvLayer1_'+str(i)))

    #crop and final linear convolution with stride=1
    dec_mods[j].add(tf.keras.layers.CenterCrop(p_crop[0] + 2*p_fin[j],
                                                    p_crop[1]+ 2*p_fin[j],
                            name='Dec_' + str(j)+'_Crop_'+str(i)))
    dec_mods[j].add(tf.keras.layers.Conv2D(filters=U.shape[3],
                                            kernel_size=ker_size[j], 
                                            activation='linear',padding=pad_dec,strides=1,
                                                name='Dec_' + str(j)+'_Final_Layer'))

# run the model once to print summary
enc0, dec0 = model(U_train[0], enc_mods, dec_mods)
print('latent   space size:', N_latent)
print('physical space size:', U[0].flatten().shape)
print('')
for j in range(N_parallel):
    enc_mods[j].summary()
for j in range(N_parallel):
    dec_mods[j].summary()

##### Visualise Error #####

# Load best model
# Restore the checkpoint (this will restore the optimizer, encoder, and decoder states)
#how to load saved model
models_dir = model_path
a = [None]*N_parallel
b = [None]*N_parallel
for i in range(N_parallel):
    a[i] = tf.keras.models.load_model(models_dir + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5', 
                                          custom_objects={"PerPad2D": PerPad2D})
for i in range(N_parallel):
    b[i] = tf.keras.models.load_model(models_dir + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5',
                                          custom_objects={"PerPad2D": PerPad2D})

validation_data = True
test_data       = True
all_data        = True

if validation_data:
    print('VALIDATION DATA')
    truth = U_vv_scaled
    decoded = model(truth,a,b)[1]
    print(np.shape(decoded), np.shape(truth))

    test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                                b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 
    print(test_times[0], test_times[-1])

    decoded = decoded.numpy()

    decoded_unscaled = ss_inverse_transform(decoded, scaler)
    truth_unscaled = ss_inverse_transform(truth, scaler)
    print('shape of validation prediction', np.shape(decoded_unscaled))
    print('shape of validation truth', np.shape(truth_unscaled))

    #SSIM
    test_ssim = compute_ssim_for_4d(truth_unscaled, decoded_unscaled)

    #MSE
    mse = MSE(truth_unscaled, decoded_unscaled)

    #NRMSE
    nrmse = NRMSE(truth_unscaled, decoded_unscaled)

    #EVR 
    evr = EVR_recon(truth_unscaled, decoded_unscaled)

    print("nrmse:", nrmse)
    print("mse:", mse)
    print("test_ssim:", test_ssim)
    print("EVR:", evr)

    import json
    # Full path for saving the file
    output_file = "validation_metrics.json"

    output_path_met = os.path.join(output_path, output_file)

    metrics = {
    "MSE": mse,
    "NRMSE": nrmse,
    "SSIM": test_ssim,
    "EVR": evr,
    }

    with open(output_path_met, "w") as file:
        json.dump(metrics, file, indent=4)

    ### plot from index 0
    index = 0
    print(np.shape(truth_unscaled[index:index+500]))
    print(np.shape(decoded_unscaled[index:index+500]))
    print(np.shape(x))
    print(np.shape(test_times[index:index+500]))
    plot_reconstruction_and_error(truth_unscaled[index:index+500], decoded_unscaled[index:index+500], 32, 0, test_times[index:index+500], f"/validation_{index}")

if test_data:
    print('TEST DATA')
    truth = U_tv_scaled
    decoded = model(truth,a,b)[1]
    print(np.shape(decoded), np.shape(truth))

    test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                                b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 
    print(test_times[0], test_times[-1])

    decoded = decoded.numpy()

    decoded_unscaled = ss_inverse_transform(decoded, scaler)
    truth_unscaled = ss_inverse_transform(truth, scaler)
    print('shape of test prediction', np.shape(decoded_unscaled))
    print('shape of test truth', np.shape(truth_unscaled))

    #SSIM
    test_ssim = compute_ssim_for_4d(truth_unscaled, decoded_unscaled)

    #MSE
    mse = MSE(truth_unscaled, decoded_unscaled)

    #NRMSE
    nrmse = NRMSE(truth_unscaled, decoded_unscaled)

    #EVR 
    evr = EVR_recon(truth_unscaled, decoded_unscaled)

    print("nrmse:", nrmse)
    print("mse:", mse)
    print("test_ssim:", test_ssim)
    print("EVR:", evr)

    import json
    # Full path for saving the file
    output_file = "test_metrics.json"

    output_path_met = os.path.join(output_path, output_file)

    metrics = {
    "MSE": mse,
    "NRMSE": nrmse,
    "SSIM": test_ssim,
    "EVR": evr,
    }

    with open(output_path_met, "w") as file:
        json.dump(metrics, file, indent=4)

    ### plot from index 0
    index = 0
    plot_reconstruction_and_error(truth_unscaled[index:index+500], decoded_unscaled[index:index+500], 32, 0, test_times[index:index+500], f"/test_{index}")


if all_data:
    print('ALL DATA')
    scaled_data = ss_transform(U, scaler)
    truth = scaled_data
    decoded = model(truth,a,b)[1]
    print(np.shape(decoded), np.shape(truth))

    test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                                b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 
    print(test_times[0], test_times[-1])

    decoded = decoded.numpy()

    decoded_unscaled = ss_inverse_transform(decoded, scaler)
    truth_unscaled = ss_inverse_transform(truth, scaler)
    print('shape of test prediction', np.shape(decoded_unscaled))
    print('shape of test truth', np.shape(truth_unscaled))

    #SSIM
    test_ssim = compute_ssim_for_4d(truth_unscaled, decoded_unscaled)

    #MSE
    mse = MSE(truth_unscaled, decoded_unscaled)

    #NRMSE
    nrmse = NRMSE(truth_unscaled, decoded_unscaled)

    #EVR 
    evr = EVR_recon(truth_unscaled, decoded_unscaled)

    print("nrmse:", nrmse)
    print("mse:", mse)
    print("test_ssim:", test_ssim)
    print("EVR:", evr)

    import json
    # Full path for saving the file
    output_file = "all_metrics.json"

    output_path_met = os.path.join(output_path, output_file)

    metrics = {
    "MSE": mse,
    "NRMSE": nrmse,
    "SSIM": test_ssim,
    "EVR": evr,
    }

    with open(output_path_met, "w") as file:
        json.dump(metrics, file, indent=4)

    ### plot from index 0
    index = 0
    plot_reconstruction_and_error(truth_unscaled, decoded_unscaled, 32, 0, test_times, f"/all_{index}")
