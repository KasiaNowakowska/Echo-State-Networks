"""
python script for convolutional autoencoder.

Usage: CAE.py [--input_path=<input_path> --output_path=<output_path> --CAE_model_path=<CAE_model_path> --CAE_hyperparam_file=<CAE_hyperparam_file> --ESN_hyperparam_file=<ESN_hyperparam_file> --number_of_tests=<number_of_tests>]

Options:
    --input_path=<input_path>                   file path to use for data
    --output_path=<output_path>                 file path to save images output [default: ./images]
    --CAE_model_path=<CAE_model_path>           file path to location of job 
    --CAE_hyperparam_file=<CAE_hyperparam_file> file with hyperparmas from CAE
    --ESN_hyperparam_file=<ESN_hyperparam_file> file with hyperparams for ESN
    --number_of_tests=<number_of_tests>  number of tests [default: 5]
"""

# import packages
import time

import os
import sys
sys.path.append(os.getcwd())
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
#import pandas as pd
import numpy as np
import h5py
import yaml
import skopt
from skopt.space import Real

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

import matplotlib.colors as mcolors
import h5py
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
from skopt.plots import plot_convergence
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
from scipy.io import loadmat, savemat
import sys
sys.stdout.reconfigure(line_buffering=True)

exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('run functions files')

import wandb
#wandb.login()

input_path = args['--input_path']
output_path = args['--output_path']
CAE_model_path = args['--CAE_model_path']
ESN_model_path = output_path
CAE_hyperparam_file = args['--CAE_hyperparam_file']
ESN_hyperparam_file = args['--ESN_hyperparam_file']
number_of_tests = int(args['--number_of_tests'])

output_path = output_path + '/further_analysis/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

with open(ESN_hyperparam_file, "r") as f:
    hyperparams = json.load(f)
    Nr = hyperparams["Nr"]
    train_len = hyperparams["N_train"]
    val_len = hyperparams["N_val"]
    test_len = hyperparams["N_test"]
    washout_len = hyperparams["N_washout"]
    washout_len_val = hyperparams.get("N_washout_val", washout_len)
    t_lyap = hyperparams["t_lyap"]
    normalisation = hyperparams["normalisation"]
    ens = hyperparams["ens"]
    ensemble_test = hyperparams["ensemble_test"]
    n_tests = hyperparams["n_tests"]
    grid_x = hyperparams["grid_x"]
    grid_y = hyperparams["grid_y"]
    added_points = hyperparams["added_points"]
    val = hyperparams["val"]
    noise = hyperparams["noise"]
    alpha = hyperparams["alpha"]
    alpha0 = hyperparams["alpha0"]
    n_forward = hyperparams["n_forward"]

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
    print('encoded shape', np.shape(encoded))

    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(encoded, training=is_train)
    print('decoded shape', np.shape(decoded))
    

    return encoded, decoded
  
@tf.function #this creates the tf graph  
def Decoder(encoded, dec_mods, is_train=False):
    '''
    Multiscale autoencoder, taken from Hasegawa 2020. The contribution of the CNNs at different
    scales are simply summed.
    '''
    
    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(encoded, training=is_train)

    return decoded

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

    return loss

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

def plot_reconstruction_and_error(original, reconstruction, z_value, t_value, time_vals, file_str):
    abs_error = np.abs(original-reconstruction)
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
        fig.colorbar(c1, ax=ax[1])
        ax[1].set_title('reconstruction')
        c3 = ax[2].pcolormesh(time_vals, x, abs_error[:, :, z_value].T, cmap='Reds')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title('error')
        for v in range(2):
            ax[v].set_ylabel('x')
        ax[-1].set_xlabel('time')
        fig.savefig(output_path+file_str+'_snapshot_recon_error.png')
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
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:,i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(x, z, abs_error[t_value,:,:, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('z')
            ax[-1].set_xlabel('x')
            fig.savefig(output_path+file_str+name+'_hovmoller_recon_error.png')
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
            ax[0].set_title('true')
            c2 = ax[1].pcolormesh(time_vals, x, reconstruction[:, :, z_value, i].T, vmin=minm, vmax=maxm)
            fig.colorbar(c1, ax=ax[1])
            ax[1].set_title('reconstruction')
            c3 = ax[2].pcolormesh(time_vals, x,  abs_error[:,:,z_value, i].T, cmap='Reds')
            fig.colorbar(c3, ax=ax[2])
            ax[2].set_title('error')
            for v in range(2):
                ax[v].set_ylabel('x')
            ax[-1].set_xlabel('time')
            fig.savefig(output_path+file_str+name+'_snapshot_recon_error.png')
            plt.close()

def global_parameters(data):
    if data.ndim == 4:
        print("data is 4D.")
    else:
        print("wrong format needs to be 4D.")

    avg_q = np.mean(data[:,:,:,0], axis=(1,2))
    ke = 0.5*data[:,:,:,1]*data[:,:,:,1]
    avg_ke = np.mean(ke, axis=(1,2))
    global_params = np.zeros((data.shape[0],2))
    global_params[:,0] = avg_ke
    global_params[:,1] = avg_q

    return global_params

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
    mask_expanded       =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
    mask_expanded_recon =  np.repeat(mask_reconstructed[:, :, :, np.newaxis], 4, axis=-1) # Shape: (256, 64, 1)
    
    return active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon

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
        
    print('shape before inverse scaling', np.shape(data_reshape))

    data_unscaled = scaler.inverse_transform(data_reshape)
    data_unscaled = data_unscaled.reshape(data.shape)
    
    if data_unscaled.ndim == 4:
        print('unscaled and reshaped to 4 dimensions')
    else:
        print('not unscaled properly')
        
    return data_unscaled

#### LOAD DATA ####
variables = ['q_all', 'w_all', 'u_all', 'b_all']
names = ['q', 'w', 'u', 'b']
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
snapshots = 16000
data_set, time_vals = load_data_set(input_path+'/data_4var_5000_30000.h5', variables, snapshots)
print(np.shape(data_set))
dt = time_vals[1]-time_vals[0]

reduce_data_set = reduce_domain2 = reduce_domain = False
if reduce_data_set:
    data_set = data_set[:, 32:96, :, :]
    x = x[32:96]
    print('reduced domain shape', np.shape(data_set))
    print('reduced x domain', np.shape(x))
    print('reduced x domain', len(x))
    print(x[0], x[-1])

### global ###
truth_global = global_parameters(data_set)
global_labels=['KE', 'q']

U = data_set
dt = time_vals[1]-time_vals[0]
print('dt:', dt)

# Load hyperparameters from a YAML file
with open(CAE_hyperparam_file, 'r') as file:
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

#### load in data ###
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

U_tt_reshape = U_tt.reshape(-1, U_tt.shape[-1])

# fit the scaler
scaler = StandardScaler()
scaler.fit(U_tt_reshape)
print('means', scaler.mean_)

#transform training, val and test sets
U_tt_scaled = scaler.transform(U_tt_reshape)

#reshape 
U_tt_scaled = U_tt_scaled.reshape(U_tt.shape)

U_train     = split_data(U_tt_scaled, b_size, n_batches).astype('float32') #to be used for randomly shuffled batches

del U_tt, U_tt_scaled

## scale all the data ##
U_scaled = ss_transform(U, scaler)

# LOAD THE MODEL
# we do not have pooling and upsampling, instead we use stride=2
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
    
# Load best model
# Restore the checkpoint (this will restore the optimizer, encoder, and decoder states)
#how to load saved model
models_dir = CAE_model_path
a = [None]*N_parallel
b = [None]*N_parallel
for i in range(N_parallel):
    a[i] = tf.keras.models.load_model(models_dir + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5', 
                                          custom_objects={"PerPad2D": PerPad2D})
for i in range(N_parallel):
    b[i] = tf.keras.models.load_model(models_dir + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5',
                                          custom_objects={"PerPad2D": PerPad2D})

##### Compute encoded time_series #####
#set U to the standard scaler version
U = U_scaled
train_leng = 0
N_pos = 2000
k     = (U.shape[0] - train_leng)//N_pos

U_enc = np.empty((k*N_pos, N_1[1], N_1[2], N_1[3]))
for i in range(k):
        U_enc[i*N_pos:(i+1)*N_pos]= model(U[i*N_pos:(i+1)*N_pos], a, b)[0]

with h5py.File(output_path+'/encoded_data'+str(N_latent)+'.h5', 'w') as df:
    df['U_enc'] = U_enc


###### ESN #######
upsample   = 1
data_len   = 10000
transient  = 0

dt = dt*upsample
n_components = N_latent

dim      = N_latent #N_latent
act      = 'tanh'
U        = np.array(U_enc[transient:transient+data_len:upsample])
shape    = U.shape
print(shape)

U = U.reshape(shape[0], shape[1]*shape[2]*shape[3])

# number of time steps for washout, train, validation, test
t_lyap    = t_lyap
dt        = dt
N_lyap    = int(t_lyap//dt)
print('N_lyap', N_lyap)
N_washout = washout_len*N_lyap #75
N_washout_val = int(washout_len_val*N_lyap)
N_train   = train_len*N_lyap #600
N_val     = val_len*N_lyap #45
N_test    = test_len*N_lyap #45

indexes_to_plot = np.array([1, 2, 8, 16, 32, dim] ) -1
indexes_to_plot = indexes_to_plot[indexes_to_plot <= (dim-1)]

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)

# standardisation 
norm_std = U_data.std(axis=0)
normalisation = normalisation #on, off, standard

print('norm', norm)
print('u_mean', u_mean)
print('norm_std', norm_std)

# washout
U_washout = U[:N_washout].copy()
# data to be used for training + validation
U_tv  = U[N_washout:N_washout+N_train-1].copy() #inputs
Y_tv  = U[N_washout+1:N_washout+N_train].copy() #data to match at next timestep

# adding noise to training set inputs with sigma_n the noise of the data
# improves performance and regularizes the error as a function of the hyperparameters
fig,ax = plt.subplots(len(indexes_to_plot), figsize=(12,6), tight_layout=True, sharex=True)
for m in range(len(indexes_to_plot)):
    index = indexes_to_plot[m]
    ax[m].plot(U_tv[:N_val,index], c='b', label='Non-noisy')
seed = 42   #to be able to recreate the data, set also seed for initial condition u0
rnd1  = np.random.RandomState(seed)
noisy = True
if noisy:
    data_std = np.std(U,axis=0)
    sigma_n = noise #1e-3     #change this to increase/decrease noise in training inputs (up to 1e-1)
    for i in range(n_components):
        U_tv[:,i] = U_tv[:,i] \
                        + rnd1.normal(0, sigma_n*data_std[i], N_train-1)
    for m in range(len(indexes_to_plot)):
        index = indexes_to_plot[m]
        ax[m].plot(U_tv[:N_val,index], 'r--', label='Noisy')
        ax[m].grid()
        ax[m].set_title('mode %i' % (index+1))
ax[0].legend()
fig.savefig(output_path + '/noise_addition_sigman%.4f.png' % sigma_n)
plt.close()

#### ESN hyperparameters #####
if normalisation == 'on':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm))]) #input bias (average absolute value of the inputs)
elif normalisation == 'standard':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm_std))]) #input bias (average absolute value of the inputs)
elif normalisation == 'off':
    bias_in   = np.array([np.mean(np.abs(U_data))]) #input bias (average absolute value of the inputs)
bias_out  = np.array([1.]) #output bias

N_units      = Nr #neurons
connectivity = 3   
sparseness   = 1 - connectivity/(N_units-1) 

fln = ESN_model_path+'/ESN_matrices.mat'
data = loadmat(fln)
print(data.keys())

Winn             = data['Win'][0] #gives Winn
fix_hyp          = data['fix_hyp']
bias_in_value    = data['fix_hyp'][:,0]
N_washout        = data['fix_hyp'][:,1][0]
opt_hyp          = data['opt_hyp']
Ws               = data['W'][0]
Woutt            = data['Wout']
norm             = data['norm'][0]
bias_in          = np.ones((1))*bias_in_value

print('fix_hyp shape:', np.shape(fix_hyp))
print('bias_in_value:', bias_in_value, 'N_washout:', N_washout)
print('shape of bias_in:', np.shape(bias_in))
print('opt_hyp shape:', np.shape(opt_hyp))
print('W shape:', np.shape(Ws))
print('Win shape:', np.shape(Winn))
print('Wout shape:', np.shape(Woutt))
print('norm:', norm)
print('u_mean:', u_mean)
print('shape of norm:', np.shape(norm))

test_interval = True
validation_interval = False
statistics_interval = False

if validation_interval:
    ##### quick test #####
    print('VALIDATION (TEST)')
    print(N_washout_val)
    N_test   = 20                    #number of intervals in the test set
    if reduce_domain2:
        N_tstart = N_washout
    else:
        N_tstart = int(N_washout)                    #where the first test interval starts
    N_intt   = int(test_len*N_lyap)            #length of each test set interval

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

    ensemble_test = ensemble_test

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = 3
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print('index:', N_tstart + i*N_intt)
            print('start_time:', time_vals[N_tstart + i*N_intt])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt].copy()
            Y_t       = U[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt].copy()

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t,xa,Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
            print(np.shape(Yh_t))
            if i == 0:
                ens_pred[:, :, j] = Yh_t
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]
            nrmse_error[i, :] = Y_err

            ##### reconstructions ####
            # 1. reshape for decoder
            Y_t_reshaped = Y_t.reshape(Y_t.shape[0], N_1[1], N_1[2], N_1[3])
            Yh_t_reshaped = Yh_t.reshape(Yh_t.shape[0], N_1[1], N_1[2], N_1[3])
            # 2. put through decoder
            reconstructed_truth = Decoder(Y_t_reshaped,b).numpy()
            reconstructed_predictions = Decoder(Yh_t_reshaped,b).numpy()
            # 3. scale back
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            # metrics
            nrmse = NRMSE(reconstructed_truth, reconstructed_predictions)
            mse   = MSE(reconstructed_truth, reconstructed_predictions)
            evr   = EVR_recon(reconstructed_truth, reconstructed_predictions)
            SSIM  = compute_ssim_for_4d(reconstructed_truth, reconstructed_predictions)

            if len(variables) == 4:
                active_array, active_array_reconstructed, mask, mask_expanded_recon = active_array_calc(reconstructed_truth, reconstructed_predictions, z)
                accuracy = np.mean(active_array == active_array_reconstructed)
                if np.any(mask):  # Check if plumes exist
                    masked_truth = reconstructed_truth[mask]
                    masked_pred = reconstructed_predictions[mask]
                    
                    print("Shape truth after mask:", masked_truth.shape)
                    print("Shape pred after mask:", masked_pred.shape)

                    # Compute NRMSE only if mask is not empty
                    nrmse_plume = NRMSE(masked_truth, masked_pred)
                else:
                    print("Mask is empty, no plumes detected.")
                    nrmse_plume = 0  # Simply add 0 to maintain shape
            else:
                nrmse_plume = np.inf

            print('NRMSE', nrmse)
            print('MSE', mse)
            print('EVR_recon', evr)
            print('SSIM', SSIM)
            print('NRMSE plume', nrmse_plume)

            # Full path for saving the file
            output_file = 'ESN_test_metrics_ens%i_test%i.json' % (j,i)

            output_path_met = os.path.join(output_path, output_file)

            metrics = {
            "test": int(i),
            "no. modes": int(n_components),
            "EVR": float(evr),
            "MSE": float(mse),
            "NRMSE": float(nrmse),
            "SSIM": float(SSIM),
            "NRMSE plume": float(nrmse_plume),
            "PH": float(PH[i]),
            }

            with open(output_path_met, "w") as file:
                json.dump(metrics, file, indent=4)

            ens_nrmse[j]       += nrmse
            ens_ssim[j]        += SSIM
            ens_nrmse_plume[j] += nrmse_plume
            ens_evr[j]         += evr
            ens_PH2[j]         += PH[i]

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    if ensemble_test % 1 == 0:
                        
                        #### modes prediction ####
                        fig,ax =plt.subplots(len(indexes_to_plot),sharex=True, tight_layout=True)
                        print(np.shape(U_wash))
                        xx = np.arange(U_wash[:,0].shape[0])/N_lyap
                        print(np.shape(xx))
                        print(np.shape(xx), xx[0], xx[-1])
                        for v in range(len(indexes_to_plot)):
                            index = indexes_to_plot[v]
                            ax[v].plot(xx,U_wash[:,index], 'b', label='True')
                            ax[v].plot(xx,Uh_wash[:-1,index], '--r', label='ESN')
                            ax[v].grid()
                            ax[v].set_ylabel('mode %i' % (index+1))
                        ax[-1].set_xlabel('Time[Lyapunov Times]')
                        if i==0:
                            ax[0].legend(ncol=2)
                        fig.suptitle('washout_ens%i_test%i' % (j,i))
                        fig.savefig(output_path+'/washout_validation_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(len(indexes_to_plot),sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        for v in range(len(indexes_to_plot)):
                            index = indexes_to_plot[v]
                            ax[v].plot(xx,Y_t[:,index], 'b')
                            ax[v].plot(xx,Yh_t[:,index], '--r')
                            ax[v].grid()
                            ax[v].set_ylabel('mode %i' % (index+1))
                        ax[-1].set_xlabel('Time [Lyapunov Times]')
                        fig.savefig(output_path+'/prediction_validation_ens%i_test%i.png' % (j,i))
                        plt.close()
                        
                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(xx,Y_err, 'b')
                        ax.axhline(y=threshold_ph, xmin=xx[0], xmax=xx[-1])
                        ax.grid()
                        ax.set_ylabel('PH')
                        ax.set_xlabel('Time')
                        fig.savefig(output_path+'/PH_validation_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(np.linalg.norm(Xa1[:, :N_units], axis=1))
                        ax.grid()
                        ax.set_ylabel('res_states')
                        fig.savefig(output_path+'/res_states_washout_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(xx, np.linalg.norm(Xa2[:, :N_units], axis=1))
                        ax.grid()
                        ax.set_ylabel('res_states')
                        fig.savefig(output_path+'/res_states_validation_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(time_vals[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt], np.linalg.norm(Xa1[:-1, :N_units], axis=1), color='red')
                        ax.plot(time_vals[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt], np.linalg.norm(Xa2[:, :N_units], axis=1), color='blue')
                        ax.grid()
                        ax.set_ylabel('res_norm')
                        fig.savefig(output_path+'/resnorm_validation_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(time_vals[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt], np.linalg.norm(U_wash, axis=1), color='red')
                        ax.plot(time_vals[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt], np.linalg.norm(Y_t, axis=1), color='blue')
                        ax.grid()
                        ax.set_ylabel('input_norm')
                        fig.savefig(output_path+'/inputnorm_validation_ens%i_test%i.png' % (j,i))
                        plt.close()

                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, 1*N_lyap, xx, 'ESN_validation_ens%i_test%i' %(j,i))

                        if len(variables) == 4:
                            fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
                            c1 = ax[0].contourf(xx, x, active_array[:,:, 32].T, cmap='Reds')
                            fig.colorbar(c1, ax=ax[0])
                            ax[0].set_title('true')
                            c2 = ax[1].contourf(xx, x, active_array_reconstructed[:,:, 32].T, cmap='Reds')
                            fig.colorbar(c1, ax=ax[1])
                            ax[1].set_title('reconstruction')
                            for v in range(2):
                                ax[v].set_xlabel('time')
                                ax[v].set_ylabel('x')
                            fig.savefig(output_path+f"/active_plumes_validation_ens{j}_test{i}.png")
                            plt.close()
                        else:
                            print('no image')

        # accumulation for each ensemble member
        ens_nrmse[j]       = ens_nrmse[j] / N_test
        ens_nrmse_plume[j] = ens_nrmse_plume[j] / N_test
        ens_ssim[j]        = ens_ssim[j] / N_test
        ens_evr[j]         = ens_evr[j] / N_test
        ens_PH2[j]         = ens_PH2[j] / N_test  
             
    # Full path for saving the file
    output_file_ALL = 'ESN_validation_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    flatten_PH = ens_PH.flatten()
    print('flat PH', flatten_PH)

    metrics_ens_ALL = {
    "mean PH": np.mean(ens_PH2),
    "lower PH": np.quantile(flatten_PH, 0.75),
    "uppper PH": np.quantile(flatten_PH, 0.25),
    "median PH": np.median(flatten_PH),
    "mean NRMSE": np.mean(ens_nrmse),
    "mean NRMSE plume": np.mean(ens_nrmse_plume),
    "mean EVR": np.mean(ens_evr),
    "mean ssim": np.mean(ens_ssim),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished validations')


if test_interval:
    ##### quick test #####
    print('TESTING')
    N_test   = 20                    #number of intervals in the test set
    if reduce_domain:
        N_tstart = N_train + N_washout_val
    elif reduce_domain2:
        N_tstart = N_train + N_washout_val
    else:
        N_tstart = int(N_train + N_washout) #850    #where the first test interval starts
    N_intt   = int(test_len*N_lyap)             #length of each test set interval

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

    ensemble_test = ensemble_test

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = N_test
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print(N_tstart + i*N_intt)
            print('start_time:', time_vals[N_tstart + i*N_intt])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt].copy()
            Y_t       = U[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt].copy()

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t, xa, Xa2        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)
            print(np.shape(Yh_t))
            if i == 0:
                ens_pred[:, :, j] = Yh_t
            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]
            nrmse_error[i, :] = Y_err

            ##### reconstructions ####
            # 1. reshape for decoder
            Y_t_reshaped = Y_t.reshape(Y_t.shape[0], N_1[1], N_1[2], N_1[3])
            Yh_t_reshaped = Yh_t.reshape(Yh_t.shape[0], N_1[1], N_1[2], N_1[3])
            # 2. put through decoder
            reconstructed_truth = Decoder(Y_t_reshaped,b).numpy()
            reconstructed_predictions = Decoder(Yh_t_reshaped,b).numpy()
            # 3. scale back
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            # metrics
            nrmse = NRMSE(reconstructed_truth, reconstructed_predictions)
            mse   = MSE(reconstructed_truth, reconstructed_predictions)
            evr   = EVR_recon(reconstructed_truth, reconstructed_predictions)
            SSIM  = compute_ssim_for_4d(reconstructed_truth, reconstructed_predictions)

            if len(variables) == 4:
                active_array, active_array_reconstructed, mask, mask_expanded_recon = active_array_calc(reconstructed_truth, reconstructed_predictions, z)
                accuracy = np.mean(active_array == active_array_reconstructed)
                if np.any(mask):  # Check if plumes exist
                    masked_truth = reconstructed_truth[mask]
                    masked_pred = reconstructed_predictions[mask]
                    
                    print("Shape truth after mask:", masked_truth.shape)
                    print("Shape pred after mask:", masked_pred.shape)

                    # Compute NRMSE only if mask is not empty
                    nrmse_plume = NRMSE(masked_truth, masked_pred)
                else:
                    print("Mask is empty, no plumes detected.")
                    nrmse_plume = 0  # Simply add 0 to maintain shape
            else:
                nrmse_plume = np.inf

            print('NRMSE', nrmse)
            print('MSE', mse)
            print('EVR_recon', evr)
            print('SSIM', SSIM)
            print('NRMSE plume', nrmse_plume)

            # Full path for saving the file
            output_file = 'ESN_test_metrics_ens%i_test%i.json' % (j,i)

            output_path_met = os.path.join(output_path, output_file)

            metrics = {
            "test": int(i),
            "no. modes": int(n_components),
            "EVR": float(evr),
            "MSE": float(mse),
            "NRMSE": float(nrmse),
            "SSIM": float(SSIM),
            "NRMSE plume": float(nrmse_plume),
            "PH": float(PH[i]),
            }

            with open(output_path_met, "w") as file:
                json.dump(metrics, file, indent=4)

            ens_nrmse[j]       += nrmse
            ens_ssim[j]        += SSIM
            ens_nrmse_plume[j] += nrmse_plume
            ens_evr[j]         += evr
            ens_PH2[j]         += PH[i]

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    if ensemble_test % 1 == 0:
                        
                        #### modes prediction ####
                        fig,ax =plt.subplots(len(indexes_to_plot),sharex=True, tight_layout=True)
                        xx = np.arange(U_wash[:,-2].shape[0])/N_lyap
                        print(np.shape(xx), xx[0], xx[-1])
                        for v in range(len(indexes_to_plot)):
                            index = indexes_to_plot[v]
                            ax[v].plot(xx,U_wash[:,index], 'b', label='True')
                            ax[v].plot(xx,Uh_wash[:-1,index], '--r', label='ESN')
                            ax[v].grid()
                            ax[v].set_ylabel('mode %i' % (index+1))
                        ax[-1].set_xlabel('Time[Lyapunov Times]')
                        if i==0:
                            ax[0].legend(ncol=2)
                        fig.suptitle('washout_ens%i_test%i' % (j,i))
                        fig.savefig(output_path+'/washout_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(len(indexes_to_plot),sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        for v in range(len(indexes_to_plot)):
                            index = indexes_to_plot[v]
                            ax[v].plot(xx,Y_t[:,index], 'b')
                            ax[v].plot(xx,Yh_t[:,index], '--r')
                            ax[v].grid()
                            ax[v].set_ylabel('mode %i' % (index+1))
                        ax[-1].set_xlabel('Time [Lyapunov Times]')
                        fig.savefig(output_path+'/prediction_ens%i_test%i.png' % (j,i))
                        plt.close()
                        
                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(xx,Y_err, 'b')
                        ax.axhline(y=threshold_ph, xmin=xx[0], xmax=xx[-1])
                        ax.grid()
                        ax.set_ylabel('PH')
                        ax.set_xlabel('Time')
                        fig.savefig(output_path+'/PH_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(np.linalg.norm(Xa1[:, :N_units], axis=1))
                        ax.grid()
                        ax.set_ylabel('res_states')
                        fig.savefig(output_path+'/res_states_test_washout_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(xx, np.linalg.norm(Xa2[:, :N_units], axis=1))
                        ax.grid()
                        ax.set_ylabel('res_states')
                        fig.savefig(output_path+'/res_states_test_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(time_vals[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt], np.linalg.norm(Xa1[:-1, :N_units], axis=1), color='red')
                        ax.plot(time_vals[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt], np.linalg.norm(Xa2[:, :N_units], axis=1), color='blue')
                        ax.grid()
                        ax.set_ylabel('res_norm')
                        fig.savefig(output_path+'/resnorm_test_ens%i_test%i.png' % (j,i))
                        plt.close()

                        fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ax.plot(time_vals[N_tstart - N_washout_val +i*N_intt : N_tstart + i*N_intt], np.linalg.norm(U_wash, axis=1), color='red')
                        ax.plot(time_vals[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt], np.linalg.norm(Y_t, axis=1), color='blue')
                        ax.grid()
                        ax.set_ylabel('input_norm')
                        fig.savefig(output_path+'/inputnorm_test_ens%i_test%i.png' % (j,i))
                        plt.close()

                        # reconstruction after scaling
                        print('reconstruction and error plot')
                        plot_reconstruction_and_error(reconstructed_truth, reconstructed_predictions, 32, 1*N_lyap, xx, 'ESN_ens%i_test%i' %(j,i))

                        if len(variables) == 4:
                            fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
                            c1 = ax[0].contourf(xx, x, active_array[:,:, 32].T, cmap='Reds')
                            fig.colorbar(c1, ax=ax[0])
                            ax[0].set_title('true')
                            c2 = ax[1].contourf(xx, x, active_array_reconstructed[:,:, 32].T, cmap='Reds')
                            fig.colorbar(c1, ax=ax[1])
                            ax[1].set_title('reconstruction')
                            for v in range(2):
                                ax[v].set_xlabel('time')
                                ax[v].set_ylabel('x')
                            fig.savefig(output_path+f"/active_plumes_ens{j}_test{i}.png")
                            plt.close()
                        else:
                            print('no image')

        # accumulation for each ensemble member
        ens_nrmse[j]       = ens_nrmse[j] / N_test
        ens_nrmse_plume[j] = ens_nrmse_plume[j] / N_test
        ens_ssim[j]        = ens_ssim[j] / N_test
        ens_evr[j]         = ens_evr[j] / N_test
        ens_PH2[j]         = ens_PH2[j] / N_test  
             
    # Full path for saving the file
    output_file_ALL = 'ESN_test_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    flatten_PH = ens_PH.flatten()
    print('flat PH', flatten_PH)

    metrics_ens_ALL = {
    "mean PH": np.mean(ens_PH2),
    "lower PH": np.quantile(flatten_PH, 0.75),
    "uppper PH": np.quantile(flatten_PH, 0.25),
    "median PH": np.median(flatten_PH),
    "mean NRMSE": np.mean(ens_nrmse),
    "mean NRMSE plume": np.mean(ens_nrmse_plume),
    "mean EVR": np.mean(ens_evr),
    "mean ssim": np.mean(ens_ssim),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished testing')

if statistics_interval:
    #### STATISTICS ####
    output_path = output_path + '/statistics/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('made directory')

    N_test   = 1                    #number of intervals in the test set
    N_tstart = int(N_washout)   #where the first test interval starts
    N_intt   = 15*N_lyap             #length of each test set interval
    N_washout = int(N_washout)
    N_gap = int(N_lyap)

    print('N_tstart:', N_tstart)
    print('N_intt:', N_intt)
    print('N_washout:', N_washout)

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
    threshold_ph = 0.2

    ensemble_test = ensemble_test

    ens_pred_global = np.zeros((N_intt, 2, N_test, ensemble_test))
    true_POD_global = np.zeros((N_intt, 2, N_test))
    true_global     = np.zeros((N_intt, 2, N_test))
    ens_PH          = np.zeros((N_intt, ensemble_test))
    ens_nrmse_global= np.zeros((ensemble_test))
    ens_mse_global  = np.zeros((ensemble_test))

    for j in range(ensemble_test):

        print('Realization    :',j+1)

        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

        # to store prediction horizon in the test set
        PH             = np.zeros(N_test)
        nrmse_error    = np.zeros((N_test, N_intt))

        # to plot results
        plot = True
        Plotting = True
        if plot:
            n_plot = 3
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print(N_tstart + i*N_gap)
            print('start time of test', time_vals[N_tstart + i*N_gap])
            # data for washout and target in each interval
            U_wash    = U[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap].copy()
            Y_t       = U[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt].copy()

            #washout for each interval
            Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
            Uh_wash = np.dot(Xa1, Wout)

            # Prediction Horizon
            Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
            print(np.shape(Yh_t))

            Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
            PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
            ens_PH[i,j] = PH[i]

            ##### reconstructions ####
            # 1. reshape for decoder
            Y_t_reshaped = Y_t.reshape(Y_t.shape[0], N_1[1], N_1[2], N_1[3])
            Yh_t_reshaped = Yh_t.reshape(Yh_t.shape[0], N_1[1], N_1[2], N_1[3])
            # 2. put through decoder
            reconstructed_truth = Decoder(Y_t_reshaped,b).numpy()
            reconstructed_predictions = Decoder(Yh_t_reshaped,b).numpy()
            # 3. scale back
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)

            ## global parameters ##
            PODtruth_global    = global_parameters(reconstructed_truth)
            predictions_global = global_parameters(reconstructed_predictions)
            ens_pred_global[:,:,i,j] = predictions_global
            true_POD_global[:,:,i] = PODtruth_global
            true_global[:,:,i] = truth_global[N_tstart + i*N_gap: N_tstart + i*N_gap + N_intt]

            # metrics
            nrmse_global = NRMSE(PODtruth_global, predictions_global)
            mse_global   = MSE(PODtruth_global, predictions_global)


            ens_nrmse_global[j]+= nrmse_global
            ens_mse_global[j]  += mse_global

            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    if ensemble_test % 1 == 0:
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        ### global prediction ###
                        fig, ax = plt.subplots(2, figsize=(12,6), sharex=True, tight_layout=True)
                        for v in range(2):
                            ax[v].plot(xx, PODtruth_global[:,v], label='POD')
                            ax[v].plot(xx, predictions_global[:,v], label='ESN')
                            ax[v].grid()
                            ax[v].legend()
                            ax[v].set_ylabel(global_labels[v])
                        ax[-1].set_xlabel('LT')
                        fig.savefig(output_path+f"/global_prediciton_ens{j}_test{i}.png")
                        plt.close()

            from scipy.stats import gaussian_kde
            fig, ax = plt.subplots(2, figsize=(8,6))
            for v in range(2):
                kde_true  = gaussian_kde(PODtruth_global[:,v])
                kde_pred  = gaussian_kde(predictions_global[:,v])

                var_vals_true      = np.linspace(min(PODtruth_global[:,v]), max(PODtruth_global[:,v]), 1000)  # X range
                pdf_vals_true      = kde_true(var_vals_true)
                var_vals_pred      = np.linspace(min(predictions_global[:,v]), max(predictions_global[:,v]), 1000)  # X range
                pdf_vals_pred      = kde_pred(var_vals_pred)

                
                ax[v].plot(var_vals_true, pdf_vals_true, label="truth")
                ax[v].plot(var_vals_pred, pdf_vals_pred, label="prediction")
                ax[v].grid()
                ax[v].set_ylabel('Denisty')
                ax[v].set_xlabel('Value')
                ax[v].legend()
            fig.savefig(output_path+f"/stats_pdf_global_ens{j}_test{i}.png")

            fig, ax = plt.subplots(1, figsize=(8,6))
            ax.scatter(Y_t[:,0], Y_t[:,1], label='truth')
            ax.scatter(Yh_t[:,0], Yh_t[:,1], label='prediction')
            ax.grid()
            ax.set_xlabel('mode 1')
            ax.set_ylabel('mode 2')
            ax.legend()
            fig.savefig(output_path+f"/trajectories_ens{j}_test{i}.png")

            fig, ax = plt.subplots(len(indexes_to_plot), figsize=(12, 12))
            for v, element in enumerate(indexes_to_plot):
                kde_true  = gaussian_kde(Y_t[:,element])
                kde_pred  = gaussian_kde(Yh_t[:,element])

                var_vals_true      = np.linspace(min(Y_t[:,element]), max(Y_t[:,element]), 1000)  # X range
                pdf_vals_true      = kde_true(var_vals_true)
                var_vals_pred      = np.linspace(min(Yh_t[:,element]), max(Yh_t[:,element]), 1000)  # X range
                pdf_vals_pred      = kde_pred(var_vals_pred)

                ax[v].plot(var_vals_true, pdf_vals_true, label="truth")
                ax[v].plot(var_vals_pred, pdf_vals_pred, label="prediction")
                ax[v].grid()
                ax[v].set_ylabel('Denisty')
                ax[v].set_xlabel(f"Mode {v}")
                ax[v].legend()
            fig.savefig(output_path+f"/stats_pdf_modes_ens{j}_test{i}.png")

        # accumulation for each ensemble member
        ens_nrmse_global[j]= ens_nrmse_global[j]/N_test

    # Full path for saving the file
    output_file_ALL = 'ESN_statistics_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    metrics_ens_ALL = {
    "mean global NRMSE": np.mean(ens_nrmse_global),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished statistics')
