"""
python script for convolutional autoencoder.

Usage: CAE.py [--input_path=<input_path> --output_path=<output_path> --model_path=<model_path> --hyperparam_file=<hyperparam_file>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --model_path=<model_path>            file path to location of job 
    --hyperparam_file=<hyperparam_file>  file with hyperparmas
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
wandb.login()

input_path = args['--input_path']
output_path = args['--output_path']
model_path = args['--model_path']
hyperparam_file = args['--hyperparam_file']

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


#### LOAD DATA AND POD ####
name='combined'
data_set, x, z, time_vals = load_data(input_path+'/plume_wave_dataset.h5', name)
print('shape of data', np.shape(data_set))

data_set = data_set.reshape(len(time_vals), len(x), len(z), 1)
print('shape of reshaped data set', np.shape(data_set))

U = data_set
dt = time_vals[1]-time_vals[0]
print('dt:', dt)
variables = 1

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

#### load in data ###
#b_size      = wandb.config.b_size   #batch_size
n_batches   = int((U.shape[0]/b_size) *0.6)  #number of batches #20
val_batches = int((U.shape[0]/b_size) *0.2)    #int(n_batches*0.2) # validation set size is 0.2 the size of the training set #2
test_batches = int((U.shape[0]/b_size) *0.2)
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

# Placeholder for standardized data
U_tt_scaled = np.zeros_like(U_tt)
U_vv_scaled = np.zeros_like(U_vv)
U_tv_scaled = np.zeros_like(U_tv)
scalers = [StandardScaler() for _ in range(variables)]

# Apply StandardScaler separately to each channel
for v in range(variables):
    # Reshape training data for the current variable
    reshaped_train_channel = U_tt[:, :, :, v].reshape(-1, U_tt.shape[1] * U_tt.shape[2])

    print('shape of data for ss', np.shape(reshaped_train_channel))

    # Fit the scaler on the training data for the current variable
    scaler = scalers[v]
    scaler.fit(reshaped_train_channel)

    # Standardize the training data
    standardized_train_channel = scaler.transform(reshaped_train_channel)

    # Reshape the standardized data back to the original shape (batches, batch_size, x, z)
    U_tt_scaled[:, :, :, v] = standardized_train_channel.reshape(U_tt.shape[0], U_tt.shape[1], U_tt.shape[2])
    
    # Standardize the validation data using the same scaler
    reshaped_val_channel = U_vv[:, :, :, v].reshape(-1, U_vv.shape[1] * U_vv.shape[2])
    standardized_val_channel = scaler.transform(reshaped_val_channel)
    U_vv_scaled[:, :, :, v] = standardized_val_channel.reshape(U_vv.shape[0], U_vv.shape[1], U_vv.shape[2])

    # Standardize the test data using the same scaler
    reshaped_test_channel = U_tv[:, :, :, v].reshape(-1, U_tv.shape[1] * U_tv.shape[2])
    standardized_test_channel = scaler.transform(reshaped_test_channel)
    U_tv_scaled[ :, :, :, v] = standardized_test_channel.reshape(U_tv.shape[0], U_tv.shape[1], U_tv.shape[2])

test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                         b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 

for v in range(variables):
    fig, ax = plt.subplots(1)
    c=ax.pcolormesh(test_times, x, U_tv_scaled[:,:,32,v].T)
    fig.colorbar(c, ax=ax)        
    fig.savefig(output_path+'/test_scaling%i.png' % v)

    fig, ax = plt.subplots(1)
    c=ax.pcolormesh(U_tt_scaled[:,:,32,v].T)
    fig.colorbar(c, ax=ax)        
    fig.savefig(output_path+'/train_scaling%i.png' % v)

    fig, ax = plt.subplots(1)
    c=ax.pcolormesh(test_times, x, U_tv[:,:,32,v].T)
    fig.colorbar(c, ax=ax)        
    fig.savefig(output_path+'/test_unscaled%i.png' % v)

    fig, ax = plt.subplots(1)
    c=ax.pcolormesh(U_tt[:,:,32,v].T)
    fig.colorbar(c, ax=ax)        
    fig.savefig(output_path+'/train_unscaled%i.png' % v)

U_train     = split_data(U_tt_scaled, b_size, n_batches).astype('float32') #to be used for randomly shuffled batches
U_val       = split_data(U_vv_scaled, b_size, val_batches).astype('float32')

del U_vv, U_tt, U_vv_scaled, U_tt_scaled

# define the model
# we do not have pooling and upsampling, instead we use stride=2

#lat_dep       = wandb.config.lat_dep       #latent space depth
#kernel_choice = wandb.config.kernel_choice
n_fil         = [6,12,24,lat_dep]          #number of filters ecnoder
n_dec         = [24,12,6,3]                #number of filters decoder
#N_parallel    = wandb.config.N_parallel    #number of parallel CNNs for multiscale
ker_size      = [(3,3), (5,5), (7,7)]      #kernel sizes
if N_parallel == 1:
    ker_size  = [ker_size[kernel_choice]]
#N_layers      = wandb.config.N_layers    #number of layers in every CNN
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

        #stride=2 padding and conv
        enc_mods[j].add(PerPad2D(padding=p_size[j], asym=True,
                                          name='Enc_' + str(j)+'_PerPad_'+str(i)))
        # Modify the last encoder layer to downsample more aggressively
        if i == N_layers - 1:  # Last layer before latent space
            enc_mods[j].add(tf.keras.layers.Conv2D(
                filters=n_fil[i], kernel_size=ker_size[j],
                activation=act, padding=pad_enc, strides=4,  # Increased stride
                name='Enc_' + str(j)+'_ConvLayer_'+str(i)))
        else:
            enc_mods[j].add(tf.keras.layers.Conv2D(
                filters=n_fil[i], kernel_size=ker_size[j],
                activation=act, padding=pad_enc, strides=2,  # Keep stride-2 for earlier layers
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
        
        # Modify the first transpose convolution to compensate for the encoder change
        if i == 0:  # First upsampling layer
            dec_mods[j].add(tf.keras.layers.Conv2DTranspose(filters=n_dec[i],
                                          output_padding=None, kernel_size=ker_size[j],
                                          activation=act, padding=pad_dec, strides=4,  # Increased stride
                                          name='Dec_' + str(j)+'_ConvLayer_'+str(i)))
        else:
            # Transpose convolution with normal stride = 2 
            dec_mods[j].add(tf.keras.layers.Conv2DTranspose(filters=n_dec[i],
                                          output_padding=None, kernel_size=ker_size[j],
                                          activation=act, padding=pad_dec, strides=2,  # Regular stride
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

#### NRMSE across x,z and averaged in time ####
truth = U_tv_scaled
decoded = model(truth,a,b)[1].numpy()
print(np.shape(decoded), np.shape(truth))

test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                         b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 
print(test_times[0], test_times[-1])

#SSIM
test_ssim = compute_ssim_for_4d(truth, decoded)

#MSE
mse = MSE(truth, decoded)

#NRMSE
nrmse = NRMSE(truth, decoded)

#EVR 
evr = EVR_recon(truth, decoded)

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


print('finished job')

decoded = decoded
decoded_unscaled = np.zeros_like(decoded)
for v in range(variables):
    reshaped_channel_decoded = decoded[:, :, :, v].reshape(-1, decoded.shape[1] * decoded.shape[2])
    unscaled_decoded_channel = scalers[v].inverse_transform(reshaped_channel_decoded)
    decoded_unscaled[:, :, :, v] = unscaled_decoded_channel.reshape(decoded.shape[0], decoded.shape[1], decoded.shape[2])

truth_unscaled = np.zeros_like(truth)
for v in range(variables):
    reshaped_channel_truth = truth[:, :, :, v].reshape(-1, truth.shape[1] * truth.shape[2])
    unscaled_truth_channel = scalers[v].inverse_transform(reshaped_channel_truth)
    truth_unscaled[:, :, :, v] = unscaled_truth_channel.reshape(truth.shape[0], truth.shape[1],truth.shape[2])
    
n       =  4

start   = b_size*n_batches*skip+b_size*val_batches*skip  #b_size*n_batches*skip+b_size*val_batches*skip #start after validation set

skips = 10
for i in range(n):
    fig, ax = plt.subplots(3,variables, figsize=(12,6), tight_layout=True)
    index = 0 + skips*i

    fig, ax =plt.subplots(1, figsize=(6,4), tight_layout=True)
    plot_reconstruction(truth_unscaled, decoded_unscaled, 32, index, name)


