"""
python script for convolutional autoencoder.

Usage: CAE.py [--input_path=<input_path> --output_path=<output_path> --model_path=<model_path> --hyperparam_file=<hyperparam_file> --job_id=<job_id> --sweep_id=<sweep_id> --reduce_domain=<reduce_domain> --encoded_data=<encoded_data>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --model_path=<model_path>            file path to location of job 
    --hyperparam_file=<hyperparam_file>  file with hyperparmas
    --job_id=<job_id>                    job_id
    --sweep_id=<sweep_id>                sweep_id
    --reduce_domain=<reduce_domain>      domain reduced True or False [default: False]
    --encoded_data=<encoded data>        data already encoded True or False [default: False]
"""


# import packages
import time

import sys

import time

import os
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
#import pandas as pd
import numpy as np
import csv
import yaml
import math

from docopt import docopt
args = docopt(__doc__)

os.environ["OMP_NUM_THREADS"] = "-1" #set cores for numpy
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
from Eval_Functions import *
from Plotting_Functions import *
from sklearn.metrics import precision_score, recall_score, f1_score

import sys
sys.stdout.reconfigure(line_buffering=True)

import wandb
#wandb.login()

input_path = args['--input_path']
output_path = args['--output_path']
model_path = args['--model_path']
hyperparam_file = args['--hyperparam_file']
job_id = args['--job_id']
sweep_id = args['--sweep_id']

encoded_data = args['--encoded_data']
if encoded_data == 'False':
    encoded_data = False
    print('data not already encoded so', encoded_data)
elif encoded_data == 'True':
    encoded_data = True
    print('data already encoded so', encoded_data)

reduce_domain = args['--reduce_domain']
if reduce_domain == 'False':
    reduce_domain = False
    print('domain not reduced', reduce_domain)
elif reduce_domain == 'True':
    reduce_domain = True
    print('domain reduced', reduce_domain)

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

#### LOAD DATA ####
Data = 'RB'
if Data == 'ToyData':
    name = names = variables = ['combined']
    n_components = 3
    num_variables = 1
    snapshots = snapshots_load = 25000
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

reduce_domain = reduce_domain

if reduce_domain:
    # data_set = data_set[200:424,60:80,:,:] # 408 so we have 13 batches 12 for training and 1 for 'validation'
    # x = x[60:80]
    # time_vals = time_vals[200:424]
    # print('reduced domain shape', np.shape(data_set))
    # print('reduced x domain', np.shape(x))
    # print('reduced x domain', len(x))
    # print(x[0], x[-1])

    data_set = data_set[170:410,32:108,:,:] ## 370 but to make divisble by 16 here and add 2 extra batches for val and test
    x = x[32:108]
    time_vals = time_vals[170:410]
    print('reduced domain shape', np.shape(data_set))
    print('reduced x domain', np.shape(x))
    print('reduced x domain', len(x))
    print(x[0], x[-1])

#### plot dataset ####
if Data == 'ToyData':
    fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
    c1 = ax.pcolormesh(time_vals, x, data_set[:,:,32,0].T)
    fig.colorbar(c1, ax=ax)
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

U = data_set

import yaml

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


#### load in data ###
if reduce_domain:
    n_batches   = 13 #int((U.shape[0]/b_size) *0.7)  #number of batches #20
    val_batches = 1 #int((U.shape[0]/b_size) *0.2)    #int(n_batches*0.2) # validation set size is 0.2 the size of the training set #2
    test_batches = 1#int((U.shape[0]/b_size) *0.1)
else:
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
U_tt_scaled = U_tt_scaled.reshape(U_tt.shape)
U_train     = split_data(U_tt_scaled, b_size, n_batches).astype('float32')

global_stds = [np.std(U_tt[..., c]) for c in range(U_tt.shape[-1])]

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

Loss_Mse    = tf.keras.losses.MeanSquaredError()


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

output_path = output_path+'/Active_Thresholds/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

data_set       = U
data_scaled    = ss_transform(U, scaler)
time_vals      = time_vals

snapshots_POD  = 11200
print('snapshots for CAE', snapshots_POD)
data_set       = data_set[:snapshots_POD]
data_scaled    = data_scaled[:snapshots_POD]
time_vals      = time_vals[:snapshots_POD]

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


### ----- CAE ------
print('running decoder... ')

batch_size = 1000
T = data_scaled.shape[0]
decoded_batches = []

for start in range(0, T, batch_size):
    print(start)
    end = min(start + batch_size, T)
    batch = data_scaled[start:end]
    
    decoded_batch = model(batch, a, b)[1].numpy()  # run decoder on batch
    decoded_batches.append(decoded_batch)

# Combine all batches
decoded = np.concatenate(decoded_batches, axis=0)

decoded_unscaled = ss_inverse_transform(decoded, scaler)

print('finding active array ...')
_, active_array_reconstructed, _, mask_reconstructed = active_array_calc(data_set, decoded_unscaled, z)
mask_original     = mask_expanded[..., 0]

# print('finding new_array ...')
# new_array_reconstructed = np.zeros((len(time_vals), len(x) // 4, len(z) // 4))

# # Iterate over the time dimension
# for t in range(len(time_vals)):
#     # Iterate over the 64 subgrids along the z-axis (along the x-dimension)
#     for i in range(0, len(x), 4):
#         # Iterate over the 16 subgrids along the x-axis (along the z-dimension)
#         for j in range(0, len(z), 4):
#             # Check if any value in the 4x4 subgrid is 1
#             if np.any(active_array_reconstructed[t, i:i+4, j:j+4] == 1):
#                 new_array_reconstructed[t, i // 4, j // 4] = 1

### ---- plot figures -----
# Binary colormap: white=0, red=1
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
data_reconstructed = decoded_unscaled[index:index+500]
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

### ---- change the thresholds ----
# from itertools import product

# RH_thresholds = [0.91, 0.92, 0.93, 0.94]
# w_thresholds  = [-0.02, -0.01, 0]
# b_thresholds  = [-0.02, -0.01, 0]

# threshold_grid = list(product(RH_thresholds, w_thresholds, b_thresholds))
# total_tests = len(RH_thresholds) * len(w_thresholds) * len(b_thresholds)

# results = []

# test_index = 1
# T_total = data_scaled.shape[0]
# batch_size = 500  # adjust based on memory
# block_size_x, block_size_z = 4, 4
# nx, nz = 256, 64
# nx_new, nz_new = nx // block_size_x, nz // block_size_z

# for RH_thresh, w_thresh, b_thresh in threshold_grid:
#     print(f"Test {test_index} of {total_tests}")

#     # Temporary storage for all batches
#     all_downsampled = []

#     # Loop over batches to compute active array and downsample
#     for start in range(0, T_total, batch_size):
#         end = min(start + batch_size, T_total)
#         batch_set = data_set[start:end]
#         batch_reconstructed = decoded_unscaled[start:end]

#         # Compute active array for this batch & threshold
#         _, active_array_batch, _, _ = active_array_calc_softer(
#             batch_set, batch_reconstructed, z,
#             RH_threshold=RH_thresh, w_threshold=w_thresh, b_threshold=b_thresh
#         )
#         active_array_batch = active_array_batch.astype(np.bool_)  # save memory

#         # Vectorized 4x4 downsampling
#         T_batch = active_array_batch.shape[0]
#         downsampled_batch = active_array_batch.reshape(
#             T_batch, nx_new, block_size_x, nz_new, block_size_z
#         ).max(axis=(2, 4))  # shape (T_batch, 64, 16)

#         all_downsampled.append(downsampled_batch)

#     # Concatenate all batches for this threshold
#     full_downsampled = np.concatenate(all_downsampled, axis=0)

#     # Compute metrics once for the full dataset
#     metrics = plume_detection_metrics(new_array, full_downsampled)
#     metrics.update({'RH_thresh': RH_thresh, 'w_thresh': w_thresh, 'b_thresh': b_thresh})
#     results.append(metrics)

#     test_index += 1

# # Make sure results is not empty
# if len(results) == 0:
#     raise ValueError("No results to save!")

# # Field names (columns) inferred from keys of the first dict
# fieldnames = list(results[0].keys())

# # Save to CSV
# with open(output_path+'plume_metrics.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()       # write column headers
#     writer.writerows(results)  # write all rows

# print("Saved metrics to plume_metrics.csv")

### --- plot with softer threshold ---

index = 5000
data_set = data_set[index:index+500]
data_reconstructed = decoded_unscaled[index:index+500]
time_vals = time_vals[index:index+500]

RH_thresh = 0.93
w_thresh  = -0.01
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
