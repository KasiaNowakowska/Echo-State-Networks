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
#import pandas as pd
import numpy as np
import h5py
import yaml

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

#### LOAD DATA ####
Data = 'ToyData'
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

reduce_domain = reduce_domain

if reduce_domain:
    data_set = data_set[200:424,60:80,:,:] # 408 so we have 13 batches 12 for training and 1 for 'validation'
    x = x[60:80]
    time_vals = time_vals[200:424]
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
    n_batches   = 12 #int((U.shape[0]/b_size) *0.7)  #number of batches #20
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

global_stds = [np.std(U_tt[..., c]) for c in range(U_tt.shape[-1])]

del U_vv, U_tt, U_tt_scaled

## scale all the data ##
U_scaled = ss_transform(U, scaler)

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

validation_data = False
test_data = True
all_data = True
visualisation = False
encoded_data_investigation = False

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

    #Plume NRMSE
    if len(variables) == 4:
        active_array, active_array_reconstructed, mask, mask_reconstructed = active_array_calc(truth_unscaled, decoded_unscaled, z)
        print(np.shape(active_array))
        print(np.shape(mask))
        nrmse_plume             = NRMSE(truth_unscaled[:,:,:,:][mask], decoded_unscaled[:,:,:,:][mask])

        mask_original     = mask[..., 0]
        nrmse_sep_plume   = NRMSE_per_channel_masked(truth_unscaled, decoded_unscaled, mask_original, global_stds) 


    import json
    # Full path for saving the file
    output_file = "validation_metrics.json"

    output_path_met = os.path.join(output_path, output_file)

    metrics = {
    "MSE": mse,
    "NRMSE": nrmse,
    "SSIM": test_ssim,
    "EVR": evr,
    "plume NRMSE": nrmse_plume,
    }

    with open(output_path_met, "w") as file:
        json.dump(metrics, file, indent=4)

    ### plot from index 0
    index = 0
    plot_reconstruction_and_error(truth_unscaled[index:index+500], decoded_unscaled[index:index+500], 32, 0, f"/validation_{index}")

    index = 500
    plot_reconstruction_and_error(truth_unscaled[index:index+500], decoded_unscaled[index:index+500], 32, 0, f"/validation_{index}")

    index = 1000
    plot_reconstruction_and_error(truth_unscaled[index:index+500], decoded_unscaled[index:index+500], 32, 0, f"/validation_{index}")

#### TESTING UNSEEN DATA ####
if test_data:
    print('TEST DATA')
    truth = U_tv_scaled
    print('shape of truth', np.shape(truth))
    chunk_size = truth.shape[0]
    timesteps = U_tv_scaled.shape[0]
    total_chunks = timesteps // chunk_size
    print('timestpes', timesteps)
    print('chunks', chunk_size)

    test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                                b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 
    print(test_times[0], test_times[-1])

    metrics_list = []
    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        print(f"Processing chunk {chunk_idx+1}/{total_chunks}: timesteps {start} to {end}")

        U_scaled = truth[start:end]
        print('shape of U_scaled', np.shape(U_scaled))

        decoded = model(U_scaled, a, b)[1].numpy()
        decoded_unscaled = ss_inverse_transform(decoded, scaler)
        truth_unscaled = ss_inverse_transform(truth, scaler)

        # Compute metrics
        print('shape of decoded unscaled', np.shape(decoded_unscaled))
        print('shape of truth_unscaled', np.shape(truth_unscaled))
        test_ssim = compute_ssim_for_4d(truth_unscaled, decoded_unscaled)
        mse = MSE(truth_unscaled, decoded_unscaled)
        nrmse = NRMSE(truth_unscaled, decoded_unscaled)
        evr = EVR_recon(truth_unscaled, decoded_unscaled)
        nrmse_sep = NRMSE_per_channel(truth_unscaled, decoded_unscaled)

        chunk_metrics = {
            "chunk": chunk_idx,
            "MSE": mse,
            "NRMSE": nrmse,
            "SSIM": test_ssim,
            "EVR": evr,
            "NRMSE sep": nrmse_sep,
        }

        # Optional plume NRMSE if variables == 4
        if len(variables) == 4:
            _, _, mask, _ = active_array_calc(truth_unscaled, decoded_unscaled, z)
            print('shape of masked area', np.shape(truth_unscaled[mask]))
            nrmse_plume = NRMSE(truth_unscaled[mask], decoded_unscaled[mask])

            mask_original     = mask[..., 0]
            nrmse_sep_plume   = NRMSE_per_channel_masked(truth_unscaled, decoded_unscaled, mask_original, global_stds) 

            chunk_metrics["plume NRMSE"] = nrmse_plume
            chunk_metrics["plume sep NRMSE"] = nrmse_sep_plume
        else:
            chunk_metrics["plume NRMSE"] = 0
            chunk_metrics["plume sep NRMSE"] = 0

        # Save metrics per chunk
        metrics_list.append(chunk_metrics)

        # Save reconstruction plot and line plot for a sample timestep
        if chunk_idx == 0:  # Example: only for first chunk
            plot_reconstruction_and_error(
                truth_unscaled[:500],
                decoded_unscaled[:500],
                32, 75, x, z, time_vals[:500], variables,
                output_path+f"/test_all_chunk_{chunk_idx}"
            )

            np.save(output_path+'/CAE_decoded.npy', decoded_unscaled)
            np.save(output_path+'/true_data.npy', truth_unscaled)

            fig, ax = plt.subplots(1)
            ax.plot(decoded_unscaled[0, :, 32, 0], label='dec')
            ax.plot(truth_unscaled[0, :, 32, 0], label='true')
            plt.legend()
            fig.savefig(os.path.join(output_path, f'lines_chunk_{chunk_idx}.png'))

    # Save all metrics to JSON after loop
    output_file = "test_metrics_testdata_data_chunked.json"
    output_path_met = os.path.join(output_path, output_file)

    with open(output_path_met, "w") as f:
        json.dump(metrics_list, f, indent=4)

    # Compute averaged metrics across all chunks
    average_metrics = {}
    metric_keys = [k for k in metrics_list[0].keys() if k != "chunk"]

    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        average_metrics[key] = sum(values) / len(values)

    # Save both raw chunk metrics and averages
    final_metrics = {
        "chunk_metrics": metrics_list,
        "average_metrics": average_metrics
    }

    # Save to JSON
    output_file = "test_metrics_testdata_data_chunked.json"
    output_path_met = os.path.join(output_path, output_file)

    with open(output_path_met, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("Saved averaged and per-chunk metrics.")


if all_data:
    chunk_size = 1000
    timesteps = snapshots_load
    total_chunks = timesteps // chunk_size

    metrics_list = []
    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        print(f"Processing chunk {chunk_idx+1}/{total_chunks}: timesteps {start} to {end}")

        U_chunk = U[start:end]
        U_scaled = ss_transform(U_chunk, scaler)
        truth = U_scaled

        decoded = model(U_scaled, a, b)[1].numpy()
        decoded_unscaled = ss_inverse_transform(decoded, scaler)
        truth_unscaled = ss_inverse_transform(truth, scaler)

        # Compute metrics
        print('shape of decoded unscaled', np.shape(decoded_unscaled))
        print('shape of truth_unscaled', np.shape(truth_unscaled))
        test_ssim = compute_ssim_for_4d(truth_unscaled, decoded_unscaled)
        mse = MSE(truth_unscaled, decoded_unscaled)
        nrmse = NRMSE(truth_unscaled, decoded_unscaled)
        evr = EVR_recon(truth_unscaled, decoded_unscaled)
        nrmse_sep = NRMSE_per_channel(truth_unscaled, decoded_unscaled)

        chunk_metrics = {
            "chunk": chunk_idx,
            "MSE": mse,
            "NRMSE": nrmse,
            "SSIM": test_ssim,
            "EVR": evr,
            "NRMSE sep": nrmse_sep,
        }

        # Optional plume NRMSE if variables == 4
        if len(variables) == 4:
            _, _, mask, _ = active_array_calc(truth_unscaled, decoded_unscaled, z)
            print('shape of masked area', np.shape(truth_unscaled[mask]))
            nrmse_plume = NRMSE(truth_unscaled[mask], decoded_unscaled[mask])

            mask_original     = mask[..., 0]
            nrmse_sep_plume   = NRMSE_per_channel_masked(truth_unscaled, decoded_unscaled, mask_original, global_stds) 

            chunk_metrics["plume NRMSE"] = nrmse_plume
            chunk_metrics["plume sep NRMSE"] = nrmse_sep_plume

        # Save metrics per chunk
        metrics_list.append(chunk_metrics)

        # Save reconstruction plot and line plot for a sample timestep
        if chunk_idx == 0:  # Example: only for first chunk
            plot_reconstruction_and_error(
                truth_unscaled[:500],
                decoded_unscaled[:500],
                32, 75, x, z, time_vals[:500], variables,
                output_path+f"/test_all_chunk_{chunk_idx}"
            )

            np.save(output_path+'/CAE_decoded.npy', decoded_unscaled)
            np.save(output_path+'/true_data.npy', truth_unscaled)

            fig, ax = plt.subplots(1)
            ax.plot(decoded_unscaled[0, :, 32, 0], label='dec')
            ax.plot(truth_unscaled[0, :, 32, 0], label='true')
            plt.legend()
            fig.savefig(os.path.join(output_path, f'lines_chunk_{chunk_idx}.png'))

    # Save all metrics to JSON after loop
    output_file = "test_metrics_all_data_chunked.json"
    output_path_met = os.path.join(output_path, output_file)

    with open(output_path_met, "w") as f:
        json.dump(metrics_list, f, indent=4)

    # Compute averaged metrics across all chunks
    average_metrics = {}
    metric_keys = [k for k in metrics_list[0].keys() if k != "chunk"]

    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        average_metrics[key] = sum(values) / len(values)

    # Save both raw chunk metrics and averages
    final_metrics = {
        "chunk_metrics": metrics_list,
        "average_metrics": average_metrics
    }

    # Save to JSON
    output_file = "test_metrics_all_data_chunked.json"
    output_path_met = os.path.join(output_path, output_file)

    with open(output_path_met, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("Saved averaged and per-chunk metrics.")

if visualisation:
    print('VISUALISATION')
    vis_path = output_path + '/visualisation/'

    ### make output_directories ###
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
        print('made visualisation directory')

    #### all data ####
    snapshots_vis = 1000
    U_scaled = ss_transform(U, scaler)
    truth = U_scaled[:snapshots_vis]

    U = U_scaled
    train_leng = 0
    N_pos = 200
    k     = snapshots_vis//N_pos

    encoded = np.empty((snapshots_vis, N_1[1], N_1[2], N_1[3]))
    for i in range(k):
            encoded[i*N_pos:(i+1)*N_pos]= model(U[i*N_pos:(i+1)*N_pos], a, b)[0]

    #encoded = encoded.numpy()
    print('shape of encoded data', np.shape(encoded))

    x_ls = np.arange(0, 8, 1)
    z_ls = np.arange(0, 2, 1)
    

    fig, ax = plt.subplots(5, figsize=(12,12), sharex=True)
    for v in range(5):
        if v == 0:
            c=ax[v].contourf(time_vals[:snapshots_vis], x, truth[:, :, 32, 1].T)
            ax[v].set_ylabel(f"x")
            ax[v].set_xlabel('time')
            fig.colorbar(c, ax=ax[v], label="w")
        else:
            c=ax[v].contourf(time_vals[:snapshots_vis], x_ls, encoded[:, :, 0, v-1].T)
            ax[v].set_ylabel(f"latent width")
            fig.colorbar(c, ax=ax[v], label=f"latent depth={v-1}")
    ax[-1].set_xlabel('time')
    fig.savefig(vis_path+'/hovmoller.png')

    time_vis = 100
    fig, ax = plt.subplots(5, figsize=(12,12))
    for v in range(5):
        if v == 0:
            c=ax[v].contourf(x, z, truth[time_vis, :, :, 1].T)
            ax[v].set_ylabel(f"z")
            ax[v].set_xlabel('x')
            fig.colorbar(c, ax=ax[v], label="w")
        else:
            c=ax[v].contourf(x_ls, z_ls, encoded[time_vis, :, :, v-1].T)
            ax[v].set_ylabel(f"latent height")
            fig.colorbar(c, ax=ax[v], label=f"latent depth={v-1}")
    ax[-1].set_xlabel('latent width')
    fig.savefig(vis_path+f"/snapshot_time{time_vis}.png")
    
if encoded_data_investigation:
    def NRMSE_per_sample(original, reconstructed_batch):
        """
        original: shape (1, H, W, C)
        reconstructed_batch: shape (N, H, W, C)
        
        Returns:
            nrmse_vals: numpy array of shape (N,)
        """
        n_samples = reconstructed_batch.shape[0]
        nrmse_vals = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Expand dims to keep original shape (1, H, W, C) and select one perturbed sample with shape (1, H, W, C)
            recon_sample = np.expand_dims(reconstructed_batch[i], axis=0)
            
            nrmse_vals[i] = NRMSE(original, recon_sample)
            
        return nrmse_vals


    decoder_sens_path = output_path + '/decoder_sens/'

    ### make output_directories ###
    if not os.path.exists(decoder_sens_path):
        os.makedirs(decoder_sens_path)
        print('made decoder directory')

    if encoded_data:
        with h5py.File('Ra2e8/CAE_ESN/LS64/encoded_data'+str(N_latent)+'.h5', 'r') as df:
            U_enc = np.array(df['U_enc'])
    else:
        U = U_scaled
        train_leng = 0
        N_pos = 2000
        k     = (U.shape[0] - train_leng)//N_pos

        U_enc = np.empty((k*N_pos, N_1[1], N_1[2], N_1[3]))
        for i in range(k):
                U_enc[i*N_pos:(i+1)*N_pos]= model(U[i*N_pos:(i+1)*N_pos], a, b)[0]

        with h5py.File(output_path+'/encoded_data'+str(N_latent)+'.h5', 'w') as df:
            df['U_enc'] = U_enc

    upsample   = 1
    data_len   = 11200
    transient  = 0

    dt = dt*upsample
    n_components = N_latent

    dim      = N_latent #N_latent
    act      = 'tanh'
    U        = np.array(U_enc[transient:transient+data_len:upsample])
    shape    = U.shape
    print(shape)

    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))

    time_step = 100
    z0 = U[time_step:time_step+1]
    z0 = z0.squeeze(0)

    epsilon = 0.05
    num_samples = 20

    epsilons = epsilon * np.random.randn(num_samples, z0.shape[0], z0.shape[1], z0.shape[2])
    z_perturbed = z0 + epsilons  # shape: (100, latent_dim)

    z0_flatten = z0.flatten()
    z_perturbed_flatten = z_perturbed.reshape(num_samples, -1)
    NRMSE_latent_space = np.sqrt(np.mean((z0_flatten-z_perturbed_flatten)**2, axis=1))/sigma_ph
        
    decoded = Decoder(z_perturbed, b).numpy()
    x_hat_perturbed = ss_inverse_transform(decoded, scaler)

    decoded = Decoder(z0[None, :], b).numpy()
    x_hat = ss_inverse_transform(decoded, scaler)

    print('shape of perts', np.shape(x_hat_perturbed))
    print('shape of true', np.shape(x_hat))

    errors = np.sqrt(np.mean((x_hat_perturbed - x_hat)**2, axis=(1, 2, 3)))    
    
    NRMSE_recons = NRMSE_per_sample(x_hat, x_hat_perturbed)
    epsilon_norms = np.linalg.norm(epsilons.reshape(epsilons.shape[0], -1), axis=1)

    print('shape of errors', np.shape(errors))
    print('shape of epsilon norms', np.shape(epsilon_norms))

    fig, ax = plt.subplots(1, figsize=(8,6), tight_layout=True)
    ax.scatter(epsilon_norms, errors, alpha=0.7)
    ax.set_xlabel(r"Latent petrubation norm $\epsilon$")
    ax.set_ylabel("RMSE")
    ax.grid()
    fig.savefig(decoder_sens_path+'/Decoder_sensitivity.png')

    fig, ax = plt.subplots(1, figsize=(8,6), tight_layout=True)
    ax.scatter(NRMSE_latent_space, NRMSE_recons, alpha=0.7)
    ax.set_xlabel(r"NRMSE betweeen latent spaces$")
    ax.set_ylabel("NRMSE in reconstruction")
    ax.grid()
    fig.savefig(decoder_sens_path+'/Decoder_sensitivity_NRMSE.png')

    i=1
    fig, ax =plt.subplots(3, figsize=(12,9), tight_layout=True)
    original = x_hat
    reconstruction = x_hat_perturbed
    t_value= 0
    residual = x_hat[t_value,:,:,i] - x_hat_perturbed[t_value,:,:,i]
    vmax_res = np.max(np.abs(residual))  # Get maximum absolute value
    vmin_res = -vmax_res
    c1 = ax[0].pcolormesh(x, z, original[t_value,:,:,i].T)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('reconstruction')
    c2 = ax[1].pcolormesh(x, z, reconstruction[t_value,:,:,i].T)
    fig.colorbar(c2, ax=ax[1])
    ax[1].set_title('perturbed reconstruction')
    c3 = ax[2].pcolormesh(x, z, residual.T, cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
    fig.colorbar(c3, ax=ax[2])
    ax[2].set_title('error', fontsize=18)

    fig.savefig(decoder_sens_path+'/reconstruction.png')