"""
python script for convolutional autoencoder.

Usage: CAE.py [--input_path=<input_path> --output_path=<output_path>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
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

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

#### make directory ####
from datetime import datetime 
now = datetime.now()
print(now)
# Format the date and time as YYYY-MM-DD_HH-MM-SS for a file name
formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
# Example of using the formatted datetime in a file name
output_new = f"validation_Ra2e6_{formatted_datetime}"
output_path = os.path.join(output_path, output_new)
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')


#### larger data 5000-30000 hf ####
total_num_snapshots = 2000
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
variables = num_variables = 4
variable_names = ['q', 'w', 'u', 'b']

with h5py.File(input_path+'/data_all.h5', 'r') as df:
    time_vals = np.array(df['total_time'][:total_num_snapshots])
    q = np.array(df['q_vertical'][:total_num_snapshots])
    w = np.array(df['w_vertical'][:total_num_snapshots])
    u = np.array(df['u_vertical'][:total_num_snapshots])
    b = np.array(df['b_vertical'][:total_num_snapshots])

    q = np.squeeze(q, axis=2)
    w = np.squeeze(w, axis=2)
    u = np.squeeze(u, axis=2)
    b = np.squeeze(b, axis=2)

    print(np.shape(q))

print('shape of time_vals', np.shape(time_vals))

# Reshape the arrays into column vectors
q_array = q.reshape(len(time_vals), len(x), len(z), 1)
w_array = w.reshape(len(time_vals), len(x), len(z), 1)
u_array = u.reshape(len(time_vals), len(x), len(z), 1)
b_array = b.reshape(len(time_vals), len(x), len(z), 1)

del q
del w
del u 
del b

data_all  = np.concatenate((q_array, w_array, u_array, b_array), axis=-1)

# Print the shape of the combined array
print('shape of all data and scaled data:', data_all.shape)

U = data_all
dt = time_vals[1]-time_vals[0]
print('dt:', dt)

del w_array
del q_array
del u_array
del b_array

#### Hyperparmas #####
# WandB hyperparams settings
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "Val Loss"},
    "parameters": {
        "lat_dep":{"values": [1,2]},
        "n_epochs": {"values": [151]},
        "l_rate": {"values": [0.001, 0.002]},
        "b_size": {"values": [32]},
        "lrate_mult": {"values": [0.8]},
        "N_lr": {"values": [150]},
        "N_layers": {"values": [4]},
        "N_parallel": {"values": [1]},
        "kernel_choice": {"values": [0, 1, 2]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="autoencoderRa2e6")

job = 0

def main():

    run = wandb.init()
    global job
    job += 1

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

    def NRMSE(predictions, true_values):
        "input: predictions, true_values as (time, variables)"
        variables = predictions.shape[1]
        mse = np.mean((true_values-predictions) ** 2, axis = 1)
        #print(np.shape(mse))
        rmse = np.sqrt(mse)

        std_squared = np.std(true_values, axis = 0) **2
        print(np.shape(std_squared))
        sum_std = np.mean(std_squared)
        print(sum_std)
        sqrt_std = np.sqrt(sum_std)

        nrmse = rmse/sqrt_std
        #print(np.shape(nrmse))

        return nrmse

    def NRMSE2(original_data, reconstructed_data):
        "flattened arrays (time, spatial)"
        rmse = np.sqrt(mean_squared_error(original_data, reconstructed_data))
        
        variance = np.var(original_data)
        std_dev  = np.sqrt(variance)
        
        nrmse = 1 - (rmse/std_dev)
        
        return nrmse

    def EVR(original_data, reconstructed_data):
        "flattened arrays (time, spatial)"
        numerator = np.sum((original_data - reconstructed_data) **2)
        denominator = np.sum(original_data ** 2)
        evr_reconstruction = 1 - (numerator/denominator)

        return evr_reconstruction

    from skimage.metrics import structural_similarity as ssim
    def compute_ssim_for_4d(original, decoded):
        """
        Compute the average SSIM across all timesteps and channels for 4D arrays.
        """

        # Initialize SSIM accumulator
        total_ssim = 0
        timesteps = original.shape[0]
        channels = original.shape[-1]

        for t in range(timesteps):
            for c in range(channels):
                # Extract the 2D slice for each timestep and channel
                orig_slice = original[t, :, :, c]
                dec_slice = decoded[t, :, :, c].numpy()
                
                # Compute SSIM for the current slice
                batch_ssim = ssim(orig_slice, dec_slice, data_range=orig_slice.max() - orig_slice.min(), win_size=3)
                total_ssim += batch_ssim

        # Compute the average SSIM across all timesteps and channels
        avg_ssim = total_ssim / (timesteps * channels)
        return avg_ssim

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
        mask_expanded       = np.expand_dims(mask, axis=-1)  # Shape: (256, 64, 1)
        mask_expanded       = np.repeat(mask_expanded, original_data.shape[-1], axis=-1)  # Shape: (256, 64, 4)
        mask_expanded_recon = np.expand_dims(mask_reconstructed, axis=-1)  # Shape: (256, 64, 1)
        mask_expanded_recon = np.repeat(mask_expanded_recon, reconstructed_data.shape[-1], axis=-1)  # Shape: (256, 64, 4)
    
        return active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon

    def plume_error(true_data, reconstructed_data):
        beta  = 1.201
        alpha = 3.0
        print(np.shape(true_data), flush=True)
        print(np.shape(reconstructed_data), flush=True)
        print(np.shape(z))
        
        T            = true_data[:,:,:,3] - beta*z
        T_recon      = reconstructed_data[:,:,:,3] - beta*z
        q_s          = np.exp(alpha*T)
        q_s_recon    = np.exp(alpha*T_recon)
        rh           = true_data[:,:,:,0]/q_s
        rh_recon     = reconstructed_data[:,:,:,0]/q_s_recon
        mean_b       = np.mean(true_data[:,:,:,3], axis=1, keepdims=True)
        mean_b_recon = np.mean(reconstructed_data[:,:,:,3], axis=1, keepdims=True)
        b_anom       = true_data[:,:,:,3] - mean_b
        b_anom_recon = reconstructed_data[:,:,:,3] - mean_b_recon
        w            = true_data[:,:,:,1]
        w_recon      = reconstructed_data[:,:,:,1]

        mask = (rh[:, :, :] >= 1) & (w[:, :, :] > 0) & (b_anom[:, :, :] > 0)
        mask_recon = (rh_recon[:, :, :] >= 1) & (w_recon[:, :, :] > 0) & (b_anom_recon[:, :, :] > 0)
        active_array = np.zeros((true_data.shape[0], len(x), len(z)))
        active_array[mask] = 1
        active_array_recon = np.zeros((true_data.shape[0], len(x), len(z)))
        active_array_recon[mask_recon] = 1

        accuracy = np.mean(active_array == active_array_recon)
        MAE = np.zeros((num_variables))
        print(np.shape(MAE))
        for v in range(variables):
            MAE[v] = np.mean(np.abs(true_data[:,:,:,v][mask] - reconstructed_data[:,:,:,v][mask]))
        MAE_all = np.sum(MAE)/variables
        print(accuracy, MAE_all)
        return accuracy, MAE_all

    #### load in data ###
    b_size      = wandb.config.b_size   #batch_size
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
    U_train     = split_data(U_tt, b_size, n_batches).astype('float32') #to be used for randomly shuffled batches
    # validation data
    U_vv        = np.array(U[b_size*n_batches*skip:
                             b_size*n_batches*skip+b_size*val_batches*skip])
    U_val       = split_data(U_vv, b_size, val_batches).astype('float32')
    # test data
    U_vt        = np.array(U[b_size*n_batches*skip+b_size*val_batches*skip:
                             b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip])
    U_test       = split_data(U_vt, b_size, test_batches).astype('float32')
    del U_vv, U_tt, U_vt

    # Placeholder for standardized data
    U_train_scaled = np.zeros_like(U_train)
    U_val_scaled = np.zeros_like(U_val)
    U_test_scaled = np.zeros_like(U_test)
    scalers = [StandardScaler() for _ in range(variables)]

    # Apply StandardScaler separately to each channel
    for v in range(variables):
        # Reshape training data for the current variable
        reshaped_train_channel = U_train[:, :, :, :, v].reshape(-1, U_train.shape[2] * U_train.shape[3])

        # Fit the scaler on the training data for the current variable
        scaler = scalers[v]
        scaler.fit(reshaped_train_channel)

        # Standardize the training data
        standardized_train_channel = scaler.transform(reshaped_train_channel)

        # Reshape the standardized data back to the original shape (batches, batch_size, x, z)
        U_train_scaled[:, :, :, :, v] = standardized_train_channel.reshape(U_train.shape[0], U_train.shape[1], U_train.shape[2], U_train.shape[3])
        
        # Standardize the validation data using the same scaler
        reshaped_val_channel = U_val[:, :, :, :, v].reshape(-1, U_val.shape[2] * U_val.shape[3])
        standardized_val_channel = scaler.transform(reshaped_val_channel)
        U_val_scaled[:, :, :, :, v] = standardized_val_channel.reshape(U_val.shape[0], U_val.shape[1], U_val.shape[2], U_val.shape[3])

        # Standardize the test data using the same scaler
        reshaped_test_channel = U_test[:, :, :, :, v].reshape(-1, U_test.shape[2] * U_test.shape[3])
        standardized_test_channel = scaler.transform(reshaped_test_channel)
        U_test_scaled[:, :, :, :, v] = standardized_test_channel.reshape(U_test.shape[0], U_test.shape[1], U_test.shape[2], U_test.shape[3])



    # define the model
    # we do not have pooling and upsampling, instead we use stride=2

    lat_dep       = wandb.config.lat_dep       #latent space depth
    kernel_choice = wandb.config.kernel_choice
    n_fil         = [6,12,24,lat_dep]          #number of filters ecnoder
    n_dec         = [24,12,6,3]                #number of filters decoder
    N_parallel    = wandb.config.N_parallel    #number of parallel CNNs for multiscale
    ker_size      = [(3,3), (5,5), (7,7)]      #kernel sizes
    if N_parallel == 1:
        ker_size  = [ker_size[kernel_choice]]
    N_layers      = wandb.config.N_layers    #number of layers in every CNN
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
    output = enc_mods[-1](U_train_scaled[0])
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
    enc0, dec0 = model(U_train_scaled[0], enc_mods, dec_mods)
    print('latent   space size:', N_latent)
    print('physical space size:', U[0].flatten().shape)
    print('')
    for j in range(N_parallel):
        enc_mods[j].summary()
    for j in range(N_parallel):
        dec_mods[j].summary()

    rng = np.random.default_rng() #random generator for later shufflinh

    Loss_Mse    = tf.keras.losses.MeanSquaredError()

    n_epochs    = wandb.config.n_epochs #number of epochs

    #define optimizer and initial learning rate
    optimizer  = tf.keras.optimizers.Adam(amsgrad=True) #amsgrad True for better convergence
    #print(dir(optimizer))

    l_rate     = wandb.config.l_rate
    optimizer.learning_rate = l_rate

    lrate_update = True #flag for l_rate updating
    lrate_mult   = wandb.config.lrate_mult #decrease by this factore the l_rate
    N_lr         = wandb.config.N_lr #20 #number of epochs before which the l_rate is not updated

    # quantities to check and store the training and validation loss and the training goes on
    old_loss      = np.zeros(n_epochs) #needed to evaluate training loss convergence to update l_rate
    tloss_plot    = np.zeros(n_epochs) #training loss
    vloss_plot    = np.zeros(n_epochs) #validation loss
    nrmse_plot    = np.zeros(n_epochs)
    tssim_plot    = np.zeros(n_epochs) #training loss
    vssim_plot    = np.zeros(n_epochs) #validation loss
    tacc_plot     = np.zeros(n_epochs)
    vacc_plot     = np.zeros(n_epochs)
    tMAE_plot     = np.zeros(n_epochs)
    vMAE_plot     = np.zeros(n_epochs)
    tEVR_plot     = np.zeros(n_epochs)
    vEVR_plot     = np.zeros(n_epochs)
    tNRMSE2_plot  = np.zeros(n_epochs)
    vNRMSE2_plot  = np.zeros(n_epochs)
    tMSEpl_plot   = np.zeros(n_epochs)
    vMSEpl_plot   = np.zeros(n_epochs)
    tacc2_plot    = np.zeros(n_epochs)
    vacc2_plot    = np.zeros(n_epochs)
    old_loss[0]  = 1e6 #initial value has to be high
    N_check      = 5 #10   #each N_check epochs we check convergence and validation loss
    patience     = 100 #if the val_loss has not gone down in the last patience epochs, early stop
    last_save    = patience

    t            = 1 # initial (not important value) to monitor the time of the training

    #### make directory to save ###
    job_name = 'job{0:}_batch_size{1:}_learning_rate{2:}_num_epochs{3:}_lrate_mult{4:}_N_lr{5:}_n_parallel{6:}_lat_dep{7:}_N_layers{8:}'.format(job, b_size, l_rate, n_epochs, lrate_mult, N_lr, N_parallel, lat_dep, N_layers)
    job_path = os.path.join(output_path, job_name)
    os.makedirs(job_path, exist_ok=True)

    images_dir = os.path.join(job_path, 'Images')
    # Create the directory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)

    models_dir = os.path.join(job_path, 'saved_models')
    print(models_dir)
    # Make sure the directory exists
    os.makedirs(models_dir, exist_ok=True)

    import json

    # Full path for saving the file
    output_file = "hyperparameters.json"

    output_path_hyp = os.path.join(job_path, output_file)

    hyperparameters = {
    "batch_size": b_size,
    "learning_rate": l_rate,
    "num_epochs": n_epochs,
    "optimizer": "adam",
    "lrate_mult": lrate_mult,
    "N_lr": N_lr,
    "n_parallel": N_parallel,
    "N_check:": N_check,
    "lat_dep": lat_dep, 
    "kernel_choice": kernel_choice,
    }

    with open(output_path_hyp, "w") as file:
        json.dump(hyperparameters, file, indent=4)

    for epoch in range(n_epochs):
        print('running epoch:', epoch)
        if epoch - last_save > patience: break #early stop
        
        if epoch > patience:
            if ssim_0 < 0.5: break #early stop

        #Perform gradient descent for all the batches every epoch
        loss_0         = 0
        ssim_0         = 0
        accuracy_0     = 0
        MAE_plume_0    = 0
        evr_0          = 0
        nrmse2_0       = 0
        accuracy_new_0 = 0
        mse_plume_0    = 0 
        rng.shuffle(U_train_scaled, axis=0) #shuffle batches
        for j in range(n_batches):
            loss, decoded    = train_step(U_train_scaled[j], enc_mods, dec_mods)
            loss_0          += loss

            # Compute SSIM
            original = U_train_scaled[j]  # Validation input
            #print(np.shape(original), np.shape(decoded))
            #print(f"Type of original: {type(original)}")
            #print(f"Type of decoded: {type(decoded)}")
            batch_ssim = compute_ssim_for_4d(original, decoded)
            ssim_0 += batch_ssim

            # compute plume error
            accuracy, MAE_plume  = plume_error(original, decoded)
            accuracy_0          += accuracy
            MAE_plume_0         += MAE_plume

            # new metric errors
            active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon = active_array_calc(original, decoded, z)
            accuracy_new        = np.mean(active_array == active_array_reconstructed)
            accuracy_new_0     += accuracy_new
            mse_plume           = mean_squared_error(original[mask_expanded], decoded[mask_expanded])
            mse_plume_0        += mse_plume
            
            decoded             = decoded.numpy()            
            decoded_reshaped    = decoded.reshape(decoded.shape[0], len(x) * len(z) * variables)
            original_reshaped   = original.reshape(original.shape[0], len(x) * len(z) * variables)
            evr_value           = EVR(original_reshaped, decoded_reshaped)
            nrmse2_value        = NRMSE2(original_reshaped, decoded_reshaped)
            nrmse2_0           += nrmse2_value

        #save train loss
        tloss_plot[epoch]        = loss_0.numpy()/n_batches
        tssim_plot[epoch]        = ssim_0/n_batches
        tacc_plot[epoch]         = accuracy_0/n_batches
        tMAE_plot[epoch]         = MAE_plume_0/n_batches
        tEVR_plot[epoch]         = evr_0/n_batches
        tNRMSE2_plot[epoch]      = nrmse2_0/n_batches
        tacc2_plot[epoch]        = accuracy_new_0/n_batches
        tMSEpl_plot[epoch]       = mse_plume_0/n_batches

        # every N epochs checks the convergence of the training loss and val loss
        if (epoch%N_check==0):
            print('checking convergence')
            #Compute Validation Loss
            loss_val        = 0
            ssim_val        = 0
            accuracy_val    = 0
            MAE_plume_val   = 0
            evr_val         = 0
            nrmse2_val      = 0
            accuracy_new_val= 0
            mse_plume_val   = 0 
            for j in range(val_batches):
                loss, decoded       = train_step(U_val_scaled[j], enc_mods, dec_mods,train=False)
                loss_val           += loss

                # Compute SSIM
                original = U_val_scaled[j]  # Validation input
                batch_ssim = compute_ssim_for_4d(original, decoded)
                ssim_val += batch_ssim

                # compute plume error
                accuracy, MAE_plume    = plume_error(original, decoded)
                accuracy_val          += accuracy
                MAE_plume_val         += MAE_plume

                # new metric errors
                active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon = active_array_calc(original, decoded, z)
                accuracy_new          = np.mean(active_array == active_array_reconstructed)
                accuracy_new_val     += accuracy_new
                mse_plume             = mean_squared_error(original[mask_expanded], decoded[mask_expanded])
                mse_plume_val        += mse_plume

                decoded            = decoded.numpy()              
                decoded_reshaped   = decoded.reshape(decoded.shape[0], len(x) * len(z) * variables)
                original_reshaped  = original.reshape(original.shape[0], len(x) * len(z) * variables)
                evr_value          = EVR(original_reshaped, decoded_reshaped)
                evr_val           += evr_value
                nrmse2_value       = NRMSE2(original_reshaped, decoded_reshaped)
                nrmse2_val        += nrmse2_value

            #save validation loss
            vloss_epoch        = loss_val.numpy()/val_batches
            vloss_plot[epoch]  = vloss_epoch 
            vssim_epoch        = ssim_val/val_batches
            vssim_plot[epoch]  = vssim_epoch
            vacc_epoch         = accuracy_val/val_batches
            vacc_plot[epoch]   = vacc_epoch
            vMAE_epoch         = MAE_plume_val/val_batches
            vMAE_plot[epoch]   = vMAE_epoch
            vEVR_plot[epoch]   = evr_val/val_batches
            vNRMSE2_plot[epoch]= nrmse2_val/val_batches
            tacc2_plot[epoch]  = accuracy_new_val/val_batches
            tMSEpl_plot[epoch] = mse_plume_val/val_batches

            # Decreases the learning rate if the training loss is not going down with respect to
            # N_lr epochs before
            if epoch > N_lr and lrate_update:
                print('Reverting to best model and optimizer, and reducing learning rate')
                print('epoch:', epoch, 'N_lr:', N_lr)
                #check if the training loss is smaller than the average training loss N_lr epochs ago
                tt_loss   = np.mean(tloss_plot[epoch-N_lr:epoch])
                if tt_loss > old_loss[epoch-N_lr]:
                    #if it is larger, load optimal val loss weights and decrease learning rate
                    print('LOADING MINIMUM')
                    for i in range(N_parallel):
                        enc_mods[i].load_weights(models_dir + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
                        dec_mods[i].load_weights(models_dir + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')

                    optimizer.learning_rate = optimizer.learning_rate*lrate_mult
                    optimizer.set_weights(min_weights)
                    print(f"Learning rate reduced to {optimizer.learning_rate.numpy()}")
                    old_loss[epoch-N_lr:epoch] = 1e6 #so that l_rate is not changed for N_lr steps

            #store current loss
            old_loss[epoch] = tloss_plot[epoch].copy()

            #save best model (the one with minimum validation loss)
            if epoch > 1 and vloss_plot[epoch] < \
                             (vloss_plot[:epoch-1][np.nonzero(vloss_plot[:epoch-1])]).min():

                #save model
                min_val_loss = vloss_plot[epoch]
                print('Saving Model..')
                for i in range(N_parallel):
                    enc_mods[i].save(models_dir + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5')
                    dec_mods[i].save(models_dir + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5')
                    enc_mods[i].save_weights(models_dir + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
                    dec_mods[i].save_weights(models_dir + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
                
                #saving optimizer parameters
                min_weights = optimizer.get_weights()
                hf = h5py.File(models_dir + '/opt_weights.h5','w')
                for i in range(len(min_weights)):
                    hf.create_dataset('weights_'+str(i),data=min_weights[i])
                hf.create_dataset('length', data=i)
                print(type(optimizer.learning_rate))
                l_rate_value = optimizer.learning_rate.numpy()
                hf.create_dataset('l_rate', data=l_rate_value)  
                hf.close()
                last_save = epoch #store the last time the val loss has decreased for early stop
                print(f"Model and optimizer saved at epoch {epoch} with validation loss: {min_val_loss}")

            # Print loss values and training time (per epoch)
            print('Epoch', epoch, '; Train_Loss', tloss_plot[epoch],
                  '; Val_Loss', vloss_plot[epoch],  '; Ratio', (vloss_plot[epoch])/(tloss_plot[epoch]))
            print('Time per epoch', (time.time()-t)/N_check)

            wandb.log(
                     {"Train Loss": tloss_plot[epoch],
                      "Val Loss": vloss_plot[epoch],
                      "epoch": epoch,
                      "SSIM train": tssim_plot[epoch],
                      "SSIM val": vssim_plot[epoch],
                      "Plume Acc": vacc_plot[epoch],
                      "Plume MAE": vMAE_plot[epoch],
                      "Plume MSE val": vMSEpl_plot[epoch],
                      "Plume Acc2": vacc2_plot[epoch],
                      "NRMSE2_val": vNRMSE2_plot[epoch],
                      "NRMSE2_train": tNRMSE2_plot[epoch],
                      "Plume MSE test": tMSEpl_plot[epoch]
                      })

            print('')

            t = time.time()

        if (epoch%N_check==0) and epoch != 0:
            #plot convergence of training and validation loss (to visualise convergence during training)
            fig, ax =plt.subplots(1, figsize=(6,4), tight_layout=True)
            ax.set_title('MSE convergence')
            ax.set_yscale('log')
            ax.grid(True, axis="both", which='both', ls="-", alpha=0.3)
            ax.plot(tloss_plot[np.nonzero(tloss_plot)], 'y', label='Train loss')
            ax.plot(np.arange(np.nonzero(vloss_plot)[0].shape[0])*N_check,
                        vloss_plot[np.nonzero(vloss_plot)], label='Val loss')
            ax.set_xlabel('epochs')
            ax.legend()
            fig.savefig(images_dir +'/convergence.png')

            fig, ax =plt.subplots(1, figsize=(6,4), tight_layout=True)
            ax.set_title('SSIM')
            ax.grid(True, axis="both", which='both', ls="-", alpha=0.3)
            ax.plot(tssim_plot, 'y', label='Train SSIM')
            ax.plot(np.arange(0,n_epochs,N_check),vssim_plot[::N_check], label='Val SSIM')
            ax.set_xlabel('epochs')
            ax.legend()
            fig.savefig(images_dir +'/SSIM.png')

            fig, ax =plt.subplots(1, figsize=(6,4), tight_layout=True)
            ax.set_title('Plume Accuracy')
            ax.grid(True, axis="both", which='both', ls="-", alpha=0.3)
            ax.plot(tacc_plot, 'y', label='Train Accuracy')
            ax.plot(np.arange(0,n_epochs,N_check),vacc_plot[::N_check], label='Val Accuracy')
            ax.set_xlabel('epochs')
            ax.legend()
            fig.savefig(images_dir +'/Accuracy.png')

            fig, ax =plt.subplots(1, figsize=(6,4), tight_layout=True)
            ax.set_title('Plume MAE')
            ax.grid(True, axis="both", which='both', ls="-", alpha=0.3)
            ax.plot(tMAE_plot, 'y', label='Train Plume MAE')
            ax.plot(np.arange(0,n_epochs,N_check),vMAE_plot[::N_check], label='Val Plume MAE')
            ax.set_xlabel('epochs')
            ax.legend()
            fig.savefig(images_dir +'/MAE_plume.png')


    print('finished job')

wandb.agent(sweep_id=sweep_id, function=main)

