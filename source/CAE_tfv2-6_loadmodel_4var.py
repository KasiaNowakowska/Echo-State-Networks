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
from matplotlib.colors import TwoSlopeNorm
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

import wandb
#wandb.login()

input_path = args['--input_path']
output_path = args['--output_path']
model_path = args['--model_path']
hyperparam_file = args['--hyperparam_file']

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

#### larger data 5000-30000 hf ####
total_num_snapshots = 10000
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
variables = num_variables = 4
variable_names = ['q', 'w', 'u', 'b']

with h5py.File(input_path+'/data_4var_5000_30000.h5', 'r') as df:
    time_vals = np.array(df['total_time_all'][:total_num_snapshots])
    q = np.array(df['q_all'][:total_num_snapshots])
    w = np.array(df['w_all'][:total_num_snapshots])
    u = np.array(df['u_all'][:total_num_snapshots])
    b = np.array(df['b_all'][:total_num_snapshots])

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

def main():

    #run = wandb.init()

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

    def MSE(predictions, true_values):
        "input: predictions, true_values as (time, variables)"
        variables = predictions.shape[1]
        mse = np.mean((true_values-predictions) ** 2, axis = 1)
        return mse

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
        
    def MAE(predictions, true_values):
        mae = np.mean(np.abs(true_values - predictions), axis=1)
        return mae

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
                dec_slice = decoded[t, :, :, c]
                
                # Compute SSIM for the current slice
                batch_ssim = ssim(orig_slice, dec_slice, data_range=orig_slice.max() - orig_slice.min(), win_size=3)
                total_ssim += batch_ssim

        # Compute the average SSIM across all timesteps and channels
        avg_ssim = total_ssim / (timesteps * channels)
        return avg_ssim


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
    global b_size   #batch_size
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
    U_test        = np.array(U[b_size*n_batches*skip+b_size*val_batches*skip:
                             b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip])

    # Placeholder for standardized data
    U_tt_scaled = np.zeros_like(U_tt)
    U_vv_scaled = np.zeros_like(U_vv)
    U_test_scaled = np.zeros_like(U_test)
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
        reshaped_test_channel = U_test[:, :, :, v].reshape(-1, U_test.shape[1] * U_test.shape[2])
        standardized_test_channel = scaler.transform(reshaped_test_channel)
        U_test_scaled[ :, :, :, v] = standardized_test_channel.reshape(U_test.shape[0], U_test.shape[1], U_test.shape[2])

    test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                             b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 

    for v in range(variables):
        fig, ax = plt.subplots(1)
        c=ax.pcolormesh(test_times, x, U_test_scaled[:,:,32,v].T)
        fig.colorbar(c, ax=ax)        
        fig.savefig(output_path+'/test_scaling%i.png' % v)

        fig, ax = plt.subplots(1)
        c=ax.pcolormesh(U_tt_scaled[:,:,32,v].T)
        fig.colorbar(c, ax=ax)        
        fig.savefig(output_path+'/train_scaling%i.png' % v)

        fig, ax = plt.subplots(1)
        c=ax.pcolormesh(test_times, x, U_test[:,:,32,v].T)
        fig.colorbar(c, ax=ax)        
        fig.savefig(output_path+'/test_unscaled%i.png' % v)

    U_train_scaled     = split_data(U_tt_scaled, b_size, n_batches).astype('float32') #to be used for randomly shuffled batches
    U_val_scaled       = split_data(U_vv_scaled, b_size, val_batches).astype('float32')
    del U_vv, U_tt
    
    '''
    # define the model
    # we do not have pooling and upsampling, instead we use stride=2

    global lat_dep                 #latent space depth
    global N_parallel                       #number of parallel CNNs for multiscale
    global kernel_choice
    ker_size      = [(3,3), (5,5), (7,7)]      #kernel sizes
    if N_parallel == 1:
        ker_size  = [ker_size[kernel_choice]]
    global N_layers    #number of layers in every CNN
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
        dec_mods[j].add(tf.keras.layers.CenterCrop(p_crop[0] + 2*p_fin[j], p_crop[1]+ 2*p_fin[j],
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
    truth = U_vt
    decoded = model(truth,a,b)[1].numpy()
    print(np.shape(decoded), np.shape(truth))
    
    test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                             b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 
    print(test_times[0], test_times[-1])
    
    # compute plume error
    accuracy, MAE_plume  = plume_error(truth, decoded)
    #SSIM
    test_ssim = compute_ssim_for_4d(truth, decoded)
    
    truth_reshaped = truth.reshape(test_batches*b_size, len(x) * len(z) * variables)
    decoded_reshaped = decoded.reshape(test_batches*b_size, len(x) * len(z) * variables) 
    print('shape of time:', np.shape(test_times))
    print('shape of inputs:', np.shape(truth))
    print('shape of reconstructions:', np.shape(decoded))

    #MSE
    mse = MSE(decoded_reshaped, truth_reshaped)
    mse_avg = np.mean(mse)
    
    #MAE
    mae = MAE(decoded_reshaped, truth_reshaped)
    mae_avg = np.mean(mae)
    
    #NRMSE
    nrmse = NRMSE(decoded_reshaped, truth_reshaped)
    avg_nrmse = np.mean(nrmse)
    sq_avg_nrmse = avg_nrmse**2
    Energy = 1-sq_avg_nrmse
    curve = 1-Energy
    
    print("nrmse:", avg_nrmse)
    print("mse:", mse_avg)
    print("mae:", mae_avg)
    print("accuracy:", accuracy)
    print("mae_plume:", MAE_plume)
    print("test_ssim:", test_ssim)
    print("Energy:", Energy)
    print("Curve:", curve)
    
    import json

    # Full path for saving the file
    output_file = "metrics.json"

    output_path_met = os.path.join(output_path, output_file)

    metrics = {
    "mse": mse_avg,
    "nrmse": avg_nrmse,
    "mae": mae_avg,
    "Energy": Energy,
    "Curve": curve,
    "accuracy": accuracy,
    "MAE_plume": MAE_plume,
    "test_ssim": test_ssim,
    }

    with open(output_path_met, "w") as file:
        json.dump(metrics, file, indent=4)

    ##### across time ####
    fig, ax = plt.subplots(3,variables, figsize=(12,6), tight_layout=True, sharex=True)
    z_index = 32
    error = truth[:, :, z_index, :] - decoded[:, :, z_index, :]
    for v in range(variables):
        minm = min(np.min(truth[:, :, z_index, v]), np.min(decoded[:, :, z_index, v]))
        maxm = max(np.max(truth[:, :, z_index, v]), np.max(decoded[:, :, z_index, v]))
        print(minm, maxm)
        c1 = ax[0,v].pcolormesh(test_times[:], x, truth[:, :, z_index, v].T, vmin=minm, vmax=maxm)
        c2 = ax[1,v].pcolormesh(test_times[:], x, decoded[:, :, z_index, v].T, vmin=minm, vmax=maxm)
        c3 = ax[2,v].pcolormesh(test_times[:], x, error[:, :, v].T, cmap='RdBu', norm=TwoSlopeNorm(vmin=-np.max(np.abs(error)), vcenter=0, vmax=np.max(np.abs(error))))
        ax[0,v].set_title(variable_names[v])

        fig.colorbar(c1, label='True')
        fig.colorbar(c2, label='reconstructed')
        fig.colorbar(c3, label='error')

    for i in range(2):
        ax[i,0].set_ylabel('x')
    for i in range(variables):
        ax[0,i].set_xlabel('time')
    fig.savefig(output_path+'/time.png')

    ##### across x,z ####
    #plot n snapshots and their reconstruction in the test set.
    n       = 4

    start   = b_size*n_batches*skip+b_size*val_batches*skip  #b_size*n_batches*skip+b_size*val_batches*skip #start after validation set

    skips = 50
    for i in range(n):
        fig, ax = plt.subplots(3,variables, figsize=(12,6), tight_layout=True)
        index = 0 + skips*i
        print(index)
        u     = truth[index]
        u_dec = decoded[index]
        for v in range(variables):
            vmax = u[:,:,v].max()
            vmin = u[:,:,v].min()
            minm = min(np.min(u[:, :, v]), np.min(u_dec[:, :, v]))
            maxm = max(np.max(u[:, :, v]), np.max(u_dec[:, :, v]))
            NAE = np.abs(u - u_dec)/(vmax-vmin)
            print(minm, maxm)
            print('NMAE:', NAE[:,:, v].mean())
            c1 = ax[0,v].pcolormesh(x, z, u[:, :, v].T, vmin=minm, vmax=maxm)
            c2 = ax[1,v].pcolormesh(x, z, u_dec[:, :, v].T, vmin=minm, vmax=maxm)
            c3 = ax[2,v].pcolormesh(x, z, NAE[:, :, v].T, cmap='RdBu')

            fig.colorbar(c1, label='True')
            fig.colorbar(c2, label='reconstructed')
            fig.colorbar(c3, label='absolute error')

            fig.suptitle('time:%i' %test_times[index])
        fig.savefig(output_path+'/xz%i.png' % test_times[index])

    fig, ax = plt.subplots(3,variables, figsize=(12,9), tight_layout=True)
    index = 30
    print(index)
    u     = truth[index]
    u_dec = decoded[index]
    for v in range(variables):
        vmax = u[:,:,v].max()
        vmin = u[:,:,v].min()
        minm = min(np.min(u[:, :, v]), np.min(u_dec[:, :, v]))
        maxm = max(np.max(u[:, :, v]), np.max(u_dec[:, :, v]))
        NAE = np.abs(u - u_dec)/(vmax-vmin)
        print(minm, maxm)
        print('NMAE:', NAE[:,:, v].mean())
        c1 = ax[0,v].pcolormesh(x, z, u[:, :, v].T, vmin=minm, vmax=maxm)
        c2 = ax[1,v].pcolormesh(x, z, u_dec[:, :, v].T, vmin=minm, vmax=maxm)
        c3 = ax[2,v].pcolormesh(x, z, NAE[:, :, v].T, cmap='RdBu')

        fig.colorbar(c1, label='True')
        fig.colorbar(c2, label='reconstructed')
        fig.colorbar(c3, label='absolute error')

        fig.suptitle('time:%i' %test_times[index])
    fig.savefig(output_path+'/xz%i.png' % test_times[index])
    print('saved images')
    
    
    #### unscale ####
    truth_inverse   = ss.inverse_transform(truth_reshaped)
    decoded_inverse = ss.inverse_transform(decoded_reshaped)
    truth           = truth_inverse.reshape(test_batches*b_size, len(x) , len(z) , variables) 
    decoded         = decoded_inverse.reshape(test_batches*b_size, len(x) , len(z) , variables) 
    ##### across time ####
    fig, ax = plt.subplots(3,variables, figsize=(12,6), tight_layout=True, sharex=True)
    z_index = 32
    error = truth[:, :, z_index, :] - decoded[:, :, z_index, :]
    for v in range(variables):
        minm = min(np.min(truth[:, :, z_index, v]), np.min(decoded[:, :, z_index, v]))
        maxm = max(np.max(truth[:, :, z_index, v]), np.max(decoded[:, :, z_index, v]))
        print(minm, maxm)
        c1 = ax[0,v].pcolormesh(test_times[:], x, truth[:, :, z_index, v].T, vmin=minm, vmax=maxm)
        c2 = ax[1,v].pcolormesh(test_times[:], x, decoded[:, :, z_index, v].T, vmin=minm, vmax=maxm)
        c3 = ax[2,v].pcolormesh(test_times[:], x, error[:, :, v].T, cmap='RdBu', norm=TwoSlopeNorm(vmin=-np.max(np.abs(error)), vcenter=0, vmax=np.max(np.abs(error))))
        ax[0,v].set_title(variable_names[v])

        fig.colorbar(c1, label='True')
        fig.colorbar(c2, label='reconstructed')
        fig.colorbar(c3, label='error')

    for i in range(2):
        ax[i,0].set_ylabel('x')
    for i in range(variables):
        ax[0,i].set_xlabel('time')
        ax[0,i].set_xlim(24000,24800)
        ax[1,i].set_xlim(24000,24800)
        ax[2,i].set_xlim(24000,24800)
    fig.savefig(output_path+'/time_scaled.png')

    ##### across x,z ####
    #plot n snapshots and their reconstruction in the test set.
    n       = 4

    start   = b_size*n_batches*skip+b_size*val_batches*skip  #b_size*n_batches*skip+b_size*val_batches*skip #start after validation set

    skips = 50
    for i in range(n):
        fig, ax = plt.subplots(3,variables, figsize=(12,6), tight_layout=True)
        index = 0 + skips*i
        print(index)
        u     = truth[index]
        u_dec = decoded[index]
        for v in range(variables):
            vmax = u[:,:,v].max()
            vmin = u[:,:,v].min()
            minm = min(np.min(u[:, :, v]), np.min(u_dec[:, :, v]))
            maxm = max(np.max(u[:, :, v]), np.max(u_dec[:, :, v]))
            NAE = np.abs(u - u_dec)/(vmax-vmin)
            print(minm, maxm)
            print('NMAE:', NAE[:,:, v].mean())
            c1 = ax[0,v].pcolormesh(x, z, u[:, :, v].T, vmin=minm, vmax=maxm)
            c2 = ax[1,v].pcolormesh(x, z, u_dec[:, :, v].T, vmin=minm, vmax=maxm)
            c3 = ax[2,v].pcolormesh(x, z, NAE[:, :, v].T, cmap='RdBu')

            fig.colorbar(c1, label='True')
            fig.colorbar(c2, label='reconstructed')
            fig.colorbar(c3, label='absolute error')

            fig.suptitle('time:%i' %test_times[index])
        fig.savefig(output_path+'/xz_scaled%i.png' % test_times[index])

    fig, ax = plt.subplots(3,variables, figsize=(12,9), tight_layout=True)
    index = 30
    print(index)
    u     = truth[index]
    u_dec = decoded[index]
    for v in range(variables):
        vmax = u[:,:,v].max()
        vmin = u[:,:,v].min()
        minm = min(np.min(u[:, :, v]), np.min(u_dec[:, :, v]))
        maxm = max(np.max(u[:, :, v]), np.max(u_dec[:, :, v]))
        NAE = np.abs(u - u_dec)/(vmax-vmin)
        print(minm, maxm)
        print('NMAE:', NAE[:,:, v].mean())
        c1 = ax[0,v].pcolormesh(x, z, u[:, :, v].T, vmin=minm, vmax=maxm)
        c2 = ax[1,v].pcolormesh(x, z, u_dec[:, :, v].T, vmin=minm, vmax=maxm)
        c3 = ax[2,v].pcolormesh(x, z, NAE[:, :, v].T, cmap='RdBu')

        fig.colorbar(c1, label='True')
        fig.colorbar(c2, label='reconstructed')
        fig.colorbar(c3, label='absolute error')

        fig.suptitle('time:%i' %test_times[index])
    fig.savefig(output_path+'/xz_scaled%i.png' % test_times[index])
    print('saved images')
    '''

main()

