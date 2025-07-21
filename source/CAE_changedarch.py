"""
python script for convolutional autoencoder.

Usage: CAE.py [--input_path=<input_path> --output_path=<output_path> --hyperparam_config=<hyperparam_config> --sweep_id=<sweep_id> --project_name=<project_name> --reduce_domain=<reduce_domain>]

Options:
    --input_path=<input_path>                 file path to use for data
    --output_path=<output_path>               file path to save images output [default: ./images]
    --hyperparam_config=<hyperparam_config>   hyperparameter config file for search
    --sweep_id=<sweep_id>                     sweep_id for restarting sweep [default: None]
    --project_name=<project_name>             name of project for weights and biases
    --reduce_domain=<reduce_domain>           reduce domain size [default: False]
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
import random
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
from Eval_Functions import *
from Plotting_Functions import *

import sys
sys.stdout.reconfigure(line_buffering=True)

from tensorflow.keras import backend as K
import gc
K.clear_session()
gc.collect()

import wandb
wandb.login()

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

input_path = args['--input_path']
output_path = args['--output_path']
hyperparam_config = args['--hyperparam_config']
sweep_id = args['--sweep_id']
project_name = args['--project_name']
if project_name == None:
    print('ERROR: must supply project name for wandb')
print('sweep id', sweep_id)

reduce_domain = args['--reduce_domain']
if reduce_domain == 'False':
    reduce_domain = False
    print('domain not reduced', reduce_domain)
elif reduce_domain == 'True':
    reduce_domain = True
    print('domain reduced', reduce_domain)

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

# #### make directory ####
# from datetime import datetime 
# now = datetime.now()
# print(now)
# # Format the date and time as YYYY-MM-DD_HH-MM-SS for a file name
# formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
# # Example of using the formatted datetime in a file name
# output_new = f"validation_{formatted_datetime}"
# output_path = os.path.join(output_path, output_new)
# print(output_path)
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
#     print('made directory')

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

#### LOAD DATA AND POD ####
Data = 'RB'
if Data == 'ToyData':
    name = names = variables = ['combined']
    n_components = 3
    num_variables = 1
    snapshots = 25000
    data_set, x, z, time_vals = load_data_set_TD(input_path+'/plume_wave_dataset_smallergrid_longertime.h5', name, snapshots)
    print('shape of dataset', np.shape(data_set))
    dt = 0.05

elif Data == 'RB':
    variables = ['q_all', 'w_all', 'u_all', 'b_all']
    names = ['q', 'w', 'u', 'b']
    num_variables = 4
    x = np.load(input_path+'/x.npy')
    z = np.load(input_path+'/z.npy')
    snapshots_load = 16000
    data_set, time_vals = load_data_set_RB(input_path+'/data_4var_5000_48000.h5', variables, snapshots_load)
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
            ax[i].pcolormesh(time_vals, x, data_set[:,:,32,i].T)
            ax[i].set_ylabel('x')
            ax[i].set_title(names[i])
            ax[-1].set_xlabel('time')
    fig.savefig(output_path+'/hovmoller.png')

U = data_set

# Load the config from file
with open(hyperparam_config, "r") as f:
    sweep_configuration = json.load(f)

if sweep_id == 'None':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name) #"Ra2e8_smallerdomain"
    print('new sweep with sweep id', sweep_id)
else:
    print('restarting sweep with sweep_id', sweep_id)

output_new = f"validation_{sweep_id}"
output_path = os.path.join(output_path, output_new)
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

job = 0

def main():

    run = wandb.init()
    global job
    job += 1

    SEED = wandb.config.seed

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

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
        
    #### load in data ###
    if reduce_domain:    
        b_size      = wandb.config.b_size   #batch_size
        n_batches   = 13 #int((U.shape[0]/b_size) *0.7)  #number of batches #20
        val_batches = 1 #int((U.shape[0]/b_size) *0.2)    #int(n_batches*0.2) # validation set size is 0.2 the size of the training set #2
        test_batches = 1 # int((U.shape[0]/b_size) *0.1)
    else:
        b_size      = wandb.config.b_size   #batch_size
        n_batches   = int((U.shape[0]/b_size) *0.7)  #number of batches #20
        val_batches = int((U.shape[0]/b_size) *0.2)    #int(n_batches*0.2) # validation set size is 0.2 the size of the training set #2
        test_batches = int((U.shape[0]/b_size) *0.1)
    skip        = 1
    print(n_batches, val_batches, test_batches)

    print('number of snapshot training set', b_size*n_batches*skip*dt)
    print('number of snapshot validation set', b_size*val_batches*skip*dt)  

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

    del U_vv, U_tt, U_tv, U_vv_scaled, U_tt_scaled, U_tv_scaled

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
    
    rng = np.random.default_rng(seed=42) #random generator for later shufflinh

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
    old_loss        = np.zeros(n_epochs) #needed to evaluate training loss convergence to update l_rate
    tloss_plot      = np.zeros(n_epochs) #training loss
    vloss_plot      = np.zeros(n_epochs) #validation loss
    tssim_plot      = np.zeros(n_epochs) #training loss
    vssim_plot      = np.zeros(n_epochs) #validation loss
    tacc_plot       = np.zeros(n_epochs)
    vacc_plot       = np.zeros(n_epochs)
    tEVR_plot       = np.zeros(n_epochs)
    vEVR_plot       = np.zeros(n_epochs)
    tNRMSE_plot     = np.zeros(n_epochs)
    vNRMSE_plot     = np.zeros(n_epochs)
    tNRMSEpl_plot   = np.zeros(n_epochs)
    vNRMSEpl_plot   = np.zeros(n_epochs)
    tMSE_plot       = np.zeros(n_epochs)
    vMSE_plot       = np.zeros(n_epochs)
    tNRMSE_sep_plot = np.zeros(n_epochs)
    vNRMSE_sep_plot = np.zeros(n_epochs)

    old_loss[0]  = 1e6 #initial value has to be high
    N_check      = 5 #10   #each N_check epochs we check convergence and validation loss
    patience     = 100 #if the val_loss has not gone down in the last patience epochs, early stop
    last_save    = patience

    t            = 1 # initial (not important value) to monitor the time of the training

    #### make directory to save ###
    #job_name = 'job{0:}_batch_size{1:}_learning_rate{2:}_num_epochs{3:}_lrate_mult{4:}_N_lr{5:}_n_parallel{6:}_lat_dep{7:}_N_layers{8:}'.format(job, b_size, l_rate, n_epochs, lrate_mult, N_lr, N_parallel, lat_dep, N_layers)
    job_name = f"job_{str(wandb.run.id)}"
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
    "N_check": N_check,
    "lat_dep": lat_dep, 
    "ker_size": str(ker_size),
    "job id": str(wandb.run.id),
    "sweep id": str(wandb.run.sweep_id)
    }

    with open(output_path_hyp, "w") as file:
        json.dump(hyperparameters, file, indent=4)

    for epoch in range(n_epochs):
        print('running epoch:', epoch)
        if epoch - last_save > patience: 
            print('early stop - val loss has not decreased in last 100 epochs')
            break #early stop
        
        if epoch > patience:
            if ssim_0 < 0.5:
                print('early stop - SSIM is smaller than 0.5 after 100 epochs')
                break #early stop

        #Perform gradient descent for all the batches every epoch
        loss_0        = 0
        ssim_0        = 0
        evr_0         = 0
        nrmse_0       = 0
        accuracy_0    = 0
        nrmse_plume_0 = 0 
        mse_0         = 0
        nrmse_sep_0   = 0
        rng.shuffle(U_train, axis=0) #shuffle batches
        for j in range(n_batches):
            loss, decoded    = train_step(U_train[j], enc_mods, dec_mods)
            loss_0          += loss

            
            original = U_train[j]  # Validation input
            
            decoded_unscaled = ss_inverse_transform(decoded.numpy(), scaler)
            original_unscaled = ss_inverse_transform(original, scaler)

            # Compute SSIM
            #print(np.shape(original), np.shape(decoded))
            #print(f"Type of original: {type(original)}")
            #print(f"Type of decoded: {type(decoded)}")
            batch_ssim = compute_ssim_for_4d(original_unscaled, decoded_unscaled)
            ssim_0 += batch_ssim

            ### new metrics from POD ###
            evr_value          = EVR_recon(original_unscaled, decoded_unscaled)
            evr_0             += evr_value
            nrmse_value        = NRMSE(original_unscaled, decoded_unscaled)
            nrmse_0           += nrmse_value
            mse_value          = MSE(original_unscaled, decoded_unscaled)
            mse_0             += mse_value  
            nrmse_sep_value    = NRMSE_per_channel(original_unscaled, decoded_unscaled)
            nrmse_sep_0       += nrmse_sep_value

            if len(variables) == 4:
                active_array, active_array_reconstructed, mask, mask_expanded_recon = active_array_calc(original_unscaled, decoded_unscaled, z)
                accuracy = np.mean(active_array == active_array_reconstructed)
                if np.any(mask):  # Check if plumes exist
                    masked_truth = original_unscaled[mask]
                    masked_pred = decoded_unscaled[mask]
                    
                    print("Shape truth after mask:", masked_truth.shape)
                    print("Shape pred after mask:", masked_pred.shape)

                    # Compute NRMSE only if mask is not empty
                    nrmse_plume = NRMSE(masked_truth, masked_pred)
                else:
                    print("Mask is empty, no plumes detected.")
                    nrmse_plume = 0  # Simply add 0 to maintain shape
            else:
                nrmse_plume = 0
                accuracy    = 0

            accuracy_0             += accuracy
            nrmse_plume_0          += nrmse_plume

            print('metrics saved')

        #save train loss
        tloss_plot[epoch]        = loss_0.numpy()/n_batches
        tssim_plot[epoch]        = ssim_0/n_batches
        tacc_plot[epoch]         = accuracy_0/n_batches
        tEVR_plot[epoch]         = evr_0/n_batches
        tNRMSE_plot[epoch]       = nrmse_0/n_batches
        tacc_plot[epoch]         = accuracy_0/n_batches
        tNRMSEpl_plot[epoch]     = nrmse_plume_0/n_batches
        tMSE_plot[epoch]         = mse_0/n_batches
        tNRMSE_sep_plot[epoch]   = nrmse_sep_0/n_batches

        # every N epochs checks the convergence of the training loss and val loss
        if (epoch%N_check==0):
            print('checking convergence')
            #Compute Validation Loss
            loss_val        = 0
            ssim_val        = 0
            accuracy_val    = 0
            evr_val         = 0
            nrmse_val       = 0
            nrmse_plume_val = 0 
            mse_val         = 0
            nrmse_sep_val   = 0

            for j in range(val_batches):
                loss, decoded       = train_step(U_val[j], enc_mods, dec_mods,train=False)
                loss_val           += loss

                # Compute SSIM
                original = U_val[j]  # Validation input

                decoded_unscaled = ss_inverse_transform(decoded.numpy(), scaler)
                original_unscaled = ss_inverse_transform(original, scaler)
            
                batch_ssim = compute_ssim_for_4d(original_unscaled, decoded_unscaled)
                ssim_val += batch_ssim

                ### new metrics from POD ###
                evr_value            = EVR_recon(original_unscaled, decoded_unscaled)
                evr_val             += evr_value
                nrmse_value          = NRMSE(original_unscaled, decoded_unscaled)
                nrmse_val           += nrmse_value
                mse_value            = MSE(original_unscaled, decoded_unscaled)
                mse_val             += mse_value
                nrmse_sep_value      = NRMSE_per_channel(original_unscaled, decoded_unscaled)
                nrmse_sep_val       += nrmse_sep_value

                if len(variables) == 4:
                    active_array, active_array_reconstructed, mask, mask_expanded_recon = active_array_calc(original_unscaled, decoded_unscaled, z)
                    accuracy = np.mean(active_array == active_array_reconstructed)
                    if np.any(mask):  # Check if plumes exist
                        masked_truth = original_unscaled[mask]
                        masked_pred = decoded_unscaled[mask]
                        
                        print("Shape truth after mask:", masked_truth.shape)
                        print("Shape pred after mask:", masked_pred.shape)

                        # Compute NRMSE only if mask is not empty
                        nrmse_plume = NRMSE(masked_truth, masked_pred)
                    else:
                        print("Mask is empty, no plumes detected.")
                        nrmse_plume = 0  # Simply add 0 to maintain shape
                else:
                    nrmse_plume = 0
                    accuracy    = 0

                accuracy_val             += accuracy
                nrmse_plume_val          += nrmse_plume

            #save validation loss
            vloss_epoch            = loss_val.numpy()/val_batches
            vloss_plot[epoch]      = vloss_epoch 
            vssim_epoch            = ssim_val/val_batches
            vssim_plot[epoch]      = vssim_epoch
            vacc_epoch             = accuracy_val/val_batches
            vacc_plot[epoch]       = vacc_epoch
            vEVR_plot[epoch]       = evr_val/val_batches
            vNRMSE_plot[epoch]     = nrmse_val/val_batches
            vacc_plot[epoch]       = accuracy_val/val_batches
            vNRMSEpl_plot[epoch]   = nrmse_plume_val/val_batches
            vMSE_plot[epoch]       = mse_val/val_batches
            vNRMSE_sep_plot[epoch] = nrmse_sep_val/val_batches

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
                      "Latent Space": N_latent,
                      "SSIM train": tssim_plot[epoch],
                      "SSIM val": vssim_plot[epoch],
                      "Plume Acc": vacc_plot[epoch],
                      "NRMSE_val": vNRMSE_plot[epoch],
                      "NRMSE_train": tNRMSE_plot[epoch],
                      "Plume NRMSE val": vNRMSEpl_plot[epoch],
                      "Plume NRMSE train": tNRMSEpl_plot[epoch],
                      "EVR_val": vEVR_plot[epoch],
                      "EVR_train": tEVR_plot[epoch],
                      "MSE_val": vMSE_plot[epoch],
                      "MSE_train": tMSE_plot[epoch],
                      "NRMSE_sep_train": tNRMSE_sep_plot[epoch],
                      "NRMSE_sep_val": vNRMSE_sep_plot[epoch]
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
            ax.set_title('EVR')
            ax.grid(True, axis="both", which='both', ls="-", alpha=0.3)
            ax.plot(tEVR_plot, 'y', label='Train EVR')
            ax.plot(np.arange(0,n_epochs,N_check),vEVR_plot[::N_check], label='Val SSIM')
            ax.set_xlabel('epochs')
            ax.legend()
            fig.savefig(images_dir +'/EVR.png')
    
    K.clear_session()
    gc.collect()
    print('finished job')

wandb.agent(sweep_id=sweep_id, function=main, entity="mm17ktn-university-of-leeds", project=project_name)
