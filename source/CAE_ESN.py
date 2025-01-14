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
import sys
sys.stdout.reconfigure(line_buffering=True)

exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('run functions files')

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
data_all_reshape = data_all.reshape(len(time_vals), len(x) * len(z) * variables)
### scale data ###
ss = StandardScaler()
data_scaled_reshape = ss.fit_transform(data_all_reshape)
data_scaled = data_scaled_reshape.reshape(len(time_vals), len(x), len(z), variables)

# Print the shape of the combined array
print('shape of all data and scaled data:', data_all.shape, data_scaled.shape)

U = data_scaled
dt = time_vals[1]-time_vals[0]
print('dt:', dt)

del w_array
del q_array

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
del U_vv, U_tt

# define the model
# we do not have pooling and upsampling, instead we use stride=2


ker_size      = [(3,3), (5,5), (7,7)]      #kernel sizes
if N_parallel == 1:
    ker_size  = [ker_size[kernel_choice]]
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

    
##### Compute encoded time_series #####
train_leng = 0
N_pos = 2500
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

dim      = N_latent #N_latent
act      = 'tanh'
U        = np.array(U_enc[transient:transient+data_len:upsample])
shape    = U.shape
print(shape)

U = U.reshape(shape[0], shape[1]*shape[2]*shape[3])

# number of time steps for washout, train, validation, test
N_washout = 800
N_train   = int(5000/dt)
N_val     = int(800/dt)

indexes_to_plot = np.array([1, 2, 10, 50, 100] ) -1

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)
normalisation = 'on'

# washout
U_washout = U[:N_washout].copy()
# data to be used for training + validation
U_tv  = U[N_washout:N_washout+N_train-1].copy() #inputs
Y_tv  = U[N_washout+1:N_washout+N_train].copy() #data to match at next timestep

# plotting part of training data to visualize noise
fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
ax.plot(U_tv[:N_val,0], c='tab:blue', label='Non-noisy')

# adding noise to training set inputs with sigma_n the noise of the data
# improves performance and regularizes the error as a function of the hyperparameters

seed = 0   #to be able to recreate the data, set also seed for initial condition u0
rnd1  = np.random.RandomState(seed)
noisy = True
if noisy:
    data_std = np.std(U,axis=0)
    sigma_n = 1e-3     #change this to increase/decrease noise in training inputs (up to 1e-1)
    for i in range(dim):
        U_tv[:,i] = U_tv[:,i] \
                        + rnd1.normal(0, sigma_n*data_std[i], N_train-1)
    ax.plot(U_tv[:N_val,0], 'r--', label='Noisy')

ax.legend()
ax.grid()
fig.savefig(output_path+'/training_data.png')

#network parameters
bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm))]) #input bias (average absolute value of the inputs)
bias_out  = np.array([1.]) #output bias 

N_units      = 4000 #neurons
connectivity = 3   
sparseness   = 1 - connectivity/(N_units-1) 

tikh = np.array([1e-9,1e-12])  # Tikhonov factor (optimize among the values in this list)

#### hyperparamter search ####
n_in  = 0           #Number of Initial random points

spec_in     = 0.6    #range for hyperparameters (spectral radius and input scaling)
spec_end    = 1.1
in_scal_in  = np.log10(0.001)
in_scal_end = np.log10(5.)

# In case we want to start from a grid_search, the first n_grid_x*n_grid_y points are from grid search
n_grid_x = 5  
n_grid_y = 5
n_bo     = 5  #number of points to be acquired through BO after grid search
n_tot    = n_grid_x*n_grid_y + n_bo #Total Number of Function Evaluatuions


# computing the points in the grid
if n_grid_x > 0:
    x1    = [[None] * 2 for i in range(n_grid_x*n_grid_y)]
    k     = 0
    for i in range(n_grid_x):
        for j in range(n_grid_y):
            x1[k] = [spec_in + (spec_end - spec_in)/(n_grid_x-1)*i,
                     in_scal_in + (in_scal_end - in_scal_in)/(n_grid_y-1)*j]
            k   += 1

# range for hyperparameters
search_space = [Real(spec_in, spec_end, name='spectral_radius'),
                Real(in_scal_in, in_scal_end, name='input_scaling')]

# ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0))*\
                  Matern(length_scale=[0.2,0.2], nu=2.5, length_scale_bounds=(5e-2, 1e1)) 


#Hyperparameter Optimization using Grid Search plus Bayesian Optimization
def g(val):
    
    #Gaussian Process reconstruction
    b_e = GPR(kernel = kernell,
            normalize_y = True, #if true mean assumed to be equal to the average of the obj function data, otherwise =0
            n_restarts_optimizer = 3,  #number of random starts to find the gaussian process hyperparameters
            noise = 1e-10, # only for numerical stability
            random_state = 10) # seed
    
    
    #Bayesian Optimization
    res = skopt.gp_minimize(val,                         # the function to minimize
                      search_space,                      # the bounds on each dimension of x
                      base_estimator       = b_e,        # GP kernel
                      acq_func             = "EI",       # the acquisition function
                      n_calls              = n_tot,      # total number of evaluations of f
                      x0                   = x1,         # Initial grid search points to be evaluated at
                      n_random_starts      = n_in,       # the number of additional random initialization points
                      n_restarts_optimizer = 3,          # number of tries for each acquisition
                      random_state         = 10,         # seed
                           )   
    return res


#Number of Networks in the ensemble
ensemble = 1

data_dir = '/Run_n_units{0:}_ensemble{1:}_normalisation{2:}/'.format(N_units, ensemble, normalisation)
output_path = output_path+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')
    
data_dir = '/GP{0:}_{1:}/'.format(n_grid_x*n_grid_y, n_bo)
output_path = output_path+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

# Which validation strategy (implemented in Val_Functions.ipynb)
val      = RVC_Noise
N_fo     = 6                     # number of validation intervals
N_in     = N_washout                 # timesteps before the first validation interval (can't be 0 due to implementation)
N_fw     = (N_train-N_val-N_washout)//(N_fo-1) # how many steps forward the validation interval is shifted (in this way they are evenly spaced)
N_splits = 4                         # reduce memory requirement by increasing N_splits

#Quantities to be saved
par      = np.zeros((ensemble, 4))      # GP parameters
x_iters  = np.zeros((ensemble,n_tot,2)) # coordinates in hp space where f has been evaluated
f_iters  = np.zeros((ensemble,n_tot))   # values of f at those coordinates
minimum  = np.zeros((ensemble, 4))      # minima found per each member of the ensemble

# to store optimal hyperparameters and matrices
tikh_opt = np.zeros(n_tot)
Woutt    = np.zeros(((ensemble, N_units+1,dim)))
Winn     = [] #save as list to keep single elements sparse
Ws       = []
Xa1_states =  np.zeros(((ensemble, N_train-1, N_units+1))) ####added  

# save the final gp reconstruction for each network
gps        = [None]*ensemble

# to print performance of every set of hyperparameters
print_flag = False

# optimize ensemble networks (to account for the random initialization of the input and state matrices)
for i in range(ensemble):

    print('Realization    :',i+1)

    k   = 0

    # Win and W generation
    seed= i+1
    rnd = np.random.RandomState(seed)

    #sparse syntax for the input and state matrices
    Win  = lil_matrix((N_units,dim+1))
    for j in range(N_units):
        Win[j,rnd.randint(0, dim+1)] = rnd.uniform(-1, 1) #only one element different from zero
    Win = Win.tocsr()

    W = csr_matrix( #on average only connectivity elements different from zero
        rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1-sparseness)))

    spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
    W = (1/spectral_radius)*W #scaled to have unitary spec radius

    # Bayesian Optimization
    tt       = time.time()
    res      = g(val)
    print('Total time for the network:', time.time() - tt)


    #Saving Quantities for post_processing
    gps[i]     = res.models[-1]
    gp         = gps[i]
    x_iters[i] = np.array(res.x_iters)
    f_iters[i] = np.array(res.func_vals)
    minimum[i] = np.append(res.x,[tikh_opt[np.argmin(f_iters[i])],res.fun])
    params     = gp.kernel_.get_params()
    key        = sorted(params)
    par[i]     = np.array([params[key[2]],params[key[5]][0], params[key[5]][1], gp.noise_])

    #saving matrices
    train      = train_save_n(U_washout, U_tv, Y_tv,
                              minimum[i,2],10**minimum[i,1], minimum[i,0], minimum[i,3]) ###changed
    Woutt[i]   = train[0] ###changed
    Winn    += [Win]
    Ws      += [W]
    
    Xa1_states[i]  = train[1] ###changed


    #Plotting Optimization Convergence for each network
    print('Best Results: x', minimum[i,0], 10**minimum[i,1], minimum[i,2],
          'f', -minimum[i,-1])
    fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
    plot_convergence(res)
    fig.savefig(output_path+'/convergence_realisation%i.png' % i)
    plt.close()
    
# points to evaluate the GP at
n_length    = 100
xx, yy      = np.meshgrid(np.linspace(spec_in, spec_end,n_length), np.linspace(in_scal_in, in_scal_end,n_length))
x_x         = np.column_stack((xx.flatten(),yy.flatten()))
x_gp        = res.space.transform(x_x.tolist())     ##gp prediction needs this normalized format
y_pred      = np.zeros((ensemble,n_length,n_length))

for i in range(ensemble):
    # retrieve the gp reconstruction
    gp         = gps[i]

    pred, pred_std = gp.predict(x_gp, return_std=True)

    fig, ax  = plt.subplots(1, figsize=(12,10), tight_layout=True)

    amin = np.amin([10,f_iters.max()])

    y_pred[i] = np.clip(-pred, a_min=-amin,
                        a_max=-f_iters.min()).reshape(n_length,n_length)
                        # Final GP reconstruction for each realization at the evaluation points

    ax.set_title('Mean GP of realization \#'+ str(i+1))

    #Plot GP Mean
    ax.set_xlabel('Spectral Radius')
    ax.set_ylabel('$\log_{10}$Input Scaling')
    CS      = ax.contourf(xx, yy, y_pred[i],levels=10,cmap='Blues')
    cbar = fig.colorbar(CS, ax=ax)
    cbar.set_label('-$\log_{10}$(MSE)',labelpad=15)
    CSa     = ax.contour(xx, yy, y_pred[i],levels=10,colors='black',
                          linewidths=1, linestyles='solid')

    #   Plot the n_tot search points
    ax.scatter(x_iters[i,:n_grid_x*n_grid_y,0],
                x_iters[i,:n_grid_x*n_grid_y,1], c='r', marker='^') #grid points
    ax.scatter(x_iters[i,n_grid_x*n_grid_y:,0],
                x_iters[i,n_grid_x*n_grid_y:,1], c='lime', marker='s') #bayesian opt points

    fig.savefig(output_path + '/GP_%i.png' % i)
    plt.close()
np.save(output_path + '/f_iters.npy', f_iters)

N_test   = 1                      #number of intervals in the test set
N_tstart = 7000                   #where the first test interval starts
N_intt   = N_val                   #length of each test set interval


ensemble_test = ensemble

for j in range(ensemble_test):
    
    print('Realization    :',j+1)
    
    #load matrices and hyperparameters
    Wout     = Woutt[j].copy()
    Win      = Winn[j] #csr_matrix(Winn[j])
    W        = Ws[j]   #csr_matrix(Ws[j])    
    rho      = minimum[j,0].copy()
    sigma_in = 10**minimum[j,1].copy()
    print('Hyperparameters:',rho, sigma_in)
    
    # to store prediction horizon in the test set
    Errors   = np.zeros(N_test)
    
    # to plot results
    plot = True
    if plot:
        n_plot = 1
    
    
    #run different test intervals
    for i in range(N_test):
        
        # data for washout and target in each interval
        U_wash    = U[N_tstart - N_washout +i*N_intt : N_tstart + i*N_intt].copy()
        Y_t       = U[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt].copy() 
                
        #washout for each interval
        Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
        Uh_wash = np.dot(Xa1, Wout)
                
        # Prediction Horizon
        Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
        print('shape of decoded image after ESN:', np.shape(Yh_t))
        Errors[i]   = np.sqrt(np.mean((Y_t-Yh_t)**2))
        if plot:
            #left column has the washout (open-loop) and right column the prediction (closed-loop)
            # only first n_plot test set intervals are plotted
            if i<n_plot:
                #### modes prediction ####
                fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                xx = np.arange(U_wash[:,-2].shape[0])
                ax.plot(xx,U_wash[:,0], 'b', label='True')
                ax.plot(xx,U_wash[:, 1:3], 'b')
                ax.plot(xx,Uh_wash[:-1,0], '--r', label='ESN')
                ax.plot(xx,Uh_wash[:-1,1:3], '--r')
                ax.grid()
                ax.set_xlabel('Time')
                fig.legend()
                fig.suptitle('washout_ens%i_test%i' % (j,i))
                fig.savefig(output_path+'/washout_ens%i_test%i.png' % (j,i))
                plt.close()

                fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                xx = np.arange(Y_t[:,-2].shape[0])
                ax.plot(xx,Y_t[:,0], 'b', label='True')
                ax.plot(xx,Y_t[:,1:3], 'b')
                ax.plot(xx,Yh_t[:,0], '--r', label='ESN')
                ax.plot(xx,Yh_t[:,1:3], '--r')
                ax.grid()
                fig.legend()
                ax.set_xlabel('Time')
                fig.savefig(output_path+'/prediction_ens%i_test%i.png' % (j,i))
                plt.close()
                
                #decode
                prediction_reshaped         = Yh_t.reshape(N_intt,N_1[1], N_1[2], N_1[3])
                truth_reshaped              = Y_t.reshape(N_intt,N_1[1], N_1[2], N_1[3]) 
                print('shape of predicted latent space reshaped:', np.shape(prediction_reshaped))
                decoded_prediction          = Decoder(prediction_reshaped,b).numpy()
                decoded_truth               = Decoder(truth_reshaped,b).numpy()
                print('shape of decoded prediction:', np.shape(decoded_prediction))
                decoded_prediction_squashed = decoded_prediction.reshape(N_intt, len(x)* len(z)* variables)
                decoded_truth_squashed      = decoded_truth.reshape(N_intt, len(x)* len(z)* variables)
                    
                decoded_prediction_scaled   = ss.inverse_transform(decoded_prediction_squashed)
                decoded_truth_scaled        = ss.inverse_transform(decoded_truth_squashed)
                decoded_prediction_reshaped = decoded_prediction_scaled.reshape(N_intt, len(x), len(z), variables)
                decoded_truth_reshaped      = decoded_truth_scaled.reshape(N_intt, len(x), len(z), variables)
                    
                for v in range(variables):
                    fig, ax = plt.subplots(2, figsize=(12,6), sharex=True, tight_layout=True)
                    minm = min(np.min(decoded_truth_reshaped[:, :, 32, v]), np.min(decoded_prediction_reshaped[:, :, 32, v]))
                    maxm = max(np.max(decoded_truth_reshaped[:, :, 32, v]), np.max(decoded_prediction_reshaped[:, :, 32, v]))
                    xx = np.arange(Y_t[:,-2].shape[0])
                    c1 = ax[0].contourf(xx, x, decoded_truth_reshaped[:,:,32,v].T, vmin=minm, vmax=maxm)
                    c2 = ax[1].contourf(xx, x, decoded_prediction_reshaped[:,:,32,v].T, vmin=minm, vmax=maxm)
                    norm1 = mcolors.Normalize(vmin=minm, vmax=maxm)
                    fig.colorbar(c1, ax=ax[0], label='POD', norm=norm1, extend='both')
                    fig.colorbar(c2, ax=ax[1], label='ESN', norm=norm1, extend='both')
                    ax[1].set_xlabel('Time')
                    ax[0].set_ylabel('x')
                    ax[1].set_ylabel('x')
                    fig.savefig(output_path+'/prediction_recon_ens%i_test%i_var%i.png' % (j,i,v))
                    plt.close()
    
    import json

    # Full path for saving the file
    output_file = "metrics%i.json" % j
    
    output_path_hyp = os.path.join(output_path, output_file)
    
    hyperparameters = {
    "spec_rad": rho,
    "input_scaling": sigma_in,
    "PH": np.median(Errors),
    }
    
    with open(output_path_hyp, "w") as file:
        json.dump(hyperparameters, file, indent=4)
    
    # Percentiles of the prediction horizon
    print('PH quantiles [Lyapunov Times]:', 
          np.quantile(Errors,.75), np.median(Errors), np.quantile(Errors,.25))
    print('')