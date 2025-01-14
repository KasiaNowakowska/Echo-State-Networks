"""
python script for convolutional autoencoder.

Usage: CAE.py [--input_path=<input_path> --output_path=<output_path> --model_path=<model_path>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
    --model_path=<model_path>          file path to location of job 
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

import wandb
#wandb.login()

input_path = args['--input_path']
output_path = args['--output_path']
model_path = args['--model_path']
print(output_path)
model_path = output_path+'/saved_models/'
print(model_path)

#### larger data 5000-30000 hf ####
total_num_snapshots = 2500
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
variables = num_variables = 2
variable_names = ['q', 'w']

with h5py.File(input_path+'/data_5000_30000.h5', 'r') as df:
    time_vals = np.array(df['total_time_all'][:total_num_snapshots])
    q = np.array(df['q_all'][:total_num_snapshots])
    w = np.array(df['w_all'][:total_num_snapshots])

    q = np.squeeze(q, axis=2)
    w = np.squeeze(w, axis=2)

    q_mean = np.mean(q, axis=0)
    w_mean = np.mean(w, axis=0)
    q_std = np.std(q, axis=0)
    w_std = np.std(w, axis=0)

    q_scaled = (q - q_mean) / q_std
    w_scaled = (w - w_mean) / w_std
    print(np.shape(q))
    print(np.shape(q_scaled))

print('shape of time_vals', np.shape(time_vals))

# Reshape the arrays into column vectors
q_array = q.reshape(len(time_vals), len(x), len(z), 1)
w_array = w.reshape(len(time_vals), len(x), len(z), 1)

q_scaled = q_scaled.reshape(len(time_vals), len(x), len(z), 1)
w_scaled = w_scaled.reshape(len(time_vals), len(x), len(z), 1)

del q
del w

data_all = np.concatenate((q_array, w_array), axis=-1)
data_scaled = np.concatenate((q_scaled, w_scaled), axis=-1)
# Print the shape of the combined array
print('shape of alldata and scaled data:', data_all.shape, data_scaled.shape)

U = data_scaled
dt = time_vals[1]-time_vals[0]
print('dt:', dt)

del w_array
del q_array


#### Hyperparmas #####
# Load hyperparameters from a JSON file
with open(output_path+'/hyperparameters.json', 'r') as file:
    hyperparameters = json.load(file)

lat_dep = hyperparameters.get("lat_dep")
n_epochs = hyperparameters.get("n_epochs")
l_rate = hyperparameters.get("learning_rate")
b_size = hyperparameters.get("batch_size")
lrate_mult = hyperparameters.get("lrate_mult")
N_lr = hyperparameters.get("N_lr")
N_parallel = hyperparameters.get("n_parallel")
N_check = hyperparameters.get("N_check")
N_layers = 4

print(f"Building Model with learning_rate: {l_rate}, batch_size: {b_size}, N_parallel: {N_parallel}, latent_depth: {lat_dep}, N_layers: {N_layers}")

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
print('layers:', N_layers)
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
p_fin         = [1,2,3]         #stride = 1 periodic padding size
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
print(models_dir)
a = [None]*N_parallel
b = [None]*N_parallel
print(ker_size)
print(N_latent)
for i in range(N_parallel):
    print(ker_size[i])
    print(str(ker_size[i]))
    print(str(N_latent))
    print(models_dir + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5')
    a[i] = tf.keras.models.load_model(models_dir + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5', custom_objects={"PerPad2D": PerPad2D})
for i in range(N_parallel):
    b[i] = tf.keras.models.load_model(models_dir + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5', custom_objects={"PerPad2D": PerPad2D})

#### NRMSE across x,z and averaged in time ####
truth = U_vt
decoded = model(truth,a,b)[1].numpy()
np.shape(decoded), np.shape(truth)
truth_reshaped = truth.reshape(test_batches*b_size, len(x) * len(z) * variables)
decoded_reshaped = decoded.reshape(test_batches*b_size, len(x) * len(z) * variables)
test_times = time_vals[b_size*n_batches*skip+b_size*val_batches*skip:
                         b_size*n_batches*skip+b_size*val_batches*skip+b_size*test_batches*skip] 
print('shape of time:', np.shape(test_times))
print('shape of inputs:', np.shape(truth))
print('shape of reconstructions:', np.shape(decoded))

nrmse = NRMSE(decoded_reshaped, truth_reshaped)

avg_nrmse = np.mean(nrmse)
sq_avg_nrmse = avg_nrmse**2

Energy = 1-sq_avg_nrmse
curve = 1-Energy
print("Energy:", Energy)
print("Curve:", curve)

##### across time ####
fig, ax =plt.subplots(3,variables, figsize=(12,9), tight_layout=True, sharex=True)
z_index=32
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
n       = 3

start   = b_size*n_batches*skip+b_size*val_batches*skip  #b_size*n_batches*skip+b_size*val_batches*skip #start after validation set

skips = 50
for i in range(n):
    fig, ax = plt.subplots(3,variables, figsize=(12,9), tight_layout=True)
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
index = 200
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


'''
    for channel in range(variables):
        for i in range(n):
        
            #truth
        
            skips = 50
        
            #snapshots to plot
            u      = truth[i*skips:1+i*skips]
            print(i*skips, 1+i*skips)
            print(np.shape(u))
            vmax   = u[:,:,:,channel].max()
            vmin   = u[:,:,:,channel].min()
        
            CS0    = ax[i,0].contourf(x, z, u[0,:,:,channel].T, levels=10,
                                 cmap='coolwarm',vmin=vmin, vmax=vmax)
            cbar   = plt.colorbar(CS0, ax=ax[0,i])
            cbar.set_label('$u_{\mathrm{True}}$',labelpad=15)
            #CS     = plt.contour(x, z,u[0,:,:,0].T, levels=10,colors='black',linewidths=.5, linestyles='solid',vmin=vmin, vmax=vmax)
        
            #autoencoded
        
            u_dec  = model(u,a,b)[1][0].numpy()
            CS1     = ax[i,1].contourf(x, z, u_dec[:,:,channel].T, levels=10,
                                cmap='coolwarm',vmin=vmin, vmax=vmax)
            cbar   = plt.colorbar(CS1, ax=ax[1,i])
            cbar.set_label('$u_{\mathrm{Autoencoded}}$',labelpad=15)
            #CS     = plt.contour(x, z,u_dec[:,:,0].T, levels=10, colors='black',linewidths=.5, linestyles='solid', vmin=vmin, vmax=vmax)
        
            fig.suptitle('%s' % variable_names[channel])
        
            #error
        
            u_err  = np.abs(u_dec-u[0])/(vmax-vmin)
            print('NMAE: ', u_err[:,:,channel].mean())
        
            CS2     = ax[i,2].contourf(x, z,u_err[:,:,channel].T, levels=10, cmap='coolwarm')
            cbar   = plt.colorbar(CS2, ax=ax[2,i])
            cbar.set_label('Relative Error',labelpad=15)
            #CS     = plt.contour(x, z,u_err[:,:,0].T,levels=10,colors='black',linewidths=.5, linestyles='solid')
    fig.savefig(output_path+'/image.png')

main()

'''
       
'''

for epoch in range(1):
    print('running epoch:', epoch)
    if epoch - last_save > patience: break #early stop

    #Perform gradient descent for all the batches every epoch
    loss_0 = 0
    rng.shuffle(U_train, axis=0) #shuffle batches
    for j in range(n_batches):
            loss    = train_step(U_train[j], enc_mods, dec_mods)
            loss_0 += loss

    #save train loss
    tloss_plot[epoch]  = loss_0.numpy()/n_batches

    print('checking convergence')
    #Compute Validation Loss
    loss_val        = 0
    for j in range(val_batches):
        loss        = train_step(U_val[j], enc_mods, dec_mods,train=False)
        loss_val   += loss

    #save validation loss
    vloss_plot[epoch]  = loss_val.numpy()/val_batches

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

    print('Reverting to best model and optimizer, and reducing learning rate')
    print('epoch:', epoch, 'N_lr:', N_lr)
    for i in range(N_parallel):
        enc_mods[i].load_weights(models_dir + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
        dec_mods[i].load_weights(models_dir + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')

    optimizer.learning_rate = optimizer.learning_rate*lrate_mult
    optimizer.set_weights(min_weights)
    print(f"Learning rate reduced to {optimizer.learning_rate.numpy()}")
'''
