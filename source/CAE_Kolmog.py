import os
os.environ["OMP_NUM_THREADS"] = "1" #set cores for numpy
import numpy as np
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
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import h5py
import time
from pathlib import Path
import matplotlib as mpl


#### DATA HANDLING ####
upsample = 1
Re       = 30
data_len = 300000
transient = 10000

Nx = 48
Nu = 2
# Generate this from Gen_data.ipynb
fln = '../../skesn/data/Kolmogorov/Kolmogorov_0.1_48_34.0_100100_32.h5'
hf  = h5py.File(fln,'r')
dt  = 0.1
U   = np.array(hf.get('U')[transient:transient+data_len:upsample], dtype=np.float32)
hf.close()

N_x     = U.shape[1]
N_y     = U.shape[2]

print('U-shape:',U.shape, dt)

def split_data(U, b_size, n_batches):
    
    '''
    Splits the data in batches. Each batch is created by sampling the signal with interval
    equal to n_batches
    '''
    data   = np.zeros((n_batches, b_size, U.shape[1], U.shape[2], U.shape[3]))    
    for j in range(n_batches):
        data[j] = U[::skip][j::n_batches].copy()

    return data

b_size      = 50   #batch_size
n_batches   = 500  #number of batches
val_batches = 50   #int(n_batches*0.2) # validation set size is 0.2 the size of the training set
skip        = 10

#print(b_size*n_batches*skip*dt*upsample)

print('Train Data%  :',b_size*n_batches*skip/U.shape[0]) #how much of the data we are using for training
print('Val   Data%  :',b_size*val_batches*skip/U.shape[0])

# training data
U_tt        = np.array(U[:b_size*n_batches*skip].copy())            #to be used for random batches
U_train     = split_data(U_tt, b_size, n_batches).astype('float32') #to be used for randomly shuffled batches
# validation data
U_vv        = np.array(U[b_size*n_batches*skip:
                         b_size*n_batches*skip+b_size*val_batches*skip].copy())
U_val       = split_data(U_vv, b_size, val_batches).astype('float32')             

###### AUTOENCODER FUNCTIONS ####
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
    When asym=True on the right and lower edges an additional column/row is added
    '''
        
    if asym:
        lower_pad = image[:,:padding+1,:]
    else:
        lower_pad = image[:,:padding,:]
    
    if padding != 0:
        upper_pad     = image[:,-padding:,:]
        partial_image = tf.concat([upper_pad, image, lower_pad], axis=1)
    else:
        partial_image = tf.concat([image, lower_pad], axis=1)
        
    if asym:
        right_pad = partial_image[:,:,:padding+1] 
    else:
        right_pad = partial_image[:,:,:padding]
    
    if padding != 0:
        left_pad = partial_image[:,:,-padding:]
        padded_image = tf.concat([left_pad, partial_image, right_pad], axis=2)
    else:
        padded_image = tf.concat([partial_image, right_pad], axis=2)

    return padded_image

###### CREATE THE MODEL ######
# define the model
# we do not have pooling and upsampling, instead we use stride=2

lat_dep       = 2                          #latent space depth
n_fil         = [6,12,24,lat_dep]          #number of filters ecnoder
n_dec         = [24,12,6,3]                #number of filters decoder
N_parallel    = 3                          #number of parallel CNNs for multiscale
ker_size      = [(3,3), (5,5), (7,7)]      #kernel sizes
N_layers      = 4                          #number of layers in every CNN
act           = 'tanh'                     #activation function

pad_enc       = 'valid'         #no padding in the conv layer
pad_dec       = 'valid'
p_size        = [0,1,2]         #stride = 2 periodic padding size          
p_fin         = [1,2,3]         #stride = 1 periodic padding size
p_dec         = 1               #padding in the first decoder layer
p_crop        = U.shape[1]      #crop size of the output equal to input size


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


#explicitly obtain the size of the latent space
N_1      = enc_mods[-1](U_train[0]).shape
N_latent = N_1[-3]*N_1[-2]*N_1[-1]

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
    dec_mods[j].add(tf.keras.layers.CenterCrop(p_crop + 2*p_fin[j],
                                                   p_crop+ 2*p_fin[j],
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
for j in range(3):
    enc_mods[j].summary()
for j in range(3):
    dec_mods[j].summary()    

##### TRAIN THE MODEL #####
path = '/Kolmogorov/48_RE34_'+str(N_latent) #to save model
print(path)
rng = np.random.default_rng() #random generator for later shufflinh

Loss_Mse    = tf.keras.losses.MeanSquaredError()

n_epochs    = 101 #number of epochs

#define optimizer and initial learning rate   
optimizer  = tf.keras.optimizers.Adam(amsgrad=True) #amsgrad True for better convergence
l_rate     = 0.002
optimizer.learning_rate = l_rate

lrate_update = True #flag for l_rate updating
lrate_mult   = 0.75 #decrease by this factore the l_rate 
N_lr         = 100  #number of epochs before which the l_rate is not updated

# quantities to check and store the training and validation loss and the training goes on
old_loss      = np.zeros(n_epochs) #needed to evaluate training loss convergence to update l_rate
tloss_plot    = np.zeros(n_epochs) #training loss
vloss_plot    = np.zeros(n_epochs) #validation loss
old_loss[0]  = 1e6 #initial value has to be high
N_check      = 5   #each N_check epochs we check convergence and validation loss
patience     = 200 #if the val_loss has not gone down in the last patience epochs, early stop
last_save    = patience

t            = 1 # initial (not important value) to monitor the time of the training

for epoch in range(n_epochs):
    print("epoch", epoch)
    
    if epoch - last_save > patience: break #early stop
                
    #Perform gradient descent for all the batches every epoch
    loss_0 = 0
    rng.shuffle(U_train, axis=0) #shuffle batches
    for j in range(n_batches):
            loss    = train_step(U_train[j], enc_mods, dec_mods)
            loss_0 += loss
    
    #save train loss
    tloss_plot[epoch]  = loss_0.numpy()/n_batches     
    
    # every N epochs checks the convergence of the training loss and val loss
    if (epoch%N_check==0):
        print("checking")
        #Compute Validation Loss
        loss_val        = 0
        for j in range(val_batches):
            loss        = train_step(U_val[j], enc_mods, dec_mods,train=False)
            loss_val   += loss
        
        #save validation loss
        vloss_plot[epoch]  = loss_val.numpy()/val_batches 
        
        # Decreases the learning rate if the training loss is not going down with respect to 
        # N_lr epochs before
        if epoch > N_lr and lrate_update:
            #check if the training loss is smaller than the average training loss N_lr epochs ago
            tt_loss   = np.mean(tloss_plot[epoch-N_lr:epoch])
            if tt_loss > old_loss[epoch-N_lr]:
                #if it is larger, load optimal val loss weights and decrease learning rate
                print('LOADING MINIMUM')
                for i in range(N_parallel):
                    enc_mods[i].load_weights(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
                    dec_mods[i].load_weights(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')

                optimizer.learning_rate = optimizer.learning_rate*lrate_mult
                optimizer.set_weights(min_weights)
                print('LEARNING RATE CHANGE', optimizer.learning_rate.numpy(), deviation)
                old_loss[epoch-N_lr:epoch] = 1e6 #so that l_rate is not changed for N_lr steps
        
        #store current loss
        old_loss[epoch] = tloss_plot[epoch].copy()
        
        #save best model (the one with minimum validation loss)
        if epoch > 1 and vloss_plot[epoch] < \
                         (vloss_plot[:epoch-1][np.nonzero(vloss_plot[:epoch-1])]).min():
        
            #saving the model weights
            print('Saving Model..')
            Path(path).mkdir(parents=True, exist_ok=True) #creates directory even when it exists
            for i in range(N_parallel):
                enc_mods[i].save(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5')
                dec_mods[i].save(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5')
                enc_mods[i].save_weights(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
                dec_mods[i].save_weights(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
            
            #saving optimizer parameters
            min_weights = optimizer.get_weights()
            hf = h5py.File(path + '/opt_weights.h5','w')
            for i in range(len(min_weights)):
                hf.create_dataset('weights_'+str(i),data=min_weights[i])
            hf.create_dataset('length', data=i)
            hf.create_dataset('l_rate', data=optimizer.learning_rate)  
            hf.close()
            
            last_save = epoch #store the last time the val loss has decreased for early stop

        # Print loss values and training time (per epoch)
        print('Epoch', epoch, '; Train_Loss', tloss_plot[epoch], 
              '; Val_Loss', vloss_plot[epoch],  '; Ratio', (vloss_plot[epoch])/(tloss_plot[epoch]))
        print('Time per epoch', (time.time()-t)/N_check)
        print('')
        
        t = time.time()
        
    if (epoch%20==0) and epoch != 0:    
        #plot convergence of training and validation loss (to visualise convergence during training)
        plt.title('MSE convergence')
        plt.yscale('log')
        plt.grid(True, axis="both", which='both', ls="-", alpha=0.3)
        plt.plot(tloss_plot[np.nonzero(tloss_plot)], 'y', label='Train loss')
        plt.plot(np.arange(np.nonzero(vloss_plot)[0].shape[0])*N_check,
                 vloss_plot[np.nonzero(vloss_plot)], label='Val loss')
        plt.xlabel('epochs')
        plt.legend()    
        plt.tight_layout()
        plt.savefig('/Kolmogorov/Images/' + 'epoch%i.png' % epoch)
        plt.close()



#save loss convergence plot
hf = h5py.File(path + '/loss_conv.h5','w')
hf.create_dataset('t_loss',  data=tloss_plot[np.nonzero(tloss_plot)])
hf.create_dataset('v_loss',  data=vloss_plot[np.nonzero(vloss_plot)]) 
hf.create_dataset('N_check', data=N_check)
hf.close()
            
plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams["font.size"]  = 20
        
#plot convergence of training and validation loss
plt.figure()
plt.title('Loss convergence')
plt.yscale('log')
plt.grid(True, axis="both", which='both', ls="-", alpha=0.3)
plt.plot(tloss_plot[np.nonzero(tloss_plot)], 'y', label='Train loss')
plt.plot(np.arange(np.nonzero(vloss_plot)[0].shape[0])*N_check,
         vloss_plot[np.nonzero(vloss_plot)], label='Val loss')
plt.xlabel('epochs')
plt.legend()
plt.tight_layout(pad=0.1)
plt.savefig('/Kolmogorov/Images/' + 'loss_convergence.png')
plt.close()


#### Visulaise Error ####
#load model for the test set
path = '/Kolmogorov/48_RE34_'+str(N_latent)
print(path)
# Load best model
#how to load saved model
a = [None]*N_parallel
b = [None]*N_parallel
for i in range(N_parallel):
    a[i] = tf.keras.models.load_model(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5', 
                                          custom_objects={"PerPad2D": PerPad2D})
for i in range(N_parallel):
    b[i] = tf.keras.models.load_model(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5',
                                          custom_objects={"PerPad2D": PerPad2D})
print('model loaded')
#grid
X       = np.linspace(0,2*np.pi,N_x) 
Y       = np.linspace(0,2*np.pi,N_y) 
XX      = np.meshgrid(X, Y, indexing='ij')

#plot n snapshots and their reconstruction in the test set.
n       = 5
plt.rcParams["figure.figsize"] = (15,4*n)
plt.rcParams["font.size"] = 20
fig, ax = plt.subplots(n,3)

start   = b_size*n_batches*skip+b_size*val_batches*skip #start after validation set

for i in range(n):
    
    #truth
    plt.subplot(n,3,i*3+1)
    
    skips = 50
    
    #snapshots to plot
    u      = U[start+500+i*skips:start+501+i*skips].copy()      
    vmax   = u.max()
    vmin   = u.min()

    CS0    = plt.contourf(XX[0], XX[1],u[0,:,:,0],
                          levels=10,cmap='coolwarm',vmin=vmin, vmax=vmax)
    cbar   = plt.colorbar()
    cbar.set_label('$u_{\mathrm{True}}$',labelpad=15)
    CS     = plt.contour(XX[0], XX[1],u[0,:,:,0],
                         levels=10,colors='black',linewidths=.5, linestyles='solid',
                         vmin=vmin, vmax=vmax)
    
    #autoencoded
    plt.subplot(n,3,i*3+2)

    u_dec  = model(u,a,b)[1][0].numpy()
    CS     = plt.contourf(XX[0],XX[1],u_dec[:,:,0],
                        levels=10,cmap='coolwarm',vmin=vmin, vmax=vmax)
    cbar   = plt.colorbar()
    cbar.set_label('$u_{\mathrm{Autoencoded}}$',labelpad=15)
    CS     = plt.contour(XX[0], XX[1],u_dec[:,:,0],
                         levels=10,colors='black',linewidths=.5, linestyles='solid',
                         vmin=vmin, vmax=vmax)
    
    #error
    plt.subplot(n,3,i*3+3)

    u_err  = np.abs(u_dec-u[0])/(vmax-vmin)
    print('NMAE: ', u_err[:,:,0].mean())

    CS     = plt.contourf(XX[0], XX[1],u_err[:,:,0],levels=10,cmap='coolwarm')
    cbar   = plt.colorbar()
    cbar.set_label('Relative Error',labelpad=15)
    CS     = plt.contour(XX[0], XX[1],u_err[:,:,0],levels=10,colors='black',linewidths=.5, 
                         linestyles='solid')

fig.tight_layout(pad=0.1)
fig.savefig('Kolmogorov/Images/'+'/Autoencoder_error.png')


#### Save encodeed data for training ####
# save the encoded data for the ESN (too much memory used for GPU)
N_pos     = 5000 #split in k interval of N_pos length needed to process long timeseries
k         = 75
transient = 10000
N_len = k*N_pos
fln      = '../data/Kolmogorov/Kolmogorov_0.1_48_30.0_100100_32.h5'
hf       = h5py.File(fln,'r')
dt       = 0.1
U        = np.array(hf.get('U')[transient:transient+N_len], dtype=np.float32)
hf.close()

N_x      = U.shape[1]
N_y      = U.shape[2]

atents  = [18]
Re       = 30

for N_latent in Latents:
    path = '/Kolmogorov/48_RE30_'+str(N_latent)
    a = [None]*N_parallel
    b = [None]*N_parallel
    for i in range(N_parallel):
        a[i] = tf.keras.models.load_model(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5', 
                                              custom_objects={"PerPad2D": PerPad2D})
    for i in range(N_parallel):
        b[i] = tf.keras.models.load_model(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5',
                                              custom_objects={"PerPad2D": PerPad2D})

    N_1   = [3,3,N_latent//9]
    U_enc = np.zeros((N_len, N_1[0], N_1[1], N_1[2]))
    #encode all the data to provide time series in latent space for the ESN
    for i in range(k):
        U_enc[i*N_pos:(i+1)*N_pos]= model(U[i*N_pos:(i+1)*N_pos], a, b)[0]

    fln = '../data/Kolmogorov/48_Encoded_data_Re30_' \
                + str(N_latent) +'.h5'
    hf = h5py.File(fln,'w')
    hf.create_dataset('U_enc'      ,data=U_enc)  
    hf.close()
    print(fln)

def gradient(U,dx,dy,n_splits):
    '''Returns dissipation of U, done in n_splits'''
    
    shapes    = np.array(U.shape)
    shapes[0] = shapes[0]//n_splits
    dU_dx     = np.empty(shapes)
    dU_dy     = np.empty(shapes)
    D         = np.empty(shapes[0]*n_splits)
    
    for i in np.arange(n_splits):
        
        for j in range(shapes[1]):
            dU_dx[:,j] = (U[i*shapes[0]:(i+1)*shapes[0],(j+1)%shapes[1]] - \
                          U[i*shapes[0]:(i+1)*shapes[0],j-1])/(2*dx)
        for k in range(shapes[2]):
            dU_dy[:,:,k] = (U[i*shapes[0]:(i+1)*shapes[0],:,(k+1)%shapes[2]] - \
                            U[i*shapes[0]:(i+1)*shapes[0],:,k-1])/(2*dy)
            
        D[i*shapes[0]:(i+1)*shapes[0]] =  np.mean(dU_dx**2+dU_dy**2,
                                            axis=(1,2,3))/Re*4
              
    return D
        
plt.rcParams["figure.figsize"] = (15,4)
plt.rcParams["font.size"] = 20

#plot average dissipation rate
plt.subplot(121)
leng      = 5000
dx        = 2*np.pi/(N_x-1)
DD        = gradient(U[-leng:],dx,dx,1) #true
U_dec     = model(U[-leng:], a, b)[1]
DD_enc    = gradient(U_dec,dx,dx,1) #autoencoded
plt.plot(DD,'w')
plt.plot(DD_enc,'r--')

#plot error
plt.subplot(122)
plt.plot(np.abs(DD_enc-DD)/(DD.max()-DD.min()))
plt.tight_layout()
plt.savefig('/Kolmogorov/Images/' + 'image.png')
print(np.mean(np.abs(DD_enc-DD)/(DD.max()-DD.min())))

del U_vv, U_tt

