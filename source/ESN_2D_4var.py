print('load packages')
import os
os.environ["OMP_NUM_THREADS"] = '1' # imposes cores
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
from scipy.io import loadmat, savemat
import time as time
from skopt.plots import plot_convergence
from sklearn.preprocessing import StandardScaler
import sys
sys.stdout.reconfigure(line_buffering=True)
import json

input_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/input_data/'
output_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/'

#global tikh_opt, k, ti, tikh, U_washout, U_tv, Y_tv, U, N_in, N_fw, N_washout, N_val, N_units

exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('run functions files')

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
    
def calculate_global_q_KE(data):
    avgq = np.mean(data[:,:,:,0], axis=(1,2))
    ke = 0.5*data[:,:,:,1]*data[:,:,:,1]
    avgke = np.mean(ke, axis=(1,2))
    globalvar = np.zeros((data.shape[0], 2))
    globalvar[:,0] = avgke
    globalvar[:,1] = avgq
    return globalvar
    
def NRMSE(original_data, reconstructed_data):
    rmse = np.sqrt(mean_squared_error(original_data, reconstructed_data))
    
    variance = np.var(original_data)
    std_dev  = np.sqrt(variance)
    
    nrmse = (rmse/std_dev)
    
    return nrmse

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

    print('shape of mask in function', np.shape(mask))
    
    # Expand the mask to cover all features (optional, depending on use case)
    mask_expanded       =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
    mask_expanded_recon =  np.repeat(mask_reconstructed[:, :, :, np.newaxis], 4, axis=-1) # Shape: (256, 64, 1)
    
    return active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon

#### load Data ####
#### larger data 5000-30000 hf ####
num_snapshots = 5001
POD_snapshots = 500
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
variables = num_variables = 4
variable_names = ['q', 'w', 'u', 'b']
scaling = 'scaled'
POD = 'together'

#### larger data 5000-30000 hf ####
total_num_snapshots = num_snapshots #snap
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

data_all         = np.concatenate((q_array, w_array, u_array, b_array), axis=-1)
data_all_reshape = data_all.reshape(len(time_vals), len(x) * len(z) * variables)

data_for_POD = data_all[:POD_snapshots]
data_for_POD_reshape = data_for_POD.reshape(POD_snapshots, len(x) * len(z) * variables)
time_vals_for_POD = time_vals[:POD_snapshots]
new_data_snapshots = total_num_snapshots-POD_snapshots
new_data = data_all[POD_snapshots:]
new_data_reshape = new_data.reshape(new_data_snapshots, len(x) * len(z) * variables)
time_vals_new = time_vals[POD_snapshots:]

### scale data ###
# Placeholder for standardized data
data_for_POD_scaled = np.zeros_like(data_for_POD)
new_data_scaled = np.zeros_like(new_data)
scalers = [StandardScaler() for _ in range(variables)]

# Apply StandardScaler separately to each channel
for v in range(variables):
    # Reshape training data for the current variable
    reshaped_POD_channel = data_for_POD[:, :, :, v].reshape(-1, data_for_POD.shape[1] * data_for_POD.shape[2])

    # Fit the scaler on the training data for the current variable
    scaler = scalers[v]
    scaler.fit(reshaped_POD_channel)

    # Standardize the training data
    standardized_POD_channel = scaler.transform(reshaped_POD_channel)

    # Reshape the standardized data back to the original shape (batches, batch_size, x, z)
    data_for_POD_scaled[:, :, :, v] = standardized_POD_channel.reshape(data_for_POD.shape[0], data_for_POD.shape[1], data_for_POD.shape[2])
    
    # Standardize the new data using the same scaler
    reshaped_new_channel = new_data[:, :, :, v].reshape(-1, new_data.shape[1] * new_data.shape[2])
    standardized_new_channel = scaler.transform(reshaped_new_channel)
    new_data_scaled[:, :, :, v] = standardized_new_channel.reshape(new_data.shape[0], new_data.shape[1], new_data.shape[2])

# Print the shape of the combined array
print('shape of all data and POD data after scaling:', data_all.shape, data_for_POD_scaled.shape)

#### global ####
groundtruth_global = calculate_global_q_KE(data_all)

### reshape data and run POD ###
data_matrix = data_for_POD_scaled.reshape(POD_snapshots, -1) ##data_scaled originally
print('shape of matirx for POD', np.shape(data_matrix)) 
data_matrix_new = new_data_scaled.reshape(new_data_snapshots, -1) #

Plotting = True
Projection = True

### generate directory to save images ###
n_components=128
snapshots_path = '/POD_ESN_snapshots%i_%imodes' % (POD_snapshots, n_components[0])
output_path = output_path1+snapshots_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

from sklearn.decomposition import PCA
pca = PCA(n_components=c, svd_solver='randomized', random_state=42)
pca.fit(data_matrix)
data_reduced = pca.transform(data_matrix)  # (5000, n_modes)
data_reconstructed_reshaped = pca.inverse_transform(data_reduced)  # (5000, 256 * 2)
data_reconstructed = data_reconstructed_reshaped.reshape(POD_snapshots, 256, 64, num_variables)  # (5000, 256, 1, 2)
if scaling == 'scaled':        
    data_reconstructed_unscaled = np.zeros_like(data_for_POD)
    for v in range(variables):
        reshaped_channel_reconstructed_POD = data_reconstructed[:, :, :, v].reshape(-1, data_for_POD.shape[1] * data_for_POD.shape[2])
        unscaled_POD_channel = scalers[v].inverse_transform(reshaped_channel_reconstructed_POD)
        data_reconstructed_unscaled[:, :, :, v] = unscaled_POD_channel.reshape(data_for_POD.shape[0], data_for_POD.shape[1], data_for_POD.shape[2])

    data_reconstructed_reshaped_unscaled = data_reconstructed_unscaled.reshape(data_for_POD.shape[0], len(x)*len(z)*variables)
components = pca.components_

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
# add to list
cumulative_explained_variance_values[c_index] = cumulative_explained_variance[-1]
print('cumulative explained variance for', c, 'components is', cumulative_explained_variance[-1])


#### PROJECTION ####
data_reduced_projection             = pca.transform(data_matrix_new)
new_reconstructed_reshaped          = pca.inverse_transform(data_reduced_projection)
new_reconstructed                   = new_reconstructed_reshaped.reshape(new_data_snapshots, 256, 64, num_variables)  # (5000, 256, 1, 2)

if scaling == 'scaled':        
    data_reconstructed_unscaled = np.zeros_like(new_data)
    for v in range(variables):
        reshaped_channel_reconstructed_new = new_reconstructed[:, :, :, v].reshape(-1, new_data.shape[1] * new_data.shape[2])
        unscaled_new_channel = scalers[v].inverse_transform(reshaped_channel_reconstructed_new)
        data_reconstructed_unscaled[:, :, :, v] = unscaled_new_channel.reshape(new_data.shape[0], new_data.shape[1], new_data.shape[2])

    data_reconstructed_reshaped_unscaled = data_reconstructed_unscaled.reshape(new_data.shape[0], len(x)*len(z)*variables)

print('projected data', np.shape(data_reconstructed_unscaled))
print('projected data true', np.shape(new_data))

#### visualise reconstruction ####
if Plotting:
    for var in range(num_variables):
        fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
        minm = min(np.min(new_data[:, :, 32, var]), np.min(data_reconstructed_unscaled[:, :, 32, var]))
        maxm = max(np.max(new_data[:, :, 32, var]), np.max(data_reconstructed_unscaled[:, :, 32, var]))
        c1 = ax[0].pcolormesh(time_vals_new, x, new_data[:,:, 32, var].T, vmin=np.min(new_data[:, :, 32, var]), vmax=np.max(new_data[:, :, 32, var]))
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('true')
        c2 = ax[1].pcolormesh(time_vals_new, x, data_reconstructed_unscaled[:,:,32,var].T, vmin=np.min(new_data[:, :, 32, var]), vmax=np.max(new_data[:, :, 32, var]))
        fig.colorbar(c1, ax=ax[1])
        ax[1].set_title('reconstruction')
        for v in range(2):
            ax[v].set_xlabel('time')
            ax[v].set_ylabel('x')
            ax[v].set_xlim(time_vals_new[0], time_vals_new[499])
        fig.savefig(output_path+'/reconstruction_modes_new_set_c%i_snapshots%i_var%i.png' % (c, snap, var))
        plt.close()

# reconstuct global
if Plotting:
    # reconstuct global
    reconstructed_global_new = calculate_global_q_KE(data_reconstructed_unscaled)
    fig, ax = plt.subplots(2,1, figsize=(12,6), tight_layout=True, sharex=True)
    for i in range(2):
        ax[i].plot(time_vals_new, groundtruth_global[POD_snapshots:,i], color='tab:blue', label='truth')
        ax[i].plot(time_vals_new, reconstructed_global_new[:,i], color='tab:orange', label='reconstruction')
        ax[i].grid()
    ax[0].set_ylabel('KE')
    ax[1].set_ylabel('q')
    ax[1].set_xlabel('time')
    ax[1].legend()
    fig.savefig(output_path+'/global_reconstruction_new_set_c%i_snapshots%i.png' % (c, snap))

#### dataset generation ####
U = data_reduced_projection 
print('shape of data for ESN', np.shape(U))

# number of time steps for washout, train, validation, test
N_lyap    = 200
N_washout = 200
N_train   = 8*N_lyap
N_val     = 2*N_lyap
N_test    = 2*N_lyap
dim       = n_components

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

indexes_to_plot = np.array([1, 2, 10, 50, 100] ) -1
indexes_to_plot = indexes_to_plot[indexes_to_plot < n_components]
if n_components < 11:
    indexes_to_plot = np.array([1, 2, 5, 10] ) -1

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
    sigma_n = 1e-3     #change this to increase/decrease noise in training inputs (up to 1e-1)
    for i in range(n_components):
        U_tv[:,i] = U_tv[:,i] \
                        + rnd1.normal(0, sigma_n*data_std[i], N_train-1)
    for m in range(len(indexes_to_plot)):
        index = indexes_to_plot[m]
        ax[m].plot(U_tv[:N_val,index], 'r--', label='Noisy')
        ax[m].grid()
        ax[m].set_title('mode %i' % (index+1))
ax[0].legend()
fig.savefig(output_path + '/noise_addition.png')
plt.close()

#### ESN hyperparameters #####
if normalisation == 'on':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm))]) #input bias (average absolute value of the inputs)
elif normalisation == 'off':
    bias_in   = np.array([np.mean(np.abs(U_data))]) #input bias (average absolute value of the inputs)
bias_out  = np.array([1.]) #output bias

N_units      = 500 #neurons
connectivity = 3
sparseness   = 1 - connectivity/(N_units-1)

tikh = np.array([1e-2,1e-3,1e-6,1e-9,1e-12])  # Tikhonov factor (optimize among the values in this list)

print('tikh:', tikh)
print('N_r:', N_units, 'sparsity:', sparseness)
print('bias_in:', bias_in, 'bias_out:', bias_out)

#### Grid Search and BO #####
n_in  = 0           #Number of Initial random points

spec_in     = .1    #range for hyperparameters (spectral radius and input scaling)
spec_end    = 1.   
in_scal_in  = np.log10(0.05)
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

print('initial grid', x1)

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
                      acq_func             = "gp_hedge",       # the acquisition function
                      n_calls              = n_tot,      # total number of evaluations of f
                      x0                   = x1,         # Initial grid search points to be evaluated at
                      n_random_starts      = n_in,       # the number of additional random initialization points
                      n_restarts_optimizer = 3,          # number of tries for each acquisition
                      random_state         = 10,         # seed
                           )   
    return res

print(search_space)

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

##### visualise grid search #####
# Plot Gaussian Process reconstruction for each network in the ensemble after n_tot evaluations
# The GP reconstruction is based on the n_tot function evaluations decided in the search

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

##### quick test #####
N_test   = 1                    #number of intervals in the test set
N_tstart = N_washout + N_train     #where the first test interval starts
N_intt   = 2*N_lyap                #length of each test set interval

# #prediction horizon normalization factor and threshold
sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
threshold_ph = 0.2

ensemble_test = 1

ens_pred = np.zeros((N_intt, dim, ensemble_test))
ens_PH = np.zeros((N_test, ensemble_test))
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
    PH       = np.zeros(N_test)

    # to plot results
    plot = True
    if plot:
        n_plot = 2
        plt.rcParams["figure.figsize"] = (15,3*n_plot)
        plt.figure()
        plt.tight_layout()

    #run different test intervals
    for i in range(N_test):
        print(N_tstart + i*N_intt)
        # data for washout and target in each interval
        U_wash    = U[N_tstart - N_washout +i*N_intt : N_tstart + i*N_intt].copy()
        Y_t       = U[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt].copy()

        #washout for each interval
        Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
        Uh_wash = np.dot(Xa1, Wout)

        # Prediction Horizon
        Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
        print(np.shape(Yh_t))
        if i == 0:
            ens_pred[:, :, j] = Yh_t
        Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
        PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
        if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)

        if plot:
            #left column has the washout (open-loop) and right column the prediction (closed-loop)
            # only first n_plot test set intervals are plotted
            if i<n_plot:
                if ensemble_test % 1 == 0:
                
                    #### modes prediction ####
                    fig,ax =plt.subplots(len(indexes_to_plot),sharex=True, tight_layout=True)
                    xx = np.arange(U_wash[:,-2].shape[0])/N_lyap
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

                    ##### reconstructions ####
                    reconstructed_predictions_reshaped = pca.inverse_transform(Yh_t)
                    reconstructed_predictions_scaled   = reconstructed_predictions_reshaped.reshape(new_data_snapshots, 256, 64, num_variables)  # (5000, 256, 1, 2)

                    POD_reconstruction_reshaped = pca.inverse_transform(Y_t)
                    reconstructed_POD_scaled    = POD_reconstruction_reshaped.reshape(new_data_snapshots, 256, 64, num_variables)  # (5000, 256, 1, 2)

                    # EVR for reconstruction of POD data
                    numerator = np.sum((POD_reconstruction_reshaped - reconstructed_predictions_scaled) **2)
                    denominator = np.sum(POD_reconstruction_reshaped ** 2)
                    evr_reconstruction = 1 - (numerator/denominator)
                    
                    #MSE for reconstruction of POD data
                    mse = mean_squared_error(POD_reconstruction_reshaped, reconstructed_predictions_scaled)
                    
                    #NRMSE
                    nrmse = NRMSE(POD_reconstruction_reshaped, reconstructed_predictions_scaled)

                    reconstructed_predictions = np.zeros_like(new_data)
                    reconstructed_POD = np.zeros_like(new_data)
                    for v in range(variables):
                        reshaped_channel_reconstructed_prediction = reconstructed_predictions_scaled[:, :, :, v].reshape(-1, new_data.shape[1] * new_data.shape[2])
                        unscaled_new_channel = scalers[v].inverse_transform(reshaped_channel_reconstructed_prediction)
                        reconstructed_predictions[:, :, :, v] = unscaled_new_channel.reshape(new_data.shape[0], new_data.shape[1], new_data.shape[2])

                        reshaped_channel_reconstructed_POD = rreconstructed_POD_scaled[:, :, :, v].reshape(-1, new_data.shape[1] * new_data.shape[2])
                        unscaled_new_channel = scalers[v].inverse_transform(reshaped_channel_reconstructed_POD)
                        reconstructed_POD[:, :, :, v] = unscaled_new_channel.reshape(new_data.shape[0], new_data.shape[1], new_data.shape[2])

                    reconstructed_predictions_reshaped_unscaled = reconstructed_predictions.reshape(new_data.shape[0], len(x)*len(z)*variables)
                    reconstructed_POD_reshaped_unscaled = reconstructed_POD.reshape(new_data.shape[0], len(x)*len(z)*variables)

                    for v in range(variables):
                        fig, ax = plt.subplots(2, figsize=(12,6), sharex=True, tight_layout=True)
                        minm = min(np.min(reconstructed_POD[:, :, 32, v]), np.min(reconstructed_predictions[:, :, 32, v]))
                        maxm = max(np.max(reconstructed_POD[:, :, 32, v]), np.max(reconstructed_predictions[:, :, 32, v]))
                        c1 = ax[0].contourf(xx, x, reconstructed_POD[:,:,32,v].T, vmin=minm, vmax=maxm)
                        c2 = ax[1].contourf(xx, x, reconstructed_predictions[:,:,32,v].T, vmin=minm, vmax=maxm)
                        norm1 = mcolors.Normalize(vmin=minm, vmax=maxm)
                        fig.colorbar(c1, ax=ax[0], label='POD', norm=norm1, extend='both')
                        fig.colorbar(c2, ax=ax[1], label='ESN', norm=norm1, extend='both')
                        ax[1].set_xlabel('Time [Lyapunov Times]')
                        ax[0].set_ylabel('x')
                        ax[1].set_ylabel('x')
                        fig.savefig(output_path+'/prediction_recon_ens%i_test%i_var%i.png' % (j,i,v))
                        plt.close()


                    #### globals #####
                    reconstructed_global_new = calculate_global_q_KE(reconstructed_predictions)
                    fig, ax = plt.subplots(2,1, figsize=(12,6), tight_layout=True, sharex=True)
                    for i in range(2):
                        ax[i].plot(time_vals_new, groundtruth_global[POD_snapshots:,i], color='tab:blue', label='truth')
                        ax[i].plot(time_vals_new, reconstructed_global_new[:,i], color='tab:orange', label='reconstruction')
                        ax[i].grid()
                    ax[0].set_ylabel('KE')
                    ax[1].set_ylabel('q')
                    ax[1].set_xlabel('time')
                    ax[1].legend()
                    fig.savefig(output_path+'/global_reconstruction_new_set_c%i_snapshots%i.png' % (c, snap))
                    # plumes 
                    active_array, active_array_reconstructed, mask, mask_reconstructed = active_array_calc(new_data, reconstructed_predictions, z)
                    print(np.shape(active_array))
                    print(np.shape(mask))
                    
                    if Plotting:
                        fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
                        c1 = ax[0].contourf(time_vals_new, x, active_array[:,:, 32].T, cmap='Reds')
                        fig.colorbar(c1, ax=ax[0])
                        ax[0].set_title('true')
                        c2 = ax[1].contourf(time_vals_new, x, active_array_reconstructed[:,:, 32].T, cmap='Reds')
                        fig.colorbar(c1, ax=ax[1])
                        ax[1].set_title('reconstruction')
                        for v in range(2):
                            ax[v].set_xlabel('time')
                            ax[v].set_ylabel('x')
                            ax[v].set_xlim(time_vals_new[0], time_vals_new[499])
                        fig.savefig(output_path+'/active_plumes_new_set_c%i_snapshots%i.png' % (c, snap))
                        plt.close()
                    
                    accuracy = np.mean(active_array == active_array_reconstructed)
                    mse_plume = mean_squared_error(new_data[:,:,:,:][mask], new_reconstructed[:,:,:,:][mask])
                    MSE_plume_new[c_index] = mse_plume
                    nrmse_plume = NRMSE(new_data[:,:,:,:][mask], new_reconstructed[:,:,:,:][mask])
                    
                    #SSIM
                    ssim_value = compute_ssim_for_4d(new_data, data_reconstructed_unscaled)
                   

                    metrics = {
                    "test": i,
                    "N_tstart + i*N_intt": N_tstart + i*N_intt,
                    "PH": PH[i],
                    "EVR": evr_reconstruction,
                    "mse": mse,
                    "nrmse": nrmse,
                    "accuracy": accuracy,
                    "MSE_plume": mse_plume,
                    "NRMSE_plume": nrmse_plume,
                    "test_ssim": ssim_value,
                    }
                    
                    metrics_file = "metrics.json"
                    metrics_path = os.path.join(output_path, metrics_file)
                    with open(metrics_path, "w") as file:
                        json.dump(metrics, file, indent=4)


    # Percentiles of the prediction horizon
    print('PH quantiles [Lyapunov Times]:',
          np.quantile(PH,.75), np.median(PH), np.quantile(PH,.25))
    ens_PH[:,j] = PH
    print('')



###### SAVE RESULTS ######
#Save the details and results of the search for post-process
opt_specs = [spec_in,spec_end,in_scal_in,in_scal_end]

fln = output_path+'/ESN_' + val.__name__ + str(N_units) +'.mat'
with open(fln,'wb') as f:  # need 'wb' in Python3
    savemat(f, {"norm": norm})
    savemat(f, {"fix_hyp": np.array([bias_in[0], N_washout],dtype='float64')})
    savemat(f, {'opt_hyp': np.column_stack((minimum[:,0], 10**minimum[:,1]))})
    savemat(f, {"Win": Winn})
    savemat(f, {'W': Ws})
    savemat(f, {"Wout": Woutt})

# to load in another file
data = loadmat(fln)
Win  = data['Win'][0] #gives Winn


xx = np.arange(Y_tv[:,-2].shape[0])/N_lyap
for i in range(ensemble):
    Yh      = np.empty((N_train-1, 2))
    xa      = Xa1_states[i]
    Wout    = Woutt[i]
    Yh_test = np.dot(xa, Wout)
    fig,ax =plt.subplots(len(indexes_to_plot),sharex=True)
    for v in range(len(indexes_to_plot)):
        index = indexes_to_plot[v]
        ax[v].plot(xx,Y_tv[:,index], 'b')
        ax[v].plot(xx,Yh_test[:,index], '--r')
        ax[v].grid()
        ax[v].set_ylabel(index)
    ax[-1].set_xlabel('Time [Lyapunov Times]')
    ax[1].set_xlim(0,5)
    fig.savefig(output_path+'/training_states_ens%i.png' % i)




