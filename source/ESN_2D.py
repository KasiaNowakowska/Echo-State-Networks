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

input_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/input_data/'
output_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/'

#global tikh_opt, k, ti, tikh, U_washout, U_tv, Y_tv, U, N_in, N_fw, N_washout, N_val, N_units

exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('run functions files')

#### Load Data ####
#### larger data 5000-30000 hf ####
total_num_snapshots = 2500
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
variables = num_variables = 2
variable_names = ['q', 'w']
scaling = 'minus_mean'
POD = 'together'

if scaling == 'scaled':
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

if scaling == 'minus_mean':
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

        q_scaled = (q - q_mean) 
        w_scaled = (w - w_mean) 
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

POD_num_snapshots = 2500
data_for_POD = data_scaled[:POD_num_snapshots]
print(data_for_POD.shape)

#### global ####
groundtruth_avgq = np.mean(data_all[:,:,:,0], axis=(1,2))
groundtruth_ke = 0.5*data_all[:,:,:,1]*data_all[:,:,:,1]
groundtruth_avgke = np.mean(groundtruth_ke, axis=(1,2))
groundtruth_global = np.zeros((total_num_snapshots,2))
groundtruth_global[:,0] = groundtruth_avgke
groundtruth_global[:,1] = groundtruth_avgq

### reshape data and run POD ###
data_matrix = data_for_POD.reshape(POD_num_snapshots, -1)
print(np.shape(data_matrix))

n_components=10
if POD == 'together':
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    data_reduced = pca.fit_transform(data_matrix)  # (5000, n_modes)
    data_reconstructed_reshaped = pca.inverse_transform(data_reduced)  # (5000, 256 * 2)
    #data_reconstructed_reshaped = ss.inverse_transform(data_reconstructed_reshaped)
    data_reconstructed = data_reconstructed_reshaped.reshape(POD_num_snapshots, 256, 64, num_variables)  # (5000, 256, 1, 2)
    components = pca.components_
    print('shape of coefficients:', np.shape(data_reduced))
    #### unscale the data ####
    if scaling == 'scaled':
        data_reconstructed[:,:,:,0] = data_reconstructed[:,:,:,0] * q_std + q_mean
        data_reconstructed[:,:,:,1] = data_reconstructed[:,:,:,1] * w_std + w_mean
    elif scaling == 'minus_mean':
        data_reconstructed[:,:,:,0] = data_reconstructed[:,:,:,0] + q_mean
        data_reconstructed[:,:,:,1] = data_reconstructed[:,:,:,1] + w_mean

if POD == 'seperate':
    from sklearn.decomposition import PCA
    data_reconstructed = np.zeros((POD_num_snapshots, len(x), len(z), variables))
    data_reduced = np.zeros((POD_num_snapshots, n_components, variables))
    pca_list = []
    for v in range(variables):
        data_var = data_for_POD[:,:,:, v]
        data_var_matrix = data_var.reshape(POD_num_snapshots, -1)
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        pca_list.append(pca)
        var_reduced = pca.fit_transform(data_var_matrix)
        print(np.shape(var_reduced))
        var_reconstructed_reshaped = pca.inverse_transform(var_reduced)  # (5000, 256 * 2)
        var_reconstructed = var_reconstructed_reshaped.reshape(POD_num_snapshots, 256, 64)  # (5000, 256, 1, 2)
        data_reconstructed[:,:,:,v] = var_reconstructed
        data_reduced[:,:,v] = var_reduced
        components = pca.components_
    #### unscale the data ####
    if scaling == 'scaled':
        data_reconstructed[:,:,:,0] = data_reconstructed[:,:,:,0] * q_std + q_mean
        data_reconstructed[:,:,:,1] = data_reconstructed[:,:,:,1] * w_std + w_mean
    elif scaling == 'minus_mean':
        data_reconstructed[:,:,:,0] = data_reconstructed[:,:,:,0] + q_mean
        data_reconstructed[:,:,:,1] = data_reconstructed[:,:,:,1] + w_mean
    #### reshape data_reduced for ESN ####
    data_reduced = data_reduced.reshape(POD_num_snapshots, -1)
    print('data_reduced_ESN shape', np.shape(data_reduced))

data_dir = '/POD_ESN_scaling{0:}_POD{1:}_n_snapshots{2:}_n_modes{3:}/'.format(scaling, POD, POD_num_snapshots, n_components)
output_path = output_path+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')


#### visualise reconstruction ####
for var in range(2):
    fig, ax = plt.subplots(2, figsize=(12,6), tight_layout=True)
    minm = min(np.min(data_all[:POD_num_snapshots, :, 32, var]), np.min(data_reconstructed[:, :, 32, var]))
    maxm = max(np.max(data_all[:POD_num_snapshots, :, 32, var]), np.max(data_reconstructed[:, :, 32, var]))
    c1 = ax[0].contourf(time_vals[:POD_num_snapshots], x, data_all[:POD_num_snapshots,:, 32, var].T, vmin=minm, vmax=maxm)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('true')
    c2 = ax[1].contourf(time_vals[:POD_num_snapshots], x, data_reconstructed[:,:,32,var].T, vmin=minm, vmax=maxm)
    fig.colorbar(c1, ax=ax[1])
    ax[1].set_title('reconstruction')
    for v in range(variables):
        ax[v].set_xlabel('time')
        ax[v].set_ylabel('x')
        ax[v].set_xlim(5000, 6000)
    fig.savefig(output_path+'/reconstruction_var%i.png' % var)
    plt.close()

# reconstuct global
reconstructed_groundtruth_avgq = np.mean(data_reconstructed[:,:,:,0], axis=(1,2))
reconstructed_groundtruth_ke = 0.5*data_reconstructed[:,:,:,1]*data_reconstructed[:,:,:,1]
reconstructed_groundtruth_avgke = np.mean(reconstructed_groundtruth_ke, axis=(1,2))
reconstructed_groundtruth_global = np.zeros((POD_num_snapshots,2))
reconstructed_groundtruth_global[:,0] = reconstructed_groundtruth_avgke
reconstructed_groundtruth_global[:,1] = reconstructed_groundtruth_avgq
print(np.shape(reconstructed_groundtruth_global))
fig, ax = plt.subplots(2,1, figsize=(12,6), tight_layout=True, sharex=True)
for i in range(2):
    ax[i].plot(time_vals[:POD_num_snapshots], groundtruth_global[:POD_num_snapshots,i], color='tab:blue', label='truth')
    ax[i].plot(time_vals[:POD_num_snapshots], reconstructed_groundtruth_global[:,i], color='tab:orange', label='reconstruction')
    ax[i].grid()
ax[0].set_ylabel('KE')
ax[1].set_ylabel('q')
ax[1].set_xlabel('time')
ax[1].legend()
fig.savefig(output_path+'/global_reconstruction.png')

'''
remaining_num_snapshots = 1000 #total_num_snapshots - POD_num_snapshots
data_remaining = data_scaled[POD_num_snapshots:POD_num_snapshots+remaining_num_snapshots]

if POD == 'together':
    data_remaining_matrix = data_remaining.reshape(remaining_num_snapshots, -1)
    print('shape of data_remaining', np.shape(data_remaining_matrix))
    data_remaining_reduced = np.dot(data_remaining_matrix, components.T)

    data_remaining_reconstructed_reshaped = np.dot(data_remaining_reduced, pca.components_)
    data_remaining_reconstructed = data_remaining_reconstructed_reshaped.reshape(remaining_num_snapshots, len(x), len(z), num_variables)

if POD == 'seperate':
    data_remaining_reconstructed = np.zeros((remaining_num_snapshots, len(x), len(z), variables))
    data_remaining_reduced = np.zeros((remaining_num_snapshots, n_components, variables))
    for v in range(variables):
        data_var = data_remaining[:,:,:, v]
        var_remaining_matrix = data_var.reshape(remaining_num_snapshots, -1)
        print('shape of data_remaining', np.shape(var_remaining_matrix))
        var_remaining_reduced = np.dot(var_remaining_matrix, components.T)

        var_remaining_reconstructed_reshaped = np.dot(var_remaining_reduced, pca.components_)
        var_remaining_reconstructed = var_remaining_reconstructed_reshaped.reshape(remaining_num_snapshots, len(x), len(z))
        data_remaining_reduced[:,:,v] = var_remaining_reduced
        data_remaining_reconstructed[:,:,:,v] = var_remaining_reconstructed

#### unscale the data ####
if scaling == 'scaled':
    data_remaining_reconstructed[:,:,:,0] = data_remaining_reconstructed[:,:,:,0] * q_std + q_mean
    data_remaining_reconstructed[:,:,:,1] = data_remaining_reconstructed[:,:,:,1] * w_std + w_mean
elif scaling == 'minus_mean':
    data_remaining_reconstructed[:,:,:,0] = data_remaining_reconstructed[:,:,:,0] + q_mean
    data_remaining_reconstructed[:,:,:,1] = data_remaining_reconstructed[:,:,:,1] + w_mean

#### visualise reconstruction ####
for var in range(2):
    fig, ax = plt.subplots(2, figsize=(12,6), tight_layout=True)
    minm = min(np.min(data_all[POD_num_snapshots:POD_num_snapshots+remaining_num_snapshots, :, 32, var]), np.min(data_remaining_reconstructed[:, :, 32, var]))
    maxm = max(np.max(data_all[POD_num_snapshots:POD_num_snapshots+remaining_num_snapshots, :, 32, var]), np.max(data_remaining_reconstructed[:, :, 32, var]))
    c1 = ax[0].contourf(time_vals[POD_num_snapshots:POD_num_snapshots+remaining_num_snapshots], x, data_all[POD_num_snapshots:POD_num_snapshots+remaining_num_snapshots,:, 32, var].T, vmin=minm, vmax=maxm)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('true')
    c2 = ax[1].contourf(time_vals[POD_num_snapshots:POD_num_snapshots+remaining_num_snapshots], x, data_remaining_reconstructed[:,:,32,var].T, vmin=minm, vmax=maxm)
    fig.colorbar(c2, ax=ax[1])
    ax[1].set_title('reconstruction')
    for v in range(variables):
        ax[v].set_xlabel('time')
        ax[v].set_ylabel('x')
    #ax[0].set_xlim(8000,9000)
    #ax[1].set_xlim(8000,9000)
    fig.savefig(output_path+'/reconstruction_remaining_var%i.png' % var)
    plt.close()

'''
#### dataset generation ####
U = data_reduced
print('shape of data for ESN', np.shape(U))

# number of time steps for washout, train, validation, test
N_lyap    = 200
N_washout = 200
N_train   = 8*N_lyap
N_val     = 2*N_lyap
N_test    = 2*N_lyap
if POD == 'together':
    dim       = n_components
elif POD == 'seperate':
    dim       = n_components*2

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)
normalisation = 'off'

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
    for i in range(dim):
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
bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm))]) #input bias (average absolute value of the inputs)
bias_out  = np.array([1.]) #output bias

N_units      = 8000 #neurons
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
n_grid_x = 6  
n_grid_y = 6
n_bo     = 4  #number of points to be acquired through BO after grid search
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

print(search_space)

#Number of Networks in the ensemble
ensemble = 1

data_dir = '/Run_n_units{0:}_ensemble{1:}/'.format(N_units, ensemble)
output_path = output_path+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

# Which validation strategy (implemented in Val_Functions.ipynb)
val      = RVC_Noise
N_fo     = 5                        # number of validation intervals
N_in     = N_washout                 # timesteps before the first validation interval (can't be 0 due to implementation)
N_fw     = (N_train-N_val)//(N_fo-1) # how many steps forward the validation interval is shifted (in this way they are evenly spaced)
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

                    #reconstructions
                    if POD == 'together':
	                    reconstructed_predictions = pca.inverse_transform(Yh_t)
	                    reconstructed_predictions = reconstructed_predictions.reshape(N_intt , len(x), len(z), variables)  # (5000, 256, 1, 2)

	                    POD_reconstruction = pca.inverse_transform(Y_t)
	                    POD_reconstruction = POD_reconstruction.reshape(N_intt, len(x), len(z), variables)
                    elif POD == 'seperate':
                        reconstructed_predictions = np.zeros((N_intt, len(x), len(z), variables))
                        POD_reconstruction = np.zeros((N_intt, len(x), len(z), variables))
                        for v in range(variables):
                            print(n_components*v, n_components*(v+1))
                            pca = pca_list[v]
                            reconstructed_predictions_var = pca.inverse_transform(Yh_t[:,n_components*v:n_components*(v+1)])
                            reconstructed_predictions_var = reconstructed_predictions_var.reshape(N_intt , len(x), len(z))  # (5000, 256, 1, 2)
                            reconstructed_predictions[:,:,:,v] = reconstructed_predictions_var

                            POD_reconstruction_var = pca.inverse_transform(Y_t[:,n_components*v:n_components*(v+1)])
                            POD_reconstruction_var = POD_reconstruction_var.reshape(N_intt, len(x), len(z))
                            POD_reconstruction[:,:,:,v] = POD_reconstruction_var

                    if scaling == 'minus_mean':
                        reconstructed_predictions[:,:,:,0] = reconstructed_predictions[:,:,:,0] + q_mean
                        reconstructed_predictions[:,:,:,1] = reconstructed_predictions[:,:,:,1] + w_mean
                        POD_reconstruction[:,:,:,0] = POD_reconstruction[:,:,:,0] + q_mean
                        POD_reconstruction[:,:,:,1] = POD_reconstruction[:,:,:,1] + w_mean
                    elif scaling == 'scaled':
                        reconstructed_predictions[:,:,:,0] = reconstructed_predictions[:,:,:,0] * q_std + q_mean
                        reconstructed_predictions[:,:,:,1] = reconstructed_predictions[:,:,:,1] * w_std + w_mean
                        POD_reconstruction[:,:,:,0] = POD_reconstruction[:,:,:,0] * q_std + q_mean
                        POD_reconstruction[:,:,:,1] = POD_reconstruction[:,:,:,1] * w_std + w_mean


                    #### globals #####
                    reconstructed_prediction_avgq = np.mean(reconstructed_predictions[:,:,:,0], axis=(1,2))
                    reconstructed_prediction_ke = 0.5*reconstructed_predictions[:,:,:,1]*reconstructed_predictions[:,:,:,1]
                    reconstructed_prediction_avgke = np.mean(reconstructed_prediction_ke, axis=(1,2))

                    reconstructed_prediction_global = np.zeros((N_intt,2))
                    reconstructed_prediction_global[:,0] = reconstructed_prediction_avgke
                    reconstructed_prediction_global[:,1] = reconstructed_prediction_avgq

                    reconstructed_groundtruth_avgq = np.mean(POD_reconstruction[:,:,:,0], axis=(1,2))
                    reconstructed_groundtruth_ke = 0.5*POD_reconstruction[:,:,:,1]*POD_reconstruction[:,:,:,1]
                    reconstructed_groundtruth_avgke = np.mean(reconstructed_groundtruth_ke, axis=(1,2))

                    reconstructed_groundtruth_global = np.zeros((N_intt,2))
                    reconstructed_groundtruth_global[:,0] = reconstructed_groundtruth_avgke
                    reconstructed_groundtruth_global[:,1] = reconstructed_groundtruth_avgq

                    for v in range(variables):
                        fig, ax = plt.subplots(2, figsize=(12,6), sharex=True, tight_layout=True)
                        minm = min(np.min(POD_reconstruction[:, :, 32, v]), np.min(reconstructed_predictions[:, :, 32, v]))
                        maxm = max(np.max(POD_reconstruction[:, :, 32, v]), np.max(reconstructed_predictions[:, :, 32, v]))
                        c1 = ax[0].contourf(xx, x, POD_reconstruction[:,:,32,v].T, vmin=minm, vmax=maxm)
                        c2 = ax[1].contourf(xx, x, reconstructed_predictions[:,:,32,v].T, vmin=minm, vmax=maxm)
                        norm1 = mcolors.Normalize(vmin=minm, vmax=maxm)
                        fig.colorbar(c1, ax=ax[0], label='POD', norm=norm1, extend='both')
                        fig.colorbar(c2, ax=ax[1], label='ESN', norm=norm1, extend='both')
                        ax[1].set_xlabel('Time [Lyapunov Times]')
                        ax[0].set_ylabel('x')
                        ax[1].set_ylabel('x')
                        fig.savefig(output_path+'/prediction_recon_ens%i_test%i_var%i.png' % (j,i,v))
                        plt.close()
                        
                    fig, ax = plt.subplots(2,1, figsize=(12,6), tight_layout=True)
                    for v in range(2):
                        ax[v].plot(xx, reconstructed_groundtruth_global[:,v], color='tab:blue', label='POD')
                        ax[v].plot(xx, reconstructed_prediction_global[:,v], color='tab:orange', label='ESN')
                        ax[v].grid()
                    ax[-1].set_xlabel('Time [Lyapunov Times]')
                    ax[0].set_ylabel('KE')
                    ax[1].set_ylabel('q')
                    ax[1].legend()
                    #ax[1].set_xlim()
                    fig.savefig(output_path+'/global_prediction_ens%i_test%i.png' % (j,i))
                    plt.close()


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




