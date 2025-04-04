"""
python script for ESN.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --hyperparam_file=<hyperparam_file> --config_number=<config_number>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --hyperparam_file=<hyperparam_file>  hyperparameters for ESN
    --config_number=<config_number>      config number file
"""

print('load packages')
import os
os.environ["OMP_NUM_THREADS"] = '1' # imposes cores
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence  # Import the exception

from docopt import docopt
args = docopt(__doc__)

import json

#global tikh_opt, k, ti, tikh, U_washout, U_tv, Y_tv, U, N_in, N_fw, N_washout, N_val, N_units

exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('run functions files')

input_path = args['--input_path']
output_path = args['--output_path']
hyperparam_file = args['--hyperparam_file']
config_number= args['--config_number']

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')


with open(hyperparam_file, "r") as f:
    hyperparams = json.load(f)
    Nr = hyperparams["Nr"]
    train_len = hyperparams["N_train"]
    val_len = hyperparams["N_val"]
    test_len = hyperparams["N_test"]
    washout_len = hyperparams["N_washout"]
    t_lyap = hyperparams["t_lyap"]
    normalisation = hyperparams["normalisation"]
    ens = hyperparams["ens"]
    ensemble_test = hyperparams["ensemble_test"]
    n_tests = hyperparams["n_tests"]
    grid_x = hyperparams["grid_x"]
    grid_y = hyperparams["grid_y"]
    added_points = hyperparams["added_points"]
    val = hyperparams["val"]
    noise = hyperparams["noise"]
    N_forward = hyperparams["N_forward"]
    alpha0 = hyperparams["alpha0"]

from sklearn.metrics import mean_squared_error
def NRMSE(original_data, reconstructed_data):
    if original_data.ndim == 3:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2])
    elif original_data.ndim == 4:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2]*original_data.shape[3])
    if reconstructed_data.ndim == 3:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2])
    elif reconstructed_data.ndim == 4:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2]*reconstructed_data.shape[3])

    # Check if both data arrays have the same dimensions and the dimension is 2
    if original_data.ndim == reconstructed_data.ndim == 2:
        print("Both data arrays have the same dimensions and are 2D.")
    else:
        print("The data arrays either have different dimensions or are not 2D.")
    rmse = np.sqrt(mean_squared_error(original_data, reconstructed_data))
    
    variance = np.var(original_data)
    std_dev  = np.sqrt(variance)
    
    nrmse = (rmse/std_dev)
    
    return nrmse

def MSE(original_data, reconstructed_data):
    if original_data.ndim == 3:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2])
    elif original_data.ndim == 4:
        original_data = original_data.reshape(original_data.shape[0], original_data.shape[1]*original_data.shape[2]*original_data.shape[3])
    if reconstructed_data.ndim == 3:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2])
    elif reconstructed_data.ndim == 4:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], reconstructed_data.shape[1]*reconstructed_data.shape[2]*reconstructed_data.shape[3])

    # Check if both data arrays have the same dimensions and the dimension is 2
    if original_data.ndim == reconstructed_data.ndim == 2:
        print("Both data arrays have the same dimensions and are 2D.")
    else:
        print("The data arrays either have different dimensions or are not 2D.")
    mse = mean_squared_error(original_data, reconstructed_data)
    return mse

def ss_transform(data, scaler):
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
    if data_reshape.ndim == 2:
        print("data array is 2D.")
    else:
        print("data array is not 2D")
        
    data_scaled = scaler.transform(data_reshape)
    data_scaled = data_scaled.reshape(data.shape)
    
    if data_scaled.ndim == 4:
        print('scaled and reshaped to 4 dimensions')
    else:
        print('not scaled properly')
        
    return data_scaled
    
def ss_inverse_transform(data, scaler):
    if data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
        data_reshape = data.reshape(-1, data.shape[-1])
    if data_reshape.ndim == 2:
        print("data array is 2D.")
    else:
        print("data array is not 2D")
        
    print('shape before inverse scaling', np.shape(data_reshape))

    data_unscaled = scaler.inverse_transform(data_reshape)
    data_unscaled = data_unscaled.reshape(data.shape)
    
    if data_unscaled.ndim == 4:
        print('unscaled and reshaped to 4 dimensions')
    else:
        print('not unscaled properly')
        
    return data_unscaled

#### Load Data ####
q = np.load(input_path + 'q5000_30000.npy')
ke = np.load(input_path + 'KE5000_30000.npy')
total_time = np.load(input_path + 'total_time5000_30000.npy')

# Reshape the arrays into column vectors
ke_column = ke.reshape(len(ke), 1)
q_column = q.reshape(len(q), 1)
#evap_column = evap.reshape(len(evap), 1)

data = np.hstack((ke_column, q_column))

# Print the shape of the combined array
print(data.shape)

U = data

#### dataset generation ####
# number of time steps for washout, train, validation, test
t_lyap    = t_lyap
dt        = 1
N_lyap    = int(t_lyap//dt)
print('N_lyap', N_lyap)
N_washout = washout_len*N_lyap #75
N_train   = train_len*N_lyap #600
N_val     = val_len*N_lyap #45
N_test    = test_len*N_lyap #45
dim       = 2

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)

# standardisation 
norm_std = U_data.std(axis=0)
normalisation = normalisation #on, off, standard

print('norm', norm)
print('u_mean', u_mean)
print('norm_std', norm_std)

# washout
U_washout = U[:N_washout].copy()
# data to be used for training + validation
U_tv  = U[N_washout:N_washout+N_train-1].copy() #inputs
Y_tv  = U[N_washout+1:N_washout+N_train].copy() #data to match at next timestep


# adding noise to training set inputs with sigma_n the noise of the data
# improves performance and regularizes the error as a function of the hyperparameters
fig,ax = plt.subplots(2, figsize=(12,6), sharex=True)
ax[0].plot(U_tv[:N_val,0], c='b', label='Non-noisy')
ax[1].plot(U_tv[:N_val,1], c='b')

seed = 42   #to be able to recreate the data, set also seed for initial condition u0
rnd1  = np.random.RandomState(seed)
noisy = True
if noisy:
    data_std = np.std(data,axis=0)
    sigma_n = noise #1e-3     #change this to increase/decrease noise in training inputs (up to 1e-1)
    for i in range(dim):
        U_tv[:,i] = U_tv[:,i] \
                        + rnd1.normal(0, sigma_n*data_std[i], N_train-1)
    ax[0].plot(U_tv[:N_val,0], 'r--', label='Noisy')
    ax[1].plot(U_tv[:N_val,1], 'r--')

ax[0].legend()
fig.savefig(output_path + '/noise_addition_sigman%.2f.png' % sigma_n)
plt.close()

#### ESN hyperparameters #####
if normalisation == 'on':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm))]) #input bias (average absolute value of the inputs)
elif normalisation == 'standard':
    bias_in   = np.array([np.mean(np.abs((U_data-u_mean)/norm_std))]) #input bias (average absolute value of the inputs)
elif normalisation == 'off':
    bias_in   = np.array([np.mean(np.abs(U_data))]) #input bias (average absolute value of the inputs)
bias_out  = np.array([1.]) #output bias

N_units      = Nr #neurons
connectivity = 3
sparseness   = 1 - connectivity/(N_units-1)

tikh = np.array([1e-6,1e-9,1e-12])  # Tikhonov factor (optimize among the values in this list)

print('tikh:', tikh)
print('N_r:', N_units, 'sparsity:', sparseness)
print('bias_in:', bias_in, 'bias_out:', bias_out)

#### Grid Search and BO #####
n_in  = 0           #Number of Initial random points

spec_in     = .5    #range for hyperparameters (spectral radius and input scaling)
spec_end    = 1.2   #1
in_scal_in  = np.log10(0.1)
in_scal_end = np.log10(5.)

# In case we want to start from a grid_search, the first n_grid_x*n_grid_y points are from grid search
n_grid_x = grid_x
n_grid_y = grid_y
n_bo     = added_points  #number of points to be acquired through BO after grid search
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
ensemble = ens
alpha0 = alpha0
print('alpha=', alpha0)

# Which validation strategy (implemented in Val_Functions.ipynb)
val      = eval(val)
N_fw     = N_forward*N_lyap
N_fo     = (N_train-N_val-N_washout)//N_fw + 1 
#N_fo     = 33                     # number of validation intervals
N_in     = N_washout                 # timesteps before the first validation interval (can't be 0 due to implementation)
#N_fw     = (N_train-N_val-N_washout)//(N_fo-1) # how many steps forward the validation interval is shifted (in this way they are evenly spaced)
N_splits = 4                         # reduce memory requirement by increasing N_splits
print('Number of folds', N_fo)
print('how many steps forward validation interval moves', N_fw)
print('how many LTs forward validation interval moves', N_fw//N_lyap)

data_dir = '/Run_n_units{0:}_ensemble{1:}_normalisation{2:}_washout{3:}_config{4:}/'.format(N_units, ensemble, normalisation, washout_len, config_number)
output_path = output_path+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')


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
i = 0
attempt = 0

while i < ensemble: 
    print(f'Attempting Realization: {attempt+1} (Successful so far: {i})')

    k   = 0

    # Win and W generation
    seed= attempt+1
    rnd = np.random.RandomState(seed)

    #sparse syntax for the input and state matrices
    Win  = lil_matrix((N_units,dim+1))
    for j in range(N_units):
        Win[j,rnd.randint(0, dim+1)] = rnd.uniform(-1, 1) #only one element different from zero
    Win = Win.tocsr()

    W = csr_matrix( #on average only connectivity elements different from zero
        rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1-sparseness)))

    try:
        # Compute spectral radius
        spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
        W = (1 / spectral_radius) * W  # Scale to have unitary spectral radius
    except ArpackNoConvergence as e:
        print(f"Skipping realization {attempt+1} due to eigenvalue convergence error: {e}")
        attempt += 1  # Move to the next attempt (ensure new seed)
        continue  # Skip to the next attempt
    except Exception as e:
        print(f"Skipping realization {attempt+1} due to unexpected error: {e}")
        attempt += 1  # Move to the next attempt (ensure new seed)
        continue  # Skip to the next attempt

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

    # Full path for saving the file
    hyp_file = '_ESN_hyperparams_ens%i.json' % i

    output_path_hyp = os.path.join(output_path, hyp_file)

    hyps = {
    "test": i,
    "spec rad": minimum[i,0],
    "input scaling": 10**minimum[i,1],
    "tikh": minimum[i,2],
    "min f": minimum[i,-1],
    }

    with open(output_path_hyp, "w") as file:
        json.dump(hyps, file, indent=4)

    # Successfully completed, move to the next
    i += 1
    attempt += 1  # Always increase attempt count for new seeds

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
print('TESTING')
N_test   = n_tests                     #number of intervals in the test set
N_tstart = N_washout + N_train     #where the first test interval starts
N_intt   = test_len*N_lyap                #length of each test set interval

# #prediction horizon normalization factor and threshold
sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
threshold_ph = 0.2

ensemble_test = ensemble_test

ens_pred  = np.zeros((N_intt, dim, N_test, ensemble_test))
ens_truth = np.zeros((N_intt, dim, N_test))

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
    nrmse_error    = np.zeros((N_test, N_intt))

    # to plot results
    plot = True
    if plot:
        n_plot = 2

    #run different test intervals
    for i in range(N_test):
        print(N_tstart)
        # data for washout and target in each interval
        U_wash    = U[N_tstart - N_washout +i*N_intt : N_tstart + i*N_intt].copy()
        Y_t       = U[N_tstart + i*N_intt            : N_tstart + i*N_intt + N_intt].copy()
        print('start index:', N_tstart+i*N_intt, 'end_index:',  N_tstart+i*N_intt+N_intt)

        #washout for each interval
        Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
        Uh_wash = np.dot(Xa1, Wout)

        # Prediction Horizon
        Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
        # save predictions
        ens_truth[:, :, i] = Y_t
        ens_pred[:, :, i, j] = Yh_t

        Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
        PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
        if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)
        ens_PH[i,j] = PH[i]
        nrmse_error[i, :] = Y_err

        if plot:
            #left column has the washout (open-loop) and right column the prediction (closed-loop)
            # only first n_plot test set intervals are plotted
            if i<n_plot:
                fig,ax =plt.subplots(2,sharex=True)
                xx = np.arange(U_wash[:,-2].shape[0])/N_lyap
                for v in range(dim):
                    ax[v].plot(xx,U_wash[:,v], 'b', label='True')
                    ax[v].plot(xx,Uh_wash[:-1,v], '--r', label='ESN')
                    ax[v].grid()
                #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                ax[1].set_xlabel('Time[Lyapunov Times]')
                if v==0:
                    ax[v].legend(ncol=2)
                fig.savefig(output_path+'/washout_ens%i_test%i.png' % (j,i))
                plt.close()

                fig,ax =plt.subplots(2,sharex=True)
                xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                for v in range(dim):
                    ax[v].plot(xx,Y_t[:,v], 'b')
                    ax[v].plot(xx,Yh_t[:,v], '--r')
                    ax[v].grid()
                #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                ax[1].set_xlabel('Time [Lyapunov Times]')
                if v==0:
                    ax[v].legend(ncol=2)
                fig.savefig(output_path+'/prediction_ens%i_test%i.png' % (j,i))
                plt.close()

                fig,ax =plt.subplots(1,sharex=True, tight_layout=True)
                xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                ax.plot(xx,Y_err, 'b')
                ax.axhline(y=threshold_ph, xmin=xx[0], xmax=xx[-1])
                ax.grid()
                ax.set_ylabel('PH')
                ax.set_xlabel('Time')
                fig.savefig(output_path+'/PH_ens%i_test%i.png' % (j,i))
                plt.close()

                nrmse = NRMSE(Y_t, Yh_t)
                mse   = MSE(Y_t, Yh_t)
                print('NRMSE', nrmse)
                print('MSE', mse)

                # Full path for saving the file
                output_file = 'ESN_test_metrics_ens%i_test%i.json' % (j,i)

                output_path_met = os.path.join(output_path, output_file)

                metrics = {
                "test": i,
                "MSE": mse,
                "NRMSE": nrmse,
                "PH": PH[i],
                }

                with open(output_path_met, "w") as file:
                    json.dump(metrics, file, indent=4)

    # Percentiles of the prediction horizon
    print('PH quantiles [Lyapunov Times]:',
          np.quantile(PH,.75), np.median(PH), np.quantile(PH,.25))
    ens_PH[:,j] = PH
    print('')

    mean_nrmse_error = np.mean(nrmse_error, axis=0)
    fig, ax = plt.subplots(1, figsize=(8,6))
    ax.plot(xx, mean_nrmse_error)
    ax.grid()
    ax.set_xlabel('LT')
    ax.set_ylabel('mean nrmse error for ensemble')
    fig.savefig(output_path+f"meanerror_ensemble{j}.png")

for j in range(N_test):
    Y_t = ens_truth[:,:,j]
    Y_ht = ens_pred[:,:,j,:]
    fig, ax =plt.subplots(2, figsize=(12,6), sharex=True)
    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
    mean_ens = np.mean(Y_ht, axis=-1)
    lower = np.percentile(Y_ht, 5, axis=-1)
    upper = np.percentile(Y_ht, 95, axis=-1)
    for i in range(dim):
        ax[i].plot(xx, Y_t[:,i], color='tab:blue')
        ax[i].plot(xx, mean_ens[:,i], color='tab:orange')
        ax[i].fill_between(xx, lower[:,i], upper[:,i], color='tab:orange', alpha=0.3)
        ax[i].grid()
        ax[i].legend()
    fig.savefig(output_path+'/ens_pred_test%i.png' % j)
    plt.close()

###### SAVE RESULTS ######
#Save the details and results of the search for post-process
opt_specs = [spec_in,spec_end,in_scal_in,in_scal_end]

fln = output_path+ '/'+ val.__name__ + str(N_units) +'.mat'
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

# validation ensembles convergence
ensemble_list = np.linspace(1,ensemble+1,ensemble)
print(np.shape(ensemble_list))

lowest_MSE_values = np.zeros((ensemble))
for f in range(ensemble):
    lowest_MSE = 10**(np.min(f_iters[f, :]))
    lowest_MSE_values[f] = lowest_MSE

avg_MSE = []
median_vals = np.zeros((ensemble))
lower_vals = np.zeros((ensemble))
upper_vals = np.zeros((ensemble))
for f in range(1,ensemble+1):
    values = lowest_MSE_values[:f]
    avg_MSE.append(values)
    print(np.shape(values))
    lower = np.percentile(values, 25)
    upper = np.percentile(values, 75) 
    median = np.percentile(values, 50)
    median_vals[f-1] = median
    upper_vals[f-1] = upper
    lower_vals[f-1] = lower

fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
ax.plot(ensemble_list, lower_vals, color='tab:green', linestyle='--', alpha=0.3)
ax.plot(ensemble_list, upper_vals, color='tab:green', linestyle='--', alpha=0.3)
ax.plot(ensemble_list, median_vals, color='tab:green')
ax.fill_between(ensemble_list, lower_vals, upper_vals, color='tab:green', alpha=0.3)
ax.set_xlabel('no. of ensembles')
ax.set_ylabel('avg MSE (validation set)')
fig.savefig(output_path+'/ens_conv.png')

xx = np.arange(Y_tv[:,-2].shape[0])/N_lyap
for i in range(ensemble):
    Yh      = np.empty((N_train-1, 2))
    xa      = Xa1_states[i]
    Wout    = Woutt[i]
    Yh_test = np.dot(xa, Wout)
    fig,ax =plt.subplots(2,sharex=True)
    for v in range(dim):
        ax[v].plot(xx,Y_tv[:,v], 'b')
        ax[v].plot(xx,Yh_test[:,v], '--r')
        ax[v].grid()
    ax[1].set_xlabel('Time [Lyapunov Times]')
    ax[1].set_xlim(0,5)
    ax[0].set_ylabel('KE')
    ax[1].set_ylabel('q')
    fig.savefig(output_path+'/training_states_ens%i.png' % i)

ESN_params = 'ESN_params.json' 

output_ESN_params = os.path.join(output_path, ESN_params)
with open(output_ESN_params, "w") as f:
    json.dump(hyperparams, f, indent=4) 