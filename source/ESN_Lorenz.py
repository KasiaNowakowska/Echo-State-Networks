"""
python script for POD.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path> --hyperparam_file=<hyperparam_file> --config_number=<config_number>]

Options:
    --input_path=<input_path>            file path to use for data
    --output_path=<output_path>          file path to save images output [default: ./images]
    --hyperparam_file=<hyperparam_file>  hyperparameters for ESN
    --config_number=<config_number>      config_number [default: 0]
"""

import os
import time
import sys
sys.path.append(os.getcwd())
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sys.stdout.reconfigure(line_buffering=True)
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
from scipy.io import loadmat, savemat
from skopt.plots import plot_convergence
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import time as time
from Eval_Functions import *

exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('run functions files')

from docopt import docopt
args = docopt(__doc__)

input_path = args['--input_path']
output_path = args['--output_path']
hyperparam_file = args['--hyperparam_file']
config_number = int(args['--config_number'])

with open(hyperparam_file, "r") as f:
    hyperparams = json.load(f)
    Nr = hyperparams["Nr"]
    train_len = hyperparams["N_train"]
    val_len = hyperparams["N_val"]
    test_len = hyperparams["N_test"]
    washout_len = hyperparams["N_washout"]
    washout_len_val = hyperparams.get("N_washout_val", washout_len)
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
    alpha = hyperparams["alpha"]
    alpha0 = hyperparams["alpha0"]
    n_forward = hyperparams["n_forward"]

U_rk4 = np.load(input_path+'/U_rk4.npy')
time_vals = np.load(input_path+'/time_array.npy')
variables= ['x', 'y', 'z']

fig, ax = plt.subplots(3, figsize=(12,9), constrained_layout=True)
for v in range(3):
    ax[v].plot(time_vals, U_rk4[:,v])
    ax[v].set_ylabel('amplitude')
    ax[v].set_xlabel('time')
fig.savefig(output_path+'/LorenzAmplitude.png')

U = U_rk4
print('shape of data for ESN', np.shape(U))

# number of time steps for washout, train, validation, test
# number of time steps for washout, train, validation, test
t_lyap    = 0.9**(-1) 
dt        = 0.01
N_lyap    = int(t_lyap//dt)
print('N_lyap', N_lyap)
N_washout = int(washout_len*N_lyap) #75
N_washout_val = int(washout_len_val*N_lyap)
N_train   = train_len*N_lyap #600
N_val     = val_len*N_lyap #45
N_test    = test_len*N_lyap #45
dim       = U.shape[1]

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)

# standardisation 
norm_std = U_data.std(axis=0)
normalisation = 'standard' #on, off, standard

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
fig,ax = plt.subplots(3, figsize=(12,6), tight_layout=True, sharex=True)
for m in range(3):
    ax[m].plot(U_tv[:N_val,m], c='b', label='Non-noisy')
seed = 42   #to be able to recreate the data, set also seed for initial condition u0
rnd1  = np.random.RandomState(seed)
noisy = True
if noisy:
    data_std = np.std(U,axis=0)
    sigma_n = noise    #change this to increase/decrease noise in training inputs (up to 1e-1)
    for i in range(3):
        U_tv[:,i] = U_tv[:,i] \
                        + rnd1.normal(0, sigma_n*data_std[i], N_train-1)
    for m in range(3):
        index = m
        ax[m].plot(U_tv[:N_val,index], 'r--', label='Noisy')
        ax[m].grid()
        ax[m].set_title('%i' % (index+1))
ax[0].legend()
fig.savefig(output_path + '/noise_addition.png')
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

tikh = np.array([1e-3, 1e-6, 1e-9]) #np.array([1e-3,1e-6,1e-9,1e-12])  # Tikhonov factor (optimize among the values in this list)

print('tikh:', tikh)
print('N_r:', N_units, 'sparsity:', sparseness)
print('bias_in:', bias_in, 'bias_out:', bias_out)

#### Grid Search and BO #####
threshold_ph = 0.2
n_in  = 0           #Number of Initial random points

spec_in     = .1    #range for hyperparameters (spectral radius and input scaling)
spec_end    = 1   
in_scal_in  = np.log10(0.1)
in_scal_end = np.log10(5.0)

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
ensemble = ens

data_dir = '/Run_n_units{0:}_ensemble{1:}_normalisation{2:}_config{3:}/'.format(N_units, ensemble, normalisation, config_number)
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
val      = eval(val)
alpha = alpha
N_fw     = n_forward*N_lyap
N_fo     = (N_train-N_val-N_washout_val)//N_fw + 1 
#N_fo     = 33                     # number of validation intervals
N_in     = N_washout                 # timesteps before the first validation interval (can't be 0 due to implementation)
#N_fw     = (N_train-N_val-N_washout)//(N_fo-1) # how many steps forward the validation interval is shifted (in this way they are evenly spaced)
N_splits = 4                         # reduce memory requirement by increasing N_splits
print('Number of folds', N_fo)
print('how many steps forward validation interval moves', N_fw)
print('how many LTs forward validation interval moves', N_fw//N_lyap)

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
    # Full path for saving the file
    hyp_file =  '_ESN_hyperparams_ens%i.json' % i

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

#Save the details and results of the search for post-process
opt_specs = [spec_in,spec_end,in_scal_in,in_scal_end]

fln = output_path + '/ESN_matrices' + '.mat'
with open(fln,'wb') as f:  # need 'wb' in Python3
    savemat(f, {"norm": norm})
    savemat(f, {"fix_hyp": np.array([bias_in, N_washout],dtype='float64')})
    savemat(f, {'opt_hyp': np.column_stack((minimum[:,0], 10**minimum[:,1]))})
    savemat(f, {"Win": Winn})
    savemat(f, {'W': Ws})
    savemat(f, {"Wout": Woutt})

ESN_params = 'ESN_params.json' 

output_ESN_params = os.path.join(output_path, ESN_params)
with open(output_ESN_params, "w") as f:
    json.dump(hyperparams, f, indent=4) 

validation_interval = True
test_interval       = True

if validation_interval:
    ##### quick test #####
    print('VALIDATION (TEST)')
    print(N_washout_val)
    N_test   = N_fo                     #number of intervals in the test set
    N_tstart = N_washout_val                    #where the first test interval starts
    N_intt   = test_len*N_lyap            #length of each test set interval

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))

    ensemble_test = ensemble_test

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_test, ensemble_test))
    ens_PH2         = np.zeros((ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))

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
            n_plot = 3
            plt.rcParams["figure.figsize"] = (15,3*n_plot)
            plt.figure()
            plt.tight_layout()

        #run different test intervals
        for i in range(N_test):
            print('test number:', i+1)
            print('index of test start:', N_tstart + i*N_intt)
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
            nrmse_error[i, :] = Y_err
            ens_PH[i,j] = PH[i]

            # metrics
            nrmse = NRMSE(Y_t, Yh_t)
            mse   = MSE(Y_t, Yh_t)

            print('NRMSE', nrmse)
            print('MSE', mse)

            # Full path for saving the file
            output_file = '_ESN_validation_metrics_ens%i_test%i.json' % (j,i)

            output_path_met = os.path.join(output_path, output_file)

            metrics = {
            "test": i,
            "MSE": mse,
            "NRMSE": nrmse,
            "PH": float(PH[i]),
            }

            with open(output_path_met, "w") as file:
                json.dump(metrics, file, indent=4)

            ens_nrmse[j]       += nrmse
            ens_PH2[j]         += PH[i]
                
            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    if j % 1 == 0:
                    
                        #### prediction ####
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        fig, ax = plt.subplots(3, figsize=(12,9), sharex=True)
                        for v in range(3):
                            ax[v].plot(xx, Y_t[:,v], label='True')
                            ax[v].plot(xx, Yh_t[:,v], label='ESN')
                            ax[v].set_ylabel(variables[v])
                            ax[v].grid()
                        ax[-1].set_xlabel('Lyapunov Time')
                        ax[-1].legend()
                        fig.savefig(output_path+f"/prediction_validation_ens{j}_test{i}.png")

        # accumulation for each ensemble member
        ens_nrmse[j]       = ens_nrmse[j] / N_test
        ens_PH2[j]         = ens_PH2[j] / N_test  
             
    # Full path for saving the file
    output_file_ALL = 'ESN_validation_metrics_all.json' 

    output_path_met_ALL = os.path.join(output_path, output_file_ALL)

    flatten_PH = ens_PH.flatten()
    print('flat PH', flatten_PH)

    metrics_ens_ALL = {
    "mean PH": np.mean(ens_PH2),
    "lower PH": np.quantile(flatten_PH, 0.75),
    "uppper PH": np.quantile(flatten_PH, 0.25),
    "median PH": np.median(flatten_PH),
    "mean NRMSE": np.mean(ens_nrmse),
    }

    with open(output_path_met_ALL, "w") as file:
        json.dump(metrics_ens_ALL, file, indent=4)
    print('finished validations')

if test_interval:
    ##### quick test #####
    print('TESTING')
    print(N_washout_val)
    N_test   = 1                    #number of intervals in the test set
    N_tstart = N_train + N_washout_val                    #where the first test interval starts
    N_intt   = test_len*N_lyap            #length of each test set interval

    # #prediction horizon normalization factor and threshold
    sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))

    ensemble_test = ensemble_test

    ens_pred        = np.zeros((N_intt, dim, ensemble_test))
    ens_PH          = np.zeros((N_intt, ensemble_test))
    ens_nrmse       = np.zeros((ensemble_test))
    ens_ssim        = np.zeros((ensemble_test))
    ens_evr         = np.zeros((ensemble_test))
    ens_nrmse_plume = np.zeros((ensemble_test))

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
            n_plot = 3
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
            nrmse_error[i, :] = Y_err

            # metrics
            nrmse = NRMSE(Y_t, Yh_t)
            mse   = MSE(Y_t, Yh_t)

            print('NRMSE', nrmse)
            print('MSE', mse)

            # Full path for saving the file
            output_file = '_ESN_test_metrics_ens%i_test%i.json' % (j,i)

            output_path_met = os.path.join(output_path, output_file)

            metrics = {
            "test": i,
            "MSE": mse,
            "NRMSE": nrmse,
            }

            with open(output_path_met, "w") as file:
                json.dump(metrics, file, indent=4)
                
            if plot:
                #left column has the washout (open-loop) and right column the prediction (closed-loop)
                # only first n_plot test set intervals are plotted
                if i<n_plot:
                    if ensemble_test % 1 == 0:
                    
                        #### prediction ####
                        xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                        fig, ax = plt.subplots(3, figsize=(12,9), sharex=True)
                        for v in range(3):
                            ax[v].plot(xx, Y_t[:,v], label='True')
                            ax[v].plot(xx, Yh_t[:,v], label='ESN')
                            ax[v].set_ylabel(variables[v])
                            ax[v].grid()
                        ax[-1].set_xlabel('Lyapunov Time')
                        ax[-1].legend()
                        fig.savefig(output_path+f"/prediction_test_ens{j}_test{i}.png")

                        