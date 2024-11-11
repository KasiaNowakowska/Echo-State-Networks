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

print("Current time:", time.time())

data_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/input_data/'
model_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/Run_n_units1000/'
output_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/Run_n_units1000/testing_independent/' #### change testing type 

print('running functions')
exec(open("Val_Functions.py").read())
exec(open("Functions.py").read())
print('functions run, still printing')

#### Load Data ####
q = np.load(data_path + 'q5000_30000.npy')
ke = np.load(data_path + 'KE5000_30000.npy')
total_time = np.load(data_path + 'total_time5000_30000.npy')

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
N_lyap    = 400
N_washout = 400
N_train   = 30*N_lyap
N_val     = 3*N_lyap
N_test    = 3*N_lyap
dim       = 2

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)

# washout
U_washout = U[:N_washout].copy()
# data to be used for training + validation
U_tv  = U[N_washout:N_washout+N_train-1].copy() #inputs
Y_tv  = U[N_washout+1:N_washout+N_train].copy() #data to match at next timestep


# adding noise to training set inputs with sigma_n the noise of the data
# improves performance and regularizes the error as a function of the hyperparameters
seed = 42   #to be able to recreate the data, set also seed for initial condition u0
rnd1  = np.random.RandomState(seed)
noisy = True
if noisy:
    data_std = np.std(data,axis=0)
    sigma_n = 1e-3     #change this to increase/decrease noise in training inputs (up to 1e-1)
    for i in range(dim):
        U_tv[:,i] = U_tv[:,i] \
                        + rnd1.normal(0, sigma_n*data_std[i], N_train-1)

# load in file
# parameters from matrix 
val      = RVC_Noise
N_units  = 500
bias_out  = np.array([1.]) #output bias
print('bias_out:', bias_out)

fln = model_path+'/ESN_' + val.__name__ + str(N_units) +'.mat'
data = loadmat(fln)
print(data.keys())

Winn             = data['Win'][0] #gives Winn
fix_hyp          = data['fix_hyp']
bias_in_value    = data['fix_hyp'][:,0]
N_washout        = data['fix_hyp'][:,1]
opt_hyp          = data['opt_hyp']
Ws               = data['W'][0]
Woutt            = data['Wout']
norm             = data['norm'][0]
bias_in          = np.ones((1))*bias_in_value

print('fix_hyp shape:', np.shape(fix_hyp))
print('bias_in_value:', bias_in_value, 'N_washout:', N_washout)
print('shape of bias_in:', np.shape(bias_in))
print('opt_hyp shape:', np.shape(opt_hyp))
print('W shape:', np.shape(Ws))
print('Win shape:', np.shape(Winn))
print('Wout shape:', np.shape(Woutt))
print('norm:', norm)
print('u_mean:', u_mean)
print('shape of norm:', np.shape(norm))



##### quick test #####
N_test   = 9                       #number of intervals in the test set
N_tstart = N_washout + N_train     #where the first test interval starts
N_intt   = 3*N_lyap                #length of each test set interval

N_tstart = int(N_tstart[0])
N_washout = int(N_washout[0])

print('N_tstart:', N_tstart)
print('N_intt:', N_intt)
print('N_washout:', N_washout)

# #prediction horizon normalization factor and threshold
sigma_ph     = np.sqrt(np.mean(np.var(U,axis=1)))
threshold_ph = 0.2

ensemble_test = 50

ens_pred = np.zeros((N_intt, dim, N_test, ensemble_test))
true_data = np.zeros((N_intt, dim, N_test))
ens_PH = np.zeros((N_test, ensemble_test))

method = 'independent'

for j in range(ensemble_test):

    print('Realization    :',j+1)

    if method == 'independent':
        #load matrices and hyperparameters
        Wout     = Woutt[j].copy()
        Win      = Winn[j] #csr_matrix(Winn[j])
        W        = Ws[j]   #csr_matrix(Ws[j])
        rho      = opt_hyp[j,0].copy()
        sigma_in = opt_hyp[j,1].copy()
        print('Hyperparameters:',rho, sigma_in)

    print(np.shape(Wout), np.shape(Win), np.shape(W))
    print(np.shape(U))

    # to store prediction horizon in the test set
    PH       = np.zeros(N_test)

    # to plot results
    plot = True
    if plot:
        n_plot = 2

    #run different test intervals
    for i in range(N_test):
        print('starting index for testing:', N_tstart + i*N_intt)
        # data for washout and target in each interval
        U_wash    = U[N_tstart - N_washout +i*(N_intt//3) : N_tstart + i*(N_intt//3)].copy()
        Y_t       = U[N_tstart + i*(N_intt//3)            : N_tstart + i*(N_intt//3) + N_intt].copy()
        print(np.shape(U_wash), np.shape(np.zeros(N_units)))
        
        #washout for each interval
        Xa1     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
        Uh_wash = np.dot(Xa1, Wout)

        # Prediction Horizon
        Yh_t        = closed_loop(N_intt-1, Xa1[-1], Wout, sigma_in, rho)[0]
        #save prediciton and true data
        ens_pred[:,:,i,j] = Yh_t
        if j == 0:
            true_data[:,:,i] = Y_t
            print('ens=', j, 'true data saved')
        #find PH
        Y_err       = np.sqrt(np.mean((Y_t-Yh_t)**2,axis=1))/sigma_ph
        PH[i]       = np.argmax(Y_err>threshold_ph)/N_lyap
        if PH[i] == 0 and Y_err[0]<threshold_ph: PH[i] = N_intt/N_lyap #(in case PH is larger than interval)

        if plot:
            #left column has the washout (open-loop) and right column the prediction (closed-loop)
            # only first n_plot test set intervals are plotted
            if i<n_plot:
                fig,ax =plt.subplots(2,sharex=True)
                xx = np.arange(U_wash[:,-2].shape[0])/N_lyap
                for v in range(dim):
                    ax[v].plot(xx,U_wash[:,v], color='tab:blue', label='True')
                    ax[v].plot(xx,Uh_wash[:-1,v], color='tab:orange', label='ESN')
                    ax[v].grid()
                #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                ax[1].set_xlabel('Time[Lyapunov Times]')
                ax[0].set_ylabel('KE')
                ax[1].set_ylabel('q')
                if v==0:
                    ax[v].legend(ncol=2)
                fig.savefig(output_path + '/washout_ens{:02d}_test{:02d}.png'.format(j, i))
                plt.close()

                fig,ax =plt.subplots(2,sharex=True)
                xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                for v in range(dim):
                    ax[v].plot(xx,Y_t[:,v], color='tab:blue', label='True')
                    ax[v].plot(xx,Yh_t[:,v], color='tab:orange', label='ESN')
                    ax[v].grid()
                #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                ax[1].set_xlabel('Time [Lyapunov Times]')
                ax[0].set_ylabel('KE')
                ax[1].set_ylabel('q')
                if v==0:
                    ax[v].legend(ncol=2)
                fig.savefig(output_path + '/prediction_ens{:02d}_test{:02d}.png'.format(j, i))
                plt.close()

                fig,ax =plt.subplots(1,sharex=True)
                xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
                ax.scatter(Y_t[:,1],Y_t[:,0], color='tab:blue', label='True', marker = '.')
                ax.scatter(Yh_t[:,1],Yh_t[:,0], color='tab:orange', label='ESN', marker='x')
                ax.grid()
                #ax[1].set_ylim(Y_t.min()-.1, Y_t.max()+.1)
                ax.set_ylabel('KE')
                ax.legend(ncol=2)
                fig.savefig(output_path + '/phasespace_ens{:02d}_test{:02d}.png'.format(j, i))
                plt.close()

    # Percentiles of the prediction horizon
    print('PH quantiles [Lyapunov Times]:',
          np.quantile(PH,.75), np.median(PH), np.quantile(PH,.25))
    ens_PH[:,j] = PH
    print('')
    
np.save(output_path+'/PH.npy', ens_PH)
np.save(output_path+'/ens_pred.npy', ens_pred)
np.save(output_path+'/true_data.npy', true_data)

#ens_pred = np.zeros((N_intt, dim, N_test, ensemble_test))

for i in range(N_test):
    fig, ax =plt.subplots(2, figsize=(12,6), sharex=True)
    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
    Yh_t_ens = ens_pred[:,:,i,:]
    mean_ens = np.mean(Yh_t_ens, axis=-1)
    lower = np.percentile(Yh_t_ens, 5, axis=-1)
    upper = np.percentile(Yh_t_ens, 95, axis=-1)
    Y_t = true_data[:,:,i]
    print('shape of mean:', np.shape(mean_ens))
    for v in range(dim):
        print(v)
        ax[v].plot(xx, Y_t[:,v], color='tab:blue', label='True')
        ax[v].plot(xx, mean_ens[:,v], color='tab:orange', label='mean prediction')
        ax[v].fill_between(xx, lower[:,v], upper[:,v], color='tab:orange', alpha=0.3, label='90% confidence interval')
        ax[v].grid()
        ax[v].legend()
    ax[1].set_xlabel('Lyapunov Time')
    ax[0].set_ylabel('KE')
    ax[1].set_ylabel('q')
    fig.savefig(output_path+'/ens_pred_test%i.png' % i)
    plt.close()

fig, ax =plt.subplots(1, figsize=(3,6))
avg_PH = np.mean(ens_PH, axis=0)
np.save(output_path+y'/avg_PH.npy', avg_PH)
# Create a violin plot
ax.violinplot(avg_PH, showmeans=False, showmedians=True)
fig.savefig(output_path+'/violin_plot.png')




