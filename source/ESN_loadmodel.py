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
from Eval_Functions import *
from sklearn.preprocessing import StandardScaler
import sys
sys.stdout.reconfigure(line_buffering=True)
import json

print("Current time:", time.time())

data_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/input_data/'
model_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/Run_gp_hedge_2_n_units1000_ensemble100_n_tot40_normalisationon_standardisationoff_sigma_n1.0e-03_12400LTs_10folds/'
output_path = '/nobackup/mm17ktn/ESN/Echo-State-Networks/source/Run_gp_hedge_2_n_units1000_ensemble100_n_tot40_normalisationon_standardisationoff_sigma_n1.0e-03_12400LTs_10folds/testing/' #### change testing type 

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

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
N_train   = 31*N_lyap
N_val     = 3*N_lyap
N_test    = 3*N_lyap
dim       = 2

# compute normalization factor (range component-wise)
U_data = U[:N_washout+N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M-m 
u_mean = U_data.mean(axis=0)
normalisation = 'on'
standardisation = 'off'
if standardisation == 'on':
    ss = StandardScaler()
    U = ss.fit_transform(U)
    #data = data[::5]
    #time_vals = time_vals[::5]
    print(np.shape(U))

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
N_units  = 1000
bias_out  = np.array([1.]) #output bias
print('bias_out:', bias_out)

fln = model_path+'/' + val.__name__ + str(N_units) +'.mat'
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

f_iters = np.load(model_path+'/f_iters.npy')
print(np.shape(f_iters))
lowest_MSE_values = np.zeros((100))
for f in range(100):
    lowest_MSE = np.min(f_iters[f, :])
    lowest_MSE_values[f] = lowest_MSE
# Get the indices of the 50 smallest values
indices_50_smallest = np.argsort(lowest_MSE_values)[:50]


##### quick test #####
N_test    = 10                      #number of intervals in the test set
N_tstart  = 50*N_lyap     #where the first test interval starts
N_intt    = 3*N_lyap                #length of each test set interval
N_gap     = N_lyap
N_washout = 1*N_lyap

print('N_tstart:', N_tstart)
print('N_intt:', N_intt)
print('N_washout:', N_washout)

# #prediction horizon normalization factor and threshold
sigma_ph        = np.sqrt(np.mean(np.var(U,axis=1)))
sigma_ph_scaled = np.sqrt(np.mean(np.var((U-u_mean) / norm,axis=1)))
threshold_phs = [0.05, 0.1, 0.2, 0.5]
#threshold_ph = 0.2

ensemble_test = 100

ens_pred = np.zeros((N_intt, dim, N_test, ensemble_test))
true_data = np.zeros((N_intt, dim, N_test))
ens_PH = np.zeros((N_test, ensemble_test, len(threshold_phs) ))
ens_PH_scaled = np.zeros((N_test, ensemble_test, len(threshold_phs) ))

method = 'independent'

PTs = [0, 1, 2]
flag_pred = np.empty((len(PTs), ensemble_test, N_test), dtype=object)

#for j in range(ensemble_test):
for j, element in enumerate(indices_50_smallest):

    print('Realization    :',j+1)

    if method == 'independent':
        #load matrices and hyperparameters
        Wout     = Woutt[element].copy()
        Win      = Winn[element] #csr_matrix(Winn[j])
        W        = Ws[element]   #csr_matrix(Ws[j])
        rho      = opt_hyp[element,0].copy()
        sigma_in = opt_hyp[element,1].copy()
        print('Hyperparameters:',rho, sigma_in)

    print(np.shape(Wout), np.shape(Win), np.shape(W))
    print(np.shape(U))

    # to store prediction horizon in the test set
    PH        = np.zeros((N_test, len(threshold_phs) ))
    PH_scaled = np.zeros((N_test, len(threshold_phs) ))

    # to plot results
    plot = False
    if plot:
        n_plot = 2
    plot_further = True

    #run different test intervals
    for i in range(N_test):
        print('starting index for testing:', N_tstart + i*N_intt)
        # data for washout and target in each interval
        U_wash    = U[N_tstart - N_washout +i*N_gap : N_tstart + i*N_gap].copy()
        Y_t       = U[N_tstart + i*N_gap           : N_tstart + i*N_gap + N_intt].copy()
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
        for k_index, threshold_ph in enumerate(threshold_phs):
            PH[i, k_index]       = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH[i, k_index] == 0 and Y_err[0]<threshold_ph: PH[i, k_index] = N_intt/N_lyap #(in case PH is larger than interval)
        
        # find PH scaled
        Yh_t_scaled        = (Yh_t - u_mean) / norm
        Y_t_scaled         = (Y_t - u_mean) / norm
        Y_err_scaled       = np.sqrt(np.mean((Y_t_scaled-Yh_t_scaled)**2,axis=1))/sigma_ph_scaled
        for k_index, threshold_ph in enumerate(threshold_phs):
            PH_scaled[i, k_index]       = np.argmax(Y_err_scaled>threshold_ph)/N_lyap
            if PH_scaled[i, k_index] == 0 and Y_err_scaled[0]<threshold_ph: PH[i, k_index] = N_intt/N_lyap #(in case PH is larger than interval)


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

        if plot_further:
                for p in range(len(PTs)):
                    PT = int(PTs[p]*N_lyap)
                    threshold_e = 0.00010
                    print(PT, threshold_e)
                    print('shape of y_t:', np.shape(Y_t))
                    #if Y_t[0,0] < threshold_e:
                    true_onset = onset_truth(Y_t, PT, N_lyap, threshold_e)
                    pred_onset = onset_prediction(Yh_t, PT, N_lyap, threshold_e)
                    flag = onset_ensemble(true_onset, pred_onset)
                    flag_pred[p,j,i] = flag
                    
    
                

    # Percentiles of the prediction horizon
    print('PH quantiles [Lyapunov Times]:',
          np.quantile(PH[:,-1],.75), np.median(PH[:,-1]), np.quantile(PH[:,-1],.25))
    ens_PH[:,j,:] = PH
    ens_PH_scaled[:,j,:] = PH_scaled
    print('')


'''
np.save(output_path+'/PH.npy', ens_PH)
np.save(output_path+'/PH_scaled.npy', ens_PH_scaled)
np.save(output_path+'/ens_pred.npy', ens_pred)
np.save(output_path+'/true_data.npy', true_data)

#ens_pred = np.zeros((N_intt, dim, N_test, ensemble_test))

from scipy.stats import gmean
'''
'''
for i in range(N_test):
    fig, ax =plt.subplots(2, figsize=(12,6), sharex=True)
    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
    Yh_t_ens = ens_pred[:,:,i,:]
    mean_ens = np.mean(Yh_t_ens, axis=-1)
    median_ens = np.percentile(Yh_t_ens, 50, axis=-1)
    lower = np.percentile(Yh_t_ens, 5, axis=-1)
    upper = np.percentile(Yh_t_ens, 95, axis=-1)
    Y_t = true_data[:,:,i]
    print('shape of mean:', np.shape(mean_ens))
    for v in range(dim):
        print(v)
        ax[v].plot(xx, Y_t[:,v], color='tab:blue', label='True')
        ax[v].plot(xx, median_ens[:,v], color='tab:orange', label='median prediction')
        ax[v].fill_between(xx, lower[:,v], upper[:,v], color='tab:orange', alpha=0.3, label='90% confidence interval')
        ax[v].grid()
        ax[v].legend()
    ax[1].set_xlabel('Lyapunov Time')
    ax[0].set_ylabel('KE')
    ax[1].set_ylabel('q')
    fig.savefig(output_path+'/ens_pred_median_test%i.png' % i)
    plt.close()

for i in range(N_test):
    fig, ax =plt.subplots(2, figsize=(12,6), sharex=True)
    xx = np.arange(Y_t[:,-2].shape[0])/N_lyap
    Yh_t_ens = ens_pred[:,:,i,:]
    mean_ens = np.mean(Yh_t_ens, axis=-1)
    median_ens = np.percentile(Yh_t_ens, 50, axis=-1)
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
    fig.savefig(output_path+'/ens_pred_mean_test%i.png' % i)
    plt.close()


avg_PH = np.mean(ens_PH, axis=0)
threshold_phs = np.array(threshold_phs)
np.save(output_path+'/avg_PH.npy', avg_PH)
print(threshold_phs)
for index, element in enumerate(threshold_phs):
    fig, ax =plt.subplots(1, figsize=(3,6), constrained_layout=True)
    #np.save(output_path+'/avg_PH{:.2f}.npy'.format(element), avg_PH[:,index])
    # Create a violin plot
    ax.violinplot(ens_PH[:,:,index].flatten(), showmeans=False, showmedians=True)
    ax.set_ylabel('PH')
    fig.savefig(output_path+'/violin_plot_flatten{:.2f}_tests{:.1f}.png'.format(element, N_test))

avg_PH_scaled = np.mean(ens_PH_scaled, axis=0)
np.save(output_path+'/avg_PH_scaled.npy', avg_PH_scaled)
for index, element in enumerate(threshold_phs):
    fig, ax =plt.subplots(1, figsize=(3,6), constrained_layout=True)
    # Create a violin plot
    ax.violinplot(ens_PH_scaled[:, :,index].flatten(), showmeans=False, showmedians=True)
    ax.set_ylabel('PH')
    fig.savefig(output_path+'/violin_plot_scaled_flatten{:.2f}_tests{:.1f}.png'.format(element, N_test))
'''
for index, element in enumerate(threshold_phs):
    fig, ax = plt.subplots(1, figsize=(3, 6), constrained_layout=True)
    # Save average PH data (commented out in your code)
    # np.save(output_path+'/avg_PH{:.2f}.npy'.format(element), avg_PH[:, index])
    
    # Create a box-and-whisker plot
    ax.boxplot(ens_PH[:, :, index].flatten(), vert=True,
               showmeans=True, meanline=True)  # Set `showmeans=True` if you want mean
    
    ax.set_ylabel('PH')
    fig.savefig(output_path + '/box_plot_flatten{:.2f}_tests{:.1f}.png'.format(element, N_test))
    print(f"Data stats for index {index}: min={np.min(ens_PH[:,:,index])}, max={np.max(ens_PH[:,:,index])}, "
          f"median={np.median(ens_PH[:,:,index])}, Q1={np.percentile(ens_PH[:,:,index], 25)}, Q3={np.percentile(ens_PH[:,:,index], 75)}")
'''
fig, ax = plt.subplots(1, figsize=(12,3))
xx = np.arange(U[N_tstart:N_tstart + (N_test+1)*N_gap,-2].shape[0])/N_lyap
ax.plot(xx, U[N_tstart:N_tstart + (N_test+1)*N_gap,0])
print(len(xx))
for i in range(N_test):
    print(i*N_gap)
    ax.scatter(xx[i*N_gap], U[N_tstart + i*N_gap,0], marker='x', color='tab:red')
ax.grid()
ax.set_xlabel('Lyapunov Time')
ax.set_ylabel('KE')
fig.savefig(output_path+'/initial_test_points.png')

print(flag_pred)
P = np.zeros((len(PTs), N_test))
R = np.zeros((len(PTs), N_test))
F = np.zeros((len(PTs), N_test))
TP_list = np.zeros((len(PTs), N_test))
FP_list = np.zeros((len(PTs), N_test))
FN_list = np.zeros((len(PTs), N_test))
TN_list = np.zeros((len(PTs), N_test))
Totals = np.zeros((len(PTs), 3)) # P, R and F

# Count 'TP' in each column (across ensemble members)
for p in range(len(PTs)):
    for i in range(N_test):
        TP = np.sum(flag_pred[p, :, i] == 'TP') #predicted true and acc true
        FP = np.sum(flag_pred[p, :, i] == 'FP') #predicted true and acc false
        FN = np.sum(flag_pred[p, :, i] == 'FN') #predicted false but acc true
        TN = np.sum(flag_pred[p, :, i] == 'TN') #predicted false and acc false

        TP_list[p, i] = TP
        FP_list[p, i] = FP
        FN_list[p, i] = FN
        TN_list[p, i] = TN

        P[p, i] = TP/(TP+FP)
        R[p, i] = TP/(TP+FN)
        F[p, i] = 2/(1/P[p, i] + 1/R[p, i])

        if i == 0:
            metrics = {
                "test": i,
                "PH": PH[i],
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "P": P[p,i],
                "R": R[p,i],
                "F-score": F[p,i],
                }

            metrics_file = "metrics%i.txt" % i
            metrics_path = os.path.join(output_path, metrics_file)

            # Save to a text file
            with open(metrics_path, "w") as file:
                for key, value in metrics.items():
                    file.write(f"{key}: {value}\n")

    print(TP_list, FP_list, FN_list, TN_list)
    print(P, F, R)


    total_TP = np.sum(TP_list[p,:])
    total_FP = np.sum(FP_list[p,:])
    total_FN = np.sum(FN_list[p,:])
    total_TN = np.sum(TN_list[p,:])

    total_P = total_TP/(total_TP+total_FP)
    total_R = total_TP/(total_TP+total_FN)
    total_F = 2/(1/total_P + 1/total_R)

    Totals[p, 0] = total_P
    Totals[p, 1] = total_R
    Totals[p, 2] = total_F

    metrics = {
    "index": p,
    "PI_start": PTs[p],
    "TP": total_TP,
    "FP": total_FP,
    "FN": total_FN,
    "TN": total_TN,
    "P": total_P,
    "R": total_R,
    "F-score": total_F,
    }

    metrics_file = "metrics_all_TP%i.txt" % p 
    metrics_path = os.path.join(output_path, metrics_file)

    # Save to a text file
    with open(metrics_path, "w") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

fig, ax = plt.subplots(1, figsize=(12, 3), constrained_layout=True)
ax.scatter(PTs, Totals[:,0], marker='x', label='Precision')
ax.scatter(PTs, Totals[:,1], marker='.', label='Recall')
ax.scatter(PTs, Totals[:,2], marker='+', label='F1-score')
ax.legend()
ax.grid()
ax.set_xlabel('Prediction Interval')
ax.set_ylabel('Score')
fig.savefig(output_path+'/scores_anyIC.png')

fig, ax = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
score_metrics = [P, R, F]
names = ['Precision', 'Recall', 'F1-score']
for index, element in enumerate(score_metrics):
    print(score_metrics[index])
    mean = np.nanmean(score_metrics[index], axis=1)
    median = np.nanpercentile(score_metrics[index], 50, axis=1)
    p25 = np.nanpercentile(score_metrics[index], 25, axis=1)
    p75 = np.nanpercentile(score_metrics[index], 75, axis=1)
    lower_error = median - p25
    upper_error = p75 - median

    ax[index].errorbar(PTs, median, yerr=[lower_error, upper_error], fmt='-o', capsize=5, capthick=2)
    ax[index].scatter(PTs, mean, marker='x')

    #ax[index].legend()
    ax[index].grid()
    ax[index].set_xlabel('Prediction Interval')
    ax[index].set_ylabel(names[index])
    ax[index].set_ylim(-0.05,1.05)
fig.savefig(output_path+'/scores_errors_anyIC.png')


# find FFT
def FFT(variable, time):
    variable = variable - np.mean(variable)
    fft = np.fft.fft(variable)
    fft = np.fft.fftshift(fft)
    end = time[-1]
    start = time[0]
    m = np.fft.fftfreq(len(time), d=(end-start)/len(time))
    m = np.fft.fftshift(m)
    #m = 2*np.pi*m/(1)
    magnitude_w = np.abs(fft)
    psd = magnitude_w**2
    return psd, m

xx = np.arange(Y_t[:,-2].shape[0])/N_lyap

KE_true_psd, KE_true_freq = FFT(true_data[:,0,0], xx)
KE_pred_psd, KE_pred_freq = FFT(ens_pred[:,0,0,0], xx)

q_true_psd, q_true_freq  = FFT(true_data[:,1,0], xx)
q_pred_psd, q_pred_freq = FFT(ens_pred[:,1,0,0], xx)

half = len(KE_true_freq)//2

fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
ax[0].plot(KE_true_freq[half:half+100], KE_true_psd[half:half+100], label='True', color='tab:blue')
ax[0].plot(KE_pred_freq[half:half+100], KE_pred_psd[half:half+100], label='Prediction', color='tab:orange', linestyle='--')
ax[1].plot(q_true_freq[half:half+100], q_true_psd[half:half+100], label='True', color='tab:blue')
ax[1].plot(q_pred_freq[half:half+100], q_pred_psd[half:half+100], label='Prediction', color='tab:orange', linestyle='--')
for v in range(2):
    ax[v].grid()
    ax[v].legend()
    ax[v].set_xlabel('Frequency')
    ax[v].set_ylabel('PSD')
fig.savefig(output_path+'/FFTs.png')
'''
'''
precision_means = np.nanmean(P, axis=1)
recall_means    = np.nanmean(R, axis=1)
Fscore_means    = np.nanmean(F, axis=1)

p_medians = np.nanpercentile(P, 50, axis=1)
p_percentile_25 = np.nanpercentile(P, 25, axis=1)
p_percentile_75 = np.nanpercentile(P, 75, axis=1)

r_medians = np.nanpercentile(R, 50, axis=1)
r_percentile_25 = np.nanpercentile(R, 25, axis=1)
r_percentile_75 = np.nanpercentile(R, 75, axis=1)

f_medians = np.nanpercentile(F, 50, axis=1)
f_percentile_25 = np.nanpercentile(F, 25, axis=1)
f_percentile_75 = np.nanpercentile(F, 75, axis=1)

fig, ax = plt.subplots(3, figsize=(12,9), sharex=True, tight_layout=True)
xxx = PTs
print(xxx)
ax[0].errorbar(xxx, p_medians, yerr=[p_percentile_25, p_percentile_75], fmt='-o', capsize=5, capthick=2)
ax[1].errorbar(xxx, r_medians, yerr=[r_percentile_25, r_percentile_75], fmt='-o', capsize=5, capthick=2)
ax[2].errorbar(xxx, f_medians, yerr=[f_percentile_25, f_percentile_75], fmt='-o', capsize=5, capthick=2)
ax[-1].set_xlabel('Prediction Time')
ax[0].set_ylabel('Precision')
ax[1].set_ylabel('Recall')
ax[2].set_ylabel('F-score')
fig.savefig(output_path+'/CE.png')

P = np.zeros((len(PTs), ensemble_test))
R = np.zeros((len(PTs), ensemble_test))
F = np.zeros((len(PTs), ensemble_test))
TP_list = np.zeros((len(PTs), ensemble_test))
FP_list = np.zeros((len(PTs), ensemble_test))
FN_list = np.zeros((len(PTs), ensemble_test))
# Count 'TP' in each column (across ensemble members)
for p in range(len(PTs)):
    for j in range(ensemble_test):
        TP = np.sum(flag_pred[p, j, :] == 'TP')
        FP = np.sum(flag_pred[p, j, :] == 'FP')
        FN = np.sum(flag_pred[p, j, :] == 'FN')

        TP_list[p, j] = TP
        FP_list[p, j] = FP
        FN_list[p, j] = FN

        P[p, j] = TP/(TP+FP)
        R[p, j] = TP/(TP+FN)
        F[p, j] = 2/(1/P[p, j] + 1/R[p, j])

ens = 3
p_medians = np.nanpercentile(P[:,ens], 50)
p_percentile_25 = np.nanpercentile(P[:,ens], 25)
p_percentile_75 = np.nanpercentile(P[:,ens], 75)

r_medians = np.nanpercentile(R[:,ens], 50)
r_percentile_25 = np.nanpercentile(R[:,ens], 25)
r_percentile_75 = np.nanpercentile(R[:,ens], 75)

f_medians = np.nanpercentile(F[:,ens], 50)
f_percentile_25 = np.nanpercentile(F[:,ens], 25)
f_percentile_75 = np.nanpercentile(F[:,ens], 75)

print('pmedians shape:', np.shape(p_medians))
fig, ax = plt.subplots(3, figsize=(12,9), sharex=True, tight_layout=True)
xxx = PTs
print(xxx)
ax[0].errorbar(xxx, p_medians, yerr=[p_percentile_25, p_percentile_75], fmt='-o', capsize=5, capthick=2)
ax[1].errorbar(xxx, r_medians, yerr=[r_percentile_25, r_percentile_75], fmt='-o', capsize=5, capthick=2)
ax[2].errorbar(xxx, f_medians, yerr=[f_percentile_25, f_percentile_75], fmt='-o', capsize=5, capthick=2)
ax[-1].set_xlabel('Prediction Time')
ax[0].set_ylabel('Precision')
ax[1].set_ylabel('Recall')
ax[2].set_ylabel('F-score')
fig.savefig(output_path+'/CE_ens_member3.png')
'''