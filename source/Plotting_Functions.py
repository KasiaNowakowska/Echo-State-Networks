import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import time
from scipy.stats import gaussian_kde

def plot_modes_washout(truth, prediction, xx, i, j, indexes_to_plot, file_name, Modes=True):
    fig, ax = plt.subplots(len(indexes_to_plot), figsize=(8,6), sharex=True, tight_layout=True)
    if len(indexes_to_plot) == 1:
        ax.plot(xx, truth, 'b', label='True')
        ax.plot(xx, prediction[:-1], '--r', label='ESN')
    else:
        for v in range(len(indexes_to_plot)):
            index = indexes_to_plot[v]
            ax[v].plot(xx,truth[:,index], 'b', label='True')
            ax[v].plot(xx,prediction[:-1,index], '--r', label='ESN')
            ax[v].grid()
            if Modes == True:
                ax[v].set_ylabel('Mode %i' % (index+1))
            else:
                ax[v].set_ylabel('LV %i' % (index+1))
        ax[-1].set_xlabel('Time[Lyapunov Times]')
        if v==0:
            ax[0].legend(ncol=2)
        fig.suptitle('washout_ens%i_test%i' % (j,i))
        fig.savefig(file_name+'_ens%i_test%i.png' % (j,i))
        plt.close()
        
def plot_modes_prediction(truth, prediction, xx, i, j, indexes_to_plot, file_name, Modes=True):
    fig, ax = plt.subplots(len(indexes_to_plot), figsize=(8,6), sharex=True, tight_layout=True)
    if len(indexes_to_plot) == 1:
        ax.plot(xx, truth, 'b', label='True')
        ax.plot(xx, prediction[:-1], '--r', label='ESN')
        ax.legend()
        ax.grid()
    else:
        for v in range(len(indexes_to_plot)):
            index = indexes_to_plot[v]
            ax[v].plot(xx,truth[:,index], 'b', label='True')
            ax[v].plot(xx,prediction[:,index], '--r', label='ESN')
            ax[v].grid()
            if Modes == True:
                ax[v].set_ylabel('Mode %i' % (index+1))
            else:
                ax[v].set_ylabel('LV %i' % (index+1))
        ax[-1].set_xlabel('Time[Lyapunov Times]')
        if v==0:
            ax[0].legend(ncol=2)
        fig.suptitle('washout_ens%i_test%i' % (j,i))
        fig.savefig(file_name+'_ens%i_test%i.png' % (j,i))
        plt.close()

def plot_PH(error, threshold_ph, xx, i, j, file_name):
    fig,ax =plt.subplots(1, figsize=(8,6),sharex=True, tight_layout=True)
    ax.plot(xx,error, 'b')
    ax.axhline(y=threshold_ph, xmin=xx[0], xmax=xx[-1])
    ax.grid()
    ax.set_ylabel('PH')
    ax.set_xlabel('Time [Lyapunov Times]')
    fig.savefig(file_name+'_ens%i_test%i.png' % (j,i))
    plt.close()

def plot_reservoir_states_norm(Xa1, Xa2, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, N_units, file_name):
    fig,ax = plt.subplots(1,figsize=(8,6),sharex=True, tight_layout=True)
    ax.plot(time_vals[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap], np.linalg.norm(Xa1[:-1, :N_units], axis=1), color='red', label='washout')
    ax.plot(time_vals[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt], np.linalg.norm(Xa2[:, :N_units], axis=1), color='blue', label='prediction')
    ax.grid()
    ax.legend()
    ax.set_ylabel('res_norm')
    ax.set_ylabel('Time')
    fig.savefig(file_name+'_ens%i_test%i.png' % (j,i))
    plt.close()

def plot_input_states_norm(U_wash, Y_t, time_vals, N_tstart, N_washout_val, i, j, N_gap, N_intt, file_name):
    fig,ax =plt.subplots(1,figsize=(8,6),sharex=True, tight_layout=True)
    ax.plot(time_vals[N_tstart - N_washout_val +i*N_gap : N_tstart + i*N_gap], np.linalg.norm(U_wash, axis=1), color='red', label='washout')
    ax.plot(time_vals[N_tstart + i*N_gap            : N_tstart + i*N_gap + N_intt], np.linalg.norm(Y_t, axis=1), color='blue', label='prediciton')
    ax.grid()
    ax.legend()
    ax.set_ylabel('input_norm')
    ax.set_ylabel('Time')
    fig.savefig(file_name+'_ens%i_test%i.png' % (j,i))
    plt.close()                   

def plot_active_array(active_array, active_array_reconstructed, x, xx, i, j, variables, file_name):
    if len(variables) == 4:
        fig, ax = plt.subplots(2, figsize=(12,12), tight_layout=True)
        c1 = ax[0].contourf(xx, x, active_array[:,:, 32].T, cmap='Reds')
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('true')
        c2 = ax[1].contourf(xx, x, active_array_reconstructed[:,:, 32].T, cmap='Reds')
        fig.colorbar(c1, ax=ax[1])
        ax[1].set_title('reconstruction')
        for v in range(2):
            ax[v].set_xlabel('time')
            ax[v].set_ylabel('x')
        fig.savefig(file_name+f"_ens{j}_test{i}.png")
        plt.close()
    else:
        print('no image')

def plot_global_prediction_ts(truth, prediciton, xx, i, j, file_name):
    fig, ax = plt.subplots(2, figsize=(12,6), sharex=True, tight_layout=True)
    for v in range(2):
        ax[v].plot(xx, truth[:,v], label='POD')
        ax[v].plot(xx, prediciton[:,v], label='ESN')
        ax[v].grid()
        ax[v].legend()
    ax[0].set_ylabel('KE')
    ax[1].set_ylabel('q')
    ax[-1].set_xlabel('LT')
    fig.savefig(file_name+f"_ens{j}_test{i}.png")
    plt.close()

def plot_global_prediction_ps(truth, prediciton, i, j, file_name):
    fig, ax = plt.subplots(1, figsize=(8,6), tight_layout=True)
    ax.scatter(truth[:,1], truth[:,0], label='POD')
    ax.scatter(prediciton[:,1], prediciton[:,0], label='ESN')
    ax.grid()
    ax.set_ylabel('KE')
    ax.set_xlabel('q')
    fig.savefig(file_name+f"_ens{j}_test{i}.png")
    plt.close()

def stats_pdf_modes(Y_t, Yh_t, indexes_to_plot, i, j, file_name, Modes=True):
    fig, ax = plt.subplots(len(indexes_to_plot), figsize=(12, 12), tight_layout=True)
    for v, element in enumerate(indexes_to_plot):
        kde_true  = gaussian_kde(Y_t[:,element])
        kde_pred  = gaussian_kde(Yh_t[:,element])

        var_vals_true      = np.linspace(min(Y_t[:,element]), max(Y_t[:,element]), 1000)  # X range
        pdf_vals_true      = kde_true(var_vals_true)
        var_vals_pred      = np.linspace(min(Yh_t[:,element]), max(Yh_t[:,element]), 1000)  # X range
        pdf_vals_pred      = kde_pred(var_vals_pred)

        ax[v].plot(var_vals_true, pdf_vals_true, label="truth")
        ax[v].plot(var_vals_pred, pdf_vals_pred, label="prediction")
        ax[v].grid()
        ax[v].set_ylabel('Denisty')
        if Modes == True:
            ax[v].set_xlabel(f"Mode {element}")
        else:
            ax[v].set_xlabel(f"LV {element}")
        ax[v].legend()
    fig.savefig(file_name+f"_ens{j}_test{i}.png")

def stats_pdf_global(truth, prediction, i, j, file_name):
    fig, ax = plt.subplots(2, figsize=(8,6))
    for v in range(2):
        kde_true  = gaussian_kde(truth[:,v])
        kde_pred  = gaussian_kde(prediction[:,v])

        var_vals_true      = np.linspace(min(truth[:,v]), max(truth[:,v]), 1000)  # X range
        pdf_vals_true      = kde_true(var_vals_true)
        var_vals_pred      = np.linspace(min(prediction[:,v]), max(prediction[:,v]), 1000)  # X range
        pdf_vals_pred      = kde_pred(var_vals_pred)

        
        ax[v].plot(var_vals_true, pdf_vals_true, label="truth")
        ax[v].plot(var_vals_pred, pdf_vals_pred, label="prediction")
        ax[v].grid()
        ax[v].set_ylabel('Denisty')
        ax[v].legend()
    ax[0].set_xlabel('KE')
    ax[1].set_xlabel('q')
    fig.savefig(file_name+f"_ens{j}_test{i}.png")

def plotting_number_of_plumes(truth, prediction, xx, i, j, file_name):
    fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
    ax.plot(xx, truth[:,0], label='Truth')
    ax.plot(xx, prediction[:,0], label='Prediction')
    ax.set_xlabel('Time[Lyapunov Times]')
    ax.set_ylabel('Number of Plumes')
    ax.grid()
    ax.legend()
    fig.savefig(file_name+f"_ens{j}_ens{i}.png")