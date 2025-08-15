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
        ax[-1].set_xlabel('Time [Lyapunov Times]')
        if v==0:
            ax[0].legend(ncol=2)
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
        ax[-1].set_xlabel('Time [Lyapunov Times]')
        ax[0].legend(ncol=2)
        fig.savefig(file_name+'_ens%i_test%i.png' % (j,i))
        fig.savefig(file_name+'_ens%i_test%i.eps' % (j,i), format='eps')
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
    fig, ax = plt.subplots(len(indexes_to_plot), figsize=(12, 3*len(indexes_to_plot)), tight_layout=True)
    for v, element in enumerate(indexes_to_plot):
        kde_true  = gaussian_kde(Y_t[:,element])
        kde_pred  = gaussian_kde(Yh_t[:,element])

        var_vals_true      = np.linspace(min(Y_t[:,element]), max(Y_t[:,element]), 1000)  # X range
        pdf_vals_true      = kde_true(var_vals_true)
        var_vals_pred      = np.linspace(min(Yh_t[:,element]), max(Yh_t[:,element]), 1000)  # X range
        pdf_vals_pred      = kde_pred(var_vals_pred)

        ax[v].plot(var_vals_true, pdf_vals_true, label="Truth")
        ax[v].plot(var_vals_pred, pdf_vals_pred, label="ESN")
        ax[v].grid()
        ax[v].set_ylabel('Denisty', fontsize=16)
        ax[v].tick_params(axis='both', labelsize=12)
        if Modes == True:
            ax[v].set_xlabel(f"Mode {element+1}", fontsize=16)
        else:
            ax[v].set_xlabel(f"LV {element+1}", fontsize=16)
    ax[0].legend(fontsize=14)
    fig.savefig(file_name+f"_ens{j}_test{i}.png")
    plt.close()

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
    plt.close()

def plotting_number_of_plumes(truth, prediction, xx, i, j, file_name):
    fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
    ax.plot(xx, truth, label='Truth')
    ax.plot(xx, prediction, label='Prediction')
    ax.set_xlabel('Time[Lyapunov Times]')
    ax.set_ylabel('Number of Plumes')
    ax.grid()
    ax.legend()
    fig.savefig(file_name+f"_ens{j}_ens{i}.png")
    plt.close()

def hovmoller_plus_plumes(truth_data, prediction_data, truth_features, prediction_features, xx, x, variable, i, j, file_name):
    truth_data      = truth_data[..., variable]
    prediction_data = prediction_data[..., variable]

    z_value =32
    
    fig, ax = plt.subplots(3, figsize=(12, 9), tight_layout=True)
    
    minm = min(np.min(truth_data[:, :, z_value]), np.min(prediction_data[:, :, z_value]))
    maxm = max(np.max(truth_data[:, :, z_value]), np.max(prediction_data[:, :, z_value]))  

    c1 = ax[0].pcolormesh(xx, x, truth_data[:, :, z_value].T, vmin=minm, vmax=maxm)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('true')
    c2 = ax[1].pcolormesh(xx, x, prediction_data[:, :, z_value].T, vmin=minm, vmax=maxm)
    fig.colorbar(c2, ax=ax[1])
    ax[1].set_title('reconstruction')

    # Ensure proper scaling: features (0–64) → x indices (0–len(x)-1)
    def scale_feature_to_x_index(feature_array):
        idx = ((feature_array / 64.0) * (len(x) - 1)).astype(int)
        return np.clip(idx, 0, len(x) - 1)

    idx_avg_true = scale_feature_to_x_index(truth_features[:, 2])
    idx_avg_pred = scale_feature_to_x_index(prediction_features[:, 2])
    idx_spread_true = scale_feature_to_x_index(truth_features[:, 3])
    idx_spread_pred = scale_feature_to_x_index(prediction_features[:, 3])


    x_avg_true   = x[idx_avg_true]
    x_avg_pred   = x[idx_avg_pred]
    x_spread_true = x[idx_spread_true]
    x_spread_pred = x[idx_spread_pred]

    ax[0].plot(xx, x_avg_true, color='red', linewidth=1.5, label='avg_x') 
    ax[1].plot(xx, x_avg_pred, color='red', linewidth=1.5, label='avg_x') 

    ax[0].fill_between(xx, x_avg_true - x_spread_true, x_avg_true + x_spread_true,
                 color='red', alpha=0.3, label='x_spread')
    ax[1].fill_between(xx, x_avg_pred - x_spread_pred, x_avg_pred + x_spread_pred,
                 color='red', alpha=0.3, label='x_spread')

    pred_counts_rounded = np.round(prediction_features[:, 0]).astype(int)
    true_counts = truth_features[:, 0].astype(int)

    ax[2].plot(xx, true_counts, label='truth')
    ax[2].plot(xx, pred_counts_rounded, label='ESN')
    ax[2].legend()
    cb = fig.colorbar(c1, ax=ax[2])
    cb.ax.set_visible(False)
    ax[2].grid()

    for v in range(2):
        ax[v].set_ylabel('x', fontsize=14)
    ax[2].set_ylabel('Number of Plumes')
    ax[-1].set_xlabel('Time [Lyapunov Times]')

    fig.savefig(file_name+f"_ens{j}_test{i}.png")
    plt.close(fig)

def hovmoller_plus_plume_pos(truth_data, prediction_data, truth_features, prediction_features, xx, x, variable, i, j, file_name):
    truth_data      = truth_data[..., variable]
    prediction_data = prediction_data[..., variable]

    z_value =32
    
    fig, ax = plt.subplots(2, figsize=(12, 9), tight_layout=True)
    
    minm = min(np.min(truth_data[:, :, z_value]), np.min(prediction_data[:, :, z_value]))
    maxm = max(np.max(truth_data[:, :, z_value]), np.max(prediction_data[:, :, z_value]))  

    c1 = ax[0].pcolormesh(xx, x, truth_data[:, :, z_value].T, vmin=minm, vmax=maxm)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('True')
    c2 = ax[1].pcolormesh(xx, x, prediction_data[:, :, z_value].T, vmin=minm, vmax=maxm)
    fig.colorbar(c2, ax=ax[1])
    ax[1].set_title('Reconstruction')
    for v in range(5):
        x_vals_truth        = truth_features[:, v]
        x_vals_pred         = prediction_features[:, v]
        valid_mask_truth    = x_vals_truth > 0
        valid_mask_pred     = x_vals_pred > 0
        ax[0].scatter(xx[valid_mask_truth], x_vals_truth[valid_mask_truth], marker='x', color='tab:orange')
        ax[1].scatter(xx[valid_mask_pred], x_vals_pred[valid_mask_pred], marker='x', color='tab:orange')
    for l in range(2):
        ax[l].set_ylabel('x', fontsize=14)
    ax[-1].set_xlabel('Time [Lyapunov Times]')
    
    fig.savefig(file_name+f"_ens{j}_test{i}.png")
    plt.close(fig)

def hovmoller_plus_plume_sincospos(truth_data, prediction_data, truth_features, prediction_features, xx, x, variable, i, j, file_name, x_domain=(0,20)):
    x_min, x_max = x_domain
    
    truth_data      = truth_data[..., variable]
    prediction_data = prediction_data[..., variable]

    z_value =32
    
    # Precompute max_strength across all plumes and truth/prediction
    all_strength_truth = []
    all_strength_pred = []

    for v in range(3):
        all_strength_truth.append(truth_features[:, v*3 + 2])
        all_strength_pred.append(prediction_features[:, v*3 + 2])

    all_strength_truth = np.concatenate(all_strength_truth)
    all_strength_pred = np.concatenate(all_strength_pred)

    max_strength = max(np.max(all_strength_truth), np.max(all_strength_pred))

    fig, ax = plt.subplots(2, figsize=(12, 9), tight_layout=True)
    
    minm = min(np.min(truth_data[:, :, z_value]), np.min(prediction_data[:, :, z_value]))
    maxm = max(np.max(truth_data[:, :, z_value]), np.max(prediction_data[:, :, z_value]))  

    c1 = ax[0].pcolormesh(xx, x, truth_data[:, :, z_value].T, vmin=minm, vmax=maxm)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('True')
    c2 = ax[1].pcolormesh(xx, x, prediction_data[:, :, z_value].T, vmin=minm, vmax=maxm)
    fig.colorbar(c2, ax=ax[1])
    ax[1].set_title('Reconstruction')

    # Plot scatter overlays, keeping a reference to one mappable for KE colorbar
    scatter_cmap = 'plasma'  # stands out from viridis background
    sc_ref = None
    
    for v in range(3):
        cos_vals_truth = truth_features[:, v*3]      # cos
        sin_vals_truth = truth_features[:, v*3 + 1]  # sin
        strength_truth = truth_features[:, v*3 + 2]  # KE

        cos_vals_pred = prediction_features[:, v*3]      # cos
        sin_vals_pred = prediction_features[:, v*3 + 1]  # sin
        strength_pred = prediction_features[:, v*3 + 2]  # KE

        # Recover downsampled x position
        angles_truth = np.arctan2(sin_vals_truth, cos_vals_truth)  # [-π, π]
        angles_truth[angles_truth < 0] += 2*np.pi  # wrap negative angles
        x_vals_truth = x_min + (x_max - x_min) * angles_truth / (2*np.pi)

        # Recover downsampled x position
        angles_pred = np.arctan2(sin_vals_pred, cos_vals_pred)  # [-π, π]
        angles_pred[angles_pred < 0] += 2*np.pi  # wrap negative angles
        x_vals_pred = x_min + (x_max - x_min) * angles_pred / (2*np.pi)

        valid_mask_truth = strength_truth > 0
        valid_mask_pred  = strength_pred > 0

        sc1 = ax[0].scatter(
            xx[valid_mask_truth],
            x_vals_truth[valid_mask_truth],
            c=strength_truth[valid_mask_truth],
            cmap=scatter_cmap,         # try 'plasma', 'coolwarm', etc.
            vmin=0, vmax=max_strength,
            marker='x'
        )

        sc2 = ax[1].scatter(
            xx[valid_mask_pred],
            x_vals_pred[valid_mask_pred],
            c=strength_pred[valid_mask_pred],
            cmap=scatter_cmap,
            vmin=0, vmax=max_strength,
            marker='x'
        )

        # Keep reference for single shared colorbar
        if sc_ref is None:
            sc_ref = sc1

    # Single KE colorbar for scatter overlays
    cbar = fig.colorbar(sc_ref, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Plume Strength (KE)', fontsize=12)

    for l in range(2):
        ax[l].set_ylabel('x', fontsize=14)
    ax[-1].set_xlabel('Time [Lyapunov Times]')

    fig.savefig(file_name+f"_ens{j}_test{i}.png")
    plt.close(fig)