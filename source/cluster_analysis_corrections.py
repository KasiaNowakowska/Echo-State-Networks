import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, SymLogNorm
import os
import pathlib
import string
from sklearn.decomposition import PCA
import joblib
from collections import Counter, defaultdict
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec

#POD
input_path_testdata = 'Ra2e8/POD_ESN/Thesis/64modes/Run_n_units1000_ensemble5_normalisationon_washout3_config4101/GP36_4/further_analysis/corrections_clusters'
#CAE
#input_path_testdata  = 'Ra2e8/CAE_ESN/Thesis/LS64/Run_n_units1000_ensemble5_normalisationon_washout3_config8002/GP36_4/further_analysis/corrections_clusters'

input_path = '../../input_data/'

# FUNCTIONS

# Assume x_coords, z_coords are the physical grids
def filter_first_timesteps_with_tolerance_robust(plume_records, x_coords, z_coords,
                                                 x_tol=0.2, z_tol=0.02, x_periodic=True):
    """
    Robust filtering: only first timestep of each plume event,
    consecutive timesteps within tolerance are treated as same plume.
    """
    plume_records = sorted(plume_records, key=lambda r: r['time'])
    x_max = x_coords[-1] + (x_coords[1]-x_coords[0])
    active_events = []  # list of [last_x, last_z, last_t]
    filtered_records = []

    for rec in plume_records:
        x_c = rec['x_c']
        z_c = rec['z_c']
        t = rec['time']
        matched = False

        for evt in active_events:
            last_x, last_z, last_t = evt

            # Compute distance with periodic x if needed
            if x_periodic:
                x_dist = min(abs(x_c - last_x), x_max - abs(x_c - last_x))
            else:
                x_dist = abs(x_c - last_x)
            z_dist = abs(z_c - last_z)

            # Same event if within tolerance and consecutive timestep
            if x_dist <= x_tol and z_dist <= z_tol and t - last_t == 1:
                # Update event
                evt[0] = x_c
                evt[1] = z_c
                evt[2] = t
                matched = True
                break

        if not matched:
            # New event
            filtered_records.append(rec)
            active_events.append([x_c, z_c, t])

    return filtered_records

def find_nearest_idx(coord, coord_array):
    return np.argmin(np.abs(coord_array - coord))

def extract_preinit_variables(plume_records, variables_dict, aggregation_dict, x_coords, z_coords,
                              preinit_window=8, postinit_window=8, x_radius=1, z_radius=1):
    nx, nz = len(x_coords), len(z_coords)
    preinit_data = {'plume': [], 'control': []}
    
    # --- Helper function ---
    def extract_window(record):
        t0 = record['time']
        t_start = max(0, t0 - preinit_window)
        #t_end = t0 + 1  # include initiation timestep
        t_end = min(t0 + postinit_window + 1,  list(variables_dict.values())[0].shape[0])

        x_idx = find_nearest_idx(record['x_c'], x_coords)
        z_idx = find_nearest_idx(record['z_c'], z_coords)
        x_slice = slice(max(0, x_idx - x_radius), min(nx, x_idx + x_radius + 1))
        z_slice = slice(max(0, z_idx - z_radius), min(nz, z_idx + z_radius + 1))

        vars_dict = {}

        for var_name, var_array in variables_dict.items():
            aggregation = aggregation_dict.get(var_name, 'mean')  # default to mean
            if var_array.ndim == 3:
                if aggregation == 'mean':
                    ts = var_array[t_start:t_end, x_slice, z_slice].mean(axis=(1,2))
                elif aggregation == 'min':
                    ts = var_array[t_start:t_end, x_slice, z_slice].min(axis=(1,2))
                elif aggregation == 'max':
                    ts = var_array[t_start:t_end, x_slice, z_slice].max(axis=(1,2))
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")

                # Surface value
                if var_name in ['CAPE', 'CIN', 'CIN-KE']:
                    surface_ts = var_array[t_start:t_end, x_idx, 0]
                else:
                    surface_ts = np.full(ts.shape, np.nan)

            elif var_array.ndim == 1:
                # Global variable
                ts = var_array[t_start:t_end]
                surface_ts = ts.copy()  # for global vars, surface = same

            else:
                raise ValueError(f"Unsupported variable shape: {var_array.shape}")

            vars_dict[var_name] = ts
            vars_dict[var_name+'_surface'] = surface_ts

        return vars_dict, t0

    # --- Extract plume pre-init variables ---
    for record in plume_records:
        vars_dict, t0 = extract_window(record)
        preinit_data['plume'].append({
            'time': t0,
            'x_c': record['x_c'],
            'z_c': record['z_c'],
            'vars': vars_dict,
        })

    return preinit_data

def compute_threshold_stats(preinit_data, t0=8):
    """
    Computes percentage-based physical thresholds per cluster.
    Returns: dict[cluster] -> dict of percentages.
    """

    stats = defaultdict(lambda: {
        "q>0.290":0,
        "KE<1e-4":0,
        "CAPE>0.03":0,
        "dudx<-0.2":0,
        "cin_ke<-0.0005":0,
        "CIN_surface<0.002":0,
        "CAPE_surface>0.06":0,
        "CINKE_decreasing":0,
        "dudx_decreasing": 0,
        "CAPE_increasing": 0,
        "CIN_surface_decreasing":0,
        "CAPE_surface_increasing":0,
        "count":0
    })
    
    for plume in preinit_data['plume']:
        c = plume['cluster']
        stats[c]["count"] += 1
        
        # scalar at t0
        q            = plume['vars']['q_global'][t0]
        ke           = plume['vars']['KE_global'][t0]
        cape         = plume['vars']['CAPE'][t0]
        dudx         = plume['vars']['dudx'][t0]
        
        cin_ke       = plume['vars']['CIN-KE']
        cin_surface  = plume['vars']['CIN_surface']
        cape_surface = plume['vars']['CAPE_surface']
        cin_ke0       = cin_ke[t0]
        cape_surface_t0 = cape_surface[t0]
        cin_surface_t0  = cin_surface[t0]
        
        # trends up to t0
        x = np.arange(t0+1)
        slope_cin_ke,      _, _, _, _ = linregress(x, cin_ke[:t0+1])
        slope_cin_surface, _, _, _, _ = linregress(x, cin_surface[:t0+1])
        slope_cape_surface,_, _, _, _ = linregress(x, cape_surface[:t0+1])
        slope_dudx,        _, _, _, _ = linregress(x, plume['vars']['dudx'][:t0+1])
        slope_CAPE,        _, _, _, _ = linregress(x, plume['vars']['CAPE'][:t0+1])

        if q > 0.290:
            stats[c]["q>0.290"] += 1
        if ke < 1e-4:
            stats[c]["KE<1e-4"] += 1
        if cape > 0.03:
            stats[c]["CAPE>0.03"] += 1
        if dudx < -0.2:
            stats[c]["dudx<-0.2"] += 1
        if cin_ke0 < -0.0005:
            stats[c]["cin_ke<-0.0005"] += 1
        if cin_surface_t0 < 0.002:
            stats[c]["CIN_surface<0.002"] += 1
        if cape_surface_t0 > 0.06:
            stats[c]["CAPE_surface>0.06"] += 1
        
        if slope_cin_ke < 0:
            stats[c]["CINKE_decreasing"] += 1
        if slope_dudx < 0: 
            stats[c]["dudx_decreasing"] += 1
        if slope_CAPE < 0: 
            stats[c]["CAPE_increasing"] += 1
        if slope_cin_surface < 0:
            stats[c]["CIN_surface_decreasing"] += 1
        if slope_cape_surface > 0:
            stats[c]["CAPE_surface_increasing"] += 1

    # convert counts to %
    for c in stats:
        total = stats[c]["count"]
        for key in ("q>0.290","KE<1e-4","CAPE>0.03","dudx<-0.2","cin_ke<-0.0005","CIN_surface<0.002","CAPE_surface>0.06",
                    "CINKE_decreasing", "dudx_decreasing", "CAPE_increasing", "CIN_surface_decreasing","CAPE_surface_increasing"):
            stats[c][key] = 100 * stats[c][key] / total

    return stats

def stats_to_dataframe(stats_dict, k):
    """
    Convert threshold_stats dict into a tidy pandas DataFrame.
    Adds columns for k (number of clusters) and cluster label.
    """
    records = []
    for cluster, values in stats_dict.items():
        record = values.copy()
        record['cluster'] = cluster
        record['k'] = k
        records.append(record)
    df = pd.DataFrame(records)
    # Optional: reorder columns
    cols = ['k','cluster','count','q>0.290','KE<1e-4','CAPE>0.03','dudx<-0.2',"cin_ke<-0.0005","CIN_surface<0.002","CAPE_surface>0.06",
            'CINKE_decreasing','dudx_decreasing', 'CAPE_increasing','CIN_surface_decreasing','CAPE_surface_increasing']
    df = df[cols]
    return df


# -------------------- CODE PART 1 -------------------------
# x = np.load(input_path+'/x.npy')
# z = np.load(input_path+'/z.npy')

# x_downsample = x[::4]
# z_downsample = z[::4]

# dataset         = np.load(input_path_testdata+'/all_test_data.npy')
# alt_dataset     = np.load(input_path_testdata+'/alternative_test_data.npy')
# x_positions     = np.load(input_path_testdata+'/all_x_positions_truth.npy')

# print(f"shape of dataset = {np.shape(dataset)}")
# print(f"shape of alt dataset = {np.shape(alt_dataset)}")
# print(f"shape of x_positions = {np.shape(x_positions)}")


# q = dataset[...,0]
# w = dataset[...,1]
# dudx = alt_dataset[...,0]

# with h5py.File(input_path_testdata+'/CIN_CAPE_t00000_to_00630.h5', 'r') as df:
#     print(df.keys())
#     CAPE = df['CAPE'][:] 
#     CIN  = df['CIN'][:] 
#     time_vals = df['time_vals'][:] 

# start_time = time_vals[0]
# end_time   = time_vals[-1]
# print(f"start time={time_vals[0]}, end time = {time_vals[-1]}")
# print(f"shape of CIN dataset = {np.shape(CIN)}")

# KE = 0.5 * w * w
# CIN_KE = CIN - KE 

# KE_global_all =  np.load(input_path+'/KE5000_30000.npy')
# q_global_all  =  np.load(input_path+'/q5000_30000.npy')

# global_start_idx = int(time_vals[0] - 5000)
# global_end_idx = global_start_idx + (len(time_vals) * 2)

# # 3. Slice the window and downsample from dt=1 to dt=2 in one go
# ds_KE_global = KE_global_all[global_start_idx : global_end_idx : 2]
# ds_q_global  = q_global_all[global_start_idx : global_end_idx : 2]

# print(f"Polished Global KE shape: {ds_KE_global.shape}")
# print(f"Target Spatial shape:    {len(time_vals)}")

# assert len(ds_KE_global) == len(time_vals), (
#     f"Mismatch! Sliced global array has {len(ds_KE_global)} steps, "
#     f"but your test fields expect {len(time_vals)} steps."
# )
# print("Global arrays perfectly aligned to your 630-step test cadence!")



# # Create a new dudx array of the same size as new_array
# ds_dudx    = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))
# ds_ke      = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))
# ds_cin     = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))
# ds_cape    = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))
# ds_cin_ke  = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))

# # Iterate over the time dimension
# for t in range(len(time_vals)):
#     if t % 250 == 0:
#         print(f"time_val = {time_vals[t]}")
#     # Iterate over the subgrids along the x and z dimensions
#     for i in range(0, len(x), 4):
#         for j in range(0, len(z), 4):
#             # Extract the 4x4 subgrid and compute its minimum value
#             subgrid = dudx[t, i:i+4, j:j+4]
#             ds_dudx[t, i // 4, j // 4] = np.min(subgrid)

#             subgrid = CAPE[t, i:i+4, j:j+4]
#             ds_cape[t, i // 4, j // 4] = np.max(subgrid)

#             subgrid = CIN[t, i:i+4, j:j+4]
#             ds_cin[t, i // 4, j // 4] = np.min(subgrid)

#             subgrid = KE[t, i:i+4, j:j+4]
#             ds_ke[t, i // 4, j // 4] = np.max(subgrid)

#             subgrid = CIN_KE[t, i:i+4, j:j+4]
#             ds_cin_ke[t, i // 4, j // 4] = np.min(subgrid)
 
# minm_z, maxm_z = 0, 10 
# ds_dudx_ss   = ds_dudx[...,minm_z:maxm_z]
# ds_cape_ss   = ds_cape[...,minm_z:maxm_z]
# ds_cin_ss    = ds_cin[...,minm_z:maxm_z]
# ds_ke_ss     = ds_ke[...,minm_z:maxm_z]
# ds_cin_ke_ss = ds_cin_ke[...,minm_z:maxm_z]


# plume_records = []

# # Loop over your 630 test time steps and your plume slots (e.g., max_plumes=3)
# T, P = x_positions.shape

# for t in range(T):
#     for p in range(P):
#         x_val = x_positions[t, p]
        
#         # Skip if no plume was detected in this slot at this time step
#         if np.isnan(x_val):
#             continue
            
#         # Append structured entry using your target steering level
#         plume_records.append({
#             'time': t,
#             'x_c': x_val,
#             'z_c': z_downsample[7],  # Forces your exact target level z0=4
#             'mask_indices': None     # Not needed since we have explicit centroids
#         })

# print(f"Generated {len(plume_records)} raw tracking records.")

# filtered_initiations = filter_first_timesteps_with_tolerance_robust(
#     plume_records=plume_records,
#     x_coords=x_downsample,
#     z_coords=z_downsample,
#     x_tol=0.2,   # Spatial window matching your tracking parameters
#     z_tol=0.05,
#     x_periodic=True
# )

# print(f"Isolated {len(filtered_initiations)} true initiation events across the test timeline.")

# # Filter plumes where z_c <= 0.7
# filtered_plume_records = [p for p in filtered_initiations if p['z_c'] <= 0.6]

# # Number of plumes remaining
# no_init_plumes = len(filtered_plume_records)
# print(len(filtered_plume_records))
# filtered_records = filtered_plume_records

# variables = {'dudx': ds_dudx, 'CAPE': ds_cape, 'CIN': ds_cin, 'KE': ds_ke, 'CIN-KE': ds_cin_ke, 'q_global': ds_q_global, 'KE_global': ds_KE_global}
# aggregation_dict = {'dudx': 'min', 'CAPE': 'max', 'CIN': 'min', 'KE': 'max', 'CIN-KE': 'min'}


# preinit_window = 8
# postinit_window = 8
# preinit_data = extract_preinit_variables(filtered_records, variables, aggregation_dict, x_downsample, z_downsample,
#                               preinit_window=preinit_window, postinit_window=postinit_window, x_radius=1, z_radius=1)

# variables = ['CAPE','CIN','CIN-KE','dudx',
#              'CAPE_surface','CIN_surface','KE_global','q_global']

# def build_feature_matrix(preinit_data, variables):
#     feature_list = []
    
#     for plume in preinit_data['plume']:
#         feats = []
#         for var in variables:
#             arr = np.array(plume['vars'][var])   # shape (17,)
            
#             if len(arr) == 17:
#                 feats.extend(arr)               # add all 17 values
#             else:
#                 # pad with nan to length 17 if short
#                 padded = np.full(17, np.nan)
#                 padded[:len(arr)] = arr
#                 feats.extend(padded)
                
#         feature_list.append(feats)
    
#     X = np.array(feature_list)   # shape (N_plumes, 17 * 8)
#     return X

# X = build_feature_matrix(preinit_data, variables)
# print("X shape:", X.shape)

# # 1. Load the frozen models from Chapter 3 run
# loaded_scaler   = joblib.load(input_path_testdata+'/models/chapter3_scaler.pkl')
# loaded_pca      = joblib.load(input_path_testdata+'/models/chapter3_pca.pkl')
# loaded_cluster  = joblib.load(input_path_testdata+'/models/chapter3_kmeans.pkl')

# col_mean = np.nanmean(X, axis=0)
# inds = np.where(np.isnan(X))
# X[inds] = np.take(col_mean, inds[1])

# # 2. Scale test data using the Chapter 3 scaler 
# X_new_scaled = loaded_scaler.transform(X) 
# print("Shape of test data going into KMeans:", X_new_scaled.shape) #(51, 136)

# # 3. Predict the cluster labels DIRECTLY using the 136 scaled features
# test_clusters = loaded_cluster.predict(X_new_scaled)
# print("Cluster labels calculated")

# # 4. OPTIONAL: Project to PCA ONLY for visualization/plotting
# # Your KMeans doesn't need this, but your matplotlib scatter plot does!
# X_new_pca = loaded_pca.transform(X_new_scaled) 
# print("Shape of test data projected for plotting:", X_new_pca.shape) # Should be (51, 2)


# # 1. Count the occurrences of each cluster label
# cluster_counts = Counter(test_clusters)

# # 2. Print a clean summary to your Slurm log
# for cluster_id in sorted(cluster_counts.keys()):
#     count = cluster_counts[cluster_id]
#     percentage = (count / len(test_clusters)) * 100
    
#     # Optional: Swap these out with your exact Chapter 3 regime names
#     print(f"🔹 Cluster {cluster_id}: {count} events ({percentage:.1f}%)")

# for i, plume in enumerate(preinit_data['plume']):
#     plume['cluster'] = int(test_clusters[i])

# # ----- plotting ------
# stats3 = compute_threshold_stats(preinit_data)
# #df3 = stats_to_dataframe(stats3, k=3)


# KE_init = []
# q_init = []
# clusters = []
# for p in preinit_data['plume']:
#     arr_ke = np.array(p['vars']['KE_global'])
#     arr_q  = np.array(p['vars']['q_global'])

#     # ensure valid and correct length
#     if arr_ke.ndim != 1 or arr_q.ndim != 1:
#         continue
#     if len(arr_ke) != 17 or len(arr_q) != 17:
#         continue
    
#     KE_init.append(arr_ke[8])  # timestep 0 (init)
#     q_init.append(arr_q[8])
#     clusters.append(p['cluster'])

# KE_init = np.array(KE_init)
# q_init = np.array(q_init)

# times = np.array([p['time'] for p in preinit_data['plume']])
# x_locs = np.array([p['x_c'] for p in preinit_data['plume']])
# clusters_arr = np.array(test_clusters)

# n_clusters = 3
# colors = ['tab:orange', 'tab:green', 'tab:purple'] #plt.cm.tab10.colors[1:n_clusters+1]

# fig = plt.figure(figsize=(24, 6))
# gs = GridSpec(1, 2, width_ratios=[6, 18], figure=fig)
# fontsize  = 18
# labelsize = 16

# # --- Subplot 1: q vs KE ---
# ax0 = fig.add_subplot(gs[0])
# for c in range(n_clusters):
#     idx = clusters_arr == c
#     ax0.scatter(q_init[idx], KE_init[idx], color=colors[c], label=f'Cluster {c+1}', s=80, edgecolors='k')

# ax0.plot(ds_q_global, ds_KE_global, color='tab:blue', alpha=0.5)
# ax0.set_xlabel(r"$\overline{q}$", fontsize=fontsize)
# ax0.set_ylabel(r"$\overline{KE}$", fontsize=fontsize)
# ax0.grid(True)
# ax0.legend(fontsize=fontsize)
# ax0.tick_params(axis='both', which='major', labelsize=labelsize)
# ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes, fontsize=fontsize, fontweight='bold', va='top')
# #ax0.set_title("Global KE vs q", fontsize=14)

# # --- Subplot 2: Time vs x-location ---
# ax1 = fig.add_subplot(gs[1])
# norm = SymLogNorm(linthresh=1e-4, linscale=1.0, vmin=-0.2, vmax=0.2)

# # 1) dudx
# zval=28
# c0 = ax1.pcolormesh(
#     time_vals[0:None], x,
#     dudx[0:None, :, zval].T,
#     cmap='RdBu', norm=norm, rasterized=True 
# )
# cbar0 = fig.colorbar(c0, ax=ax1)
# cbar0.set_label('$\partial u/\partial x$', fontsize=fontsize)
# cbar0.ax.tick_params(labelsize=labelsize)

# for c in range(n_clusters):
#     idx = clusters_arr == c
#     ax1.scatter(time_vals[times[idx]], x_locs[idx], color=colors[c], label=f'Cluster {c+1}', s=80, edgecolors='white')
    

# ax1.set_xlabel("Time", fontsize=fontsize)
# ax1.set_ylabel("x", fontsize=fontsize)
# ax1.tick_params(axis='both', which='major', labelsize=labelsize)
# #ax1.grid(True)
# #ax1.set_xlim(5000,10000)
# ax1.set_ylim(0,20)
# ax1.legend(fontsize=fontsize)
# ax1.text(0.01, 0.98, "(b)", transform=ax1.transAxes, fontsize=fontsize, fontweight='bold', va='top')

# #ax1.set_title("Plume x-location vs time", fontsize=14)

# plt.tight_layout()
# #plt.show()
# output_plot_path = input_path_testdata + '/ClusterPositions_Corrections_testdata.png'
# fig.savefig(output_plot_path, facecolor='white', transparent=False,  dpi=300)

# # Create a container to export your true baseline validation records
# true_export_records = []

# # Loop over your tracked, filtered initiations and pair them with their predicted cluster labels
# for idx, rec in enumerate(filtered_records):
#     true_export_records.append({
#         'event_id': idx,
#         'time_idx': int(rec['time']),           # Relative index (0 to 629)
#         'actual_time': float(time_vals[rec['time']]), # Absolute simulation time
#         'x_c': float(rec['x_c']),               # Physical horizontal position
#         'z_c': float(rec['z_c']),               # Core level height (z0=4 or z~0.4)
#         'assigned_cluster': int(test_clusters[idx]) # Chapter 3 structural regime (0, 1, or 2)
#     })

# # Define the export folder path
# export_dir = input_path_testdata
# os.makedirs(export_dir, exist_ok=True)
# export_file_path = os.path.join(export_dir, 'true_plume_initiation_records.pkl')
# # Dump the structured tracking database to disk
# joblib.dump(true_export_records, export_file_path)
# print(f"Saved portable file to:   {export_file_path}")

# --------------- CODE PART 2 ----------------------------
# master_export_path = input_path_testdata+'/test_metrics/actual_composite_records.pkl'
# composite_records_all = joblib.load(master_export_path)
# images_path = input_path_testdata+'/test_images/'

# print(f"Successfully loaded {len(composite_records_all)} total matched records")

# # View the first record in the database
# sample_record = composite_records_all[0]
# print("\n --- INSPECTING SINGLE RECORD [0] ---")
# print(f"Assigned Cluster: {sample_record['cluster']}")
# print(f"Model Lead Time : {sample_record['lead_time']} steps")
# print(f"True Time (t0)  : {sample_record['true_time']}")
# print(f"True X-coord    : {sample_record['true_x']}")
# print(f"Extracted dudx Shape: {sample_record['dudx'].shape}") # Expected: (20, 64)

# n_time_steps, n_x_points = sample_record['dudx'].shape  # Shoud extract (20, 64)

# dt = 2.0          # The cadence inside the neighborhood slice (dt=1)
# dx = 20.0 / 256.0 # Horizontal grid cell resolution spacing

# # Center axes so (0,0) marks the exact initiation focus point
# time_rel = (np.arange(n_time_steps) - 16) * dt 
# dist_rel = (np.arange(n_x_points) - (n_x_points // 2)) * dx

# min_lt, max_lt = 30, 45
# LT = 15

# print(f"Processing Lead Time window: {min_lt/LT} to {max_lt/LT} LTs...")

# cluster_means = {}
# global_max_val = 0.0

# for i in range(3):
#     # Select slices that match BOTH the cluster ID and the targeted lead time frame
#     stack_slices = [
#         rec['dudx'] for rec in composite_records_all 
#         if rec['cluster'] == i and min_lt <= rec['lead_time'] <= max_lt
#     ]
    
#     if len(stack_slices) > 0:
#         # Calculate the nan-safe mean structure for this specific cluster
#         mean_profile = np.nanmean(np.array(stack_slices), axis=0)
#         cluster_means[i] = mean_profile
        
#         # Track the absolute maximum to normalize your symmetric color bounds
#         cluster_max = np.nanmax(np.abs(mean_profile))
#         if cluster_max > global_max_val:
#             global_max_val = cluster_max
#     else:
#         cluster_means[i] = None

# if global_max_val == 0.0:
#     global_max_val = 1.0
#     print("Warning: No plumes matched this lead time criteria across any cluster.")
# print(f"Global symmetric limit calculated: ±{global_max_val:.4f}")

# norm = SymLogNorm(linthresh=0.005, linscale=1, vmin=-global_max_val, vmax=global_max_val, base=10)

# labs_top = ['(a)', '(b)', '(c)']
# fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True, tight_layout=True)

# for i in range(3):
#     # Re-verify the exact slice list for counting purposes
#     stack_slices = [
#         rec['dudx'] for rec in composite_records_all 
#         if rec['cluster'] == i and min_lt <= rec['lead_time'] <= max_lt
#     ]
#     n_plumes = len(stack_slices)

#     composite_mean_filtered = cluster_means[i]
    
#     if composite_mean_filtered is not None:
#         # Render the spatial gradient structure using the global logarithmic norm
#         im = ax[i].pcolormesh(dist_rel, time_rel, composite_mean_filtered, 
#                               shading='nearest', 
#                               cmap='RdBu_r',
#                               rasterized=True, 
#                               norm=norm)
#     else:
#         # Fill with a gray placeholder background if a cluster has zero events
#         ax[i].patch.set_facecolor('lightgray')
#         ax[i].text(0.5, 0.5, 'No Event Slices Matched', 
#                    transform=ax[i].transAxes, ha='center', va='center')
#         continue

#     # Add focus crosshair markers at the initiation pivot point
#     ax[i].scatter(0, 0, color='black', marker='x', s=200, linewidths=2, zorder=5, label='Initiation')
#     ax[i].axhline(0, color='black', linestyle=':', alpha=0.5)
#     ax[i].axvline(0, color='black', linestyle=':', alpha=0.5)

#     # Label styling and axis boundaries
#     ax[i].set_xlabel('Relative Distance', fontsize=16)
#     ax[i].text(0.02, 0.98, labs_top[i], transform=ax[i].transAxes, fontsize=16, fontweight='bold', va='top')
#     ax[i].set_ylim(-30, 4)
#     ax[i].tick_params(axis='both', which='major', labelsize=14)
#     ax[i].set_title(f"Cluster {i+1} ($n={n_plumes}$)", fontsize=14, pad=10)

# # Build and append a single coordinated global colorbar to axis pane 2
# cb = fig.colorbar(im, ax=ax[2], pad=0.02)
# cb.set_label(r'Mean Convergence $\partial u / \partial x$', fontsize=14)
# cb.ax.tick_params(labelsize=12)

# # Set common shared Y-axis tracking descriptor label
# ax[0].set_ylabel('Relative Time', fontsize=16)

# # Export the high-res figure asset for presentation/thesis incorporation
# fig.savefig(images_path+f"/actual_ClusterAnalysis_LT_{min_lt}-{max_lt}.png", facecolor='white', transparent=False, dpi=300)

# print(f"Vector composite plot successfully saved")
# #plt.show()

# # Define your three lead-time windows
# lt_windows = [
#     (0, 14, "Short Lead Times"),
#     (15, 29, "Mid Lead Times"),
#     (30, 44, "Long Lead Times")
# ]

# print("=" * 65)
# print("     VALIDATION TRACKER: NON-NAN DATA STEPS PER REGIME")
# print("=" * 65)

# for min_lt, max_lt, label in lt_windows:
#     before_counts = []
#     after_counts = []
#     total_events = 0
    
#     for rec in composite_records_all:
#         if min_lt <= rec['lead_time'] <= max_lt:
#             total_events += 1
#             # rec['dudx'] shape is (20, 64)
#             # Row index 16 is t0 (initiation). 
#             # Rows 0 to 15 are BEFORE init (16 steps total)
#             # Rows 16 to 19 are AT/AFTER init (4 steps total)
#             dudx_block = rec['dudx']
            
#             # Count along the time axis (axis 0) if ANY valid numerical value exists in the row
#             valid_time_mask = ~np.isnan(dudx_block).all(axis=1)
            
#             # Split into before and after timelines
#             valid_before = valid_time_mask[0:16]
#             valid_after  = valid_time_mask[16:20]
            
#             before_counts.append(np.sum(valid_before))
#             after_counts.append(np.sum(valid_after))
            
#     if total_events > 0:
#         mean_before = np.mean(before_counts)
#         mean_after  = np.mean(after_counts)
#         print(f"🔹 {label} (LT {min_lt}-{max_lt} steps) | Samples: n = {total_events}")
#         print(f"   • Avg steps BEFORE initiation (Max 16): {mean_before:.1f} steps")
#         print(f"   • Avg steps AFTER initiation  (Max  4): {mean_after:.1f} steps")
#     else:
#         print(f"🔹 {label} (LT {min_lt}-{max_lt} steps) | No matching events found.")
#     print("-" * 65)

# # S(t)
# # Define our target lead time strata to loop over
# lt_windows = [
#     (0, 14, "Short_Lead_Times_0-14"),
#     (15, 29, "Mid_Lead_Times_15-29"),
#     (30, 44, "Long_Lead_Times_30-44")
# ]

# # Configure physical axes resolutions
# n_time_steps = 20
# dt = 2.0  # Master time interval cadence spacing
# time_rel = (np.arange(n_time_steps) - 16) * dt  # Spans from -32.0 to +6.0

# # Define colors for consistent cluster tracking
# cluster_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}

# for min_lt, max_lt, file_label in lt_windows:
#     print(f"📈 Processing S(t) Wave-Strength for Lead Times {min_lt} to {max_lt}...")
    
#     # Pre-calculate wave strength vectors to locate global dynamic range maxima
#     precomputed_s_t = {}
#     n_counts = {}
#     max_s_val = 0.0
    
#     for i in range(3):
#         # Extract matching profile data matrices
#         stack_slices = [
#             rec['dudx'] for rec in composite_records_all 
#             if rec['cluster'] == i and min_lt <= rec['lead_time'] <= max_lt
#         ]
#         n_counts[i] = len(stack_slices)
        
#         if len(stack_slices) > 0:
#             # Step A: Compute the mean spatial structure across events
#             composite_mean_filtered = np.nanmean(np.array(stack_slices), axis=0)
            
#             # Step B: Calculate 1D standard deviation across the spatial axis (axis 1)
#             # Use nanstd to safely ignore truncated history padding
#             wave_t = np.nanstd(composite_mean_filtered, axis=1)
#             precomputed_s_t[i] = wave_t
            
#             # Track peak value to scale y-axis uniformly
#             local_max = np.nanmax(wave_t)
#             if local_max > max_s_val:
#                 max_s_val = local_max
#         else:
#             precomputed_s_t[i] = None

#     # If no data is available, set a fallback limit to prevent crash
#     if max_s_val == 0.0:
#         max_s_val = 0.5
        
#     # --- Plot Rendering ---
#     fig, ax = plt.subplots(1, figsize=(8, 4.5), tight_layout=True)
    
#     for i in range(3):
#         wave_t = precomputed_s_t[i]
#         n_plumes = n_counts[i]
        
#         if wave_t is not None:
#             ax.plot(time_rel, wave_t, 
#                     label=f"Cluster {i+1} ($n={n_plumes}$)", 
#                     color=cluster_colors[i], 
#                     linewidth=2.5)
            
#     # Add clear indicator vertical line at the exact initiation pivot
#     ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Initiation ($t_0$)')
    
#     # Graph labels and axis scaling custom tailoring
#     ax.set_xlabel('Relative Time ($t - t_0$ units)', fontsize=16)
#     ax.set_ylabel(r"Wave Strength $S(t)$", fontsize=16)
    
#     # Sync Y-axis scaling ceiling dynamically with a 10% safety headroom margin
#     ax.set_ylim(0, max_s_val * 1.1)
#     #ax.set_ylim(0, 0.01)
#     ax.set_xlim(time_rel[0], time_rel[-1])
#     #ax.set_xlim(-6,4)
    

#     ax.tick_params(axis='both', which='major', labelsize=14)
#     ax.grid(True, linestyle=':', alpha=0.6)
#     ax.legend(fontsize=12, loc='upper left', frameon=True, facecolor='white', edgecolor='lightgray')
#     ax.set_title(f"Structural Wave Profile Buildup: {file_label.replace('_', ' ')}", fontsize=14, pad=12, weight='bold')
    
#     # Export the high-resolution vector layout asset
#     fig.savefig(images_path+f"/actual_ClusterAnalysis_S_plot_tests_{min_lt}-{max_lt}.png", facecolor='white', transparent=False, dpi=300)
#     print(f"💾 Saved S(t) plot")
    
#     plt.show()

# print("\n🎉 All wave strength line graphs successfully compiled!")

# ---------- CODE PART 3 -------------
#POD
input_path_positions = 'Ra2e8/POD_ESN/Thesis/64modes/Run_n_units1000_ensemble5_normalisationon_washout3_config4101/GP36_4/further_analysis/initation_score'

x_positions_pred_all = np.load(input_path_positions+'/x_positions_pred_all.npy')
x_positions_truth_all = np.load(input_path_positions+'/x_positions_truth_all.npy')

print(f"shape of x_positions_pred_all {np.shape(x_positions_pred_all)}")

master_export_path = input_path_testdata+'/test_metrics/actual_composite_records.pkl'
composite_records_all = joblib.load(master_export_path)


def find_initiations_truth_clustered(x_positions_truth, test_idx, composite_records_all, x_min=0, x_max=20):
    """
    Find plume initiations in truth array and append their corresponding cluster ID 
    by looking up matches in composite_records_all using a periodic spatiotemporal threshold.

    Returns:
    -------
    initiations : list of (t_init, x_init, p, cluster_id)
    """
    T, P = x_positions_truth.shape
    initiations = []
    L = x_max - x_min

    for p in range(P):
        active = ~np.isnan(x_positions_truth[:, p])
        if not active.any():
            continue

        was_active = False
        for t in range(T):
            if active[t]:
                if not was_active:
                    if t > 0:  # <-- skip initiations at t=0
                        x0 = x_positions_truth[t, p]
                        
                        # --- NEW LOOKUP LOGIC ---
                        cluster_id = -1  # Default fallback if no match found
                        
                        # Search through the grouped composite records
                        for rec in composite_records_all:
                            # 1. Match the current test segment interval id
                            rec_test_id = int(rec.get('test_interval_id', rec.get('test_idx', 0)))
                            if rec_test_id != test_idx:
                                continue
                            
                            # 2. Check for an exact match on the local time index via lead_time
                            # and a very close spatial match on true_x (handling periodic domain)
                            if int(rec['lead_time']) == t:
                                dx_raw = abs(x0 - rec['true_x'])
                                dx = min(dx_raw, L - dx_raw)
                                
                                if dx < 0.1:
                                    cluster_id = int(rec['cluster'])
                                    break  # Exact match found, exit records loop
                        
                        # Only include this event if it successfully mapped to your cluster database
                        if cluster_id != -1:
                            initiations.append((t, x0, p, cluster_id))
                            
                    was_active = True
            else:
                was_active = False  # reset when plume goes away

    return initiations

def find_initiations_truth(x_positions_truth):
    """
    Find plume initiations (including re-activations) in truth array,
    but ignore cases where the plume is already active at t=0.

    x_positions_truth : ndarray (T, P)
        True plume positions over time. NaN if no plume.

    Returns:
    initiations : list of (t_init, x_init, p)
        One entry per plume slot per initiation event:
        - t_init = time index when plume first appears after being absent
        - x_init = plume position at that time
        - p      = slot index
    """
    T, P = x_positions_truth.shape
    initiations = []

    for p in range(P):
        active = ~np.isnan(x_positions_truth[:, p])
        if not active.any():
            continue

        was_active = False
        for t in range(T):
            if active[t]:
                if not was_active:
                    if t > 0:  # <-- skip initiations at t=0
                        x0 = x_positions_truth[t, p]
                        initiations.append((t, x0, p))
                    was_active = True
            else:
                was_active = False  # reset when plume goes away

    return initiations

def find_initiations_pred(x_positions_pred, x_positions_truth):
    """
    Find plume initiations (including re-activations) in pred array,
    but ignore cases where the plume is already active at t=0 if there is a 
    actual plume at t=0.

    x_positions_truth : ndarray (T, P)
        True plume positions over time. NaN if no plume.

    Returns:
    initiations : list of (t_init, x_init, p)
        One entry per plume slot per initiation event:
        - t_init = time index when plume first appears after being absent
        - x_init = plume position at that time
        - p      = slot index
    """
    T, P = x_positions_pred.shape
    initiations = []

    true_active_at_zero = ~np.isnan(x_positions_truth[0, :])
    true_active_at_zero_flag = true_active_at_zero.any()

    for p in range(P):
        active = ~np.isnan(x_positions_pred[:, p])
        if not active.any():
            continue

        was_active = False
        for t in range(T):
            if active[t]:
                if not was_active:
                    if true_active_at_zero_flag:
                        if t > 0:  # <-- skip initiations at t=0
                            x0 = x_positions_pred[t, p]
                            initiations.append((t, x0, p))
                        was_active = True
                    else:
                        x0 = x_positions_pred[t, p]
                        initiations.append((t, x0, p))
                        was_active = True
            else:
                was_active = False  # reset when plume goes away

    return initiations

def match_initiations_hits_clustered(all_pred_inits, true_inits, delta_t=1, delta_x=1, x_min=0, x_max=20):
    """
    Match predicted plume initiations to clustered true initiations (one-to-one matching)
    and compute hits and misses by cluster index, while tracking global false alarms.
    """
    # Flatten all ensemble predictions
    pred_all = np.array([pt for ens in all_pred_inits for pt in ens])
    true_all = np.array(true_inits)
    
    pred_no_inits = len(pred_all)
    true_no_inits = len(true_all)

    # Initialize dictionaries to hold independent metrics for Cluster 0, 1, and 2
    hits_by_cluster = {0: 0, 1: 0, 2: 0}
    misses_by_cluster = {0: 0, 1: 0, 2: 0}
    false_alarms = 0

    # Handle empty cases cleanly
    if len(pred_all) == 0 and len(true_all) == 0:
        return true_no_inits, pred_no_inits, hits_by_cluster, 0, misses_by_cluster
        
    if len(true_all) == 0:
        return true_no_inits, pred_no_inits, hits_by_cluster, len(pred_all), misses_by_cluster
        
    if len(pred_all) == 0:
        for c in true_all[:, 3].astype(int):
            misses_by_cluster[c] += 1
        return true_no_inits, pred_no_inits, hits_by_cluster, 0, misses_by_cluster

    # Extract coordinates (true_all has 4 columns: t, x, p, cluster)
    t_pred, x_pred = pred_all[:, 0], pred_all[:, 1]
    t_true, x_true, cluster_true = true_all[:, 0], true_all[:, 1], true_all[:, 3].astype(int)

    # Vectorized distance evaluations (Your exact logic)
    dt = np.abs(t_pred[:, None] - t_true[None, :])
    L = x_max - x_min
    dx_raw = np.abs(x_pred[:, None] - x_true[None, :])
    dx = np.minimum(dx_raw, L - dx_raw)

    # Determine validation bounding box alignment
    within_box = (dt <= delta_t) & (dx <= delta_x)

    # Bookkeeping templates
    matched_true = np.zeros(len(true_all), dtype=bool)
    matched_pred = np.zeros(len(pred_all), dtype=bool)

    # Executing strict one-to-one pairing loop
    for i_pred in range(len(pred_all)):
        candidates = np.where(within_box[i_pred, :] & (~matched_true))[0]
        if len(candidates) > 0:
            best_idx = candidates[np.argmin(np.sqrt(dt[i_pred, candidates]**2 + dx[i_pred, candidates]**2))]
            
            matched_true[best_idx] = True
            matched_pred[i_pred] = True
            
            # Key modification: increment the hit counter for this specific event's cluster type!
            c_id = cluster_true[best_idx]
            hits_by_cluster[c_id] += 1

    # False Alarms remain global (predictions that mapped to absolutely nothing)
    false_alarms = np.sum(~matched_pred)
    
    # Calculate misses by grouping unmatched true profiles into their corresponding cluster archetype
    unmatched_true_indices = np.where(~matched_true)[0]
    for idx in unmatched_true_indices:
        c_id = cluster_true[idx]
        misses_by_cluster[c_id] += 1

    return true_no_inits, pred_no_inits, hits_by_cluster, false_alarms, misses_by_cluster

# Initialize metrics tracking lists across the 40 windows
all_ens_no_true_inits = []
all_ens_no_pred_inits = []
all_ens_falsealarms = []

# Separate lists to track hits and misses for each individual cluster archetype
cluster_hits = {0: [], 1: [], 2: []}
cluster_misses = {0: [], 1: [], 2: []}

delta_t = 1
delta_x = 1

for test_idx in range(40):
    all_ens_x_init_pred = []
    x_init_truth_old = find_initiations_truth(x_positions_truth_all[:,:,test_idx,0])
    x_init_truth_new = find_initiations_truth_clustered(x_positions_truth_all[:,:,test_idx,0], test_idx, composite_records_all, x_min=0, x_max=20)
    print(f"old init: {x_init_truth_old}")
    print(f"new init: {x_init_truth_new}")

    for ens_idx in range(5):
        x_init_pred  = find_initiations_pred(x_positions_pred_all[:,:,test_idx,ens_idx], x_positions_truth_all[:,:,test_idx,ens_idx])
        all_ens_x_init_pred.append(x_init_pred)

    # Run your updated matching function
    true_no_inits, pred_no_inits, hits_by_c, false_alarms, misses_by_c = match_initiations_hits_clustered(
        all_ens_x_init_pred, 
        x_init_truth_new, 
        delta_t=delta_t, 
        delta_x=delta_x
    )

    # Track global totals
    all_ens_no_true_inits.append(true_no_inits)
    all_ens_no_pred_inits.append(pred_no_inits)
    all_ens_falsealarms.append(false_alarms)
    
    # Track independent breakdown lists per cluster type
    for c in [0, 1, 2]:
        cluster_hits[c].append(hits_by_c[c])
        cluster_misses[c].append(misses_by_c[c])
    
# --- SUMMARY SCORE RECAP OUTPUT GENERATION ---
print("="*60)
print(f"GLOBAL RESULTS MATRIX SUMMARY (delta_t={delta_t}, delta_x={delta_x})")
print(f"Total True Inits: {sum(all_ens_no_true_inits)} | Total Predictions Assessed: {sum(all_ens_no_pred_inits)}")
print(f"Global False Alarms: {sum(all_ens_falsealarms)}")
print("="*60)

for c in [0, 1, 2]:
    total_h = sum(cluster_hits[c])
    total_m = sum(cluster_misses[c])
    global_fa = sum(all_ens_falsealarms)
    global_h = sum(sum(cluster_hits[i]) for i in [0, 1, 2])
    
    # Recall (Capture accuracy for this physical plume archetype)
    recall_c = total_h / (total_h + total_m) if (total_h + total_m) > 0 else 0
    
    # Precision Contribution (Proportion of total positive assertions belonging to this cluster hit category)
    precision_c = total_h / (global_h + global_fa) if (global_h + global_fa) > 0 else 0
    
    f1_c = 2 * precision_c * recall_c / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0
    
    print(f"Cluster Profile Type {c+1} Performance Metrics:")
    print(f"  -> Total Caught Hits: {total_h} | Total Missed Plumes: {total_m}")
    print(f"  -> Conditional Archetype Recall  (R_I): {recall_c:.4f}")
    print(f"  -> Contribution System Precision (P_I): {precision_c:.4f}")
    print(f"  -> Calculated Harmonic F1-Score Value  : {f1_c:.4f}")
    print("-"*60)