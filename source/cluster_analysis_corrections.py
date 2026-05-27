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

input_path_testdata = 'Ra2e8/POD_ESN/Thesis/64modes/Run_n_units1000_ensemble5_normalisationon_washout3_config4101/GP36_4/further_analysis/corrections_clusters'
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


# CODE
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')

x_downsample = x[::4]
z_downsample = z[::4]

dataset         = np.load(input_path_testdata+'/all_test_data.npy')
alt_dataset     = np.load(input_path_testdata+'/alternative_test_data.npy')
x_positions     = np.load(input_path_testdata+'/all_x_positions_truth.npy')

print(f"shape of dataset = {np.shape(dataset)}")
print(f"shape of alt dataset = {np.shape(alt_dataset)}")
print(f"shape of x_positions = {np.shape(x_positions)}")


q = dataset[...,0]
w = dataset[...,1]
dudx = alt_dataset[...,0]

with h5py.File(input_path_testdata+'/CIN_CAPE_t00000_to_00630.h5', 'r') as df:
    print(df.keys())
    CAPE = df['CAPE'][:] 
    CIN  = df['CIN'][:] 
    time_vals = df['time_vals'][:] 

start_time = time_vals[0]
end_time   = time_vals[-1]
print(f"start time={time_vals[0]}, end time = {time_vals[-1]}")
print(f"shape of CIN dataset = {np.shape(CIN)}")

KE = 0.5 * w * w
CIN_KE = CIN - KE 

KE_global_all =  np.load(input_path+'/KE5000_30000.npy')
q_global_all  =  np.load(input_path+'/q5000_30000.npy')

global_start_idx = int(time_vals[0] - 5000)
global_end_idx = global_start_idx + (len(time_vals) * 2)

# 3. Slice the window and downsample from dt=1 to dt=2 in one go
ds_KE_global = KE_global_all[global_start_idx : global_end_idx : 2]
ds_q_global  = q_global_all[global_start_idx : global_end_idx : 2]

print(f"Polished Global KE shape: {ds_KE_global.shape}")
print(f"Target Spatial shape:    {len(time_vals)}")

assert len(ds_KE_global) == len(time_vals), (
    f"Mismatch! Sliced global array has {len(ds_KE_global)} steps, "
    f"but your test fields expect {len(time_vals)} steps."
)
print("Global arrays perfectly aligned to your 630-step test cadence!")



# Create a new dudx array of the same size as new_array
ds_dudx    = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))
ds_ke      = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))
ds_cin     = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))
ds_cape    = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))
ds_cin_ke  = np.zeros((len(time_vals), len(x_downsample), len(z_downsample)))

# Iterate over the time dimension
for t in range(len(time_vals)):
    if t % 250 == 0:
        print(f"time_val = {time_vals[t]}")
    # Iterate over the subgrids along the x and z dimensions
    for i in range(0, len(x), 4):
        for j in range(0, len(z), 4):
            # Extract the 4x4 subgrid and compute its minimum value
            subgrid = dudx[t, i:i+4, j:j+4]
            ds_dudx[t, i // 4, j // 4] = np.min(subgrid)

            subgrid = CAPE[t, i:i+4, j:j+4]
            ds_cape[t, i // 4, j // 4] = np.max(subgrid)

            subgrid = CIN[t, i:i+4, j:j+4]
            ds_cin[t, i // 4, j // 4] = np.min(subgrid)

            subgrid = KE[t, i:i+4, j:j+4]
            ds_ke[t, i // 4, j // 4] = np.max(subgrid)

            subgrid = CIN_KE[t, i:i+4, j:j+4]
            ds_cin_ke[t, i // 4, j // 4] = np.min(subgrid)
 
minm_z, maxm_z = 0, 10 
ds_dudx_ss   = ds_dudx[...,minm_z:maxm_z]
ds_cape_ss   = ds_cape[...,minm_z:maxm_z]
ds_cin_ss    = ds_cin[...,minm_z:maxm_z]
ds_ke_ss     = ds_ke[...,minm_z:maxm_z]
ds_cin_ke_ss = ds_cin_ke[...,minm_z:maxm_z]


plume_records = []

# Loop over your 630 test time steps and your plume slots (e.g., max_plumes=3)
T, P = x_positions.shape

for t in range(T):
    for p in range(P):
        x_val = x_positions[t, p]
        
        # Skip if no plume was detected in this slot at this time step
        if np.isnan(x_val):
            continue
            
        # Append structured entry using your target steering level
        plume_records.append({
            'time': t,
            'x_c': x_val,
            'z_c': z_downsample[7],  # Forces your exact target level z0=4
            'mask_indices': None     # Not needed since we have explicit centroids
        })

print(f"Generated {len(plume_records)} raw tracking records.")

filtered_initiations = filter_first_timesteps_with_tolerance_robust(
    plume_records=plume_records,
    x_coords=x_downsample,
    z_coords=z_downsample,
    x_tol=0.2,   # Spatial window matching your tracking parameters
    z_tol=0.05,
    x_periodic=True
)

print(f"Isolated {len(filtered_initiations)} true initiation events across the test timeline.")

# Filter plumes where z_c <= 0.7
filtered_plume_records = [p for p in filtered_initiations if p['z_c'] <= 0.6]

# Number of plumes remaining
no_init_plumes = len(filtered_plume_records)
print(len(filtered_plume_records))
filtered_records = filtered_plume_records

variables = {'dudx': ds_dudx, 'CAPE': ds_cape, 'CIN': ds_cin, 'KE': ds_ke, 'CIN-KE': ds_cin_ke, 'q_global': ds_q_global, 'KE_global': ds_KE_global}
aggregation_dict = {'dudx': 'min', 'CAPE': 'max', 'CIN': 'min', 'KE': 'max', 'CIN-KE': 'min'}


preinit_window = 8
postinit_window = 8
preinit_data = extract_preinit_variables(filtered_records, variables, aggregation_dict, x_downsample, z_downsample,
                              preinit_window=preinit_window, postinit_window=postinit_window, x_radius=1, z_radius=1)

variables = ['CAPE','CIN','CIN-KE','dudx',
             'CAPE_surface','CIN_surface','KE_global','q_global']

def build_feature_matrix(preinit_data, variables):
    feature_list = []
    
    for plume in preinit_data['plume']:
        feats = []
        for var in variables:
            arr = np.array(plume['vars'][var])   # shape (17,)
            
            if len(arr) == 17:
                feats.extend(arr)               # add all 17 values
            else:
                # pad with nan to length 17 if short
                padded = np.full(17, np.nan)
                padded[:len(arr)] = arr
                feats.extend(padded)
                
        feature_list.append(feats)
    
    X = np.array(feature_list)   # shape (N_plumes, 17 * 8)
    return X

X = build_feature_matrix(preinit_data, variables)
print("X shape:", X.shape)

# 1. Load the frozen models from Chapter 3 run
loaded_scaler   = joblib.load(input_path_testdata+'/models/chapter3_scaler.pkl')
loaded_pca      = joblib.load(input_path_testdata+'/models/chapter3_pca.pkl')
loaded_cluster  = joblib.load(input_path_testdata+'/models/chapter3_kmeans.pkl')

col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

# 2. Scale test data using the Chapter 3 scaler 
X_new_scaled = loaded_scaler.transform(X) 
print("Shape of test data going into KMeans:", X_new_scaled.shape) #(51, 136)

# 3. Predict the cluster labels DIRECTLY using the 136 scaled features
test_clusters = loaded_cluster.predict(X_new_scaled)
print("Cluster labels calculated")

# 4. OPTIONAL: Project to PCA ONLY for visualization/plotting
# Your KMeans doesn't need this, but your matplotlib scatter plot does!
X_new_pca = loaded_pca.transform(X_new_scaled) 
print("Shape of test data projected for plotting:", X_new_pca.shape) # Should be (51, 2)


# 1. Count the occurrences of each cluster label
cluster_counts = Counter(test_clusters)

# 2. Print a clean summary to your Slurm log
for cluster_id in sorted(cluster_counts.keys()):
    count = cluster_counts[cluster_id]
    percentage = (count / len(test_clusters)) * 100
    
    # Optional: Swap these out with your exact Chapter 3 regime names
    print(f"🔹 Cluster {cluster_id}: {count} events ({percentage:.1f}%)")

for i, plume in enumerate(preinit_data['plume']):
    plume['cluster'] = int(test_clusters[i])

# ----- plotting ------
stats3 = compute_threshold_stats(preinit_data)
#df3 = stats_to_dataframe(stats3, k=3)


KE_init = []
q_init = []
clusters = []
for p in preinit_data['plume']:
    arr_ke = np.array(p['vars']['KE_global'])
    arr_q  = np.array(p['vars']['q_global'])

    # ensure valid and correct length
    if arr_ke.ndim != 1 or arr_q.ndim != 1:
        continue
    if len(arr_ke) != 17 or len(arr_q) != 17:
        continue
    
    KE_init.append(arr_ke[8])  # timestep 0 (init)
    q_init.append(arr_q[8])
    clusters.append(p['cluster'])

KE_init = np.array(KE_init)
q_init = np.array(q_init)

times = np.array([p['time'] for p in preinit_data['plume']])
x_locs = np.array([p['x_c'] for p in preinit_data['plume']])
clusters_arr = np.array(test_clusters)

n_clusters = 3
colors = ['tab:orange', 'tab:green', 'tab:purple'] #plt.cm.tab10.colors[1:n_clusters+1]

fig = plt.figure(figsize=(24, 6))
gs = GridSpec(1, 2, width_ratios=[6, 18], figure=fig)
fontsize  = 18
labelsize = 16

# --- Subplot 1: q vs KE ---
ax0 = fig.add_subplot(gs[0])
for c in range(n_clusters):
    idx = clusters_arr == c
    ax0.scatter(q_init[idx], KE_init[idx], color=colors[c], label=f'Cluster {c+1}', s=80, edgecolors='k')

ax0.plot(ds_q_global, ds_KE_global, color='tab:blue', alpha=0.5)
ax0.set_xlabel(r"$\overline{q}$", fontsize=fontsize)
ax0.set_ylabel(r"$\overline{KE}$", fontsize=fontsize)
ax0.grid(True)
ax0.legend(fontsize=fontsize)
ax0.tick_params(axis='both', which='major', labelsize=labelsize)
ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes, fontsize=fontsize, fontweight='bold', va='top')
#ax0.set_title("Global KE vs q", fontsize=14)

# --- Subplot 2: Time vs x-location ---
ax1 = fig.add_subplot(gs[1])
norm = SymLogNorm(linthresh=1e-4, linscale=1.0, vmin=-0.2, vmax=0.2)

# 1) dudx
zval=28
c0 = ax1.pcolormesh(
    time_vals[0:None], x,
    dudx[0:None, :, zval].T,
    cmap='RdBu', norm=norm, rasterized=True 
)
cbar0 = fig.colorbar(c0, ax=ax1)
cbar0.set_label('$\partial u/\partial x$', fontsize=fontsize)
cbar0.ax.tick_params(labelsize=labelsize)

for c in range(n_clusters):
    idx = clusters_arr == c
    ax1.scatter(time_vals[times[idx]], x_locs[idx], color=colors[c], label=f'Cluster {c+1}', s=80, edgecolors='white')
    

ax1.set_xlabel("Time", fontsize=fontsize)
ax1.set_ylabel("x", fontsize=fontsize)
ax1.tick_params(axis='both', which='major', labelsize=labelsize)
#ax1.grid(True)
#ax1.set_xlim(5000,10000)
ax1.set_ylim(0,20)
ax1.legend(fontsize=fontsize)
ax1.text(0.01, 0.98, "(b)", transform=ax1.transAxes, fontsize=fontsize, fontweight='bold', va='top')

#ax1.set_title("Plume x-location vs time", fontsize=14)

plt.tight_layout()
#plt.show()
output_plot_path = input_path_testdata + '/ClusterPositions_Corrections_testdata.png'
fig.savefig(output_plot_path, facecolor='white', transparent=False,  dpi=300)

# Create a container to export your true baseline validation records
true_export_records = []

# Loop over your tracked, filtered initiations and pair them with their predicted cluster labels
for idx, rec in enumerate(filtered_records):
    true_export_records.append({
        'event_id': idx,
        'time_idx': int(rec['time']),           # Relative index (0 to 629)
        'actual_time': float(time_vals[rec['time']]), # Absolute simulation time
        'x_c': float(rec['x_c']),               # Physical horizontal position
        'z_c': float(rec['z_c']),               # Core level height (z0=4 or z~0.4)
        'assigned_cluster': int(test_clusters[idx]) # Chapter 3 structural regime (0, 1, or 2)
    })

# Define the export folder path
export_dir = input_path_testdata
os.makedirs(export_dir, exist_ok=True)
export_file_path = os.path.join(export_dir, 'true_plume_initiation_records.pkl')
# Dump the structured tracking database to disk
joblib.dump(true_export_records, export_file_path)
print(f"Saved portable file to:   {export_file_path}")

# # 1. Find the spatial index closest to z = 0.4
# # (Using your full-resolution or downsampled grids depending on dudx shape)
# z_idx = 28
# dudx_slice = dudx[:, :, z_idx]  # Shape: (Time, X)

# print(f"Plotting Hovmöller slice at z = {z[z_idx]:.3f} (Index: {z_idx})")

# # 2. Extract plot coordinates for your initiation events
# event_times = []
# event_xs = []
# event_clusters = []

# for idx, rec in enumerate(filtered_initiations):
#     # Convert relative record time index back to actual simulation time
#     t_idx = rec['time']
#     event_times.append(time_vals[t_idx])
#     event_xs.append(rec['x_c'])
#     event_clusters.append(test_clusters[idx])

# event_times = np.array(event_times)
# event_xs = np.array(event_xs)
# event_clusters = np.array(event_clusters)

# # 3. Define a distinct color palette for your clusters 
# # (Adjust colors/labels to match your Chapter 3 palette)
# unique_clusters = np.unique(test_clusters)
# cluster_colors = ['#e74c3c', '#2ecc71', '#3498db'] # Red, Green, Blue
# cmap_scatter = mcolors.ListedColormap(cluster_colors[:len(unique_clusters)])

# # 4. Set up the figure
# fig, ax = plt.subplots(figsize=(12, 8))

# # Plot background dudx convergence field (Time on Y-axis, X on X-axis)
# # Diverging colormap 'RdBu_r' makes convergence zones (negative dudx) stand out in red or blue
# X_mesh, T_mesh = np.meshgrid(x, time_vals)
# cbar_contour = ax.pcolormesh(X_mesh, T_mesh, dudx_slice, cmap='RdBu_r', 
#                              shading='auto', alpha=0.7, vmin=-0.005, vmax=0.005)

# # Add background colorbar
# cbar = fig.colorbar(cbar_contour, ax=ax, orientation='vertical', pad=0.02)
# cbar.set_label(r'$\partial u / \partial x$ (Convergence/Divergence)', rotation=270, labelpad=15)

# # 5. Overlay initiation points colored by cluster
# for cid in sorted(unique_clusters):
#     mask = (event_clusters == cid)
#     ax.scatter(event_xs[mask], event_times[mask], 
#                label=f'Regime {cid}', 
#                color=cluster_colors[cid], 
#                marker='X', s=120, edgecolor='black', zorder=5)

# # 6. Formatting details
# ax.set_title(r'Plume Initiations Overlaid on $dudx$ Field ($z \approx 0.4$)', fontsize=14, weight='bold')
# ax.set_xlabel('Horizontal Domain Space ($x$)', fontsize=12)
# ax.set_ylabel('Simulation Time ($t$)', fontsize=12)
# ax.set_xlim(x[0], x[-1])
# ax.set_ylim(time_vals[0], time_vals[-1])
# ax.legend(loc='upper right', framealpha=0.9, facecolor='white', edgecolor='gray')
# ax.grid(axis='both', linestyle='--', alpha=0.3)

# # Save the plot
# output_plot_path = input_path_testdata + '/plume_initiations_hovmoller.png'
# fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
# print(f"📊 Physical validation plot saved successfully to: {output_plot_path}")