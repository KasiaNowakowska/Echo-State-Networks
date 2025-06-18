import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def POD(data, c, x, z, variables, file_str, Plotting=False):
    if data.ndim == 3: #len(time_vals), len(x), len(z)
            data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    elif data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
            data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2]*data.shape[3])
    print('shape of data for POD:', np.shape(data_matrix))
    pca = PCA(n_components=c, svd_solver='randomized', random_state=42)
    pca.fit(data_matrix)
    data_reduced = pca.transform(data_matrix)
    print('shape of reduced_data:', np.shape(data_reduced))
    data_reconstructed_reshaped = pca.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data.shape)
    print('shape of data reconstructed:', np.shape(data_reconstructed))
    np.save(file_str+'_data_reduced.npy', data_reduced)

    components = pca.components_
    print(np.shape(components))
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print('cumulative explained variance for', c, 'components is', cumulative_explained_variance[-1])

    if Plotting:
        # Plot cumulative explained variance
        fig, ax = plt.subplots(1, figsize=(12,3))
        ax.plot(cumulative_explained_variance*100, marker='o', linestyle='-', color='b')
        ax2 = ax.twinx()
        ax2.semilogy(explained_variance_ratio*100,  marker='o', linestyle='-', color='r')
        ax.set_xlabel('Number of Modes')
        ax.set_ylabel('Cumulative Energy (%)', color='blue')
        #plt.axvline(x=truncated_modes, color='r', linestyle='--', label=f'Truncated Modes: {truncated_modes}')
        ax2.set_ylabel('Inidvidual Energy (%)', color='red')
        fig.savefig(file_str+'_EVRmodes.png')
        #plt.show()
        plt.close()

        # Plot the time coefficients and mode structures
        indexes_to_plot = np.array([1, 2, 10, 32, 64] ) -1
        indexes_to_plot = indexes_to_plot[indexes_to_plot <= (c-1)]
        print('plotting for modes', indexes_to_plot)
        print('number of modes', len(indexes_to_plot))

        #time coefficients
        fig, ax = plt.subplots(1, figsize=(12,3), tight_layout=True)
        for index, element in enumerate(indexes_to_plot):
            #ax.plot(index - data_reduced[:, element], label='S=%i' % (element+1))
            ax.plot(data_reduced[:, element], label='S=%i' % (element+1))
        ax.grid()
        ax.legend()
        #ax.set_yticks([])
        ax.set_xlabel('Time Step $\Delta t$')
        ax.set_ylabel('Time Coefficients')
        fig.savefig(file_str+'_coef.png')
        #plt.show()
        plt.close()

        # Visualize the modes
        minm = components.min()
        maxm = components.max()
        if data.ndim == 3:
            fig, ax =plt.subplots(len(indexes_to_plot), figsize=(12,6), tight_layout=True, sharex=True)
            for i in range(len(indexes_to_plot)):
                if len(indexes_to_plot) == 1:
                    mode = components[indexes_to_plot[i]].reshape(data.shape[1], data.shape[2])  # Reshape to original dimensions for visualization
                    c1 = ax.pcolormesh(x, z, mode[:, :].T, cmap='viridis', vmin=minm, vmax=maxm)  # Visualizing the first variable
                    ax.set_title('mode % i' % (indexes_to_plot[i]+1))
                    fig.colorbar(c1, ax=ax)
                    ax.set_ylabel('z')
                    ax.set_xlabel('x')
                else:
                    mode = components[indexes_to_plot[i]].reshape(data.shape[1], data.shape[2])  # Reshape to original dimensions for visualization              
                    c1 = ax[i].pcolormesh(x, z, mode[:, :].T, cmap='viridis', vmin=minm, vmax=maxm)  # Visualizing the first variable
                    #ax[i].axis('off')
                    ax[i].set_title('mode % i' % (indexes_to_plot[i]+1))
                    fig.colorbar(c1, ax=ax[i])
                    ax[i].set_ylabel('z')
                    ax[-1].set_xlabel('x')
            fig.savefig(file_str+'_modes.png')
            #plt.show()
            plt.close()
        elif data.ndim == 4:
            for v in range(len(variables)):
                fig, ax =plt.subplots(len(indexes_to_plot), figsize=(12,6), tight_layout=True, sharex=True)
                for i in range(len(indexes_to_plot)):
                    if len(indexes_to_plot) == 1:
                        mode = components[indexes_to_plot[i]].reshape(data.shape[1], data.shape[2], data.shape[3])  # Reshape to original dimensions for visualization
                        c1 = ax.pcolormesh(x, z, mode[:, :, v].T, cmap='viridis')  # Visualizing the first variable
                        ax.set_title('mode % i' % (indexes_to_plot[i]+1))
                        fig.colorbar(c1, ax=ax)
                        ax.set_ylabel('z')
                        ax.set_xlabel('x')
                    else:
                        mode = components[indexes_to_plot[i]].reshape(data.shape[1], data.shape[2], data.shape[3])  # Reshape to original dimensions for visualization              
                        c1 = ax[i].pcolormesh(x, z, mode[:, :, v].T, cmap='viridis')  # Visualizing the first variable
                        #ax[i].axis('off')
                        ax[i].set_title('mode % i' % (indexes_to_plot[i]+1))
                        fig.colorbar(c1, ax=ax[i])
                        ax[i].set_ylabel('z')
                        ax[-1].set_xlabel('x')
                fig.savefig(file_str+variables[v]+'_modes.png')
                #plt.show()
                plt.close()

    return data_reduced, data_reconstructed_reshaped, data_reconstructed, pca, cumulative_explained_variance[-1]

def inverse_POD(data_reduced, x, z, variables, pca_):
    data_reconstructed_reshaped = pca_.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data_reconstructed_reshaped.shape[0], len(x), len(z), len(variables))
    print('shape of data reconstructed:', np.shape(data_reconstructed))
    return data_reconstructed_reshaped, data_reconstructed 

def transform_POD(data, pca_):
    if data.ndim == 3: #len(time_vals), len(x), len(z)
            data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    elif data.ndim == 4: #len(time_vals), len(x), len(z), len(var)
            data_matrix = data.reshape(data.shape[0], data.shape[1]*data.shape[2]*data.shape[3])
    data_reduced = pca_.transform(data_matrix)
    print('shape of reduced_data:', np.shape(data_reduced))
    return data_reduced
