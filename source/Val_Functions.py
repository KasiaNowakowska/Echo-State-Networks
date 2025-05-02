import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import time

#Objective Functions to minimize with Bayesian Optimization

def KFC_Noise(x):
    #K-fold cross Validation
    
    global tikh_opt, k, ti 
    #tikh, U_washout, U_tv, Y_tv, U, N_in, N_fw, N_washout, N_val, N_units
    #setting and initializing
    rho      = x[0]
    sigma_in = round(10**x[1],2)    
    ti       = time.time()
    lenn     = tikh.size
    Mean     = np.zeros(lenn)
    
    #Train using tv: training+val
    Wout, LHS0, RHS0 = train_n(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)
        
    #Different Folds in the validation set
    t1   = time.time()
    for i in range(N_fo):

       #select washout and validation
        p      = N_in + i*N_fw
        Y_val  = U[N_washout + p    : N_washout + p + N_val    ].copy()
        U_wash = U[            p    : N_washout + p            ].copy()
        
        #washout
        xf     = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
        #Train: remove the validation interval
        Xt     = open_loop(Y_val, xf[:N_units], sigma_in, rho)[:-1]
        
        LHS    = LHS0 - np.dot(Xt.T, Xt)
        RHS    = RHS0 - np.dot(Xt.T, Y_val)

        for j in range(lenn):
            if j == 0: #add tikhonov to the diagonal (fast way that requires less memory)
                LHS.ravel()[::LHS.shape[1]+1] += tikh[j]
            else:
                LHS.ravel()[::LHS.shape[1]+1] += tikh[j] - tikh[j-1]
            
            Wout[j]  = np.linalg.solve(LHS, RHS)

            #Validate
            Yh_val   = closed_loop(N_val-1, xf, Wout[j], sigma_in, rho)[0]
            Mean[j] += np.log10(np.mean((Y_val-Yh_val)**2))
        
                
    if k==0: print('closed-loop time:', time.time() - t1)
    
    #select optimal tikh
    a           = np.argmin(Mean)
    tikh_opt[k] = tikh[a]
    k          +=1
    
    #print every set of hyperparameters
    if print_flag:
        print(k, ': Spectral radius, Input Scaling, Tikhonov, MSE:',
              rho, sigma_in, tikh_opt[k-1],  Mean[a]/N_fo)

    return Mean[a]/N_fo

def RVC_Noise(x):
    #Recycle Validation
    
    global tikh_opt, k, ti 
    #tikh, U_washout, U_tv, Y_tv, U, N_in, N_fw, N_washout, N_val, N_units
    #print(tikh)
    #setting and initializing
    rho      = x[0]
    sigma_in = round(10**x[1],2)
    ti       = time.time()
    lenn     = tikh.size
    Mean     = np.zeros(lenn)
    
    #Train using tv: training+val
    Wout = train_n(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)[0]

    #Different Folds in the validation set
    t1   = time.time()
    for i in range(N_fo):
        
        #select washout and validation
        p      = N_in + i*N_fw
        Y_val  = U[N_washout + p : N_washout + p + N_val].copy()
        U_wash = U[            p : N_washout + p        ].copy()
        
        #washout before closed loop
        xf = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
                  
        for j in range(lenn):
            #Validate
            Yh_val   = closed_loop(N_val-1, xf, Wout[j], sigma_in, rho)[0]
            Mean[j] += np.log10(np.mean((Y_val-Yh_val)**2))
                            
    if k==0: print('closed-loop time:', time.time() - t1)
    
    #select optimal tikh
    a           = np.argmin(Mean)
    tikh_opt[k] = tikh[a]
    k          +=1
    
    #print for every set of hyperparameters
    if print_flag:
        print(k, ': Spectral radius, Input Scaling, Tikhonov, MSE:',
              rho, sigma_in, tikh_opt[k-1],  Mean[a]/N_fo)

    return Mean[a]/N_fo

def RVC_Noise_PH(x):
    #Recycle Validation
    
    global tikh_opt, k, ti 
    #tikh, U_washout, U_tv, Y_tv, U, N_in, N_fw, N_washout, N_val, N_units
    #print(tikh)
    #setting and initializing
    rho      = x[0]
    sigma_in = round(10**x[1],2)
    ti       = time.time()
    lenn     = tikh.size
    Mean     = np.zeros(lenn)
    PH       = np.zeros(lenn)

    #Train using tv: training+val
    Wout = train_n(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)[0]

    sigma_ph     = np.sqrt(np.mean(np.var(U_tv,axis=1)))
    threshold_ph = 0.05

    #Different Folds in the validation set
    t1   = time.time()
    for i in range(N_fo):
        
        #select washout and validation
        p      = N_in + i*N_fw
        Y_val  = U[N_washout + p : N_washout + p + N_val].copy()
        U_wash = U[            p : N_washout + p        ].copy()
        
        #washout before closed loop
        xf = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
                  
        for j in range(lenn):
            #Validate
            Yh_val      = closed_loop(N_val-1, xf, Wout[j], sigma_in, rho)[0]
            Mean[j]    += np.log10(np.mean((Y_val-Yh_val)**2))
            Y_err       = np.sqrt(np.mean((Y_val-Yh_val)**2, axis=1))/sigma_ph
            PH_val      = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH_val == 0 and PH_val<threshold_ph: PH_val = N_val/N_lyap #(in case PH is larger than interval)
            PH[j]      += -PH_val

                            
    if k==0: print('closed-loop time:', time.time() - t1)
    
    #select optimal tikh
    #a           = np.argmin(Mean)
    a            = np.argmin(PH)
    tikh_opt[k] = tikh[a]
    k          +=1
    
    #print for every set of hyperparameters
    if print_flag:
        print(k, ': Spectral radius, Input Scaling, Tikhonov, MSE:',
              rho, sigma_in, tikh_opt[k-1],  Mean[a]/N_fo)

    return PH[a]/N_fo

def inverse_POD(data_reduced, pca_):
    data_reconstructed_reshaped = pca_.inverse_transform(data_reduced)
    print('shape of data reconstructed flattened:', np.shape(data_reconstructed_reshaped))
    data_reconstructed = data_reconstructed_reshaped.reshape(data_reconstructed_reshaped.shape[0], len(x), len(z), len(variables))
    print('shape of data reconstructed:', np.shape(data_reconstructed))
    return data_reconstructed_reshaped, data_reconstructed 

def active_array_calc(original_data, reconstructed_data, z):
    beta = 1.201
    alpha = 3.0
    T = original_data[:,:,:,3] - beta*z
    T_reconstructed = reconstructed_data[:,:,:,3] - beta*z
    q_s = np.exp(alpha*T)
    q_s_reconstructed = np.exp(alpha*T)
    rh = original_data[:,:,:,0]/q_s
    rh_reconstructed = reconstructed_data[:,:,:,0]/q_s_reconstructed
    mean_b = np.mean(original_data[:,:,:,3], axis=1, keepdims=True)
    mean_b_reconstructed= np.mean(reconstructed_data[:,:,:,3], axis=1, keepdims=True)
    b_anom = original_data[:,:,:,3] - mean_b
    b_anom_reconstructed = reconstructed_data[:,:,:,3] - mean_b_reconstructed
    w = original_data[:,:,:,1]
    w_reconstructed = reconstructed_data[:,:,:,1]
    
    mask = (rh[:, :, :] >= 1) & (w[:, :, :] > 0) & (b_anom[:, :, :] > 0)
    mask_reconstructed = (rh_reconstructed[:, :, :] >= 1) & (w_reconstructed[:, :, :] > 0) & (b_anom_reconstructed[:, :, :] > 0)
    
    active_array = np.zeros((original_data.shape[0], len(x), len(z)))
    active_array[mask] = 1
    active_array_reconstructed = np.zeros((original_data.shape[0], len(x), len(z)))
    active_array_reconstructed[mask_reconstructed] = 1
    
    # Expand the mask to cover all features (optional, depending on use case)
    mask_expanded       =  np.repeat(mask[:, :, :, np.newaxis], 4, axis=-1)  # Shape: (256, 64, 1)
    mask_expanded_recon =  np.repeat(mask_reconstructed[:, :, :, np.newaxis], 4, axis=-1) # Shape: (256, 64, 1)
    
    return active_array, active_array_reconstructed, mask_expanded, mask_expanded_recon

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

def spectral_corr_loss(y_true, y_pred):
    fft_true = np.abs(np.fft.fft(y_true, axis=0))
    fft_pred = np.abs(np.fft.fft(y_pred, axis=0))

    # Normalize each mode's spectrum
    fft_true /= np.sum(fft_true, axis=0, keepdims=True)
    fft_pred /= np.sum(fft_pred, axis=0, keepdims=True)

    # Compute 1 - Pearson correlation per mode and average
    num_modes = y_true.shape[1]
    losses = []
    for m in range(num_modes):
        corr = np.corrcoef(fft_true[:, m], fft_pred[:, m])[0, 1]
        losses.append(1 - corr)

    return np.mean(losses)  # smaller is better

def RVC_Noise_weighted(x):
    #Recycle Validation
    
    global tikh_opt, k, ti, pca_, z, alpha, scaler
    #tikh, U_washout, U_tv, Y_tv, U, N_in, N_fw, N_washout, N_val, N_units
    #print(tikh)
    #setting and initializing
    rho         = x[0]
    sigma_in    = round(10**x[1],2)
    ti          = time.time()
    lenn        = tikh.size
    Mean        = np.zeros(lenn)
    NRMSE_plume = np.zeros(lenn)
    
    #Train using tv: training+val
    Wout = train_n(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)[0]

    #Different Folds in the validation set
    t1   = time.time()
    for i in range(N_fo):
        
        #select washout and validation
        p      = N_in + i*N_fw
        Y_val  = U[N_washout + p : N_washout + p + N_val].copy()
        U_wash = U[            p : N_washout + p        ].copy()
        
        #washout before closed loop
        xf = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
                  
        for j in range(lenn):
            #Validate
            Yh_val   = closed_loop(N_val-1, xf, Wout[j], sigma_in, rho)[0]
            Mean[j] += np.log10(np.mean((Y_val-Yh_val)**2))
            
            _, reconstructed_truth       = inverse_POD(Y_val, pca_)
            _, reconstructed_predictions = inverse_POD(Yh_val, pca_)
            print('shape truth before scaling', np.shape(reconstructed_truth))
            print('shape pred before scaling', np.shape(reconstructed_predictions))
            reconstructed_truth = ss_inverse_transform(reconstructed_truth, scaler)
            reconstructed_predictions = ss_inverse_transform(reconstructed_predictions, scaler)
            print('shape truth after scaling', np.shape(reconstructed_truth))
            print('shape pred after scaling', np.shape(reconstructed_predictions))

            _,_, mask, _ = active_array_calc(reconstructed_truth, reconstructed_predictions, z)
            if np.any(mask):  # Check if plumes exist
                masked_truth = reconstructed_truth[mask]
                masked_pred = reconstructed_predictions[mask]
                
                print("Shape truth after mask:", masked_truth.shape)
                print("Shape pred after mask:", masked_pred.shape)

                # Compute NRMSE only if mask is not empty
                NRMSE_plume[j] += NRMSE(masked_truth, masked_pred)
            else:
                print("Mask is empty, no plumes detected.")
                NRMSE_plume[j] += 0  # Simply add 0 to maintain shape

    if k==0: print('closed-loop time:', time.time() - t1)

    #select optimal tikh
    a = np.argmin(Mean)  # Find the best tikh *before* averaging
    Mean /= N_fo
    NRMSE_plume /= N_fo
    tikh_opt[k] = tikh[a]
    k          +=1
    
    # Compute weighted loss
    weighted_loss = alpha * Mean[a] + (1 - alpha) * NRMSE_plume[a]

    #print for every set of hyperparameters
    if print_flag:
        print(k, ': Spectral radius, Input Scaling, Tikhonov, MSE:',
              rho, sigma_in, tikh_opt[k-1],  Mean[a]/N_fo)

    return weighted_loss

def RVC_Noise_FFT(x):
    #Recycle Validation
    
    global tikh_opt, k, ti, pca_, z, alpha, scaler
    #tikh, U_washout, U_tv, Y_tv, U, N_in, N_fw, N_washout, N_val, N_units
    #print(tikh)
    #setting and initializing
    rho         = x[0]
    sigma_in    = round(10**x[1],2)
    ti          = time.time()
    lenn        = tikh.size
    Mean        = np.zeros(lenn)
    spectral_loss = np.zeros(lenn)
    
    
    #Train using tv: training+val
    Wout = train_n(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)[0]

    #Different Folds in the validation set
    t1   = time.time()
    for i in range(N_fo):
        
        #select washout and validation
        p      = N_in + i*N_fw
        Y_val  = U[N_washout + p : N_washout + p + N_val].copy()
        U_wash = U[            p : N_washout + p        ].copy()
        
        #washout before closed loop
        xf = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
                  
        for j in range(lenn):
            #Validate
            Yh_val   = closed_loop(N_val-1, xf, Wout[j], sigma_in, rho)[0]
            Mean[j] += np.log10(np.mean((Y_val-Yh_val)**2))
            spectral_loss[j]   += spectral_corr_loss(Y_val, Yh_val)

    if k==0: print('closed-loop time:', time.time() - t1)

    #select optimal tikh
    a = np.argmin(Mean)  # Find the best tikh *before* averaging
    Mean /= N_fo
    spectral_loss /= N_fo
    tikh_opt[k] = tikh[a]
    k          +=1
    
    # Compute weighted loss
    weighted_loss = alpha * Mean[a] + (1 - alpha) * np.log10(spectral_loss[a] + 1e-12) #1e-12 for numerical stability

    #print for every set of hyperparameters
    if print_flag:
        print(k, ': Spectral radius, Input Scaling, Tikhonov, MSE:',
              rho, sigma_in, tikh_opt[k-1],  Mean[a]/N_fo)

    return weighted_loss