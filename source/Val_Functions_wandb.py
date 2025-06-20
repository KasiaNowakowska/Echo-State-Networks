import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import time

from Functions_wandb import *

#Objective Functions to minimize with Bayesian Optimization


def KFC_Noise(x, tikh):
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
        Y_val  = U[N_washout_val + p : N_washout_val + p + N_val].copy()
        U_wash = U[            p : N_washout_val + p        ].copy()
        
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

def RVC_Noise_weightedloss(x):
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
        Y_val  = U[N_washout_val + p : N_washout_val + p + N_val].copy()
        U_wash = U[            p : N_washout_val + p        ].copy()
        
        #washout before closed loop
        xf = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
                  
        for j in range(lenn):
            #Validate
            Yh_val   = closed_loop(N_val-1, xf, Wout[j], sigma_in, rho)[0]
            err      = (Y_val-Yh_val)**2
            
            weights       = np.exp(-np.linspace(0, 3, len(err))) #np.linspace(1.0, 0.1, len(err))  # heavier weight early
            weighted_loss = np.sum(weights[:,None] * err) / np.sum(weights)
            Mean[j]      += np.log10(weighted_loss)
                            
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


def RVC_Noise_PH(x, tikh, params):
    #Recycle Validation
    
    # Extract from params
    U_washout = params["U_washout"]
    U_tv      = params["U_tv"]
    Y_tv      = params["Y_tv"]
    U         = params["U"]
    N_in      = params["N_in"]
    N_fw      = params["N_fw"]
    N_washout = params["N_washout"]
    N_val     = params["N_val"]
    N_units   = params["N_units"]
    N_splits  = params["N_splits"]
    dim       = params["dim"]
    N_lyap    = params["N_lyap"]
    N_fo      = params["N_fo"]
    print_flag   = params.get("print_flag", False)
    threshold_ph = params["threshold_ph"]
    k         = params["k"]
    tikh_opt  = params["tikh_opt"]


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
    Wout = train_n(U_washout, U_tv, Y_tv, tikh, sigma_in, rho, params)[0]

    sigma_ph     = np.sqrt(np.mean(np.var(U_tv,axis=1)))
    threshold_ph = threshold_ph

    #Different Folds in the validation set
    t1   = time.time()
    for i in range(N_fo):
        
        #select washout and validation
        p      = N_in + i*N_fw
        Y_val  = U[N_washout + p : N_washout + p + N_val].copy()
        U_wash = U[            p : N_washout + p        ].copy()
        
        #washout before closed loop
        #print('open loop')
        xf = open_loop(U_wash, np.zeros(N_units), sigma_in, rho, params)[-1]
                  
        for j in range(lenn):
            #Validate
            Yh_val      = closed_loop(N_val-1, xf, Wout[j], sigma_in, rho, params)[0]
            Mean[j]    += np.log10(np.mean((Y_val-Yh_val)**2))
            Y_err       = np.sqrt(np.mean((Y_val-Yh_val)**2, axis=1))/sigma_ph
            PH_val      = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH_val == 0 and Y_err[0]<threshold_ph: PH_val = N_val/N_lyap #(in case PH is larger than interval)
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
        Y_val  = U[N_washout_val + p : N_washout_val + p + N_val].copy()
        U_wash = U[            p : N_washout_val + p        ].copy()
        
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
        Y_val  = U[N_washout_val + p : N_washout_val + p + N_val].copy()
        U_wash = U[            p : N_washout_val + p        ].copy()
        
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

def RVC_Noise_PH_test(x):
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
    threshold_ph = 0.2

    #Different Folds in the test set
    t1   = time.time()
    for i in range(N_fo):
        
        #select washout and test
        p      = N_in + N_train + i*N_fw
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
            if PH_val == 0 and Y_err[0]<threshold_ph: PH_val = N_val/N_lyap #(in case PH is larger than interval)
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

def RVC_Noise_test(x):
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
        p      = N_in + N_train + i*N_fw
        Y_val  = U[N_washout_val + p : N_washout_val + p + N_val].copy()
        U_wash = U[            p : N_washout_val + p        ].copy()
        
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

def RVC_Noise_modeweight_PH(x):
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
    threshold_ph = 0.2

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
            Yh_val        = closed_loop(N_val-1, xf, Wout[j], sigma_in, rho)[0]
            num_modes     = Y_val.shape[1]
            mode_weights  = np.exp(-np.arange(num_modes) / 20)
            mode_weights /= mode_weights.sum()
            
            err_sq         = (Y_val - Yh_val)**2  # shape: (time, modes)
            weighted_mse   = np.mean(np.dot(err_sq, mode_weights))  # time-averaged, mode-weighted
            Mean[j]       += np.log10(weighted_mse)

            Y_err = np.sqrt(np.dot((Y_val - Yh_val)**2, mode_weights)) / sigma_ph
            PH_val      = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH_val == 0 and Y_err[0]<threshold_ph: PH_val = N_val/N_lyap #(in case PH is larger than interval)
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

def RVC_Noise_modeweight_PH_test(x):
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
    threshold_ph = 0.2

    #Different Folds in the validation set
    t1   = time.time()
    for i in range(N_fo):
        
        #select washout and validation
        p      = N_in + N_train + i*N_fw
        Y_val  = U[N_washout_val + p : N_washout_val + p + N_val].copy()
        U_wash = U[            p : N_washout_val + p        ].copy()
        
        #washout before closed loop
        xf = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
                  
        for j in range(lenn):
            #Validate
            Yh_val        = closed_loop(N_val-1, xf, Wout[j], sigma_in, rho)[0]
            num_modes     = Y_val.shape[1]
            mode_weights  = np.exp(-np.arange(num_modes) / 20)
            mode_weights /= mode_weights.sum()
            
            err_sq         = (Y_val - Yh_val)**2  # shape: (time, modes)
            weighted_mse   = np.mean(np.dot(err_sq, mode_weights))  # time-averaged, mode-weighted
            Mean[j]       += np.log10(weighted_mse)

            Y_err = np.sqrt(np.dot((Y_val - Yh_val)**2, mode_weights)) / sigma_ph
            PH_val      = np.argmax(Y_err>threshold_ph)/N_lyap
            if PH_val == 0 and Y_err[0]<threshold_ph: PH_val = N_val/N_lyap #(in case PH is larger than interval)
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