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
    print(tikh)
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