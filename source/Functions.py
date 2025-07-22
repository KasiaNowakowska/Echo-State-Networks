import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

## ESN with bias architecture

def step(x_pre, u, sigma_in, rho):
    """ Advances one ESN time step.
        Args:
            x_pre: reservoir state
            u: input
        Returns:
            new augmented state (new state with bias_out appended)
    """
    # input is normalized and input bias added
    #print('shapes:', np.shape((u-u_mean)/norm), np.shape(bias_in))
    if normalisation == 'on':
        u_augmented = np.hstack(((u-u_mean)/norm, bias_in))
    elif normalisation == 'standard':
        u_augmented = np.hstack(((u-u_mean)/norm_std, bias_in))
    elif normalisation == 'off':
        u_augmented = np.hstack((u, bias_in))
    elif normalisation == 'standard_plusregions':
        u_augmented = np.hstack(((u[:n_components]-u_mean_pr)/norm_std_pr, u[n_components:], bias_in))
    elif normalisation == 'off_plusfeatures':
        u_pods = u[:n_components]
        u_feats = (u[n_components:] - mean_feats)/ std_feats 

        u_pods_scaled = u_pods*sigma_in 
        u_feats_scaled = u_feats*sigma_in_feats
        bias_in_scaled = bias_in*sigma_in

        u_augmented = np.hstack((u_pods_scaled, u_feats_scaled, bias_in_scaled))

    elif normalisation == 'modeweight':
        u_augmented = np.hstack((u * weights * sigma_in, bias_in))

    # reservoir update
    if normalisation == 'off_plusfeatures':
        x_post_part      = np.tanh(Win.dot(u_augmented) + W.dot(rho*x_pre))
    elif normalisation == 'modeweight':
        x_post_part      = np.tanh(Win.dot(u_augmented) + W.dot(rho*x_pre))
    else:
        x_post_part      = np.tanh(Win.dot(u_augmented*sigma_in) + W.dot(rho*x_pre))

    x_post           = (1-alpha0) * x_pre + alpha0 * x_post_part

    # output bias added
    x_augmented = np.concatenate((x_post, bias_out))

    return x_augmented

def open_loop(U, x0, sigma_in, rho):
    """ Advances ESN in open-loop.
        Args:
            U: input time series
            x0: initial reservoir state
        Returns:
            time series of augmented reservoir states
    """
    N     = U.shape[0]
    Xa    = np.empty((N+1, N_units+1))
    #print("Shape of x0:", x0.shape)           # Should be (N_units,)
    #print("Shape of bias_out:", bias_out.shape)  # Should be (1,)
    Xa[0] = np.concatenate((x0,bias_out))
    for i in np.arange(1,N+1):
        Xa[i] = step(Xa[i-1,:N_units], U[i-1], sigma_in, rho)

    return Xa

def closed_loop(N, x0, Wout, sigma_in, rho):
    """ Advances ESN in closed-loop.
        Args:
            N: number of time steps
            x0: initial reservoir state
            Wout: output matrix
        Returns:
            time series of prediction
            final augmented reservoir state
    """
    xa    = x0.copy()
    Yh    = np.empty((N+1, dim))
    Xa    = np.empty((N+1, N_units+1))
    Yh[0] = np.dot(xa, Wout)
    for i in np.arange(1,N+1):
        xa    = step(xa[:N_units], Yh[i-1], sigma_in, rho)
        Xa[i] = xa
        Yh[i] = np.dot(xa, Wout) #np.linalg.multi_dot([xa, Wout])

    return Yh, xa, Xa

def train_n(U_washout, U_train, Y_train, tikh, sigma_in, rho):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """

    ## initial washout phase
    xf = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1,:N_units]

    ## splitting training in N_splits to save memory
    LHS = 0
    RHS = 0
    N_len = (U_train.shape[0]-1)//N_splits

    for ii in range(N_splits):
        ## open-loop train phase
        t1  = time.time()
        Xa1 = open_loop(U_train[ii*N_len:(ii+1)*N_len], xf, sigma_in, rho)[1:]
        xf  = Xa1[-1,:N_units].copy()
        if ii == 0 and k==0: print('open_loop time:', (time.time()-t1)*N_splits)
        cond_number = np.linalg.cond(Xa1)
        #print("Condition number of reservoir state matrix:", cond_number)

        ##computing the matrices for the linear system
        t1  = time.time()
        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_train[ii*N_len:(ii+1)*N_len])
        if ii == 0 and k==0: print('matrix multiplication time:', (time.time()-t1)*N_splits)

    # to cover the last part of the data that didn't make it into the even splits
    if N_splits > 1:
        Xa1 = open_loop(U_train[(ii+1)*N_len:], xf, sigma_in, rho)[1:]
        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_train[(ii+1)*N_len:])

    Wout = np.empty((len(tikh),N_units+1,dim))

    # solve linear system for different Tikhonov
    for j in range(len(tikh)):
        if j == 0: #add tikhonov to the diagonal (fast way that requires less memory)
            LHS.ravel()[::LHS.shape[1]+1] += tikh[j]
        else:
            LHS.ravel()[::LHS.shape[1]+1] += tikh[j] - tikh[j-1]

        #solve linear system
        t1  = time.time()
        Wout[j] = np.linalg.solve(LHS, RHS)

        if j==0 and k==0: print('linear system time:', (time.time() - t1)*len(tikh))

    return Wout, LHS, RHS

def train_n_decoded(U_washout, U_train, Y_train, tikh, sigma_in, rho, decoder):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """

    ## initial washout phase
    xf = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1,:N_units]

    ## splitting training in N_splits to save memory
    LHS = 0
    RHS = 0
    N_len = (U_train.shape[0]-1)//N_splits

    Xa_all = []

    for ii in range(N_splits):
        ## open-loop train phase
        t1      = time.time()
        Xa1     = open_loop(U_train[ii*N_len:(ii+1)*N_len], xf, sigma_in, rho)[1:]
        xf      = Xa1[-1,:N_units].copy()
        Xa_all.append(Xa1)

        if ii == 0 and k==0: print('open_loop time:', (time.time()-t1)*N_splits)
        cond_number = np.linalg.cond(Xa1)
        #print("Condition number of reservoir state matrix:", cond_number)

        ##computing the matrices for the linear system
        t1  = time.time()
        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_train[ii*N_len:(ii+1)*N_len])
        if ii == 0 and k==0: print('matrix multiplication time:', (time.time()-t1)*N_splits)

    # to cover the last part of the data that didn't make it into the even splits
    if N_splits > 1:
        Xa1 = open_loop(U_train[(ii+1)*N_len:], xf, sigma_in, rho)[1:]
        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_train[(ii+1)*N_len:])

    Xa_all = np.vstack(Xa_all)
    Y_train_all = Y_train[:Xa_all.shape[0]]

    Wout = np.empty((len(tikh),N_units+1,dim))
    decoded_img_loss = []

    # solve linear system for different Tikhonov
    for j in range(len(tikh)):
        if j == 0: #add tikhonov to the diagonal (fast way that requires less memory)
            LHS.ravel()[::LHS.shape[1]+1] += tikh[j]
        else:
            LHS.ravel()[::LHS.shape[1]+1] += tikh[j] - tikh[j-1]

        #solve linear system
        t1  = time.time()
        Wout[j] = np.linalg.solve(LHS, RHS)

        if j==0 and k==0: print('linear system time:', (time.time() - t1)*len(tikh))

    return Wout, LHS, RHS


def train_save_n(U_washout, U_train, Y_train, tikh, sigma_in, rho, noise):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """

    ## washout phase
    xf    = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1,:N_units]

    LHS   = 0
    RHS   = 0
    Xa1_list = np.zeros((len(U_train), N_units+1))
    N_len = (U_train.shape[0]-1)//N_splits

    for ii in range(N_splits):
        t1  = time.time()
        ## open-loop train phase
        Xa1 = open_loop(U_train[ii*N_len:(ii+1)*N_len], xf, sigma_in, rho)[1:]
        xf  = Xa1[-1,:N_units].copy()

        t1  = time.time() 
        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_train[ii*N_len:(ii+1)*N_len])
        
        Xa1_list[ii*N_len:(ii+1)*N_len, :] = Xa1 

    if N_splits > 1:# to cover the last part of the data that didn't make into the even splits
        Xa1 = open_loop(U_train[(ii+1)*N_len:], xf, sigma_in, rho)[1:]
        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_train[(ii+1)*N_len:])
        
        Xa1_list[(ii+1)*N_len:, :] = Xa1 

    LHS.ravel()[::LHS.shape[1]+1] += tikh

    Wout = np.linalg.solve(LHS, RHS)

    return Wout, Xa1_list
