import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import time

def onset_truth(Y_t, PT, N_lyap, threshold_e):
    result_truth = np.any(Y_t[PT:PT+N_lyap,0] > threshold_e)
    return result_truth

def onset_prediction(Yh_t, PT, N_lyap, threshold_e):
    result_prediction = np.any(Yh_t[PT:PT+N_lyap,0] > threshold_e)
    return result_prediction

def onset_ensemble(true_onset, pred_onset):
    if true_onset == True and pred_onset == True:
        flag = 'TP'
    elif true_onset == True and pred_onset == False:
        flag = 'FN'
    elif true_onset == False and pred_onset == True:
        flag = 'FP'
    elif true_onset == False and pred_onset == False:
        flag = 'TN'
    return flag
