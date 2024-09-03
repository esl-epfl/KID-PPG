import numpy as np
import resampy

def create_input_from_array(x: np.ndarray, fs: int = 32) ->np.ndarray:
    """Create input data for kid-ppg from 1d array of PPG data

    Args:
    x (np.ndarray): Input 1D array of PPG data of shape (n,)
    fs (int): Sampling frequency of the input signal. Defaults to 32 Hz.

    Returns:
    y (np.ndarray): Pairs of data of shape [(256, 2), ..].
    """
    
    kid_ppg_fs = 32
    kid_ppg_window = 256
    kid_ppg_overlap = 64

    x = resampy.resample(x, fs, kid_ppg_fs)
    
    if len(x) % kid_ppg_window:
        x = np.concatenate((x, np.zeros(256 - (len(x) % kid_ppg_window))))
    
    y = np.lib.stride_tricks.sliding_window_view(x, kid_ppg_window)[::kid_ppg_overlap, :]
    y = np.stack((y[:-1], y[1:]), axis=2)
    
    return y
 

def create_temporal_pairs(X_in: np.ndarray, y_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create pairs from the data X associated to labels y.
    
    Pairs of [X(n), X(n + 1)] are associated to labels y(n + 1).
    
    Args:
        X_in (numpy.ndarray): Data of shape (n, ...).
        y_in (numpy.ndarray): Labels of shape (n,).

    Returns:
        temp_X (numpy.ndarray): Pairs of input data of shape (n-1, ..., 2).
        temp_y (numpy.ndarray): Associated labels of shape (n-1,).
    """
    
    temp_X = np.concatenate([X_in[:-1, ...][..., None], 
                             X_in[1:, ...][..., None]], axis = -1)
    temp_y = y_in[1:]
    
    return temp_X, temp_y

def sample_wise_z_score_normalization(X: np.ndarray):
    """Sample-wise normalization of X.

    Args:
        X (numpy.ndarray): Data of shape (n, )
    Returns:
        X (numpy.ndarray): Sample-wise normalized output data.
        ms (numpy.ndarray): Averages used for the normalization.
        stds (numpy.ndarray): Standard deviations used for the normalization.
    """
    
    ms = np.zeros((X.shape[0], 4))
    stds = np.zeros((X.shape[0], 4))
    for i in range(X.shape[0]):
        curX = X[i, ...]
        
        for j in range(4): 
            std = np.std(curX[j, ...])
            m = np.mean(curX[j, ...])
            
            curX[j, ...] = curX[j, ...] - np.mean(curX[j, ...])
            
            if std != 0:
                curX[j, ...] = curX[j, ...] / std
            
            ms[i, j] = m
            stds[i, j] = std
        
        X[i, ...] = curX
        
    return X, ms, stds

def sample_wise_z_score_denormalization(X, ms, stds):
    """Sample-wise denormalization of X. 

    Args:
        X (numpy.ndarray): Sample-wise normalized output data.
        ms (numpy.ndarray): Averages used for the normalization.
        stds (numpy.ndarray): Standard deviations used for the normalization.
    Returns:
        X (numpy.ndarray)
    """
    
    for i in range(X.shape[0]):
        curX = X[i, ...]
        
        for j in range(X.shape[1]): 
            
            if stds[i, j] != 0:
                curX[j, ...] = curX[j, ...] * stds[i, j]
            
            curX[j, ...] = curX[j, ...] + ms[i, j]
        X[i, ...] = curX
    
    return X
