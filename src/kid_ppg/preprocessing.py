import numpy as np


def create_temporal_pairs(X_in, y_in):
    
    temp_X = np.concatenate([X_in[:-1, ...][..., None], 
                             X_in[1:, ...][..., None]], axis = -1)
    temp_y = y_in[1:]
    
    return temp_X, temp_y

def channel_wise_z_score_normalization(X):
    
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

def channel_wise_z_score_denormalization(X, ms, stds):
    
    for i in range(X.shape[0]):
        curX = X[i, ...]
        
        for j in range(X.shape[1]): 
            
            if stds[i, j] != 0:
                curX[j, ...] = curX[j, ...] * stds[i, j]
            
            curX[j, ...] = curX[j, ...] + ms[i, j]
        X[i, ...] = curX
    
    return X
