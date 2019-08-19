import numpy as np
import tensorflow as tf

def tversky(v1, v2, a ,b):
    nominator= np.dot(v1,v2.T)
    norm1 = a * np.linalg.norm(v1, axis = 1)**2
    norm2 = b * np.linalg.norm(v2, axis = 1)**2
    add1 = np.zeros([np.size(norm1),np.shape(nominator)[1]])
    for num, i in enumerate(norm1):
        add1[num,:] = i + norm2
    scl_nom = nominator * (1-a-b)
    
    denominator = scl_nom + add1
    tversky_matrix = nominator/(denominator+ 1e-8)
    
    avg_tver = np.sum(tversky_matrix)/tversky_matrix.size
    max_tver = np.max(tversky_matrix)
    min_tver = np.min(tversky_matrix)
    useful_tver = np.sum(tversky_matrix[tversky_matrix>0.8])/tversky_matrix.size
    semiUseful_tver = np.sum(tversky_matrix[tversky_matrix>0.7])/tversky_matrix.size
    notUseful_tver = np.sum(tversky_matrix[tversky_matrix>0.5])/tversky_matrix.size
    return avg_tver, max_tver, min_tver, useful_tver, semiUseful_tver, notUseful_tver

def mutual_info():
    return
    
def hamming():
    return

def KL_div():
    return