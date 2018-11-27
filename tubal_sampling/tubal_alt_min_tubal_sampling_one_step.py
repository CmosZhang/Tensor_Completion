# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:23:26 2018

@author: ZCC
"""

import numpy as np
from Mat_to_Vec import Mat_to_Vec
from Vec_to_Diag import Vec_to_Diag
from blkdiag import blkdiag

def tubal_alt_min_tubal_sampling_one_step(T_omega_f, omega_f, X_f):
    
    [m, n] = T_omega_f.shape
    [m, r] = X_f.shape
    
    Y_f = np.zeros((r, n), dtype = complex)
    
    
    X_f_new = X_f
    for i in range(n-1):
        X_f_new = blkdiag(X_f_new, X_f)

    tensor_V = np.zeros((m*n, 1), dtype = complex)
    temp_Y_f = np.zeros((r*n, 1), dtype = complex)
    
    
    tensor_V = Mat_to_Vec(T_omega_f)
    
    omega_temp = np.zeros((m*n, 1), dtype = complex)
    omega_temp = Mat_to_Vec(omega_f)
    omega_f_new = Vec_to_Diag(omega_temp)
    
    temp = np.dot(omega_f_new, X_f_new)
        
    #temp_Y_f = np.dot((np.linalg.pinv(np.dot(temp.T*temp)))*temp.T)*tensor_V

    temp_Y_f,resid,rank,sigma = np.linalg.lstsq(temp, tensor_V)
   
    for j in range(n):
        Y_f[:, j] = np.squeeze(temp_Y_f[j*r:(j+1)*r])
    return Y_f