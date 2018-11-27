# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:40:16 2018

@author: ZCC
"""
'''
tubal_alt_min algorithm
'''


import numpy as np
from blkdiag import blkdiag
from arr_to_circul import arr_to_circul


def tubal_alt_min_one_step(T_omega_f, omega_f, X_f):
    
    [m, n, k] = T_omega_f.shape
    [m, r, k] = X_f.shape
    
    Y_f = np.zeros((r, n, k), dtype = complex)
    
    
    X_f_new = X_f[:, :, 0]
    for i in range(k-1):
        X_f_new = blkdiag(X_f_new, X_f[:, :, 1+i])

    tensor_V = np.zeros((k*m, 1), dtype = complex)
    temp_Y_f = np.zeros((r*k, 1), dtype = complex)
    for i in range(n):
        for j in range(k):
            tensor_V[j*m : (j+1)*m] = np.squeeze(T_omega_f[:, i, j]).reshape((m, 1))
        omega_f_3D = np.zeros((k, k, m), dtype = complex)
        omega_f_new = np.zeros((k*m, k*m), dtype = complex)
        for j in range(m):
            temp = arr_to_circul(np.squeeze(omega_f[j, i, :]))
            omega_f_3D[:, :, j] = temp.T
        for a in range(k):
            for b in range(k):
                for c in range(m):
                    row = a*m+c
                    col = b*m+c
                    omega_f_new[row, col] = omega_f_3D[a, b, c]

                    
        temp =np.dot(omega_f_new, X_f_new)
        
        #temp_Y_f = np.dot((np.linalg.pinv(np.dot(temp.T*temp)))*temp.T)*tensor_V

        temp_Y_f,resid,rank,sigma = np.linalg.lstsq(temp, tensor_V)

        for j in range(k):
            Y_f[:,i, j] = np.squeeze(temp_Y_f[j*r:(j+1)*r])
    return Y_f