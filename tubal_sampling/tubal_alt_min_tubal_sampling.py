# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:12:28 2018

@author: ZCC
"""

import numpy as np
from scipy.fftpack import fft,ifft
from t_prod import t_prod
from tubalrank import tubalrank
from tubal_alt_min_one_step import tubal_alt_min_one_step
from numpy import linalg as LA
import scipy.io as sio
import matplotlib.pyplot as plt

''' data loading '''
data = sio.loadmat('video.mat')
T = data['T']
T = T[1:50,1:50,1:20]
[m, n, k] = T.shape

'''tubal rank'''
r = tubalrank(T,1)


''' random tubal sampling'''
samplingrate = 0.7
omega_tubal = np.random.rand(m, n)<=samplingrate
omega = np.zeros((m,n,k))
for i in range(k):
    omega[:,:,i] = omega_tubal
T_omega = np.zeros((m, n, k))
T_omega = omega*T


''' fft transform'''
T_f = fft(T)
T_omega_f = fft(T_omega)
omega_f = fft(omega)


'''Given Y, do LS to get X'''
Y = np.random.rand(r, n, k)
Y_f = fft(Y)

''' do the transpose for each frontal slice'''
Y_f_trans = np.zeros((n, r, k), dtype = complex)
X_f = np.zeros((m, r, k), dtype = complex)
T_omega_f_trans  = np.zeros((n, m, k), dtype = complex)
omega_f_trans = np.zeros((n, m, k), dtype = complex)
for i in range(k):
    Y_f_trans[:, :, i] = Y_f[:, :, i].T
    T_omega_f_trans[:, :, i] = T_omega_f[:, :, i].T
    omega_f_trans[:, :, i] = omega_f[:, :, i].T

iteration = 1;
while iteration<=5:
    print('Random Element Sampling--', samplingrate,'---Round--', iteration);
    X_f_trans = tubal_alt_min_one_step(T_omega_f_trans, omega_f_trans*1/k, Y_f_trans)
    for i in range(k):
        X_f[:, :, i] = X_f_trans[:, :, i].T
    '''Given X, do LS to get Y'''
    Y_f = tubal_alt_min_one_step(T_omega_f, omega_f*1/k, X_f)
    for i in range(k):
        Y_f_trans[:, :, i] = Y_f[:, :, i].T
    iteration = iteration + 1
    
    
'''%The relative error:'''
temp = 0
X_est = ifft(X_f)
Y_est = ifft(Y_f)
T_est = t_prod(X_est,Y_est)

temp = T-T_est
error = LA.norm(temp[:])/LA.norm(T[:])
print('error:', error)


####################################################
#    plot figure
####################################################
plt.figure(1)
plt.subplot(131)
plt.imshow(T[:, :, 1],cmap = 'gray')
plt.title('Original data')
plt.figure(2)
plt.subplot(132)
plt.imshow(T_omega[:, :, 1],cmap = 'gray')
plt.title('Corrupted data')
plt.figure(3)
plt.subplot(133)
plt.imshow(T_est.real[:, :, 1], cmap = 'gray')
plt.title('Recovery data')