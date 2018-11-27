# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:32:25 2018

@author:ZCC
"""

import numpy as np
from scipy.fftpack import fft,ifft
from t_prod import t_prod
from tubalrank import tubalrank
from tubal_alt_min_tubal_sampling_one_step import tubal_alt_min_tubal_sampling_one_step
from numpy import linalg as LA
import scipy.io as sio
import matplotlib.pyplot as plt

''' data loading '''
data = sio.loadmat('video.mat')
T = data['T']
T = T[1:15,1:15,1:5]
[m, n, k] = T.shape

'''tubal rank'''
r = tubalrank(T,1)


''' random tubal sampling'''
samplingrate = 0.7
omega_temp = np.random.rand(m, n)<=samplingrate
omega = np.zeros((m,n,k))
for i in range(k):
    omega[:,:,i] = omega_temp
T_omega = np.zeros((m, n, k))
T_omega = omega*T


''' fft transform'''
T_f = fft(T)
T_omega_f = fft(T_omega)
omega_f = fft(omega)


'''Alternating Minimization'''
Y = np.random.rand(r, n, k)
Y_f = fft(Y)
X_f = np.zeros((m,r,k),dtype = complex)
X_f_temp = np.zeros((m,r),dtype = complex)

max_iter = 15

for j in range(k):
    T_temp = T_omega_f[:,:,j]
   # print(T_temp.shape)
    omega_temp = omega[:,:,1]*1/k
    Y_f_temp = Y_f[:,:,j]
   # print(Y_f_temp.shape)
    T_temp_trans = T_temp.T
    omega_temp_trans = omega_temp.T
    Y_temp_trans = Y_f_temp.T
    
    for i in range(max_iter+1):
        [X_f_temp] = tubal_alt_min_tubal_sampling_one_step(T_temp_trans,
                                                           omega_temp_trans,
                                                           Y_temp_trans)
        X_f[:,:,j] = X_f_temp.T
        
        [Y_f_temp] = tubal_alt_min_tubal_sampling_one_step(T_temp,omega_temp.T,X_f_temp.T)
        Y_f[:,:,j] = Y_f_temp


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