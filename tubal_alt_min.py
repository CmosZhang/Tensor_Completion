# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:32:25 2018

@author:ZCC
"""

import numpy as np
from scipy.fftpack import fft,ifft
from t_prod import t_prod
from tubalrank import tubalrank
from tubal_alt_min_one_step import tubal_alt_min_one_step
from numpy import linalg as LA


''' data loading '''
'''
data = sio.loadmat('video.mat')
T = data['T']
T = T[1:50,1:50,1:20]
[m, n, k] = T.shape
'''
m = 60
n = 60
k = 20
r = 5

T = t_prod(np.random.rand(m, r, k),np.random.rand(r, n, k))

'''tubal rank'''
r = tubalrank(T,1)


''' random element sampling'''
samplingrate = 0.7
omega = np.random.rand(m, n, k)<=samplingrate
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
while iteration<=15:
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

'''
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
'''