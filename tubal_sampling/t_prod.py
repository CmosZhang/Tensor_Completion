# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:55:56 2018

@author: ZCC
"""

'''
tensor product
时域相卷，频域相乘
'''

import numpy as np
from scipy.fftpack import fft,ifft
def t_prod(A,B):
    [a1,a2,a3] = A.shape
    [b1,b2,b3] = B.shape
    A = fft(A)
    B = fft(B)
    C = np.zeros((a1,b2,b3), dtype = complex)
    for i in range(b3):
        C[:, :,i] = np.dot(A[:, :, i], B[:, :, i])
    C = ifft(C)
    return C   