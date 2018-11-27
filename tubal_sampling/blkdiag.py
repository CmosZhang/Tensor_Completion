# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:58:42 2018

@author: ZCC
"""

'''
将一个 tensor 排列成 block 形式
'''

import numpy as np


def blkdiag(X, x):
    [n1, n2] = X.shape
    [m1, m2] = x.shape
    row = n1 + m1
    col = n2 + m2
    X_new = np.zeros((row, col), dtype = complex)
    X_new[0:n1, 0:n2] = X
    X_new[n1:, n2:] = x
    return X_new