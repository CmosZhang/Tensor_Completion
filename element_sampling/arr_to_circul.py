# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:04:41 2018

@author: ZCC
"""

import numpy as np


def arr_to_circul(x):
    n = len(x)
    X = np.zeros((n, n), dtype = complex)
    for i in range(n):
        for j in range(n):
            k = i + j
            s = k
            if k>n-1:
                s = k-n
            X[i, s] = x[j]
    return X