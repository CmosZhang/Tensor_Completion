# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:27:51 2018

@author: ZCC
"""

'''
向量化
'''


import numpy as np

def Mat_to_Vec(X):
    [a,b] = X.shape
    x = np.zeros((a*b,1),dtype = complex)
    for i in range(b):
        for j in range(a):
            x[j+i*a] = X[j,i]
    return x


'''
A = np.random.rand(3,4)
print(A)
aa = Mat_to_Vec(A)
#print(aa)
b = np.diagonal(aa)
print(b)
'''