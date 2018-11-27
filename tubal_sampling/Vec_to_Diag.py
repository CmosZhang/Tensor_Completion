# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:37:31 2018

@author: ZCC
"""

'''
将一个列向量转换为对角矩阵
'''


import numpy as np



def Vec_to_Diag(x):
    [a,b] = x.shape
    X = np.zeros((a,a),dtype = complex)
    for i in range(a):
        X[i,i] = x[i]
    return X


'''
A = np.random.rand(3,4)
print(A)
aa = Mat_to_Vec(A)
#print(aa)
b = Vec_to_Diag(aa)
print(b)
'''