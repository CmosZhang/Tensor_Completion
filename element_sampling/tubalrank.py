# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:22:21 2018

@author: ZCC
"""
'''
计算tensor的tubalrank,根据CDF（分布累计函数来求，阈值设为90%）
计算随机点缺失的tubal-rank.
'''


import numpy as np
from numpy import linalg as LA
from t_svd import t_svd

def tubalrank(T,sampleRate):
    [m,n,k]=T.shape
    Omega = np.random.rand(m,n,k)<sampleRate
    T_omega = Omega*T
    [U,S,V]=t_svd(T_omega)
    sz = min(m,n)
    S1 = S[1:sz,1:sz,:]
    
    tubal=np.zeros((1,sz))
    sumN=0
    for j in range(0,sz-1):
        tubal[:,j] = tubal[:,j] + LA.norm(np.squeeze(S1[j,j,:]))   
        sumN = tubal[:,j] + sumN
    r=0
    sum=0
    CDF = np.zeros((1,sz))
    for i in range(0,sz-1):
        sum=sum+tubal[:,i]
        CDF[:,i]=100*sum/sumN
        if CDF[:,i]>90 and r==0:
            r=i+1
    return r