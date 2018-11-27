# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:53:00 2018

@author: ZCC
"""

import numpy as np
from t_prod import t_prod
from t_svd import t_svd
from tubalrank import tubalrank


m = 60
n = 60
k = 20
r = 5

'''
%low-tubal-rank tensor
%T = rand(m,n,k);  %a ranom tensor: m * n * k
%T = t_svd_threshold(T,r);  %make it to be tubal-rank = r
'''


T = t_prod(np.random.rand(m, r, k),np.random.rand(r, n, k))

r = tubalrank(T)
print(r)