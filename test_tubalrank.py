# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:05:26 2018

@author: ZCC
"""

import numpy as np
from t_prod import t_prod
from tubalrank import tubalrank

m = 60
n = 40
k = 20
r = 5

T = t_prod(np.random.rand(m, r, k),np.random.rand(r, n, k))

r1 = tubalrank(T,1)
print(r1)