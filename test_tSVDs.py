# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:24:41 2018

@author: ZCC
"""

import matplotlib.pyplot as plt
import numpy as np


from tSVDs import t_SVDs
from t_prod import t_prod
import scipy.io as sio

def t_svds_test():
    M = sio.loadmat('video.mat')
    M = M['T']
    a1,b1,c1  = t_SVDs(M, 30)
    M_svds = t_prod(t_prod(a1,b1), c1).real
    err = M-M_svds
    print('The RSE =',np.linalg.norm(err)/np.linalg.norm(M))
    print('Transformed Tensor is equal to the origin:', np.allclose(M, M_svds))
    plt.subplot(121)
    plt.imshow(M_svds[:, :, 1],cmap = 'gray')
    plt.title('t_svds result')
    plt.subplot(122)
    plt.imshow(M[:, :, 1], cmap = 'gray')
    plt.title('original')
if __name__=='__main__':
    t_svds_test()