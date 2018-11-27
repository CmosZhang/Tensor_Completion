# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:37:02 2018

@author: ZCC
"""

import numpy as np
from numpy import linalg as la
from scipy.fftpack import fft,ifft


def t_svd(M):
	[n1 ,n2 ,n3] = M.shape
	D = np.zeros((n1 ,n2 ,n3), dtype = complex)
	D = fft(M)   
	Uf = np.zeros((n1,n1,n3), dtype = complex)
	Thetaf = np.zeros((n1,n2,n3), dtype = complex)
	Vf = np.zeros((n2,n2,n3), dtype = complex)	

	for i in range(n3):
		temp_U ,temp_Theta, temp_V = la.svd(D[: ,: ,i], full_matrices=True);
		Uf[: ,: ,i] = temp_U;
		Thetaf[:n2, :n2, i] = np.diag(temp_Theta)
		Vf[:, :, i] = temp_V;
	U = np.zeros((n1,n1,n3))
	Theta = np.zeros((n1,n2,n3))
	V = np.zeros((n2,n2,n3))
	U = ifft(Uf).real
	Theta = ifft(Thetaf).real
	V = ifft(Vf).real
	return U, Theta, V