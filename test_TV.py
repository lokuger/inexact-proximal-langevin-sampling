#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 20:09:00 2022

@author: lorenzkuger
"""

import potentials as pot
import distributions as pds
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def main():
    n1 = 128
    n2 = 128
    TV = pot.TotalVariation(n1, n2)
    u = np.zeros((n1,n2))
    for i1 in np.arange(n1):
        for i2 in np.arange(n2):
            R2 = 1/2*(i1-37)**2 + (i2-80)**2
            if R2 > 20**2 and R2 < 30**2:
                u[i1,i2] = 1
    u = np.reshape(u,(-1,1))
    u_noisy = u + 0.25*rnd.normal(size=(n1*n2,1))
    
    gamma = 1
    epsilon = 1e-4*n1*n2
    maxiter = 1e4
    u_denoise = TV.inexact_prox(u_noisy, gamma, epsilon, maxiter,verbose=True)
    
    plt.imshow(u_denoise,cmap='Greys')
    plt.colorbar()
    plt.show()
    

if __name__ == "__main__":
    main()
    