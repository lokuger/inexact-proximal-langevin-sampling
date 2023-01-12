#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 08:51:53 2023

@author: lorenzkuger
"""

from numpy.random import default_rng
import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt

import potentials as pot
import distributions as pds
from sapg import sapg

def blur(n, b):
    """compute the blur operator a, its transpose a.t and the maximum eigenvalue 
    of ata.
    Carfeul, this assumes a quadratic n x n image
    blur length b must be integer and << n to prevent severe ill-posedness"""
    h = np.ones((1, b))
    lh = h.shape[1]
    h = h / np.sum(h)
    h = np.pad(h, ((0,0), (0, n-b)), mode='constant')
    h = np.roll(h, -int((lh-1)/2))
    h = np.matmul(h.T, h)
    H_FFT = np.fft.fft2(h)
    HC_FFT = np.conj(H_FFT)
    a = lambda x : np.real(np.fft.ifft2(H_FFT * np.fft.fft2(x)))
    at = lambda x : np.real(np.fft.ifft2(HC_FFT * np.fft.fft2(x)))
    max_eigval = power_method(a, at, n, 1e-4, int(1e3), 0)
    return a,at,max_eigval
    
def power_method(a, at, n, tol, max_iter, verbose):
    """power method to compute the maximum eigenvalue of the linear op at*a"""
    x = np.random.normal(size=(n,n))
    x = x/np.linalg.norm(x.ravel())
    val, val_old = 1, 1
    for k in range(max_iter):
        x = at(a(x))
        val = np.linalg.norm(x.ravel())
        rel_var = np.abs(val-val_old)/val_old
        val_old = val
        x = x/val
        if rel_var < tol:
            break
    return val

def main():
    """ generate data - image denoising """
    rng = default_rng(38456809)
    x = iio.imread('test_images/barbara.png').astype(float)
    # careful, we assume a quadratic n x n images for simplicity. 
    # Might have to rewrite parts of the files to handle non-quadratic ones
    n = x.shape[0]
    
    tv = pot.total_variation(n, n, scale=1)
    tv_groundtruth = tv(x)
    
    """ Generate noisy observation: normally distributed/Gaussian noise model"""
    a,at,max_ev = blur(n,b=9)                       # blur operator
    # a,at,max_ev = lambda x : x, lamba x : x, 1    # only denoising
    
    y = a(x)
    noise_snr = 20 # [dB]
    noise_std = np.std(y)*10**(-noise_snr/20)
    y = y + noise_std*rng.normal(size=(n,n))
    
    """ define posterior. For SAPG, use reg parameter theta = 1 since it 
    handles the changing parameter inside the algorithm """
    unscaled_posterior = pds.l2_deblur_tv(n, n, a, at, y, noise_std=noise_std, mu_tv=1)
    # metaparameter of the posterior: L = Lipschitz constant of nabla F, necessary for stepsize
    L = max_ev/noise_std**2
   
    """ determine optimal theta using SAPG """
    iter_outer = 40
    iter_burnin = 20
    theta0 = 0.01
    s = sapg(iter_wu=20,
             iter_outer=iter_outer,
             iter_burnin=iter_burnin,
             iter_inner=1,
             tau=0.2/L,
             delta=lambda k: 0.1/(theta0*n**2) * (k+1)**(-0.8),
             x0=y,
             theta0=theta0,
             theta_min=0.001,
             theta_max=1,
             epsilon_prox=1e-1,
             pd=unscaled_posterior)
    s.simulate()
    
    """ -- plots to check that SAPG converged -- """
    # plt.plot(s.logpi_wu, label='log-likelihood warm-up samples')
    # plt.legend()
    # plt.show()
    
    # # thetas
    # plt.plot(s.theta,label='theta_n')
    # plt.plot(n1*n2/s.mean_g, label='dim/g(u_n)', color='orange')
    # plt.plot(np.arange(s.iter_burnin,s.iter_outer), s.mean_theta, label='theta_bar',color='green')
    # plt.legend()
    # plt.show()
    
    # # values g(X_n)
    # plt.plot(s.mean_g, label='g(u_n)')
    # plt.hlines(tv_groundtruth,0,iter_outer+1, label='g(u_true)')
    # plt.legend()
    # plt.show()
    
    
    
    
if __name__ == '__main__':
    main()