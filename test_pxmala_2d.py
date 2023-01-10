#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:06:03 2022

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from pxmala import PxMALA
import distributions as pds

def main():
    """ generate data """
    rng = default_rng(38456809)
    n1,n2 = 64,64
    im_true = np.zeros((n1,n2))
    n_peaks,mean_peak = 10,15
    S0,S1 = rng.integers(n1,size=(n_peaks,)),rng.integers(n2,size=(n_peaks,))
    H = rng.poisson(mean_peak,size=(n_peaks,))
    for i in np.arange(n_peaks):
        im_true[S0[i],S1[i]] = H[i]
    # draw a noisy version of u_true by adding normally distributed noise
    std_noise = 2
    im_noisy = im_true + std_noise*rng.normal(size=(n1,n2))
    
    """ define posterior """
    u_noisy = np.reshape(im_noisy,(-1,1))
    mu_l1 = 2
    posterior = pds.L2Loss_SparsityReg(n1*n2,l2shift=u_noisy,l2scale=std_noise,l1reg=mu_l1)
    # metaparameter of the distribution: L = Lipschitz constant of nabla F
    L = 1/std_noise**2
    
    """<><><> compute the MAP using FISTA and check that it properly denoises the image <><><>"""
    u_fista = u_noisy
    v_fista = u_fista
    iter_fista, tau_fista = 10, std_noise**2
    t = 1
    for k in np.arange(iter_fista):
        # gradient forward step
        w = v_fista - tau_fista*1/std_noise**2 * (v_fista-u_noisy)
        # proximal gradient backward step
        u_new = np.maximum(np.abs(w)-tau_fista*mu_l1, 0) * np.sign(w)
        t_new = (1+np.sqrt(1+4*t**2))/2
        v_fista = u_new + (t-1)/t_new * (u_new-u_fista)
        u_fista = u_new
        t = t_new
    
    u_map = u_fista
    im_map = np.reshape(u_map,(n1,n2))
    fig,ax = plt.subplots(1,3,figsize=(6,19))
    ax[0].imshow(im_true,cmap='Greys')
    ax[1].imshow(im_noisy,cmap='Greys')
    ax[2].imshow(im_map,cmap='Greys')
    plt.show()
    print('True support:')
    print(np.nonzero(im_true)[0])
    print(np.nonzero(im_true)[1])
    print('Denoised support:')
    print(np.nonzero(im_map)[0])
    print(np.nonzero(im_map)[1])
    
    """ sample using px-mala """
    tau_pxmala = 0.9/L*1e-7
    max_iter_pxmala = 1000
    n_samples_pxmala = 100
    # initialize at the MAP to minimize burn-in time
    #x0_pxmala = np.zeros((n1*n2,n_samples_pxmala)) 
    x0_pxmala = u_map*np.ones((1,n_samples_pxmala))
    pxmala = PxMALA(max_iter_pxmala, tau_pxmala, x0_pxmala, pd = posterior)
    
    x_pxmala,acceptance = pxmala.simulate(return_all=True)
    print('Acceptance rate after {} iterations on {} samples: {}%'.format(max_iter_pxmala,n_samples_pxmala,100*acceptance))
    mean_samples = np.mean(x_pxmala,axis=(1,2))
    im_mean = np.reshape(mean_samples,(n1,n2))
    std_samples = np.std(x_pxmala,axis=(1,2))
    im_std= np.reshape(std_samples,(n1,n2))
    fig,ax = plt.subplots(1,2,figsize=(7,15))
    ax[0].imshow(im_mean)
    ax[1].imshow(im_std)
    plt.show()
    return x_pxmala

if __name__ == '__main__':
    x_pxmala = main()
    
    
    
    
    
    
    
    
    
    
    
    
    