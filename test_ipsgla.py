#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:23:01 2023

@author: lorenzkuger
"""

import rwt
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import imageio.v2 as iio

import distributions as pds
import potentials as pot
from sapg import *
from inexact_psgla import *

def main_l1():
    """ generate data - sparse image for l1-restoration"""
    rng = default_rng(38456809)
    n1,n2 = 64,64
    im_true = np.zeros((n1,n2))
    n_peaks,mean_peak = 10,15
    S0,S1 = rng.integers(n1,size=(n_peaks,)),rng.integers(n2,size=(n_peaks,))
    H = rng.poisson(mean_peak,size=(n_peaks,))
    for i in np.arange(n_peaks):
        im_true[S0[i],S1[i]] = H[i]
    # plt.imshow(im_true, cmap='Greys_r',vmin=0,vmax=1)
    # plt.colorbar()
    # plt.title('True image')
    # plt.show()
    
    l1norm = pot.L1loss_scaled(n1*n2, b=1)
    l1_groundtruth = l1norm(np.reshape(im_true,(-1,1)))[0,0]
    print('l1-norm of ground truth = {}'.format(l1_groundtruth))
    mu_prop = n1*n2/l1_groundtruth
    print('Hence optimal l1 regularization parameter should be theta = d / ||u_true||_1 = {:.2f}'.format(mu_prop))
    
    """ Normally distributed/Gaussian noise model"""
    # noise_psnr = 30 # [dB]
    # noise_std = 10**(-noise_psnr/20)
    noise_snr = 10 # [dB]
    noise_std = np.std(im_true)*10**(-noise_snr/20)
    im_noisy = im_true + noise_std*rng.normal(size=(n1,n2))
    plt.imshow(im_noisy, cmap='Greys_r',vmin=0,vmax=255)
    plt.colorbar()
    plt.title('Noisy image')
    plt.show()
    
    """ define posterior """
    u_noisy = np.reshape(im_noisy,(-1,1))
    mu_l1 = mu_prop
    posterior = pds.L2Loss_SparsityReg(n1*n2, l2shift=u_noisy, l2scale=noise_std, l1reg=mu_l1)
    # metaparameter of the distribution: L = Lipschitz constant of nabla F, necessary for stepsize
    L = 1/noise_std**2
    
    """ compute the MAP using the l1 prox and check that it properly denoises the image & that the regularization parameter is plausible """
    u_denoised = l1norm.prox(u_noisy, gamma = mu_l1*noise_std**2)
    im_denoised = np.reshape(u_denoised, (n1,n2))
    plt.imshow(im_denoised, cmap='Greys_r',vmin=0,vmax=1)
    plt.colorbar()
    plt.title('Denoised image')
    plt.show()
    
    """ sample using inexact psgla to make sure the sampler works and step size is right """
    max_iter = 30
    tau = 0.9/L
    x0 = np.zeros_like(u_denoised) #u_denoised
    epsilon = 1e-1
    ipsgla = IPSGLA(max_iter, tau, x0, epsilon, pd=posterior)
    u_samples = ipsgla.simulate(return_all=True)
    log_probs_samples = np.zeros((max_iter+1,))
    for i in np.arange(max_iter+1):
        v = u_samples[:,:,i]
        log_probs_samples[i] = -posterior.F(v)-posterior.G(v)
    plt.plot(log_probs_samples)


def main_tv():
    """ generate data - image denoising """
    rng = default_rng(38456809)
    im_true = iio.imread('test_images/barbara.png').astype(float)
    plt.imshow(im_true, cmap='Greys_r',vmin=0,vmax=255)
    plt.colorbar()
    plt.title('True image')
    plt.show()
    n1,n2 = im_true.shape
    
    tv = pot.TotalVariation_scaled(n1, n2, scale=1)
    tv_groundtruth = tv(np.reshape(im_true,(-1,1)))
    print('TV-seminorm of ground truth = {}'.format(tv_groundtruth))
    mu_prop = n1*n2/tv_groundtruth
    print('Hence optimal TV regularization parameter should be theta = d / TV(u_true) = {:.2f}'.format(mu_prop))
    
    """ Normally distributed/Gaussian noise model"""
    # noise_psnr = 30 # [dB]
    # noise_std = 10**(-noise_psnr/20)
    noise_snr = 30 # [dB]
    noise_std = np.std(im_true)*10**(-noise_snr/20)
    im_noisy = im_true + noise_std*rng.normal(size=(n1,n2))
    plt.imshow(im_noisy, cmap='Greys_r',vmin=0,vmax=255)
    plt.colorbar()
    plt.title('Noisy image')
    plt.show()
    
    """ define posterior """
    u_noisy = np.reshape(im_noisy,(-1,1))
    mu_tv = mu_prop
    posterior = pds.L2Loss_TVReg(n1, n2, l2shift=u_noisy, l2scale=noise_std, mu_TV=mu_tv)
    # metaparameter of the posterior: L = Lipschitz constant of nabla F, necessary for stepsize
    L = 1/noise_std**2
    
    
    """ compute the MAP using the TV prox and check that it properly denoises the image & that the regularization parameter is plausible """
    # u_denoised = tv.inexact_prox(u_noisy, gamma=mu_prop*noise_std**2, epsilon=1e-3, maxiter=50)
    # im_denoised = np.reshape(u_denoised, (n1,n2))
    # plt.imshow(im_denoised, cmap='Greys_r',vmin=0,vmax=255)
    # plt.colorbar()
    # plt.title('Denoised image')
    # plt.show()
    
    """ sample using inexact psgla to make sure the sampler works and step size is right """
    max_iter = 10
    tau = 0.9/L
    x0 = u_noisy
    epsilon = 1e-1
    ipsgla = IPSGLA(max_iter, tau, x0, epsilon, pd=posterior)
    u_samples = ipsgla.simulate(return_all=True)
    log_probs_samples = np.zeros((max_iter+1,))
    for i in np.arange(max_iter+1):
        v = u_samples[:,:,i]
        log_probs_samples[i] = -posterior.F(v)-posterior.G(v)
    plt.plot(log_probs_samples)
    plt.show()

if __name__ == '__main__':
    main_tv()
    
    
    
    
    
    
    
    
    
    
    
    
    