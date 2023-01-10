#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:04:45 2023

@author: lorenzkuger

This is a script that tests the SAPG routine from 
"Maximum likelihood estimation of regularisation parameters in high-dimensional inverse problems: an empirical Bayesian approach
Part I: Methodology and Experiments", 2020
by Vidal, De Bortoli, Pereyra, Durmus

The algorithm automatically estimates the regularisation parameter for a 
variational minimization problem with exactly known forward operator by 
computing the marginal MLE of the regularization parameter. 
For the inexact proximal Langevin algorithms, we need this as a side routine
that estimates the regularisation parameter, so that we can then run the 
inexact samplers on the correct distributions. This particularly ensures
comparability to the literature and eliminates the need to 'guess' the 
parameter using heuristics.
"""

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
    n1,n2 = im_true.shape
    
    tv = pot.TotalVariation_scaled(n1, n2, scale=1)
    tv_groundtruth = tv(np.reshape(im_true,(-1,1)))
    mu_prop = n1*n2/tv_groundtruth
    
    """ Normally distributed/Gaussian noise model"""
    # noise_psnr = 30 # [dB]
    # noise_std = 10**(-noise_psnr/20)
    noise_snr = 15 # [dB]
    noise_std = np.std(im_true)*10**(-noise_snr/20)
    im_noisy = im_true + noise_std*rng.normal(size=(n1,n2))
    
    """ define posterior. For SAPG, use reg parameter theta = 1 and include the
    parameter in the algorithm itself by multiplying the step size with the 
    parameter in the prox evaluation """
    u_noisy = np.reshape(im_noisy,(-1,1))
    posterior = pds.L2Loss_TVReg(n1, n2, l2shift=u_noisy, l2scale=noise_std, mu_TV=1)
    # metaparameter of the posterior: L = Lipschitz constant of nabla F, necessary for stepsize
    L = 1/noise_std**2
   
    """ determine optimal theta using SAPG """
    iter_outer = 50
    theta0 = 0.01
    sapg = SAPG(iter_wu=200, 
                iter_outer=iter_outer, 
                iter_inner=1, 
                tau=0.2/L, 
                delta=lambda n: 0.1/(theta0*n1*n2) * (n+1)**(-0.8), 
                x0=u_noisy, 
                theta0=theta0, 
                thetamin=0.001, 
                thetamax=1, 
                epsilon_prox=1e-1, 
                pd=posterior)
    sapg.simulate()
    
    
    
    
    """ images & plotting """
    # plt.imshow(im_true, cmap='Greys_r',vmin=0,vmax=255)
    # plt.colorbar()
    # plt.title('True image')
    # plt.show()
    
    # plt.imshow(im_noisy, cmap='Greys_r',vmin=0,vmax=255)
    # plt.colorbar()
    # plt.title('Noisy image')
    # plt.show()
    
    # u_denoised = tv.inexact_prox(u_noisy, gamma=10, epsilon=1e-3, maxiter=50)
    # im_denoised = np.reshape(u_denoised, (n1,n2))
    # plt.imshow(im_denoised, cmap='Greys_r',vmin=0,vmax=255)
    # plt.colorbar()
    # plt.title('Denoised image')
    # plt.show()
    
    # log pi during warm-up:
    plt.plot(sapg.logpi_wu, label='log-likelihood warm-up samples')
    plt.legend()
    plt.show()
    
    # thetas
    plt.plot(sapg.theta,label='theta_n')
    plt.plot(n1*n2/sapg.meansG, label='dim/TV(u_n)', color='orange')
    plt.plot(sapg.mean_theta, label='theta_bar',color='green')
    plt.hlines(mu_prop, 0, iter_outer, label='dim/TV(u_true)',color='red')
    plt.legend()
    plt.show()
    
    # values g(X_n)
    plt.plot(sapg.meansG, label='TV(u_n)')
    plt.hlines(tv_groundtruth,0,iter_outer+1, label='TV(u_true)')
    plt.legend()
    plt.show()
    
    

if __name__ == '__main__':
    main_tv()
    
    
    
    
    
    
    
    
    
    
    
    
    