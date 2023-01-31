#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:39:11 2023

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import sys, getopt, os
import 

from pdla import pdla
from pdhg import pdhg
from sapg import sapg
import potentials as pot
import distributions as pds

#%% initial parameters: test image, computation settings etc.
params = {
    'iterations': 1000,
    'testfile_path': 'test_images/fibo.jpeg',
    'efficient': True,
    'verbose': True
    }

#%% auxiliary functions
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

def my_imshow(im, label, vmin=0, vmax=1):
    plt.imshow(im, cmap='Greys_r',vmin=vmin,vmax=vmax)
    plt.title(label)
    plt.colorbar()
    plt.show()

#%% Main method
def main():
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./results/rof'):
        os.makedirs('./results/rof')
    image_dir = './results/rof/{}'.format(params['testfile_path'].split('/')[-1].split('.')[0])
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    #%% generate data - image deblurring
    rng = default_rng(13401)
    verb = params['verbose']
    try:
        x = cv2.imread(params['testfile_path']).astype(float)
    except AttributeError:
        print('Provided test image did not exist under that path, aborting.')
        sys.exit()
    # handle images that are too large or colored
    if x.shape[0] > 512 or x.shape[1] > 512: x = cv2.resize(x, dsize=(512,512))
    if len(x.shape) == 3 and x.shape[2] == 3: x = 0.299*x[...,0] + 0.587*x[...,1] + 0.114*x[...,2]

    x = x-np.min(x)
    x = x/np.max(x)
    # assume quadratic images
    n = x.shape[0]
    
    #%% Generate noisy observation for l1-tv using additive Laplace noise or salt-pepper noise
    blur_width = 5
    a,at,max_ev = blur(n,blur_width)
    
    noise_std = 0.15
    lamda = 6
    # normally distributed noise
    y = a(x) + noise_std*rng.normal(size=x.shape)
    
    #%% denoise the image (=compute MAP using PDHG)
    if verb: sys.stdout.write('Compute ROF model MAP - '); sys.stdout.flush()
    tv = pot.total_variation(n, n, scale=lamda)
    u,_ = tv.inexact_prox(y, gamma=noise_std**2, epsilon=1e-5, max_iter=400, verbose=verb)
    if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
    
    #%% define the posterior and compute the optimal regularization parameter using SAPG
    unscaled_posterior = pds.l2_deblur_tv(n, n, a, at, y, noise_std=noise_std, mu_tv=1)
    # metaparameter of the posterior: L = Lipschitz constant of nabla F, necessary for stepsize
    L = max_ev/noise_std**2
    
    theta0 = 0.01
    # empirically, for blur b=10 we need ~1000 warm up iterations with tau = 0.9/L. 
    # for blur=5 roughly 500 warm up iterations
    # For b=0 almost immediate warm-up since the noisy image seems to be in a region of high probability
    s = sapg(iter_wu=500,iter_outer=25,iter_burnin=10,iter_inner=1,
             tau=0.9/L,delta=lambda k: 0.1/(theta0*n**2)*(k+1)**(-0.8),
             x0=y,theta0=theta0,theta_min=0.001,theta_max=1e2,
             epsilon_prox=3e-2,pd=unscaled_posterior)
    s.simulate()
    
    #%% sample using PDLA
    k = tv._imgrad
    kt = tv._imdiv
    x0, y0 = np.zeros_like(x), np.zeros((2,)+x.shape)
    tau = 1/np.sqrt(8)
    sigma = 1/(tau*8)
    n_samples = params['iterations']
    burnin = 50
    f = pot.l2_l1_norm(n, n, scale=lamda)
    g = pot.l2_loss_homoschedastic(y=y, sigma2=noise_std**2)
    eff = params['efficient']
    
    sampler = pdla(x0, y0, tau, sigma, n_samples, burnin, f, g, k, kt, efficient=eff)
    
    if verb: sys.stdout.write('Sample from ROF posterior - '); sys.stdout.flush()
    sampler.simulate(verbose=verb)
    if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
    
    #%% plots
    # diagnostic plot, making sure the sampler looks plausible
    plt.plot(np.arange(1,n_samples+1), sampler.logpi_vals)
    plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n)')
    plt.show()
    
    # images
    my_imshow(x, 'ground truth')
    my_imshow(y, 'noisy image')
    my_imshow(u,'ROF MAP (dual AGD, mu_TV = {})'.format(lamda))
    my_imshow(sampler.mean, 'Sample Mean')
    my_imshow(sampler.std, 'Sample standard deviation', np.min(sampler.std), np.max(sampler.std))
    my_imshow(sampler.var, 'Sample variance', np.min(sampler.var), np.max(sampler.var))
    
    #%% saving
    cv2.imwrite(image_dir+'/ground_truth.png',x*256)
    cv2.imwrite(image_dir+'/noisy.png',y*256)
    cv2.imwrite(image_dir+'/rof_map.png',u*256)
    cv2.imwrite(image_dir+'/rof_posterior_mean.png',sampler.mean*256)
    r1 = sampler.std - np.min(sampler.std)
    cv2.imwrite(image_dir+'/rof_posterior_std.png',r1/np.max(r1)*256)
    r2 = sampler.var - np.min(sampler.var)
    cv2.imwrite(image_dir+'/rof_posterior_var.png',r2/np.max(r2)*256)
    
    
#%% help function for calling from command line
def print_help():
    print('<>'*39)
    print('Run primal dual Langevin algorithm to generate samples from ROF posterior.')
    print('<>'*39)
    print(' ')
    print('Options:')
    print('    -h (--help): Print help.')
    print('    -i (--iterations=): Number of iterations of the Markov chain')
    print('    -f (--testfile_path=): Path to test image file')
    print('    -e (--efficient): Storage-efficient mode, where we dont save samples but only compute a runnning mean and standard deviation during the algorithm')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:f:ev",
                                   ["help","iterations=","testfile_path=",
                                    "efficient","verbose"])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-i", "--iterations"):
            params['iterations'] = int(arg)
        elif opt in ("-f", "--testfile_path"):
            params['testfile_path'] = arg
        elif opt in ("-e","--efficient"):
            params['efficient'] = True
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    