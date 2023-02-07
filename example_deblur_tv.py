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
#from time import time
from skimage import io, transform

from inexact_pla import inexact_pla
#from sapg import sapg
from pdhg import pdhg#, acc_pdhg
import potentials as pot
import distributions as pds

#%% initial parameters: test image, computation settings etc.
params = {
    'iterations': 10000,
    'testfile_path': 'test_images/cameraman.tif',
    'blurtype': 'gaussian',
    'bandwidth': 3,
    'noise_std': 0.01,
    'logepsilon': -9,
    'efficient': True,
    'verbose': True
    }

#%% auxiliary functions
def blur_unif(n, b):
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
    ata = lambda x : np.real(np.fft.ifft2(H_FFT * HC_FFT * np.fft.fft2(x)))
    max_eigval = power_method(ata, n, 1e-4, int(1e3))
    return a,at,max_eigval

def blur_gauss(n, sigma):
    """compute the blur operator a, its transpose a.t and the maximum eigenvalue 
    of ata.
    Carfeul, this assumes a quadratic n x n image, with n even
    blur standard dev is assumed to be given in #pixels"""
    t = np.arange(-n/2+1,n/2+1)
    h = np.exp(-t**2/(2*sigma**2))
    h = h / np.sum(h)
    h = np.roll(h, -int(n/2)+1)
    h = h[np.newaxis,:] * h[:,np.newaxis]
    H_FFT = np.fft.fft2(h)
    HC_FFT = np.conj(H_FFT)
    a = lambda x : np.real(np.fft.ifft2(H_FFT * np.fft.fft2(x)))
    at = lambda x : np.real(np.fft.ifft2(HC_FFT * np.fft.fft2(x)))
    ata = lambda x : np.real(np.fft.ifft2(H_FFT * HC_FFT * np.fft.fft2(x)))
    max_eigval = power_method(ata, n, 1e-4, int(1e3))
    return a,at,max_eigval
    
def power_method(ata, n, tol, max_iter, verbose=False):
    """power method to compute the maximum eigenvalue of the linear op at*a"""
    x = np.random.normal(size=(n,n))
    x = x/np.linalg.norm(x.ravel())
    val, val_old = 1, 1
    for k in range(max_iter):
        x = ata(x)
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

#%% Main method - generate results directories
def main():
    if not os.path.exists('./results'): os.makedirs('./results')
    if not os.path.exists('./results/deblur_tv'): os.makedirs('./results/deblur_tv')
    blur_dir = './results/deblur_tv/{}'.format(params['blurtype'])
    if not os.path.exists(blur_dir): os.makedirs(blur_dir)
    bandwidth_dir = blur_dir + '/blur{}'.format(params['bandwidth'])
    if not os.path.exists(bandwidth_dir): os.makedirs(bandwidth_dir)
    accuracy_dir = bandwidth_dir + '/logepsilon{}'.format(params['logepsilon'])
    if not os.path.exists(accuracy_dir): os.makedirs(accuracy_dir)
    results_dir = bandwidth_dir + '/{}'.format(params['testfile_path'].split('/')[-1].split('.')[0])
    if not os.path.exists(results_dir): os.makedirs(results_dir)
        
    #%% Ground truth
    rng = default_rng(348591)
    verb = params['verbose']
    try:
        x = io.imread(params['testfile_path'],as_gray=True).astype(float)
    except FileNotFoundError:
        print('Provided test image did not exist under that path, aborting.')
        sys.exit()
    # handle images that are too large or colored
    if x.shape[0] > 512 or x.shape[1] > 512: x = transform.resize(x, (512,512))
    x = x-np.min(x)
    x = x/np.max(x)
    # assume quadratic images
    n = x.shape[0]
    
    tv = pot.total_variation(n, n, scale=1)
    #tv_groundtruth = tv(x)
    
    #%% Forward model & corrupted data
    blur_width = params['bandwidth']
    if params['blurtype'] == 'gaussian': 
        a,at,max_ev = blur_gauss(n,blur_width) 
    elif params['blurtype'] == 'uniform':
        a,at,max_ev = blur_unif(n,blur_width)
    else:
        print('Unknown blur type, aborting')
        sys.exit()
    
    # a,at,max_ev = lambda x : x, lambda x : x, 1
    
    noise_std = params['noise_std']
    y = a(x) + noise_std*rng.normal(size=x.shape)
    L = max_ev/noise_std**2
    
    # show ground truth and corrupted image
    my_imshow(x, 'ground truth')
    my_imshow(y, 'noisy image')
    
    #%% SAPG - compute the optimal regularization parameter
    # unscaled_posterior = pds.l2_deblur_tv(n, n, a, at, y, noise_std=noise_std, mu_tv=1)
    # # metaparameter of the posterior: L = Lipschitz constant of nabla F, necessary for stepsize
    
    # theta0 = 1
    # # empirically, for blur b=10 we need ~1000 warm up iterations with tau = 0.9/L. 
    # # for blur=5 roughly 500 warm up iterations
    # # For b=0 almost immediate warm-up since the noisy image seems to be in a region of high probability
    # s = sapg(iter_wu=25,iter_outer=60,iter_burnin=10,iter_inner=1,
    #           tau=0.9/L,delta=lambda k: 0.2/(theta0*n**2)*(k+1)**(-0.8),
    #           x0=x,theta0=theta0,theta_min=0.01,theta_max=1e2,
    #           epsilon_prox=3e-2,pd=unscaled_posterior)
    # ###################### change initialization later : iter_wu back to 500, x0 back to y
    # s.simulate()
    # mu_tv = s.mean_theta[-1]
    
    ##### -- plots to check that SAPG converged --
    # # log pi values during warm-up Markov chain
    # plt.plot(s.logpi_wu, label='log-likelihood warm-up samples')
    # plt.legend()
    # plt.show()
    
    # # thetas
    # plt.plot(s.theta,label='theta_n')
    # plt.plot(n**2/s.mean_g, label='dim/g(u_n)', color='orange')
    # plt.plot(np.arange(s.iter_burnin+1,s.iter_outer+1), s.mean_theta, label='theta_bar',color='green')
    # plt.legend()
    # plt.show()
    
    # # values g(X_n)
    # plt.plot(s.mean_g, label='g(u_n)')
    # plt.hlines(tv_groundtruth,0,len(s.mean_g)+1, label='g(u_true)')
    # plt.legend()
    # plt.show()
    
    #%% regularization parameter
    # mu_tv = s.mean_theta[-1]          # computed by SAPG
    mu_tv = 2.8                         # set by hand
    
    #%% MAP computation - L2-TV deblurring
    # deblur using PDHG in the version f(Kx) + g(x) + h(x) with smooth h
    # splitting is f(Kx) = TV(x) and h(x) = smooth L2-data term
    # this matches example 5.7 - PD-explicit in Chambolle+Pock 2016
    x0, y0 = np.zeros(x.shape), np.zeros((2,)+x.shape)
    tau, sigma = 1/(np.sqrt(8)+L), 1/np.sqrt(8)
    n_iter = 1000
    f = pot.l2_l1_norm(n, n, scale=mu_tv)
    k,kt = tv._imgrad, tv._imdiv
    g = pot.zero()
    h = pot.l2_loss_reconstruction_homoschedastic(y, noise_std**2, a, at)
    pd = pdhg(x0, y0, tau, sigma, n_iter, f, k, kt, g, h)
    
    if verb: sys.stdout.write('Compute MAP - '); sys.stdout.flush()
    u = pd.compute(verbose=True)
    if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
    
    # quicker than PDHG for the denoising-only case: use aGD on dual or aPDHG
    # u,_ = tv.inexact_prox(y, gamma=mu_tv*noise_std**2, epsilon=1e-5, max_iter=500, verbose=verb)
    
    my_imshow(u,'MAP (PDHG, mu_TV = {:.1f})'.format(mu_tv))
    print('MAP: mu_TV = {:.1f};\tPSNR: {:.2f}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((u-x)**2))))
    
    #%% sample using inexact PLA
    x0 = np.zeros_like(x)
    tau = 1/L
    epsilon = 10**params['logepsilon']
    n_samples = params['iterations']
    burnin = 1000*blur_width # rough approximation to the necessary burn-in in our setting (empirically observed)
    posterior = pds.l2_deblur_tv(n, n, a, at, y, noise_std=noise_std, mu_tv=mu_tv)
    eff = params['efficient']
    
    ipla = inexact_pla(x0, tau, epsilon, n_samples, burnin, posterior, rng=rng, efficient=eff)
    if verb: sys.stdout.write('Sample from posterior - '); sys.stdout.flush()
    ipla.simulate(verbose=verb)
    if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
    
    #%% plots
    # diagnostic plot, making sure the sampler looks plausible
    plt.plot(np.arange(1,n_samples+1), ipla.logpi_vals)
    plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [All]')
    plt.show()
    plt.plot(np.arange(burnin+1,n_samples+1), ipla.logpi_vals[burnin:])
    plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [after burn-in]')
    plt.show()
    
    my_imshow(ipla.mean, 'Sample Mean, log10(epsilon)={}'.format(params['logepsilon']))
    logstd = np.log10(ipla.std)
    my_imshow(logstd, 'Sample standard deviation (log10)', np.min(logstd), np.max(logstd))
    print('Total no. iterations to compute proximal mappings: {}'.format(ipla.num_prox_its_total))
    print('No. iterations per sampling step: {:.1f}'.format(ipla.num_prox_its_total/n_samples))
    
    #%% saving
    io.imsave(results_dir+'/ground_truth.png',x*256)
    io.imsave(results_dir+'/noisy.png',y*256)
    io.imsave(results_dir+'/rof_map.png',u*256)
    io.imsave(results_dir+'/rof_posterior_mean.png',ipla.mean*256)
    r1 = ipla.std - np.min(ipla.std)
    io.imsave(results_dir+'/rof_posterior_std.png',r1/np.max(r1)*256)
    
    
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
    print('    -e (--efficientOff): Turn off storage-efficient mode, where we dont save samples but only compute a runnning mean and standard deviation during the algorithm. This can be used if we need the samples for some other reason (diagnostics etc). Then modify the code first')
    print('    -b (--blur=): Type of blurring. 0 = No blur, denoising only; 1 = Gaussian [default]; 2 = Uniform')
    print('    -w (--width=): Bandwidth of blur, only applicable if blurtype > 0. For Gaussian, this is the std of the blur kernel, for uniform this is the size of the mask')
    print('    -s (--std=): Standard deviation of the noise added to the blurred image. The true image is always scaled to [0,1], so noise should be chosen accordingly depending on what blur type is used and how hard you want the problem to be. :)')
    print('    -l (--logepsilon=): log-10 of the accuracy parameter epsilon. The method will report the total number of iterations in the proximal computations for this epsilon = 10**logepsilon in verbose mode')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:f:eb:w:s:l:v",
                                   ["help","iterations=","testfile_path=",
                                    "efficientOff","blur=","width=","std=",
                                    "logepsilon=","verbose"])
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
        elif opt in ("-e","--efficientOff"):
            params['efficient'] = False
        elif opt in ("-b", "--blur"):
            if int(arg) == 0: params['blurtype'] = 'none'
            elif int(arg) == 1: params['blurtype'] = 'gaussian'
            elif int(arg) == 2: params['blurtype'] = 'uniform'
        elif opt in ("-w", "--width"):
            params['bandwidth'] = int(arg)
        elif opt in ("-s", "--std"):
            params['noise_std'] = float(arg)
        elif opt in ("-l", "--logepsilon"):
            params['logepsilon'] = int(arg)
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    