#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 08:51:53 2023

@author: lorenzkuger
"""

from numpy.random import default_rng
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import multiprocessing as mp
from contextlib import closing
import sys,getopt

# own imports
import potentials as pot
import distributions as pds
from sapg import sapg
from inexact_pla import inexact_pla

#%% parameters
params = {
    'iterations': 1000,
    'num_chains': 1,
    'testfile_path' : 'test_images/cameraman.tif',     # relative path to test image
    'num_cores' : 1,
    'parallel': False,
    'verbose': False
}

#%% auxiliary functions: blurring operator, power method for largest eigenvalue estimation etc
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

#%% main method instantiating the samplers and calling the simulation
def main():
    """ generate data - image denoising """
    rng = default_rng(38456809)
    x = iio.imread(params['testfile_path']).astype(float)
    # careful, we assume a quadratic n x n images for simplicity. 
    # Might have to rewrite parts of the files to handle non-quadratic ones
    n = x.shape[0]
    
    tv = pot.total_variation(n, n, scale=1)
    tv_groundtruth = tv(x)
    
    """ Generate noisy observation: normally distributed/Gaussian noise model"""
    blur_width = 10
    a,at,max_ev = blur(n,blur_width)                        # blur operator
    # a,at,max_ev = lambda x : x, lambda x : x, 1    # only denoising
    
    y = a(x)
    noise_snr = 40 # [dB] of blurred snr
    noise_std = np.std(y)*10**(-noise_snr/20)
    y = y + noise_std*rng.normal(size=(n,n))
    
    """ define posterior. For SAPG, use reg parameter theta = 1 since it 
    handles the changing parameter inside the algorithm """
    unscaled_posterior = pds.l2_deblur_tv(n, n, a, at, y, noise_std=noise_std, mu_tv=1)
    # metaparameter of the posterior: L = Lipschitz constant of nabla F, necessary for stepsize
    L = max_ev/noise_std**2
   
    """ determine optimal theta using SAPG """
    iter_outer = 100
    iter_burnin = 50
    theta0 = 0.01
    # empirically, for blur b=10 we need ~800 warm up iterations with tau = 0.9/L. 
    # for blur=5 roughly 400 warm up iterations
    # For b=0 almost immediate warm-up since the noisy image seems to be in a region of high probability
    s = sapg(iter_wu=750,
             iter_outer=iter_outer,
             iter_burnin=iter_burnin,
             iter_inner=1,
             tau=0.9/L,
             delta=lambda k: 0.1/(theta0*n**2) * (k+1)**(-0.8),
             x0=y,
             theta0=theta0,
             theta_min=0.001,
             theta_max=1,
             epsilon_prox=1e-1,
             pd=unscaled_posterior)
    s.simulate()
    # the computed optimal regularization parameter
    mu_tv = s.mean_theta[-1]
    
    # plt.imshow(x,cmap='Greys_r')
    # plt.title('True image')
    # plt.colorbar()
    # plt.show()
    
    # plt.imshow(y,cmap='Greys_r')
    # plt.title('Blurred & noisy image')
    # plt.colorbar()
    # plt.show()
    
    # """ -- plots to check that SAPG converged -- """
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
    # plt.hlines(tv_groundtruth,0,iter_outer+1, label='g(u_true)')
    # plt.legend()
    # plt.show()
    
    posterior = pds.l2_deblur_tv(n, n, a, at, y, noise_std=noise_std, mu_tv=mu_tv)
    x0 = s.x # use last SAPG iterate as initializer, alternatively run separate warm-up
    n_iter = params['iterations']
    tau = 0.9/L
    epsilon_prox = 1e-1
    ipla = inexact_pla(n_iter, tau=0.9/L, x0=x0, epsilon=epsilon_prox, pd=posterior)
    ipla.simulate()
    
    # compute and save mmse and std images
    mmse_samples = np.mean(ipla.x,axis=2)
    std_samples = np.std(ipla.x,axis=2)
    
    plt.imshow(mmse_samples, cmap='Greys_r')
    plt.colorbar()
    plt.title('Sample mean')
    plt.show()
    
    plt.imshow(std_samples, cmap='Greys_r')
    plt.colorbar()
    plt.title('Sample standard deviation')
    plt.show()
    
    result_path = 'results/blur{}/snr{}'.format(blur_width,noise_snr)
    Path(result_path).mkdir(exist_ok=True,parents=True)
    iio.imwrite(result_path+'/mmse.png',mmse_samples.astype('uint8'))
    iio.imwrite(result_path+'/std.png',std_samples.astype('uint8'))
    
    
def print_help():
    print('<>'*30)
    print(' Run inexact PLA to generate samples for TV deblurring posterior ')
    print('<>'*30)
    print(' ')
    print('Options:')
    print('    -h (--help): Print help.')
    print('    -i (--iterations): Number of iterations of each Markov chain')
    print('    -z (--num_chains=): Number of Markov chains to run')
    print('    -f (--testfile_path=): Path to test image file')
    print('    -p (--parallel): Use parallel processing for several chains.')
    print('    -c (--num_cores=): Number of cores to use in parallel processing (default=1).')
    print('    -v (--verbose): Verbose mode.')

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:z:f:pc:v",
                                   ["help","iterations=","num_chains=",
                                    "testfile_path=","parallel","num_cores=",
                                    "verbose"])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-i", "--iterations"):
            params['iterations'] = int(arg)
        elif opt in ("-z", "--num_chains"):
            params['num_chains'] = float(arg)
        elif opt in ("-f", "--testfile_path"):
            params['testfile_path'] = arg
        elif opt in ("-p", "--parallel"):
            parallel = True
        elif opt in ("-c", "--num_cores"):
            params['num_cores'] = int(arg)
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()