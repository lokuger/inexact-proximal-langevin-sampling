#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:26:10 2023

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import sys, getopt, os
#from time import time
from skimage import io, transform
import pywt

from inexact_pla import inexact_pla
import potentials as pot
import distributions as pds

#%% initial parameters: test image, computation settings etc.
params = {
    'iterations': 500,
    'testfile_path': 'test-images/fibo2.jpeg',
    'blur_width': 15,
    'noise_std': 0.05,
    'log_epsilon': -np.Inf, # -0.1, -0.5, -2.0, -np.Inf
    'step': 'large',
    'verbose': True,
    'result_root': './results/deblur-wavelets',
    }

step_factors = {
    'large': 1,
    'small': 0.5,
    'tiny': 0.1
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

def my_imsave(im, filename, vmin=-0.02, vmax=1.02):
    im = np.clip(im,vmin,vmax)
    im = np.clip((im-vmin)/(vmax-vmin) * 256,0,255).astype('uint8')
    io.imsave(filename, im)
    
def my_imshow(im, label, cbarfile=None, vmin=-0.02, vmax=1.02, cbar=False):
    fig = plt.figure()
    plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
    q = plt.imshow(im, cmap='Greys_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(label)
    if cbar: fig.colorbar(q)
    plt.show()
    
    # draw a new figure and replot the colorbar there
    # fig,ax = plt.subplots(figsize=(2,3))
    # plt.colorbar(q,ax=ax)
    # ax.remove()
    # plt.savefig(cbarfile,bbox_inches='tight')
    
def wavelet_operators(n, slices, wav_type, wav_level):
    a = lambda x : pywt.coeffs_to_array(pywt.wavedec2(x, wav_type, level=wav_level))[0]
    at = lambda c : pywt.waverec2(pywt.array_to_coeffs(c, slices, output_format='wavedec2'), wav_type)
    return a, at #dwt, idwt

#%% main
def main():
    #%% generate results directories
    result_root = params['result_root']
    
    test_image_name = params['testfile_path'].split('/')[-1].split('.')[0]
    accuracy = 'exact-prox' if params['log_epsilon'] == -np.Inf else 'log-epsilon{}'.format(params['log_epsilon'])
    file_specifier = '{}_{}_{}-samples'.format(test_image_name,accuracy,params['iterations'])
    results_file = result_root+'/'+file_specifier+'.npy'
    mmse_file = result_root+'/mmse_'+file_specifier+'.png'
    
    if not os.path.exists(results_file):    
        #%% generate ground truth and noisy image
        rng = default_rng(952765)
        verb = params['verbose']
        try:
            x = io.imread(params['testfile_path'],as_gray=True).astype(float)
        except FileNotFoundError:
            print('Provided test image did not exist under that path, aborting.')
            sys.exit()
        # handle images that are too large
        Nmax = 512
        if x.shape[0] > Nmax or x.shape[1] > Nmax: x = transform.resize(x, (Nmax,Nmax))
        x = x-np.min(x)
        x = x/np.max(x)
        n = x.shape[0] # assume quadratic images for now, not much changes otherwise
        
        # %% Forward model
        # blur operator
        a, at, max_ev_ata = blur_unif(n, params['blur_width'])
        
        # noisy data
        noise_std = params['noise_std']
        y = a(x) + noise_std*rng.normal(size=x.shape)
        
        # wavelet operator
        wav_type = 'haar'
        wav_level = 8
        c_x, slices = pywt.coeffs_to_array(pywt.wavedec2(x, wav_type, level=wav_level))
        dwt, idwt = wavelet_operators(n, slices, wav_type, wav_level)
        c_y = dwt(y)
        b, bt = lambda x : a(idwt(x)), lambda x : dwt(at(x))
        
        # show ground truth and corrupted image
        my_imshow(x, 'ground truth')
        my_imshow(y, 'noisy image')
        
        #%% MAP computation - using ISTA
        mu_l1 = 5.0
        posterior = pds.l2_deblur_l1prior(b, bt, max_ev_ata, y, noise_std=noise_std, mu_l1=mu_l1)
        
        # compute map using ista (pdhg also possible and has its own class)
        k = 0
        tau_ista = 1/posterior.f.L
        c_u = c_y
        if verb: sys.stdout.write('Run ISTA: {:3d}% '.format(0)); sys.stdout.flush()
        iter_ista = 50
        while k < iter_ista:
            if verb: sys.stdout.write('\b'*5 + '{:3d}% '.format(int(k/iter_ista*100))); sys.stdout.flush()
            k += 1
            c_u = posterior.g.prox(c_u - tau_ista * posterior.f.grad(c_u), tau_ista)
        u = idwt(c_u)
        if verb: sys.stdout.write('\b'*5 + '100%\n'); sys.stdout.flush()
        
        my_imshow(u,'MAP (ISTA, mu_l1 = {:.2f})'.format(mu_l1))
        if verb: print('MAP estimate PSNR: {:.4f}'.format(10*np.log10(np.max(x)**2/np.mean((u-x)**2))))
        
        #%% sample Wavelet coefficients using inexact PLA
        c0 = np.copy(c_y)
        n_samples = params['iterations']
        burnin = 500
        tau_max = 1/posterior.f.L
        tau = step_factors[params['step']] * tau_max
        
        if params['log_epsilon'] == -np.Inf:
            ipla = inexact_pla(c0, n_samples, burnin, posterior, step_size=tau, rng=rng, efficient=True, exact=True)
        else:
            epsilon = 10**params['log_epsilon']
            ipla = inexact_pla(c0, n_samples, burnin, posterior, step_size=tau, epsilon_prox=epsilon, rng=rng, efficient=True)
        
        if verb: sys.stdout.write('Sample from posterior - '); sys.stdout.flush()
        ipla.simulate(verbose=verb)
        
        #%% plots
        # diagnostic plot, making sure the sampler looks plausible
        plt.figure()
        plt.plot(np.arange(1,n_samples+1), ipla.logpi_vals)
        plt.title('- log(pi(X_n)) = F(X_n) + G(X_n) [All]')
        plt.show()
        plt.figure()
        plt.plot(np.arange(burnin+1,n_samples+1), ipla.logpi_vals[burnin:])
        plt.title('- log(pi(X_n)) = F(X_n) + G(X_n) [after burn-in]')
        plt.show()
        
        mmse_coeffs = ipla.mean
        im_mmse = idwt(mmse_coeffs)
        my_imshow(im_mmse, 'MMSE coeffs, log10(epsilon)={}'.format(params['log_epsilon']))
        print('MMSE estimate PSNR: {:.4f}'.format(10*np.log10(np.max(x)**2/np.mean((im_mmse-x)**2))))
        print('Total no. iterations to compute proximal mappings: {}'.format(ipla.num_prox_its_total))
        print('No. iterations per sampling step: {:.1f}'.format(ipla.num_prox_its_total/n_samples))
        
        #%% saving
        np.save(results_file,(x,y,u,im_mmse))
    else:
        x,y,u,mn = np.load(results_file)
        my_imshow(x, 'ground truth')
        my_imshow(y, 'noisy')
        my_imshow(u, 'map')
        my_imshow(mn, 'mmse estimate, logeps={:.1f}'.format(params['log_epsilon']))
        print('MMSE estimate PSNR: {:.4f}'.format(10*np.log10(np.max(x)**2/np.mean((mn-x)**2))))
        
        # image details for paper close-up
        my_imsave(x,result_root+'/ground_truth.png')
        my_imsave(y,result_root+'/noisy.png')
        my_imsave(u,result_root+'/map.png')
        my_imsave(mn,mmse_file)
        
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
    print('    -e (--efficient_off): Turn off storage-efficient mode, where we dont save samples but only compute a runnning mean and standard deviation during the algorithm. This can be used if we need the samples for some other reason (diagnostics etc). Then modify the code first')
    print('    -n (--neg_log_epsilon=): negative log-10 of the accuracy parameter epsilon. The method will report the total number of iterations in the proximal computations for this epsilon = 10**-neg_log_epsilon in verbose mode')
    print('    -s (--step=): \'large\' for 1/L or \'small\' for 0.5/L')
    print('    -d (--result_dir=): root directory for results. Default: ./results/deblur-wavelets')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:f:n:s:d:v",
                                   ["help","iterations=","testfile_path=","neg_log_epsilon=","step=","result_dir=","verbose"])
    except getopt.GetoptError as e:
        print(e.msg)
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
        elif opt in ("-n", "--neg_log_epsilon"):
            params['log_epsilon'] = -float(arg)
        elif opt in ["-s", "--step="]:
            params['step'] = arg
        elif opt in ("-d","--result_dir"):
            params['result_root'] = arg
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    