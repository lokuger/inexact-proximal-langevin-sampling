#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 18:32:15 2023

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import sys, getopt, os
#from time import time
from skimage import io, transform

from inexact_pla import inexact_pla
# from sapg import sapg
import potentials as pot
import distributions as pds

#%% initial parameters: test image, computation settings etc.
params = {
    'iterations': 1000,
    'testfile_path': 'test_images/wheel.png',
    'noise_std': 0.2,
    'logepsilon': -0.5,
    'efficient': True,
    'verbose': True
    }

#%% auxiliary functions
def my_imshow(im, label, vmin=0, vmax=1):
    plt.imshow(im, cmap='Greys_r',vmin=vmin,vmax=vmax)
    plt.title(label)
    plt.colorbar()
    plt.show()

#%% Main method - generate results directories
def main():
    if not os.path.exists('./results'): os.makedirs('./results')
    if not os.path.exists('./results/denoise_tv'): os.makedirs('./results/denoise_tv')
    accuracy_dir = './results/denoise_tv/logepsilon{}'.format(params['logepsilon'])
    if not os.path.exists(accuracy_dir): os.makedirs(accuracy_dir)
    sample_dir = accuracy_dir + '/{}samples'.format(params['iterations'])
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    results_dir = sample_dir + '/{}'.format(params['testfile_path'].split('/')[-1].split('.')[0])
    if not os.path.exists(results_dir): os.makedirs(results_dir)
        
    #%% Ground truth
    results_file = results_dir+'/result_images.npy'
    if not os.path.exists(results_file):    
        rng = default_rng(6346534)
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
        # tv_groundtruth = tv(x)
        
        #%% Forward model & corrupted data
        # a,at,_ = lambda x : x, lambda x : x, 1
        
        noise_std = params['noise_std']
        y = x + noise_std*rng.normal(size=x.shape)
        L = 1/noise_std**2
        
        # show ground truth and corrupted image
        my_imshow(x, 'ground truth')
        my_imshow(y, 'noisy image')
        
        #%% SAPG - compute the optimal regularization parameter
        # unscaled_posterior = pds.l2_denoise_tv(n, n, y, noise_std=noise_std, mu_tv=1)
        
        # theta0 = 10
        # # empirically, for blur b=10 we need ~1000 warm up iterations with tau = 0.9/L. 
        # # for blur=5 roughly 500 warm up iterations
        # # For b=0 almost immediate warm-up since the noisy image seems to be in a region of high probability
        # s = sapg(iter_wu=200,iter_outer=400,iter_burnin=50,iter_inner=1,
        #           tau=0.9/L,delta=lambda k: 10/(theta0*n**2)*(k+1)**(-0.8),
        #           x0=x,theta0=theta0,theta_min=0.1,theta_max=1e3,
        #           epsilon_prox=1e-2,pd=unscaled_posterior)
        # ###################### change initialization later : iter_wu back to 500, x0 back to y
        # s.simulate()
        # mu_tv = s.mean_theta[-1]
        
        # #### -- plots to check that SAPG converged --
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
        mu_tv = 3.56                        # set by hand, optimized for highest PSNR of MAP
        
        #%% MAP computation - L2-TV denoising (ROF)
        if verb: sys.stdout.write('Compute MAP - '); sys.stdout.flush()
        u,_ = tv.inexact_prox(y, gamma=mu_tv*noise_std**2, epsilon=1e-5, max_iter=500, verbose=verb)
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
        
        my_imshow(u,'MAP (dual aGD, mu_TV = {:.1f})'.format(mu_tv))
        print('MAP: mu_TV = {:.2f};\tPSNR: {:.4f}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((u-x)**2))))
        
        #%% sample using inexact PLA
        x0 = np.copy(y)
        tau = 1/L
        epsilon = 10**params['logepsilon']
        n_samples = params['iterations']
        burnin = 50 # burnin for denoising is usually short since noisy data is itself in region of high probability of the posterior
        posterior = pds.l2_denoise_tv(n, n, y, noise_std=noise_std, mu_tv=mu_tv)
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
        np.save(results_file,(x,y,u,ipla.mean,ipla.std))
        io.imsave(results_dir+'/ground_truth.png',np.clip(x*256,0,255).astype(np.uint8))
        io.imsave(results_dir+'/noisy.png',np.clip(y*256,0,255).astype(np.uint8))
        io.imsave(results_dir+'/map.png',np.clip(u*256,0,255).astype(np.uint8))
        io.imsave(results_dir+'/posterior_mean.png',np.clip(ipla.mean*256,0,255).astype(np.uint8))
        r1 = ipla.std - np.min(ipla.std)
        io.imsave(results_dir+'/posterior_std.png',np.clip(r1/np.max(r1)*256,0,255).astype(np.uint8))
    else:
        x,y,u,mn,std = np.load(results_file)
        my_imshow(x, 'ground truth')
        my_imshow(y, 'noisy image')
        my_imshow(u, 'MAP ROF')
        my_imshow(mn, 'posterior mean')
        my_imshow(np.log10(std), 'posterior std',-1.1,-0.5)
    
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
    print('    -s (--std=): Standard deviation of the noise added to the blurred image. The true image is always scaled to [0,1], so noise should be chosen accordingly depending on what blur type is used and how hard you want the problem to be. :)')
    print('    -l (--logepsilon=): log-10 of the accuracy parameter epsilon. The method will report the total number of iterations in the proximal computations for this epsilon = 10**logepsilon in verbose mode')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:f:eb:w:s:l:v",
                                   ["help","iterations=","testfile_path=",
                                    "efficientOff","std=",
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
        elif opt in ("-s", "--std"):
            params['noise_std'] = float(arg)
        elif opt in ("-l", "--logepsilon"):
            params['logepsilon'] = int(arg)
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    