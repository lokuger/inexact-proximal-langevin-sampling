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
    'iterations': 10000,
    'testfile_path': 'test-images/wheel.png',
    'noise_std': 0.2,
    'log_epsilon': -2.0,
    'step': 'large',
    'efficient': True,
    'verbose': True,
    'result_root': './results/denoise-tv',
    }

#%% auxiliary functions
#  change those
def my_imsave(im, filename, vmin=-0.02, vmax=1.02):
    im = np.clip(im,vmin,vmax)
    im = np.clip((im-vmin)/(vmax-vmin) * 256,0,255).astype('uint8')
    io.imsave(filename, im)
    
def my_imshow(im, label, vmin=-0.02, vmax=1.02, cbar=False):
    fig = plt.figure()
    plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
    q = plt.imshow(im, cmap='Greys_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    if cbar: fig.colorbar(q)
    plt.show()
    
    # draw a new figure and replot the colorbar there
    # fig,ax = plt.subplots(figsize=(2,3))
    # plt.colorbar(q,ax=ax)
    # ax.remove()
    # plt.savefig(cbarfile,bbox_inches='tight')

#%% Main method - generate results directories
def main():
    result_root = params['result_root']
    
    test_image_name = params['testfile_path'].split('/')[-1].split('.')[0]
    accuracy = 'log-epsilon{}'.format(params['log_epsilon'])
    file_specifier = '{}_{}_{}-samples'.format(test_image_name,accuracy,params['iterations'])
    results_file = result_root+'/'+file_specifier+'.npy'
    mmse_file = result_root+'/mmse_'+file_specifier+'.png'  
    mmse_detail_file = result_root+'/mmse_detail_'+file_specifier+'.png'  
    logstd_file = result_root+'/logstd_'+file_specifier+'.png'  
    logstd_detail_file = result_root+'/logstd_detail_'+file_specifier+'.png'  
    
    #%% Ground truth
    if not os.path.exists(results_file):    
        rng = default_rng(6346534)
        verb = params['verbose']
        try:
            x = io.imread(params['testfile_path'],as_gray=True).astype(float)
        except FileNotFoundError:
            print('Provided test image did not exist under that path, aborting.')
            sys.exit()
        # handle images that are too large or colored
        Nmax = 512
        if x.shape[0] > Nmax or x.shape[1] > Nmax: x = transform.resize(x, (Nmax,Nmax))
        x = x-np.min(x)
        x = x/np.max(x)
        n = x.shape[0] # assume quadratic image, otherwise change some implementation details
        
        tv = pot.total_variation(n, n, scale=1)
        # tv_groundtruth = tv(x)
        
        #%% Forward model & corrupted data
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
        mu_tv = 4.3                        # set by hand, optimized for highest PSNR of MAP
            
        #%% MAP computation - L2-TV denoising (ROF)
        if verb: sys.stdout.write('Compute MAP - '); sys.stdout.flush()
        u,its_map,_ = tv.inexact_prox(y, gamma=mu_tv*noise_std**2, epsilon=1e-8, max_iter=100, verbose=verb)
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
        
        my_imshow(u,'MAP (dual aGD, mu_TV = {:.1f})'.format(mu_tv))
        my_imshow(u[314:378,444:508],'MAP details')
        print('MAP: mu_TV = {:.2f};\tPSNR: {:.4f}, #steps: {}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((u-x)**2)),its_map))
        
        #%% sample using inexact PLA
        x0 = np.copy(y)
        if params['step'] == 'large':
            tau = 1/L
        elif params['step'] == 'small':
            tau = 0.5/L
        C = tau*mu_tv*tv(u)
        epsilon = C*(10**params['log_epsilon'])
        burnin = 50 # burnin for denoising is usually short
        n_samples = params['iterations']+burnin
        posterior = pds.l2_denoise_tv(n, n, y, noise_std=noise_std, mu_tv=mu_tv)
        eff = params['efficient']
        
        ipla = inexact_pla(x0, n_samples, burnin, posterior, step_size=tau, rng=rng, epsilon_prox=epsilon, efficient=eff)
        if verb: sys.stdout.write('Sample from posterior - '); sys.stdout.flush()
        ipla.simulate(verbose=verb)
        
        #%% plots
        # diagnostic plot, making sure the sampler looks plausible
        # plt.figure()
        # plt.plot(np.arange(1,n_samples+1), ipla.logpi_vals)
        # plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [All]')
        # plt.show()
        # plt.figure()
        # plt.plot(np.arange(burnin+1,n_samples+1), ipla.logpi_vals[burnin:])
        # plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [after burn-in]')
        # plt.show()
        
        my_imshow(ipla.mean, 'Sample Mean, log10(epsilon)={}'.format(params['log_epsilon']))
        my_imshow(ipla.mean[314:378,444:508],'sample mean details')
        logstd = np.log10(ipla.std)
        my_imshow(logstd, 'Sample standard deviation (log10)', np.min(logstd), np.max(logstd))
        my_imshow(logstd[314:378,444:508],'sample std details', np.min(logstd), np.max(logstd))
        print('Total no. iterations to compute proximal mappings: {}'.format(ipla.num_prox_its_total))
        print('No. iterations per sampling step: {:.1f}'.format(ipla.num_prox_its_total/n_samples))
        
        #%% saving
        np.save(results_file,(x,y,u,ipla.mean,ipla.std))
    else:
        x,y,u,mn,std = np.load(results_file)
        logstd = np.log10(std)
        
        # my_imshow(x, 'ground truth')
        # my_imshow(y, 'noisy')
        # my_imshow(u, 'map')
        my_imshow(mn, 'mean')
        my_imshow(logstd, 'logstd',-1.15,-0.58)
        print('MMSE estimate PSNR: {:.4f}'.format(10*np.log10(np.max(x)**2/np.mean((mn-x)**2))))
        
        my_imsave(x,result_root+'/ground_truth.png')
        my_imsave(y,result_root+'/noisy.png')
        my_imsave(u,result_root+'/map.png')
        my_imsave(mn, mmse_file)
        my_imsave(logstd, logstd_file,-1.15,-0.58)
        # image details for paper close-up
        my_imsave(x[314:378,444:508], result_root+'/ground_truth_detail.png')
        my_imsave(y[314:378,444:508], result_root+'/noisy_detail.png')
        my_imsave(u[314:378,444:508], result_root+'/map_detail.png')
        my_imsave(mn[314:378,444:508], mmse_detail_file)
        my_imsave(logstd[314:378,444:508], logstd_detail_file, -1.15,-0.58)
        
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
    print('    -l (--log_epsilon=): log-10 of the accuracy parameter epsilon. The method will report the total number of iterations in the proximal computations for this epsilon = 10**log_epsilon in verbose mode')
    print('    -s (--step=): \'large\' for 1/L or \'small\' for 0.5/L')
    print('    -d (--result_dir=): root directory for results. Default: ./results/deblur-wavelets')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:f:el:s:d:v",
                                   ["help","iterations=","testfile_path=","efficient_off","log_epsilon=","step=","result_dir=","verbose"])
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
        elif opt in ("-e","--efficient_off"):
            params['efficient'] = False
        elif opt in ("-l", "--log_epsilon"):
            params['log_epsilon'] = float(arg)
        elif opt in ["-s", "--step="]:
            params['step'] = arg
        elif opt in ("-d","--result_dir"):
            params['result_root'] = arg
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    