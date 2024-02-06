#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:45:27 2023

@author: kugerlor
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
    'iterations': 500,
    'testfile_path': 'test-images/wheel.png',
    'noise_std': 0.2,
    'log_epsilon': 1.0,
    'step': 'large',
    'efficient': True,
    'verbose': True,
    'result_root': './results/denoise-tv',
    }

#%% auxiliary functions
def read_image(Nmax):
    try:
        x = io.imread(params['testfile_path'],as_gray=True).astype(float)
    except FileNotFoundError:
        print('Provided test image did not exist under that path, aborting.')
        sys.exit()
    # handle images that are too large or colored
    if x.shape[0] > Nmax or x.shape[1] > Nmax: x = transform.resize(x, (Nmax,Nmax))
    x = x-np.min(x)
    x = x/np.max(x)
    n = x.shape[0] # assume quadratic image, otherwise change some implementation details
    return x,n

def my_imsave(im, filename, vmin=-0.02, vmax=1.02):
    im = np.clip(im,vmin,vmax)
    im = np.clip((im-vmin)/(vmax-vmin) * 256,0,255).astype('uint8')
    io.imsave(filename, im)
    
def my_imshow(im, label, vmin=-0.02, vmax=1.02, cbar=False):
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

#%% Main method - generate results directories
def main():
    result_root = params['result_root'] + '/decay-steps'
    
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
        x, n = read_image(Nmax=512)
        
        tv = pot.total_variation(n, n, scale=1)
        
        #%% Forward model & corrupted data
        noise_std = params['noise_std']
        y = x + noise_std*rng.normal(size=x.shape)  # noisy observation
        L = 1/noise_std**2                          # log-likelihood gradient Lipschitz const
        mu_tv = 8                                   # regularization parameter
        
        #%% compute MAP estimate - L2-TV denoising (= ROF model)
        if verb: sys.stdout.write('Compute MAP - '); sys.stdout.flush()
        u,its_map,_ = tv.inexact_prox(y, gamma=mu_tv*noise_std**2, epsilon=1e-8, max_iter=100, verbose=verb)
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
        
        # show ground truth, noisy observation and MAP estimate
        my_imshow(x, 'ground truth')
        my_imshow(y, 'noisy image')
        my_imshow(u,'MAP (dual aGD, mu_TV = {:.1f})'.format(mu_tv))
        my_imshow(u[314:378,444:508],'MAP details')
        print('MAP: mu_TV = {:.2f};\tPSNR: {:.4f}, #steps: {}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((u-x)**2)),its_map))
        
        #%% sample using inexact PLA
        x0 = np.copy(y)
        burnin = 1                 # burnin iterations, for denoising usually short (one gradient step with step size 1/L lands at y)
        n_samples = params['iterations']+burnin
        steps = 1/L*np.ones((n_samples,))
        for i in np.arange(1,n_samples):
            steps[i] = np.minimum(steps[i-1],np.maximum(100/(i*L),steps[i-1]/(1+L)))
        tau = lambda n : steps[n]
        epsilon = lambda n : 10/(n+1)
        posterior = pds.l2_denoise_tv(n, n, y, noise_std=noise_std, mu_tv=mu_tv)
        eff = params['efficient']
        
        # output_means = np.reshape(np.reshape(np.array([1,2,5]),(1,-1))*np.reshape(10**np.arange(6),(-1,1)),(-1,))     # 1,2,5,10,20,50,... until max number of samples is reached
        # output_means = output_means[output_means<=n_samples]
        ######################## TODO see comment next line
        ipla = inexact_pla(x0, n_samples, burnin, posterior, step_size=tau, rng=rng, epsilon_prox=epsilon, efficient=eff, output_means=None) ###################### # implement a custom evaluatable stopping criterion
        if verb: sys.stdout.write('Sample from posterior - '); sys.stdout.flush()
        ipla.simulate(verbose=verb)
        
        running_means = ipla.output_means
        n_means,I_running_means = running_means.shape[-1],ipla.I_output_means
        
        #%% plots
        # diagnostic plot, making sure the sampler looks plausible
        plt.figure()
        plt.plot(np.arange(1,n_samples+1), ipla.logpi_vals)
        plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [All]')
        plt.show()
        plt.figure()
        plt.plot(np.arange(burnin+1,n_samples+1), ipla.logpi_vals[burnin:])
        plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [after burn-in]')
        plt.show()
        
        my_imshow(ipla.mean, 'Sample Mean, log10(epsilon)={}'.format(params['log_epsilon']))
        my_imshow(ipla.mean[314:378,444:508],'sample mean details')
        logstd = np.log10(ipla.std)
        my_imshow(logstd, 'Sample standard deviation (log10)', np.min(logstd), np.max(logstd))
        my_imshow(logstd[314:378,444:508],'sample std details', np.min(logstd), np.max(logstd))
        print('Total no. iterations to compute proximal mappings: {}'.format(ipla.num_prox_its_total))
        print('No. iterations per sampling step: {:.1f}'.format(ipla.num_prox_its_total/(n_samples-burnin)))
        
        #%% saving
        with open(results_file,'wb') as f:
            np.save(f,(x,y,u,ipla.mean,ipla.std))   # ground truth, noisy, map, sample mean and sample std
            np.save(f,running_means)            # running means
            np.save(f,I_running_means)          # indices to which these means belong
    else:
        with open(results_file,'rb') as f:
            x,y,u,mn,std = np.load(f)               # ground truth, noisy, map, sample mean and sample std
            running_means = np.load(f)              # running means
            I_running_means = np.load(f)            # indices to which these means belong
        logstd = np.log10(std)
        
        with open(result_root+'/'+'{}_log-epsilon-2.0_10000-samples.npy'.format(test_image_name),'rb') as f:
            _,_,_,mmse,_ = np.load(f)               # ground truth, noisy, map, sample mean and sample std
        
        n_means = running_means.shape[-1]
        mmse_err = np.zeros((n_means,))
        # for i in [6]:# np.arange(n_means):
            # my_imshow(running_means[...,i], 'Running mean at iteration {}'.format(I_running_means[i])) 
            # my_imshow(running_means[314:378,444:508,i], 'Running mean at iteration {}'.format(I_running_means[i])) 
        #     mmse_err[i] = np.sqrt(np.sum((mmse-running_means[...,i])**2)/np.sum((mmse)**2))
        # ax = plt.axes()
        # ax.set_title('MMSE Error')
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        # ax.plot(I_running_means-50,mmse_err)
        
        logstd_min = np.min(logstd)
        logstd_max = np.max(logstd)
        
        # my_imshow(x, 'ground truth')
        # my_imshow(y, 'noisy')
        # my_imshow(u, 'map')
        # my_imshow(mn, 'mean')
        my_imshow(logstd, 'logstd', -1.8, -0.6)
        print('MMSE estimate PSNR: {:.4f}'.format(10*np.log10(np.max(x)**2/np.mean((mn-x)**2))))
        
        my_imsave(x,result_root+'/ground_truth.png')
        my_imsave(y,result_root+'/noisy.png')
        my_imsave(u,result_root+'/map.png')
        my_imsave(mn, mmse_file)
        my_imsave(logstd, logstd_file, -1.8, -0.6)
        # image details for paper close-up
        my_imsave(x[314:378,444:508], result_root+'/ground_truth_detail.png')
        my_imsave(y[314:378,444:508], result_root+'/noisy_detail.png')
        my_imsave(u[314:378,444:508], result_root+'/map_detail.png')
        my_imsave(mn[314:378,444:508], mmse_detail_file)
        my_imsave(logstd[314:378,444:508], logstd_detail_file, -1.8, -0.6)
        
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
    