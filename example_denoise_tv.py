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

from inexact_pgla import inexact_pgla
# from sapg import sapg
import potentials as pot
import distributions as pds

#%% initial parameters: test image, computation settings etc.
params = {
    'iterations': 100,
    'testfile_path': 'test-images/wheel.png',
    'noise_std': 0.2,
    'log_eta': -2.0,
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
    result_root = params['result_root']
    
    test_image_name = params['testfile_path'].split('/')[-1].split('.')[0]
    accuracy = 'log-eta{}'.format(params['log_eta'])
    file_specifier = '{}_{}_{}-samples'.format(test_image_name,accuracy,params['iterations'])
    results_file = result_root+'/'+file_specifier+'.npy'
    # results_file= result_root+'/wheel_mmse_reference.npy'
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
        
        mu_tv = 8
        tv = pot.total_variation(n, n, scale=mu_tv)
        # tv_groundtruth = tv(x)
        
        #%% Forward model & corrupted data
        noise_std = params['noise_std']
        y = x + noise_std*rng.normal(size=x.shape)
        L = 1/noise_std**2
        
        posterior = pds.l2_denoise_tv(n, n, y, noise_std=noise_std, mu_tv=mu_tv)
        
        # show ground truth and corrupted image
        # my_imshow(x, 'ground truth')
        # my_imshow(y, 'noisy image')
            
        #%% MAP computation - L2-TV denoising (ROF)
        C = mu_tv*tv(y)
        if verb: sys.stdout.write('Compute MAP - '); sys.stdout.flush()
        u,its = tv.inexact_prox(y, gamma=noise_std**2, epsilon=C*1e-4, max_iter=500, verbose=verb) # epsilon=1e2 corresponds to approx. 200 FISTA iterations
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
        
        # my_imshow(u,'MAP (FISTA on dual, mu_TV = {:.1f})'.format(mu_tv))
        # my_imshow(u[314:378,444:508],'FISTA MAP details')
        print('MAP: mu_TV = {:.2f};\tPSNR: {:.4f}, #steps: {}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((u-x)**2)),its))
        
        #%% sample using inexact PLA
        x0 = np.copy(y)
        tau = 1/L
        epsilon = C*(10**params['log_eta'])
        burnin = np.intc(1/(tau*L)) # burnin for denoising is usually short, one gradient step with step size 1/L lands at y
        n_samples = params['iterations']+burnin
        eff = params['efficient']
        
        output_means = np.reshape(np.reshape(np.array([1,2,5]),(1,-1))*np.reshape(10**np.arange(6),(-1,1)),(-1,))     # 1,2,5,10,20,50,... until max number of samples is reached
        output_means = output_means[output_means<=n_samples]
        
        ipgla = inexact_pgla(x0, n_samples, burnin, posterior, step_size=tau, rng=rng, epsilon_prox=epsilon, efficient=eff, output_means=output_means)
        if verb: sys.stdout.write('Sample from posterior - '); sys.stdout.flush()
        ipgla.simulate(verbose=verb)
        
        running_means = ipgla.output_means
        n_means,I_running_means = running_means.shape[-1],ipgla.I_output_means
        
        #%% plots
        # diagnostic plots to make sure the sampler looks plausible
        # plt.figure()
        # plt.plot(np.arange(1,n_samples+1), ipla.logpi_vals)
        # plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [All]')
        # plt.show()
        # plt.figure()
        # plt.plot(np.arange(burnin+1,n_samples+1), ipla.logpi_vals[burnin:])
        # plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [after burn-in]')
        # plt.show()
        
        my_imshow(ipgla.mean, 'Sample Mean, epsilon=C*eta with log10(eta)={}'.format(params['log_eta']))
        my_imshow(ipgla.mean[314:378,444:508],'sample mean details')
        logstd = np.log10(ipgla.std)
        my_imshow(logstd, 'Sample standard deviation (log10)', np.min(logstd), np.max(logstd))
        my_imshow(logstd[314:378,444:508],'sample std details', np.min(logstd), np.max(logstd))
        print('Total no. iterations to compute proximal mappings: {}'.format(ipgla.num_prox_its_total))
        print('No. iterations per sampling step: {:.1f}'.format(ipgla.num_prox_its_total/(ipgla.n_iter-ipgla.burnin)))
        
        #%% saving
        with open(results_file,'wb') as f:
            np.save(f,(x,y,u,ipgla.mean,ipgla.std))   # ground truth, noisy, map, sample mean and sample std
            np.save(f,running_means)              # running means
            np.save(f,I_running_means)            # indices to which these means belong
    else:
        with open(results_file,'rb') as f:
            x,y,u,mn,std = np.load(f)               # ground truth, noisy, map, sample mean and sample std
            # running_means = np.load(f)              # running means
            # I_running_means = np.load(f)            # indices to which these means belong
        logstd = np.log10(std)
        
        # with open(result_root+'/'+'{}_log-epsilon-2.0_10000-samples.npy'.format(test_image_name),'rb') as f:
        #     _,_,_,mmse,_ = np.load(f)               # ground truth, noisy, map, sample mean and sample std
        
        # n_means = running_means.shape[-1]
        # mmse_err = np.zeros((n_means,))
        # for i in [6]:# np.arange(n_means):
            # my_imshow(running_means[...,i], 'Running mean at iteration {}'.format(I_running_means[i])) 
            # my_imshow(running_means[314:378,444:508,i], 'Running mean at iteration {}'.format(I_running_means[i])) 
        #     mmse_err[i] = np.sqrt(np.sum((mmse-running_means[...,i])**2)/np.sum((mmse)**2))
        # ax = plt.axes()
        # ax.set_title('MMSE Error')
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        # ax.plot(I_running_means-50,mmse_err)
        
        # logstd_min = np.min(logstd)
        # logstd_max = np.max(logstd)
        
        my_imshow(x, 'ground truth')
        my_imshow(y, 'noisy')
        my_imshow(u, 'map')
        my_imshow(mn, 'mean')
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
    print('    -l (--log_eta=): log-10 of the accuracy parameter eta. The method will report the total number of iterations in the proximal computations for this epsilon = C*10**log_eta in verbose mode')
    print('    -d (--result_dir=): root directory for results. Default: ./results/deblur-wavelets')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:f:el:d:v",
                                   ["help","iterations=","testfile_path=","efficient_off","log_epsilon=","result_dir=","verbose"])
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
        elif opt in ("-l", "--log_eta"):
            params['log_eta'] = float(arg)
        elif opt in ("-d","--result_dir"):
            params['result_root'] = arg
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    