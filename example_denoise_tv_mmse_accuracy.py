#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:40:45 2023

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
    'iterations_max': 1000,
    'testfile_path': 'test-images/wheel.png',
    'noise_std': 0.2,
    'log_epsilon': np.arange(-2,-2.9,-0.2), # originally had np.arange(-2,-4.9,-0.2)
    'mmse_accuracy': 0.05,
    'step': 'large',
    'efficient': True,
    'verbose': True,
    'result_root': './results/denoise-tv',
    }

#%% auxiliary functions
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
    plt.show(block=False)
    
    # draw a new figure and replot the colorbar there
    # fig,ax = plt.subplots(figsize=(2,3))
    # plt.colorbar(q,ax=ax)
    # ax.remove()
    # plt.savefig(cbarfile,bbox_inches='tight')

#%% Main method - generate results directories
def main():
    result_root = params['result_root']
    
    test_image_name = params['testfile_path'].split('/')[-1].split('.')[0]
    accuracy = 'mmse-accuracy{}'.format(params['mmse_accuracy'])
    file_specifier = '{}_{}'.format(test_image_name,accuracy)
    results_file = result_root+'/'+file_specifier+'.npy'
    
    if True:#not os.path.exists(results_file):
        ############# read in ground truth and generate observation #############
        rng = default_rng(6346534)
        verb = params['verbose']
        try:
            x = io.imread(params['testfile_path'],as_gray=True).astype(float)
        except FileNotFoundError:
            print('Provided test image did not exist under that path, aborting.')
            sys.exit()
        Nmax = 512
        if x.shape[0] > Nmax or x.shape[1] > Nmax: x = transform.resize(x, (Nmax,Nmax))
        x = x-np.min(x)
        x = x/np.max(x)
        n = x.shape[0] # assume quadratic image
        
        mu_tv = 8
        tv = pot.total_variation(n, n, scale=mu_tv)
        
        noise_std = params['noise_std']
        y = x + noise_std*rng.normal(size=x.shape)
        L = 1/noise_std**2
        
        posterior = pds.l2_denoise_tv(n, n, y, noise_std=noise_std, mu_tv=mu_tv)
        
        # show ground truth and corrupted image
        my_imshow(x, 'ground truth')
        my_imshow(y, 'noisy image')
            
        ############# MAP computation - L2-TV denoising (ROF) #############
        if verb: sys.stdout.write('Compute MAP - '); sys.stdout.flush()
        C = mu_tv*tv(y)
        u,its = tv.inexact_prox(y, gamma=noise_std**2, epsilon=C*1e-4, max_iter=500, verbose=verb) # epsilon=1e2 corresponds to approx. 200 FISTA iterations
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
        
        my_imshow(u,'MAP (FISTA on dual, mu_TV = {:.1f})'.format(mu_tv))
        my_imshow(u[314:378,444:508],'FISTA MAP details')
        print('MAP: mu_TV = {:.2f};\tPSNR: {:.4f}, #steps: {}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((u-x)**2)),its))
        
        ############# sample using inexact PLA #############
        with open(result_root+'/'+'{}_mmse_reference.npy'.format(test_image_name),'rb') as f:
            _,_,_,mmse_ref,_ = np.load(f)               # ground truth, noisy, map, sample mean and sample std
        my_imshow(mmse_ref,'MMSE estimate (One long IPGLA run w/ small epsilon)')
        my_imshow(mmse_ref[314:378,444:508],'MMSE details')
        print('MMSE: mu_TV = {:.2f};\tPSNR: {:.4f}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((mmse_ref-x)**2))))
        stopcrit = lambda sampler : (sampler.iter>sampler.burnin and np.sqrt(np.sum((mmse_ref - sampler.sum/(sampler.iter-sampler.burnin))**2)/np.sum((mmse_ref)**2)) < params['mmse_accuracy'])
        
        x0 = np.copy(y)
        tau = 1/L
        
        burnin = np.intc(1/(tau*L)) # burnin for denoising is usually short, one gradient step with step size 1/L lands at y
        n_samples_max = params['iterations_max']+burnin
        
        n_epsilons = params['log_epsilon'].size
        n_samples = np.zeros((n_epsilons,))
        prox_its = np.zeros((n_epsilons,))
        prox_its_per_sample = np.zeros((n_epsilons,))
        C = mu_tv*tv(y)
        for i,l in enumerate(params['log_epsilon']):
            epsilon = C*(10**l)
            
            ipgla = inexact_pla(x0, n_samples_max, burnin, posterior, step_size=('fxd',tau), rng=rng, epsilon_prox=epsilon, efficient=True, stop_crit=stopcrit)
            if verb: sys.stdout.write('\#{}: epsilon = {:.2e} - '.format(i,epsilon)); sys.stdout.flush()
            ipgla.simulate(verbose=verb)
            
            n_samples[i] = ipgla.n_iter-burnin
            prox_its[i] = ipgla.num_prox_its_total
            prox_its_per_sample[i] = prox_its[i]/n_samples[i]
            print('Number of inner iterations per proximal point: {}'.format(prox_its_per_sample[i]))
        # with open(results_file,'wb') as f:
            # np.save(f,C*(10**params['log_epsilon']))
            # np.save(f,n_samples)
            # np.save(f,prox_its)
            # np.save(f,prox_its_per_sample)
    else:
        pass
        print('Results file for this parameter already existed! Plotting the result:')
        with open(results_file,'rb') as f:
            epsilons = np.load(f)
            n_samples = np.load(f)
            prox_its = np.load(f)
            prox_its_per_sample = np.load(f)
        
        plt.figure()
        ax = plt.axes()
        ax.set_title('#Prox iterations until {}% MMSE error'.format(params['mmse_accuracy']))
        ax.set_xscale("log")
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('Total (inner) prox iterations')
        ax.plot(epsilons,prox_its,'b^-')
        
        plt.figure()
        ax = plt.axes()
        ax.set_title('#Samples until {}% MMSE error'.format(params['mmse_accuracy']))
        ax.set_xscale("log")
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('Number of (outer) Langevin iterations')
        ax.plot(epsilons,n_samples,'kv-')
        
    
        
#%% help function for calling from command line
def print_help():
    print('<>'*39)
    print('Run primal dual Langevin algorithm to generate samples from ROF posterior.')
    print('<>'*39)
    print(' ')
    print('Options:')
    print('    -h (--help): Print help.')
    print('    -i (--iterations_max=): Number of iterations of the Markov chain')
    print('    -f (--testfile_path=): Path to test image file')
    print('    -e (--efficient_off): Turn off storage-efficient mode, where we dont save samples but only compute a runnning mean and standard deviation during the algorithm. This can be used if we need the samples for some other reason (diagnostics etc). Then modify the code first')
    print('    -a (--mmse_accuracy=): Level of accuracy to which we want the reference MMSE image to be approximated')
    print('    -d (--result_dir=): root directory for results. Default: ./results/deblur-wavelets')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    # try:
    #     opts, args = getopt.getopt(sys.argv[1:],"hi:f:ea:s:d:v",
    #                                ["help","iterations_max=","testfile_path=","efficient_off","mmse_accuracy=","result_dir=","verbose"])
    # except getopt.GetoptError as e:
    #     print(e.msg)
    #     print_help()
    #     sys.exit(2)
    
    # for opt, arg in opts:
    #     if opt in ("-h", "--help"):
    #         print_help()
    #         sys.exit()
    #     elif opt in ("-i", "--iterations_max"):
    #         params['iterations_max'] = int(arg)
    #     elif opt in ("-f", "--testfile_path"):
    #         params['testfile_path'] = arg
    #     elif opt in ("-e","--efficient_off"):
    #         params['efficient'] = False
    #     elif opt in ("-a", "--mmse_accuracy"):
    #         params['mmse_accuracy'] = float(arg)
    #     elif opt in ("-d","--result_dir"):
    #         params['result_root'] = arg
    #     elif opt in ("-v", "--verbose"):
    #         params['verbose'] = True
    main()
# %%
