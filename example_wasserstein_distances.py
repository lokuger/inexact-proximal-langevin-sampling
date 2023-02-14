#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:47:26 2023

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import sys, getopt, os
from time import time
import ot

from inexact_pla import inexact_pla
from pxmala import pxmala
import potentials as pot
import distributions as pds

#%% initial parameters: test image, computation settings etc.
params = {
    'iterations_pxmala': 1000,
    'iterations_ipgla': 1000,
    'runs': 1,
    'step': 'large',
    'verbose': True,
    }

step_factors = {
    'large': 1,
    'small': 0.5,
    'smaller': 0.25
    }

#%% auxiliary functions
def create_image(n1,n2,l1scale,noise_std,rng):
    verb = params['verbose']
    x = rng.laplace(scale=l1scale, size=(n1,n2))
    y = x + noise_std*rng.normal(size=x.shape)
    
    if verb: sys.stdout.write('Compute l2-l1 MAP - '); sys.stdout.flush()
    l1prior = pot.l1_loss_unshifted_homoschedastic(l1scale)
    u = l1prior.prox(y, gamma=noise_std**2)
    if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
    
    return x, y, u

#%% Main method
def main():
    #%% output file
    # results_dir = './results/wasserstein_estimates/size{}x{}/{}iterations'.format(params['a'],params['b'],params['iterations'])
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
    # results_file = results_dir+'/W2dists.npy'
    if True: #not os.path.exists(results_file):    
        #%% generate data - artificial image with pixels drawn from Laplace distribution
        rng = default_rng(28757)
        verb = params['verbose']
        
        n1, n2 = 1,1
        l1scale = 0.5
        noise_std = 0.02
        L = 1/noise_std**2
        
        tau_ipgla = 1/L * step_factors[params['step']]
        epsilon_ipgla = 1
        T = 10
        n_runs = params['runs']
        Wd = np.zeros((n_runs,))
        
        for i_run in np.arange(n_runs):
            print('Run #{}'.format(i_run))
            x_true, x_noisy, x_l2l1map = create_image(n1, n2, l1scale, noise_std, rng)
            
            #%% sample using PxMALA
            x0_pxmala = np.copy(x_l2l1map)
            tau_pxmala = 8e-4 # tune this by hand to achieve a satisfactory acceptance rate (roughly 50%-65%)
            n_samples_pxmala = int(T/tau_pxmala)#params['iterations_pxmala']
            burnin_pxmala = 200
            posterior = pds.l2loss_l1prior(y=x_noisy, noise_std=noise_std, mu_l1=l1scale)
            sampler_unbiased = pxmala(x0_pxmala, tau_pxmala, n_samples_pxmala, burnin_pxmala, pd=posterior, rng=rng)
        
            if verb: sys.stdout.write('Sample without bias from ROF posterior - '); sys.stdout.flush()
            sampler_unbiased.simulate(verbose=verb)
            if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
            
            #%% sample using inexact PLA
            x0_ipgla = np.copy(x_noisy)
            burnin_ipgla = 200
            n_samples_ipgla = int(T/tau_ipgla) + burnin_ipgla #params['iterations_ipgla']
            
            sampler = inexact_pla(x0_ipgla, tau_ipgla, epsilon_ipgla, n_samples_ipgla, burnin_ipgla, posterior, rng=rng)
            if verb: sys.stdout.write('Sample from posterior - '); sys.stdout.flush()
            sampler.simulate(verbose=verb)
            if verb: sys.stdout.write('Done.\nCompute Wasserstein-2 distance. '); sys.stdout.flush()
            
            #%% compute Wasserstein distances between PDLA and Px-MALA samples
            a = np.reshape(sampler.x[...,burnin_ipgla:],(n1*n2,n_samples_ipgla-burnin_ipgla+1)).T
            b = np.reshape(sampler_unbiased.x[...,burnin_pxmala:],(n1*n2,n_samples_pxmala-burnin_pxmala+1)).T
            if n1 == 1 and n2 == 1:
                W = ot.emd2_1d(a,b)
            else:
                M = ot.dist(a,b)
                W = ot.emd2([],[],M) # empty lists indicate uniform weighting of the samples
            # print('Time to compute distance matrix: {:.2f}s. Time to compute W2 distance: {:.2f}s'.format(time2-time1,time3-time2))
            print(W)
            Wd[i_run] = W
        print('Mean distance: {}'.format(np.mean(Wd)))
        # np.save(results_file, Wd)
    else:
        pass
        # Wd = np.load(results_file)
        # Tau = np.load(results_dir+'/steps.npy')
        # W2_min = np.min(Wd,axis=1)
        # W2_max = np.max(Wd,axis=1)
        # W2_mean = np.mean(Wd,axis=1)
        # W2_std = np.std(Wd,axis=1)
        # ax = plt.axes()
        # ax.set_title('Wasserstein-2 distances over {} runs'.format(params['iterations']))
        # ax.set_xscale("log")
        # ax.set_xlabel(r'$\tau$')
        # ax.set_yscale("log")
        # ax.set_ylabel(r'$\mathcal{W}_2^2(\bar{\mu}_N^{\mathrm{PDLA}},\mu^{\mathrm{MH}}_{M})$')
        # ax.set_xlim(3e-4,1.1e0)
        # ax.set_ylim(7e-6,1.1e1)
        # ax.errorbar(Tau[3:], W2_mean[3:], np.stack((W2_mean-W2_min, W2_max-W2_mean))[:,3:], fmt='-o', capsize=3, label='W2 distances')
        # ax.plot(Tau[3:], Tau[3:]*W2_max[-1]/Tau[-1], label=r'$\mathcal{O}(\tau)$', color='black')
        # ax.legend()
        # plt.savefig(results_dir+'/plot{}runs.pdf'.format(params['iterations']))
    
    #%% plots
    # plt.imshow(x)
    # plt.title('Ground truth')
    # plt.show()
    # plt.imshow(u)
    # plt.title('MAP')
    # plt.show()
    # plt.imshow(sampler.mean)
    # plt.title('PDLA')
    # plt.show()
    # plt.imshow(sampler_unbiased.mean)
    # plt.title('PxMALA')
    # plt.show()
    
#%% help function for calling from command line
def print_help():
    print('<>'*39)
    print('Run primal dual Langevin algorithm to generate samples from L1-TV posterior.')
    print('<>'*39)
    print(' ')
    print('Options:')
    print('    -h (--help): Print help.')
    print('    -i (--iterations=): Number of iterations of the Markov chain')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:v",
                                   ["help","iterations=","verbose"])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-i", "--iterations"):
            params['iterations'] = int(arg)
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    
    
    