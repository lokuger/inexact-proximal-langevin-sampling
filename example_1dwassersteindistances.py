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
    'n_chains_ipgla': 1000,
    'iterations_pxmala': 100000,
    'verbose': False,
    }


#%% Main method
def main():
    results_dir = './results/wasserstein_estimates)'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    #%% generate data - artificial image with pixels drawn from Laplace distribution
    rng = default_rng(246264)
    verb = params['verbose']
    
    l1scale = 1
    noise_std = 1
    L = 1/noise_std**2
    x_prior = rng.laplace(loc=0,scale=l1scale,size=(1,1))
    x_noisy = x_prior + rng.normal(loc=0.0, scale=noise_std)
    posterior = pds.l2_l1prior(y=x_noisy, noise_std=noise_std, mu_l1=l1scale)
    
    #%% sample unbiasedly from posterior using PxMALA
    x0_pxmala = x_noisy*np.ones((1,1))
    tau_pxmala = 1.2 # tune this by hand to achieve a satisfactory acceptance rate (roughly 50%-65%)
    n_samples_pxmala = params['iterations_pxmala']#int(10*T/tau_pxmala)
    burnin_pxmala = 1000 #int(10*T_burnin/tau_pxmala)
    posterior = pds.l2_l1prior(y=x_noisy, noise_std=noise_std, mu_l1=l1scale)
    sampler_unbiased = pxmala(x0_pxmala, tau_pxmala, n_samples_pxmala+burnin_pxmala, burnin_pxmala, pd=posterior, rng=rng, efficient=False)

    if verb: sys.stdout.write('Sample without bias from ROF posterior - '); sys.stdout.flush()
    sampler_unbiased.simulate(verbose=verb)
    if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
    
    # format samples for Wasserstein distance computation
    s_pxmala = np.reshape(sampler_unbiased.x,(n_samples_pxmala,))
    
    #%% Sample biasedly from posterior using (exact) PGLA
    # run many different chains and extract the law at Kth iterate to validate the theoretical convergence results
    n_chains = params['n_chains_ipgla']
    s_ipgla = np.zeros((n_chains,))
    tau_ipgla = 0.01/L
    x0_ipgla = x_noisy
    W2sq_init = ot.lp.emd2_1d(s_pxmala,np.reshape(x0_ipgla,(-1,)))
    K = np.unique(np.intc(np.round(10**np.arange(start=0,stop=4.1,step=0.2))))
    
    epsilons = np.append(10**(-np.arange(0.0,2.1,1.0)),0)
    n_epsilons = len(epsilons)
    
    W2sq = np.zeros((len(K),n_epsilons))
    W2sq_ub_theory = np.zeros((len(K),n_epsilons))
    
    samples_K = np.zeros((len(K),n_chains,n_epsilons))
    for ic in np.arange(n_chains):
        for ie, e in enumerate(epsilons):
            if e == 0:
                sampler = inexact_pla(x0_ipgla, np.max(K), 0, posterior, step_size=tau_ipgla, rng=rng, exact=True, efficient=True, output_iterates=K)
            else:
                sampler = inexact_pla(x0_ipgla, np.max(K), 0, posterior, step_size=tau_ipgla, rng=rng, epsilon_prox=e, efficient=True, output_iterates=K)
            sampler.simulate(verbose=verb)
            samples_K[:,ic,ie] = sampler.output_iterates
        
    
    ax1 = plt.axes()
    ax1.set_title('Wasserstein-2 distances')
    ax1.set_xscale("log")
    ax1.set_xlabel(r'$K$')
    ax1.set_yscale("log")
    ax1.set_ylabel(r'$\mathcal{W}_2^2(\mu^{k},\mu^{\ast})$')
    ax1.set_xlim(np.min(K)-0.2,np.max(K)+0.2)
    colors = ['b','r','m','g']
    for ie,e in enumerate(epsilons):
        for ik,k in enumerate(K):
            s_ipgla = np.reshape(samples_K[ik,:,ie],(-1,))
            W2sq[ik,ie] = ot.emd2_1d(s_pxmala,s_ipgla)
            W2sq_ub_theory[ik,ie] = (1 - L * tau_ipgla)**k * W2sq_init + tau_ipgla/L * (2*L*1 + 1) + 2*e/L
        s = '{:.0e}'.format(e) if e > 0 else '0'
        ax1.plot(K,W2sq[:,ie],colors[ie],label=r'$\mathcal{W}_2^2(\mu^K,\mu^\ast), \epsilon = $'+'{:s}'.format(s))
        line_style = colors[ie]+'--'
        ax1.plot(K,W2sq_ub_theory[:,ie],line_style,label=r'upper bound $\epsilon = ${:s}'.format(s))
    ax1.legend()
    # ax.errorbar(taus, W2_mean, np.stack((W2_mean-W2_min, W2_max-W2_mean)), fmt='-o', capsize=3, label='W2 distances')
    # # ax.plot(Tau[3:], Tau[3:]*W2_max[-1]/Tau[-1], label=r'$\mathcal{O}(\tau)$', color='black')
    # ax.legend()
    # # np.save(results_file, Wd)
    plt.savefig(results_dir+'/W2_plots.pdf')
    print('Done')

    
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
    
    
    