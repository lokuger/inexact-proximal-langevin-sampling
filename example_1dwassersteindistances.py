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
    'n_chains_ipgla': 10000,
    'iterations_pxmala': 100000,
    'verbose': False,
    'step_type': 'fixed', # 'decay'
    'error_type': 'decay', # 'decay'
    'epsilon': 0.1,
    'rate': -0.2,
    }


#%% Main method
def main():
    results_dir = './results/wasserstein_estimates/steps_'+params['step_type']+'_errors_'+params['error_type']+'/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    errs_fixed = 'error_type' == 'fixed'
    if errs_fixed: 
        results_file = results_dir+'/W2dists_'+str(params['epsilon'])+'.npy'
        results_file_ub = results_dir+'/W2dists_'+str(params['epsilon'])+'_ub.npy'
    else:
        results_file = results_dir+'/W2dists_'+str(params['rate'])+'.npy'
        results_file_ub = results_dir+'/W2dists_'+str(params['rate'])+'_ub.npy'
        
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
    sampler_unbiased.simulate(verbose=True)
    if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
    
    # format samples for Wasserstein distance computation
    s_pxmala = np.reshape(sampler_unbiased.x,(n_samples_pxmala,))
    
    #%% Sample biasedly from posterior using (exact) PGLA
    # run many different chains and extract the law at Kth iterate to validate the theoretical convergence results
    n_chains = params['n_chains_ipgla']
    x0_ipgla = x_noisy
    W2sq_init = ot.lp.emd2_1d(s_pxmala,np.reshape(x0_ipgla,(-1,)))
    K = np.unique(np.intc(np.round(10**np.arange(start=0,stop=3.1,step=0.2))))
    Kmax = np.max(K)
    
    if params['step_type'] == 'fixed':
        tau_ipgla = 0.01/L
    else:
        tau_array = np.zeros((Kmax,))
        tau_array[0] = 1/L
        for i in np.arange(1,Kmax):
            tau_array[i] = np.minimum(tau_ipgla[i-1],np.maximum(1/(L*i),tau_ipgla[i-1]/(1+L)))
        tau_ipgla = lambda n: tau_array[n]
        
    if errs_fixed:
        epsilons = np.append(10**(-np.arange(0.0,2.1,1.0)),0)
        n_epsilons = len(epsilons)
        samples_K = np.zeros((len(K),n_chains,n_epsilons))
        for ic in np.arange(n_chains):
            for ie, e in enumerate(epsilons):
                if e == 0:
                    sampler = inexact_pla(x0_ipgla, Kmax, 0, posterior, step_size=tau_ipgla, rng=rng, exact=True, efficient=True, output_iterates=K)
                else:
                    sampler = inexact_pla(x0_ipgla, Kmax, 0, posterior, step_size=tau_ipgla, rng=rng, epsilon_prox=e, efficient=True, output_iterates=K)
                sampler.simulate(verbose=verb)
                samples_K[:,ic,ie] = sampler.output_iterates
        W2sq = np.zeros((len(K),n_epsilons))
        W2sq_ub_theory = np.zeros((len(K),n_epsilons))
    else:
        epsilon_rate = params['rate']
        samples_K = np.zeros((len(K),n_chains))
        for ic in np.arange(n_chains):
            epsilon_prox = lambda n: n**epsilon_rate
            sampler = inexact_pla(x0_ipgla, Kmax, 0, posterior, step_size=tau_ipgla, rng=rng, epsilon_prox=epsilon_prox, efficient=True, output_iterates=K)
            sampler.simulate(verbose=verb)
            samples_K[:,ic] = sampler.output_iterates
                
        # compute cumulative sums of epsilons for theoretical upper bounds
        sums_epsilons = np.zeros((len(K),))
        i,s = 0,0
        for k in np.arange(1,Kmax+1):
            s += k**epsilon_rate
            if k in K:
                sums_epsilons[i] = s
                i+=1
        W2sq = np.zeros((len(K),))
        W2sq_ub_theory = np.zeros((len(K),))
    
    # ax1 = plt.axes()
    # ax1.set_title('Wasserstein-2 distances')
    # ax1.set_xscale("log")
    # ax1.set_xlabel(r'$K$')
    # ax1.set_yscale("log")
    # ax1.set_ylabel(r'$\mathcal{W}_2^2(\mu^{K},\mu^{\ast})$')
    # ax1.set_xlim(np.min(K)-0.2,np.max(K)+0.2)
    # colors = ['b','r','m','g']
    if errs_fixed:
        for ie,e in enumerate(epsilons):
            for ik,k in enumerate(K):
                s_ipgla = np.reshape(samples_K[ik,:,ie],(-1,))
                W2sq[ik,ie] = ot.emd2_1d(s_pxmala,s_ipgla)
                W2sq_ub_theory[ik,ie] = (1 - L * tau_ipgla)**k * W2sq_init + tau_ipgla/L * (2*L*1 + 1) + 2*e/L
            # s = '{:.0e}'.format(e) if e > 0 else '0'
            # ax1.plot(K,W2sq[:,ie],colors[ie],label=r'$\mathcal{W}_2^2(\mu^K,\mu^\ast), \epsilon = $'+'{:s}'.format(s))
            # line_style = colors[ie]+'--'
            # ax1.plot(K,W2sq_ub_theory[:,ie],line_style,label=r'upper bound $\epsilon = ${:s}'.format(s))
    else:
        for ik,k in enumerate(K):
            s_ipgla = np.reshape(samples_K[ik,:],(-1,))
            W2sq[ik] = ot.emd2_1d(s_pxmala,s_ipgla)
            W2sq_ub_theory[ik] = (1 - L * tau_ipgla)**k * W2sq_init + tau_ipgla/L * (2*L*1 + 1) + 2/(L*k)*sums_epsilons[ik]
        # s = r'$k^{'+'{:.1f}'.format(r)+'}$'
        # ax1.plot(K,W2sq[:,ir],colors[ir],label=r'$\mathcal{W}_2^2(\mu^K,\mu^\ast), \epsilon_k \propto $'+'{:s}'.format(s))
        # line_style = colors[ir]+'--'
        # ax1.plot(K,W2sq_ub_theory[:,ir],line_style,label=r'upper bound $\epsilon_k \propto ${:s}'.format(s))            
    # ax1.legend()
    
    # # ax.errorbar(taus, W2_mean, np.stack((W2_mean-W2_min, W2_max-W2_mean)), fmt='-o', capsize=3, label='W2 distances')
    # # # ax.plot(Tau[3:], Tau[3:]*W2_max[-1]/Tau[-1], label=r'$\mathcal{O}(\tau)$', color='black')
    # # ax.legend()
    np.save(results_file, W2sq)
    np.save(results_file_ub, W2sq_ub_theory)
    plt.savefig(results_dir+'/W2_plots.pdf')
    print('Done')

    
#%% help function for calling from command line
def print_help():
    print('<>'*39)
    print('Run inexact proximal Langevin algorithm to generate samples from L1-TV posterior.')
    print('<>'*39)
    print(' ')
    print('Options:')
    print('    -h (--help): Print this help.')
    print('    -s (--stepsdecay): Instead of fixed step sizes use decaying step sizes, choice as in remark in paper')
    print('    -e (--errorsdecay): Instead of fixed errors choose decaying errors (modify file for the precise rate)')
    print('    -r (--rate=): If decaying errors, set their rate here')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hser:v",
                                   ["help","stepsdecay","errorsdecay","rate=","verbose"])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-s", "--stepsdecay"):
            params['step_type'] = 'decay'
        elif opt in ("-e", "--errorsdecay"):
            params['error_type'] = 'decay'
        elif opt in ("-r", "--rate="):
            params['rate'] = np.float16(arg)
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    
    
    