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
    'n_chains_ipgla': 100,
    'iterations_pxmala': 100000,
    'verbose': True,
    'step_type': 'fixed', # 'decay'
    'inexactness_type': 'decay', # 'fixed'
    'epsilon': 0.1,
    'rate': -0.6,
    'result_root': './results/1dwasserstein',
    }


#%% Main method
def main():
    step_fixed = (params['step_type'] == 'fixed')
    errs_fixed = (params['inexactness_type'] == 'fixed')
    
    result_root = params['result_root']
    results_dir = result_root+'/steps_'+params['step_type']+'_inexactness_'+params['inexactness_type']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    if errs_fixed: 
        results_file = results_dir+'/W2dists_epsilon'+str(params['epsilon'])+'.npy'
        results_file_ub = results_dir+'/W2dists_ub_epsilon'+str(params['epsilon'])+'.npy'
        steps_file = results_dir+'/steps'+str(params['epsilon'])+'.npy'
    else:
        results_file = results_dir+'/W2dists_rate'+str(params['rate'])+'.npy'
        results_file_ub = results_dir+'/W2dists_ub_rate'+str(params['rate'])+'.npy'
        steps_file = results_dir+'/steps'+str(params['rate'])+'.npy'
    
    #if os.path.exists(results_file) and os.path.exists(results_file_ub):
    #    print('Results file for these parameters already exist')
    #else:
    if True:
        #%% generate data - artificial image with pixels drawn from Laplace distribution
        # fix the seed here so that the posterior remains the same for different runs of the script
        rng = default_rng(65654)
        verb = params['verbose']
        
        l1scale = 1
        noise_std = 1
        L = 1/noise_std**2
        x_prior = rng.laplace(loc=0,scale=l1scale,size=(1,1))
        x_noisy = x_prior + rng.normal(loc=0.0, scale=noise_std)
        posterior = pds.l2_l1prior(y=x_noisy, noise_std=noise_std, mu_l1=l1scale)
        
        #%% sample unbiasedly from posterior using PxMALA
        # might be interesting to remove the following line. This generates the same Brownian motion for different 
        # runs of the script and shows the effect of the error level even more obviously. 
        # But could also be confusing when the error curves look very similar for different error levels/runs
        rng = default_rng()
        
        x0_pxmala = x_noisy*np.ones((1,1))
        tau_pxmala = 1.2 # tune this by hand to achieve a satisfactory acceptance rate (roughly 50%-65%)
        n_samples_pxmala = params['iterations_pxmala'] # int(10*T/tau_pxmala)
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
        x0_ipgla = x_noisy
        W2sq_init = ot.lp.emd2_1d(s_pxmala,np.reshape(x0_ipgla,(-1,)))
        K = np.unique(np.intc(np.round(10**np.arange(start=0,stop=3.1,step=0.2))))
        Kmax = np.max(K)
        
        # set fixed step size or compute the decaying sequence
        if step_fixed:
            tau_ipgla = 0.01/L
        else:
            tau_array = np.zeros((Kmax,))
            tau_array[0] = 1/L
            for i in np.arange(1,Kmax):
                tau_array[i] = np.minimum(tau_ipgla[i-1],np.maximum(1/(L*i),tau_ipgla[i-1]/(1+L)))
            tau_ipgla = lambda n: tau_array[n]
        
        if errs_fixed:
            epsilon = params['epsilon']
            samples_K = np.zeros((len(K),n_chains))
            if verb: sys.stdout.write('Run inexact PLA chains: {:3d}% '.format(0)); sys.stdout.flush()
            for ic in np.arange(n_chains):
                progress = int(ic/n_chains*100)
                if verb: sys.stdout.write('\b'*5 + '{:3d}% '.format(progress)); sys.stdout.flush()
                if epsilon == 0:
                    sampler = inexact_pla(x0_ipgla, Kmax, 0, posterior, step_size=tau_ipgla, rng=rng, exact=True, efficient=True, output_iterates=K)
                else:
                    sampler = inexact_pla(x0_ipgla, Kmax, 0, posterior, step_size=tau_ipgla, rng=rng, epsilon_prox=epsilon, efficient=True, output_iterates=K)
                sampler.simulate(verbose=False)
                samples_K[:,ic] = sampler.output_iterates
            W2sq = np.zeros((len(K),))
            W2sq_ub_theory = np.zeros((len(K),))
            if verb: sys.stdout.write('\b'*5 + 'Done.\n'); sys.stdout.flush()
        else:
            epsilon_rate = params['rate']
            samples_K = np.zeros((len(K),n_chains))
            if verb: sys.stdout.write('Run inexact PLA chains: {:3d}% '.format(0)); sys.stdout.flush()
            for ic in np.arange(n_chains):
                progress = int(ic/n_chains*100)
                if verb: sys.stdout.write('\b'*5 + '{:3d}% '.format(progress)); sys.stdout.flush()
                epsilon_prox = lambda n: n**epsilon_rate
                sampler = inexact_pla(x0_ipgla, Kmax, 0, posterior, step_size=tau_ipgla, rng=rng, epsilon_prox=epsilon_prox, efficient=True, output_iterates=K)
                sampler.simulate(verbose=False)
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
            if verb: sys.stdout.write('\b'*5 + 'Done.\n'); sys.stdout.flush()
        
        if verb: sys.stdout.write('Compute Wasserstein distances...'); sys.stdout.flush()
        if errs_fixed:
            for ik,k in enumerate(K):
                s_ipgla = np.reshape(samples_K[ik,:],(-1,))
                W2sq[ik] = ot.emd2_1d(s_pxmala,s_ipgla)
                W2sq_ub_theory[ik] = (1 - L * tau_ipgla)**k * W2sq_init + tau_ipgla/L * (2*L*1 + 1) + 2*epsilon/L
        else:
            for ik,k in enumerate(K):
                s_ipgla = np.reshape(samples_K[ik,:],(-1,))
                W2sq[ik] = ot.emd2_1d(s_pxmala,s_ipgla)
                W2sq_ub_theory[ik] = (1 - L * tau_ipgla)**k * W2sq_init + tau_ipgla/L * (2*L*1 + 1) + 2/(L*k)*sums_epsilons[ik]
        if verb: sys.stdout.write('\b'*2 + ' Done.\nSaving...'); sys.stdout.flush()
        np.save(results_file, W2sq)
        np.save(results_file_ub, W2sq_ub_theory)
        np.save(steps_file, K)
        if verb: sys.stdout.write('\b'*2 + ' Done.\n'); sys.stdout.flush()

    
#%% help function for calling from command line
def print_help():
    print('<>'*39)
    print('Run inexact proximal Langevin algorithm to generate samples from a synthetic 1D posterior.')
    print('<>'*39)
    print(' ')
    print('Options:')
    print('    -h (--help): Print this help.')
    print('    -n (--n_chains_ipgla): Number of chains to run with IPGLA')
    print('    -m (--n_samples_pxmala): Number of (unbiased) samples to generate by Px-MALA (1000 burnin + this value)')
    print('    -s (--step_decays): Instead of fixed step sizes use decaying step sizes, choice as in remark in paper')
    print('    -i (--inexactness_decays): Instead of fixed epsilon choose decaying inexactness level')
    print('    -r (--rate=): If decaying inexactness level, set the decay rate here')
    print('    -e (--epsilon=): If fixed inexactness level, set the fixed level here')
    print('    -d (--result_dir=): If fixed inexactness level, set the fixed level here')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hn:m:sir:e:d:v",
                                   ["help","n_chains_ipgla=","n_samples_pxmala=","step_decays","inexactness_decays","rate=","epsilon=","result_dir=","verbose"])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-n","--n_chains_ipgla"):
            params['n_chains_ipgla'] = int(arg)
        elif opt in ("-m","--n_samples_pxmala"):
            params['iterations_pxmala'] = int(arg)
        elif opt in ("-s", "--step_decays"):
            params['step_type'] = 'decay'
        elif opt in ("-i", "--inexactness_decays"):
            params['inexactness_type'] = 'decay'
        elif opt in ("-r", "--rate"):
            params['rate'] = np.float16(arg)
        elif opt in ("-e","--epsilon"):
            params['epsilon'] = np.float16(arg)
        elif opt in ("-d","--result_dir"):
            params['result_root'] = arg
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    
    
    