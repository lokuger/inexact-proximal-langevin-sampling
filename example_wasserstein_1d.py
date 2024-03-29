#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:47:26 2023

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import sys, getopt, os
import ot

from inexact_pla import inexact_pla
from pxmala import pxmala
import distributions as pds

#%% initial parameters: test image, computation settings etc.
params = {
    'n_chains_ipgla': 500,
    'iterations_pxmala': 100000,
    'verbose': True,
    'step_type': 'decay',        # 'decay','fixed'
    'inexactness_type': 'none', # 'fixed','decay','none'
    'epsilon': 0.01,
    'rate': -1.0,
    'result_root': './results/wasserstein-dists-validation',
    }


#%% Main method
def main():
    result_root = params['result_root']
    step_fixed = (params['step_type'] == 'fixed')
    result_dir = result_root+'/steps-'+params['step_type']+'-inexactness-'+params['inexactness_type']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    d = {'fixed': '-epsilon'+str(params['epsilon']),
         'decay': '-rate'+str(params['rate']),
         'none': ''}
    result_file = result_dir+'/W2dists'+d[params['inexactness_type']]+'.npy'
    
    if os.path.exists(result_file):
        print('Results file for these parameters already exist')
    else:
        #%% generate data - artificial parameter drawn from Laplace distribution
        # fix the seed here so that the posterior remains the same for different runs of the script, otherwise the bounds would change
        rng = default_rng(12345)
        verb = params['verbose']
        
        l1scale = 1
        noise_std = 1
        L = 1/noise_std**2
        x_prior = rng.laplace(loc=0,scale=l1scale,size=(1,1))
        x_noisy = x_prior + rng.normal(loc=0.0, scale=noise_std)
        posterior = pds.l2_l1prior(y=x_noisy, noise_std=noise_std, mu_l1=l1scale)
        
        #%% sample unbiasedly from posterior using PxMALA
        rng = default_rng()
        x0_pxmala = x_noisy*np.ones((1,1))
        tau_pxmala = 0.9 # tune this by hand to achieve a satisfactory acceptance rate (roughly 50%-65%)
        n_samples_pxmala = params['iterations_pxmala']
        burnin_pxmala = 1000
        posterior = pds.l2_l1prior(y=x_noisy, noise_std=noise_std, mu_l1=l1scale)
        sampler_unbiased = pxmala(x0_pxmala, tau_pxmala, n_samples_pxmala+burnin_pxmala, burnin_pxmala, pd=posterior, rng=rng, efficient=False)
    
        if verb: sys.stdout.write('Sample without bias from ROF posterior - '); sys.stdout.flush()
        sampler_unbiased.simulate(verbose=verb)
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
        
        # format samples for Wasserstein distance computation
        s_pxmala = np.reshape(sampler_unbiased.x,(n_samples_pxmala,))
        
        #%% Sample biasedly from posterior using PGLA
        n_chains = params['n_chains_ipgla']
        x0_ipgla = x_noisy
        W2sq_init = ot.lp.emd2_1d(s_pxmala,np.reshape(x0_ipgla,(-1,)))
        K = np.unique(np.intc(np.round(10**np.arange(start=0,stop=3.1,step=0.2))))
        Kmax = np.max(K)
        
        # set step size: either fixed or the decaying sequence from remark in paper
        if not step_fixed:
            tau_array = np.zeros((Kmax,))
            tau_array[0] = 0.01/L
            for i in np.arange(1,Kmax):
                tau_array[i] = np.minimum(tau_array[i-1],np.maximum(1/(L*i),tau_array[i-1]/(1+L)))
        tau_ipgla = 0.01/L if step_fixed else (lambda n: tau_array[n])
            
        
        # set inexactness level
        if params['inexactness_type'] == 'none':
            epsilon = 0
        elif params['inexactness_type'] == 'fixed':
            epsilon = params['epsilon']
        elif params['inexactness_type'] == 'decay':
            epsilon_rate = params['rate']
            epsilon = lambda n: n**epsilon_rate
            
        # actual simulation - draw samples using ipgla
        samples_K = np.zeros((len(K),n_chains))
        if verb: sys.stdout.write('Run inexact PLA chains: {:3d}% '.format(0)); sys.stdout.flush()
        n_prox_its = 0
        for ic in np.arange(n_chains):
            if verb: progress = int(ic/n_chains*100); sys.stdout.write('\b'*5 + '{:3d}% '.format(progress)); sys.stdout.flush()
            sampler = inexact_pla(x0_ipgla, Kmax, 0, posterior, step_size=tau_ipgla, rng=rng, epsilon_prox=epsilon, efficient=True, output_iterates=K)
            sampler.simulate(verbose=False)
            n_prox_its += sampler.num_prox_its_total
            samples_K[:,ic] = sampler.output_iterates
        print('Avg number iterations towards prox: {}'.format(n_prox_its/(Kmax*n_chains)))
        W2sq = np.zeros((len(K),))
        if verb: sys.stdout.write('\b'*5 + 'Done.\n'); sys.stdout.flush()
        
        # compute squared W2 distances
        if verb: sys.stdout.write('Compute Wasserstein distances...'); sys.stdout.flush()
        for ik,k in enumerate(K):
            s_ipgla = np.reshape(samples_K[ik,:],(-1,))
            W2sq[ik] = ot.emd2_1d(s_pxmala,s_ipgla)

        # compute upper bounds predicted by the theory (if applicable - we did not compute bounds for variable step sizes)
        if params['step_type'] == 'fixed':
            W2sq_ub_theory = np.zeros((len(K),))
            if params['inexactness_type'] in ['none','fixed']:  # part 1 of nonasymptotic theorem on fixed step sizes
                for ik,k in enumerate(K):
                    W2sq_ub_theory[ik] = (1 - L * tau_ipgla)**k * W2sq_init + tau_ipgla/L * (2*L*1 + 1) + 2*epsilon/L
            elif params['inexactness_type'] == 'decay':         # part 2 of nonasymptotic theorem on fixed step sizes
                sums_epsilons = np.zeros((len(K),))
                i,s = 0,0
                for k in np.arange(1,Kmax+1):
                    s += k**epsilon_rate
                    if k in K:
                        sums_epsilons[i] = s
                        i+=1
                for ik,k in enumerate(K):
                    W2sq_ub_theory[ik] = (1 - L * tau_ipgla)**k * W2sq_init + tau_ipgla/L * (2*L*1 + 1) + 2/(L*k)*sums_epsilons[ik]
            
        if verb: sys.stdout.write('\b'*2 + ' Done.\n'); sys.stdout.flush()
        
        with open(result_file,'wb') as f:
            np.save(f, W2sq)
            np.save(f, K)
            if params['step_type'] in ['none','fixed']:
                np.save(f, W2sq_ub_theory)
        

    
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
    print('    -s (--step=): Type of step sizes, \'fixed\' or \'decay\'')
    print('    -i (--inexactness=): Type of errors - \'fixed\', \'decay\' or \'none\'')
    print('    -r (--rate=): If decaying inexactness level, set the decay rate here')
    print('    -e (--epsilon=): If fixed inexactness level, set the fixed level here')
    print('    -d (--result_dir=): root directory for results. Default: ./results/wasserstein-dists-validation')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hn:m:s:i:r:e:d:v",
                                   ["help","n_chains_ipgla=","n_samples_pxmala=","step=","inexactness=","rate=","epsilon=","result_dir=","verbose"])
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
        elif opt in ("-s", "--step"):
            if arg in ['fixed','decay']:
                params['step_type'] = arg
            else: print('Unknown step size option'); sys.exit(3)
        elif opt in ("-i", "--inexactness"):
            if arg in ['fixed','decay','none']:
                params['inexactness_type'] = arg
            else: print('Unknown inexactness choice option'); sys.exit(3)
        elif opt in ("-r", "--rate"):
            params['rate'] = np.float16(arg)
        elif opt in ("-e","--epsilon"):
            params['epsilon'] = np.float16(arg)
        elif opt in ("-d","--result_dir"):
            params['result_root'] = arg
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    
    main()
    
    
    