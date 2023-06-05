# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 14:47:47 2023

@author: kugerlor
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
    'iterations_pxmala': 10000,
    'iterations_pgla': 10000,
    'n_runs': 10,
    'verbose': False,
    }

#%% Main method
def main():
    #%% generate data - artificial image with pixels drawn from Laplace distribution
    rng = default_rng(246264)
    verb = params['verbose']
    
    mu1 = 0
    sigma1 = np.sqrt(2)
    mu2 = 0
    sigma2 = np.sqrt(2)
    
    mu_prod = (mu1*sigma2**2 + mu2*sigma1**2)/(sigma1**2+sigma2**2)
    sigma_prod = (sigma1**2*sigma2**2)/(sigma1**2+sigma2**2)
    
    N_samples = params['iterations_pxmala']
    s_direct = rng.normal(loc=mu_prod,scale=sigma_prod,size=(N_samples,))
    
    N_runs = params['n_runs']
    # to test different levels of epsilon
    Wd = np.zeros((N_runs,))
    
    for i_run in np.arange(N_runs):
        #%% sample unbiasedly from posterior using PxMALA
        x0_pxmala = mu_prod*np.ones((1,1))
        tau_pxmala = 0.6 # tune this by hand to achieve a satisfactory acceptance rate (roughly 50%-65%)
        burnin = 1000 #int(10*T_burnin/tau_pxmala)
        posterior = pds.l2_tikhprior(mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2)
        sampler_unbiased = pxmala(x0_pxmala, tau_pxmala, N_samples+burnin, burnin, pd=posterior, rng=rng, efficient=False)
    
        if verb: sys.stdout.write('Sample without bias from ROF posterior - '); sys.stdout.flush()
        sampler_unbiased.simulate(verbose=verb)
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
        
        # format samples for Wasserstein distance computation
        s_pxmala = np.reshape(sampler_unbiased.x,(N_samples,))
        Wd[i_run] = ot.lp.emd2_1d(s_pxmala,s_direct)

    print('Done with all runs, Compare directly drawn samples with Px-MALA chain samples')
    print('\tMean Wasserstein distance: {:3f}'.format(np.mean(Wd)))
    print('\tStandard deviation is Wasserstein distances: {:3f}'.format(np.std(Wd)))
    
    
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
    
    
    