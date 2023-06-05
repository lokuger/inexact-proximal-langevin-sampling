# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 12:33:40 2023

@author: kugerlor
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import sys, getopt, os
from time import time
import ot


def main():
    rng = default_rng(2342)
    
    mu1 = 0
    sigma1 = 1
    
    mu2 = 0
    sigma2 = 1
    
    wd_true = (mu1-mu2)**2 + (sigma1-sigma2)**2
    
    N_samples = 1000
    
    N_runs = 10
    wd_samples = np.zeros((N_runs,))
    for i in np.arange(N_runs):
        samples1 = rng.normal(loc=mu1,scale=sigma1,size=(N_samples,))
        samples2 = rng.normal(loc=mu2,scale=sigma2,size=(N_samples,))
        
        wd_samples[i] = ot.lp.emd2_1d(samples1,samples2)
    
    print('True Wasserstein distance: {:3f}'.format(wd_true))
    print('Mean Samples Wasserstein disance: {:3f}'.format(np.mean(wd_samples)))
    print('Standard deviation of samples Wasserstein distances: {:3f}'.format(np.std(wd_samples)))
    

if __name__ == '__main__':
    main()