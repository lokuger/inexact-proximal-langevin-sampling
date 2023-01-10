#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:02:11 2022

@author: lorenzkuger
"""

from psgla import PSGLA
import distributions as pds
import numpy as np
from numpy.random import default_rng
rng = default_rng()
    
def main():
    (M,N) = (5,5)
    u_true = np.ones((N,1))
    K = 95*np.eye(M,N)
    b = 5*np.ones((M,1))
    v = np.random.poisson(K@u_true + b)
    poisson_dist = pds.Poisson(data=v,background=b,K=K)
    
    L = np.sum(v/(b**2)*np.reshape(np.sum(K,axis=1),(M,1)))
    tau = 1/L
    max_iter = np.round(np.max((100,0.1/tau))).astype(int)
    n_samples = 2
    x0 = rng.uniform(size=(N,n_samples))
    psgla_fixed_step = PSGLA(max_iter, tau, x0, pd = poisson_dist)
    x_fixed_step, steps_fixed, sum_steps_fixed = psgla_fixed_step.simulate()
    
    # psgla with backtracking
    psgla_backtracking = PSGLA(max_iter, -1, x0, pd = poisson_dist)
    x_backtracking, steps_backtracking, sum_steps_backtracking = psgla_backtracking.simulate()
    
if __name__ == "__main__":
    main()