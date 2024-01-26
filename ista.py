#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:59:23 2024

@author: lorenzkuger
"""

import numpy as np
import sys

class ista():
    """solve the problem
        f(x) + g(x)
    using ISTA algorithm
        x_{k+1} = prox_{tau*g}(x_k - tau*df(x_k))
    with differentiable f and non-smooth g

    Step size tau must be given as a tuple, either 
    ('fixed',c) (where c is the fixed value, e.g. 1/L if df has Lipschitz const L)
    ('bt',t0) (where t0 will be an initial guess for backtracking)
    """
    def __init__(self, x0, tau, n_iter, f, g, efficient=True):
        self.iter = 0
        self.n_iter = n_iter
        self.x_shape = x0.shape
        
        self.step_strategy = tau
        if self.step_strategy[0] == 'fixed':
            self.tau = self.step_strategy[1]
        elif self.step_strategy[0] == 'bt':
            self.tauold = self.step_strategy[1]
            self.tauall = np.zeros((n_iter,))
        self.f = f
        self.df = f.grad
        self.g = g
        try:
            self.prox_g = g.prox
        except AttributeError:
            self.prox_g = lambda x,tau: g.inexact_prox(x,tau,max_iter=50)[0]
        
        # allow a non-efficient form of the algorithm where all iterates are returned, 
        # to be able to plot/compute errors or other diagnostics afterwards
        self.eff = efficient
        if self.eff:
            self.x = np.copy(x0)
        else:
            self.x = np.zeros(self.x_shape+(self.n_iter+1,))
            self.x[...,0] = np.copy(x0)
        
    def compute(self, verbose=False):
        if verbose: sys.stdout.write('run ISTA: {:3d}% '.format(0)); sys.stdout.flush()
        while self.iter < self.n_iter:
            self.iter += 1
            if self.step_strategy[0] == 'fixed':        # step size is in self.tau in the fixed step case
                if self.eff:
                    self.x = self.prox_g(self.x - self.tau*self.df(self.x), self.tau)
                else:
                    self.x[...,self.iter] = self.prox_g(self.x[...,self.iter-1] - self.tau*self.df(self.x[...,self.iter-1]), self.tau)
            else:                                       # backtracking with initial step size guess/last step size self.tauold
                x = self.x if self.eff else self.x[...,self.iter-1]
                sd = -self.df(x)    # search direction
                m = -0.5*np.sum(sd**2)
                fx = self.f(x)
                gamma = 0.7
                tau = 2*self.tauold                     # try increasing the old value, otherwise steps will only decrease
                while True:
                    z = x + tau*sd
                    if self.f(z) > fx + tau*m:          # check armijo-goldstein cond and decrease step size
                        tau *= gamma
                    else :                              # accept step size
                        self.tauold,self.tauall[self.iter-1] = tau,tau
                        if self.eff:
                            self.x = self.prox_g(z, tau)
                        else:
                            self.x[...,self.iter] = self.prox_g(z, tau)
                        break
            
            if verbose: sys.stdout.write('\b'*5 + '{:3d}% '.format(int(self.iter/self.n_iter*100))); sys.stdout.flush()
        return self.x if self.eff else self.x[...,-1]
