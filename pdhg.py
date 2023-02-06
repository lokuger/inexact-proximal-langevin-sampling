#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 19:50:29 2023

@author: lorenzkuger
"""

import numpy as np
import sys

class pdhg():
    """solve the problem
        f(Kx) + g(x)
    using PDHG algorithm
        x_{k+1} = prox_{tau*g}(x_k - tau*Kt*y_k)
        x_r = 2*x_{k+1} - x_k
        y_{k+1} = prox_{sigma*fconj}(y_k + sigma*K*x_r)
    """
    def __init__(self, x0, y0, tau, sigma, n_iter, f, g, k, kt):
        self.iter = 0
        self.n_iter = n_iter
        self.x_shape = x0.shape
        self.y_shape = y0.shape
        self.x = np.zeros(self.x_shape+(self.n_iter+1,))
        self.y = np.zeros(self.y_shape+(self.n_iter+1,))
        self.x[...,0] = np.copy(x0)
        self.y[...,0] = np.copy(y0)
        self.tau = tau
        self.sigma = sigma
        self.f = f
        self.prox_fconj = f.conj_prox
        self.g = g
        self.prox_g = g.prox
        self.k = k
        self.kt = kt
        
    def compute(self, verbose=False):
        if verbose: sys.stdout.write('run PDHG: {:3d}% '.format(0)); sys.stdout.flush()
        while self.iter < self.n_iter:
            self.iter += 1
            self.x[...,self.iter] = self.prox_g(self.x[...,self.iter-1] - self.tau*self.kt(self.y[...,self.iter-1]), self.tau)
            self.y[...,self.iter] = self.prox_fconj(self.y[...,self.iter-1] + self.sigma*self.k(2*self.x[...,self.iter]-self.x[...,self.iter-1]), self.sigma)
            if verbose: sys.stdout.write('\b'*5 + '{:3d}% '.format(int(self.iter/self.n_iter*100))); sys.stdout.flush()
        return self.x[...,-1]