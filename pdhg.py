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
        f(Kx) + g(x) + h(x)
    using PDHG algorithm
        x_{k+1} = prox_{tau*g}(x_k - tau*(dh(x_k) + Kt*y_k))
        x_r = 2*x_{k+1} - x_k
        y_{k+1} = prox_{sigma*fconj}(y_k + sigma*K*x_r)
    with differentiable h and possibly non-smooth f and g
    """
    def __init__(self, x0, y0, tau, sigma, n_iter, f, k, kt, g, h, efficient=True):
        self.iter = 0
        self.n_iter = n_iter
        self.x_shape = x0.shape
        self.y_shape = y0.shape
        
        self.tau = tau
        self.sigma = sigma
        self.f = f
        self.prox_fconj = f.conj_prox
        self.k = k
        self.kt = kt
        self.g = g
        self.prox_g = g.prox
        self.h = h
        self.dh = h.grad
        
        # allow a non-efficient form of the algorithm where all iterates are returned, 
        # to be able to compute errors or other diagnostics afterwards
        self.eff = efficient
        if self.eff:
            self.x = np.copy(x0)
            self.y = np.copy(y0)
        else:
            self.x = np.zeros(self.x_shape+(self.n_iter+1,))
            self.y = np.zeros(self.y_shape+(self.n_iter+1,))
            self.x[...,0] = np.copy(x0)
            self.y[...,0] = np.copy(y0)
        
    def compute(self, verbose=False):
        if verbose: sys.stdout.write('run PDHG: {:3d}% '.format(0)); sys.stdout.flush()
        while self.iter < self.n_iter:
            self.iter += 1
            if self.eff:
                x_prev = self.x
                self.x = self.prox_g(self.x - self.tau*(self.dh(self.x)+self.kt(self.y)), self.tau)
                self.y = self.prox_fconj(self.y + self.sigma*self.k(2*self.x - x_prev), self.sigma)
            else:
                self.x[...,self.iter] = self.prox_g(self.x[...,self.iter-1] - self.tau*(self.dh(self.x[...,self.iter-1])+self.kt(self.y[...,self.iter-1])), self.tau)
                self.y[...,self.iter] = self.prox_fconj(self.y[...,self.iter-1] + self.sigma*self.k(2*self.x[...,self.iter]-self.x[...,self.iter-1]), self.sigma)
            if verbose: sys.stdout.write('\b'*5 + '{:3d}% '.format(int(self.iter/self.n_iter*100))); sys.stdout.flush()
        return self.x if self.eff else self.x[...,-1]
    
class acc_pdhg():
    """solve the problem
        f(Kx) + g(x) + h(x)
    using an accelerated PDHG algorithm
        x_r = x_k + theta_k*(x_k - x_{k-1})
        y_{k+1} = prox_{sigma*fconj}(y_k + sigma*K*x_r)
        x_{k+1} = prox_{tau*g}(x_k - tau*(dh(x_k) + Kt*y_{k+1}))
    with differentiable h and possibly non-smooth f and g
    
    Acceleration based on the assumption that g is mu_g strongly convex and h 
    is mu_h strongly cvx. mu_g and mu_h are assumed to be passed as parameters
    
    In Chambolle-Pock 2016, the analysis is based on the assumption that g is
    mu-strongly convex and h not strongly convex. This is equivalent, since we
    can shift the strongly convex part of h onto g and run the same algorithm
    with a slightly different step size.
    For that reason adjust in every step the true steps gamma from the running 
    step size tau (tau corresponds the step size tau in the paper)
    """
    def __init__(self, x0, y0, tau0, sigma0, n_iter, f, k, kt, g, mu_g, h, mu_h, efficient=True):
        self.iter = 0
        self.n_iter = n_iter
        self.x_shape = x0.shape
        self.y_shape = y0.shape
        
        self.tau = tau0
        self.sigma = sigma0
        self.theta = 0
        self.f = f
        self.prox_fconj = f.conj_prox
        self.k = k
        self.kt = kt
        self.g = g
        self.prox_g = g.prox
        self.mu_g = mu_g
        self.h = h
        self.dh = h.grad
        self.mu_h = mu_h
        
        # allow a non-efficient form of the algorithm where all iterates are returned
        self.eff = efficient
        if self.eff:
            self.x = np.copy(x0)
            self.x_prev = self.x
            self.y = np.copy(y0)
        else:
            self.x = np.zeros(self.x_shape+(self.n_iter+1,))
            self.y = np.zeros(self.y_shape+(self.n_iter+1,))
            self.x[...,0] = np.copy(x0)
            self.y[...,0] = np.copy(y0)
            
    def compute(self, verbose=False):
        if verbose: sys.stdout.write('run accelerated PDHG: {:3d}% '.format(0)); sys.stdout.flush()
        while self.iter < self.n_iter:
            self.iter += 1
            if self.eff:
                self.y = self.prox_fconj(self.y + self.sigma*self.k(self.x + self.theta*(self.x - self.x_prev)), self.sigma)
                self.x_prev = self.x
                gamma = self.tau/(1-self.tau*self.mu_h)     # this takes care of shifting the strong convexity constant from h to g so that tau matches the updates from the analysis
                self.x = self.prox_g(self.x - gamma*(self.dh(self.x)+self.kt(self.y)), gamma)
            else:
                x_prev = self.x[...,self.iter-2] if self.iter > 1 else self.x[...,0]
                self.y[...,self.iter] = self.prox_fconj(self.y[...,self.iter-1] + self.sigma*self.k(self.x[...,self.iter-1]+self.theta*(self.x[...,self.iter-1]-x_prev)), self.sigma)
                gamma = self.tau/(1-self.tau*self.mu_h)     # this takes care of shifting the strong convexity constant from h to g so that tau matches the updates from the analysis
                self.x[...,self.iter] = self.prox_g(self.x[...,self.iter-1] - gamma*(self.dh(self.x[...,self.iter-1])+self.kt(self.y[...,self.iter])), gamma)
            
            self.theta = 1/np.sqrt(1+(self.mu_g+self.mu_h)*self.tau)
            self.tau = self.theta * self.tau
            self.sigma = self.sigma/self.theta
            
            if verbose: sys.stdout.write('\b'*5 + '{:3d}% '.format(int(self.iter/self.n_iter*100))); sys.stdout.flush()
        return self.x if self.eff else self.x[...,-1]
    
    