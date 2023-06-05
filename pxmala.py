#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:41:15 2022

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import sys

class pxmala():
    """
    An asymptotically unbiased Langevin algorithm for log-concave distributions
    with non-smooth potentials
    
    Update step in Px-MALA:
    x_prop ~ N(prox_{tau*(F+G)}(x_k), delta*I)
    r(x_prop, x_k) := min(1, pi(x_prop)/pi(x_k) * q(x_k,x_prop)/q(x_prop,x_k) )
    with probability r(x_prop,x_k), set
        x_{k+1} = x_prop
    else
        x_{k+1} = x_k
    
    Notes:
        1) Since prox_{tau*g} = prox_{tau*(F+G)} with two functions F,G in our setup
        is hardly computable, we approximate
            prox_{tau*(F+G)}(u) = argmin{F(x) + G(x) + 1/(2*tau)*||u-x||^2}
                                ~ argmin{F(u) + <F'(u),x-u> + G(x) + 1/(2*tau)*||u-x||^2}
                                = argmin{G(x)+1/(2*tau)*||u-tau*F'(u)-x||^2}
                                = prox_{tau*G}(u-tau*F'(u))
                                               
        2) In every step, due to the definition of the transition probability,
        we have to compute prox_{tau*g}(x_prop). Save these values for the next step
        since x_prop will be the new iterates x_{k+1}, this saves computations
    """
    def __init__(self, x0, step_size, n_iter, burnin, pd, rng=None, efficient=False):
        self.n_iter = n_iter
        self.burnin = burnin
        self.iter = 0
        self.rng = rng if rng is not None else default_rng()    # for reproducibility allow to pass rng
        self.step_size = lambda n : step_size if np.isscalar(step_size) else step_size
        
        self.shape_x = x0.shape
        self.eff = efficient
        if self.eff:        # save only running sum and sum of squares to compute mean & std estimates
            self.x = np.copy(x0)
            self.sum = np.zeros(self.shape_x)
            self.sum_sq = np.zeros(self.shape_x)
        else:
            self.x = np.zeros(self.shape_x+(self.n_iter+1,))
            self.x[...,0] = x0
        
        self.f = pd.f
        self.df = pd.f.grad
        self.g = pd.g
        try:
            self.prox_is_exact = True
            self.prox_g = self.g.prox
        except AttributeError:
            self.prox_is_exact = False
            self.prox_g = self.g.inexact_prox
        # store current iterate's gradient and prox for efficiency reasons
        self.dfx = self.df(self.x[...,0])
        self.prox_val = self.prox(self.x[...,0],self.step_size(0),self.dfx)
        
        # diagnostic checks
        self.logpi_vals = np.zeros((self.n_iter,))
    
    def simulate(self, verbose=False):
        self.accepted = 0
        if verbose: sys.stdout.write('run PxMALA: {:3d}%. AR = NaN%'.format(0)); sys.stdout.flush()
        while self.iter < self.n_iter:
            self.update()
            if verbose and self.iter%20==0: 
                progress = int(self.iter/self.n_iter*100)
                ar = int(self.accepted/self.iter*100)
                sys.stdout.write('\b'*16 + '{:3d}%. AR = {:3d}% '.format(progress,ar));
                sys.stdout.flush()
        if verbose > 0: sys.stdout.write('\n'); sys.stdout.flush()
            
        if self.eff:
            # once loop is done, compute mean and variance point estimates
            N = self.n_iter-self.burnin
            self.mean = self.sum/N
            self.var = (self.sum_sq - (self.sum**2)/N)/(N-1)
            self.std = np.sqrt(self.var)
        else:
            self.x = self.x[...,self.burnin+1:]
            self.mean = np.mean(self.x,axis=-1)
            self.std = np.std(self.x,axis=-1)
            self.var = self.std**2
    
    def update(self):
        self.iter = self.iter + 1
        xi = self.rng.normal(size = self.shape_x)
        step_size = self.step_size(self.iter)
        
        x_proposal = self.prox_val + np.sqrt(2*step_size) * xi
        dfx_proposal = self.df(x_proposal)
        prox_val_proposal = self.prox(x_proposal, step_size, dfx_proposal)
        if self.eff:
            s1 = - np.sum((self.x-prox_val_proposal)**2)/(4*step_size)
            s2 = + 0.5 * np.sum(xi**2)
            s3 = - self.f(x_proposal) - self.g(x_proposal)
            s4 = + self.f(self.x) + self.g(self.x)
        else:
            s1 = - np.sum((self.x[...,self.iter-1]-prox_val_proposal)**2)/(4*step_size)
            s2 = + 0.5 * np.sum(xi**2) # + np.sum((x_proposal-self.prox_val)**2)/(4*step_size)
            s3 = - self.f(x_proposal) - self.g(x_proposal) #### -/+ g?????
            s4 = + self.f(self.x[...,self.iter-1]) + self.g(self.x[...,self.iter-1])
        s = s1+s2+s3+s4
        q = np.exp(s) 
        p = 1 if q == float('inf') else np.minimum(1,q)
        
        r = self.rng.binomial(1, p)
        if r==1:
            self.accepted += 1
            if self.eff:
                self.x = x_proposal
            else:
                self.x[...,self.iter] = x_proposal
            self.prox_val = prox_val_proposal
            self.dfx = dfx_proposal
        else:
            if not self.eff:
                self.x[...,self.iter] = self.x[...,self.iter-1]
        
    def prox(self,u,step_size,dfu):
        w = u - step_size * dfu
        if self.prox_is_exact:
            y = self.prox_g(w, gamma=step_size)
        else:
            y, _ = self.prox_g(w, gamma=step_size, epsilon=None, maxiter=1e2, verbose=False)
        return y
    
                