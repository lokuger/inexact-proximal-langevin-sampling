#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:41:15 2022

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

class pxmala():
    def __init__(self, n_iter, tau, x0, pd):
        # we're computing samples in self.x with shape (d, n_samples)
        # expect input x0 to be (d, n_samples) as well
        self.n_iter = n_iter
        self.iter = 0
        self.tau = tau
        self.shape_x = x0.shape
        self.x0 = np.copy(x0)
        self.x = np.zeros((self.shape_x[0],self.shape_x[1],self.n_iter+1))
        self.x[:,:,0] = self.x0
        
        self.pd = pd
        self.f = pd.f
        self.df = self.f.grad
        try:
            self.targetpdf = pd.pdf
        except AttributeError:
            self.targetpdf = pd.unscaled_pdf
        try:
            self.prox_is_exact = True
            self.prox_g = pd.g.prox
        except AttributeError:
            self.prox_is_exact = False
            self.prox_g = pd.g.inexact_prox
        self.rng = default_rng()
        
        # iterates
        self.dfx = self.df(self.x[:,:,0])
        self.prox_vals = self.prox(self.x[:,:,0],self.dfx)
    
    def simulate(self):
        self.accepted = 0
        while self.iter < self.n_iter:
            self.update()
    
    def update(self):
        """
        Update step in Px-MALA:
        x_prop ~ N(prox_{tau*g}(x_k), delta*I)
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
        self.iter = self.iter + 1
        xi = self.rng.normal(size = self.shape_x)
        x_proposal = self.prox_vals + np.sqrt(2*self.tau) * xi
        dfx_proposal = self.df(x_proposal)
        prox_vals_proposal = self.prox(x_proposal, dfx_proposal)
        s1 = - np.sum((self.x[:,:,self.iter-1]-prox_vals_proposal)**2)/(4*self.tau)
        s2 = + np.sum((x_proposal-self.prox_vals)**2)/(4*self.tau)
        s3 = - self.pd.f(x_proposal) + self.pd.g(x_proposal)
        s4 = + self.pd.f(self.x[:,:,self.iter-1]) + self.pd.g(self.x[:,:,self.iter-1])
        s = s1+s2+s3+s4
        q = np.exp(s)
        p = 1 if q == float('inf') else np.minimum(1,q)
        
        r = self.rng.binomial(1, p)
        if r==1:
            self.accepted += 1
            self.x[:,:,self.iter] = x_proposal
            self.prox_vals = prox_vals_proposal
            self.dfx = dfx_proposal
        else:
            self.x[:,:,self.iter] = self.x[:,:,self.iter-1]
        
    def prox(self,u,dfu):
        w = u - self.tau * dfu
        if self.prox_is_exact:
            y = self.prox_g(w, gamma=self.tau)
        else:
            # is there some easy way to estimate duality gap bound epsilon here? Maybe duality gap at first iteration?
            y, _ = self.prox_g(w, gamma=self.tau, epsilon=None, maxiter=1e2, verbose=False)
        return y
    
    
    
                