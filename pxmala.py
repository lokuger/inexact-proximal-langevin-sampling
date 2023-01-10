#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:41:15 2022

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

class PxMALA():
    def __init__(self, max_iter, tau, x0, pd):
        # we're computing samples in self.x with shape (d, n_samples)
        # expect input x0 to be (d, n_samples) as well
        self.max_iter = max_iter
        self.tau = tau
        self.iter = 0
        self.T = 0
        self.d, self.n_samples = x0.shape
        self.x0 = np.copy(x0)
        
        self.pd = pd
        self.F = pd.F
        self.dF = self.F.grad
        try:
            self.targetpdf = pd.pdf
        except AttributeError:
            self.targetpdf = pd.unscaled_pdf
        try:
            self.proxExact = True
            self.proxG = pd.G.prox
        except AttributeError:
            self.proxExact = False
            self.proxG = pd.G.inexact_prox
        self.rng = default_rng()
        
        # iterates
        self.x = np.copy(x0)
        self.dFx = self.dF(self.x)
        self.proxVal = self.proxApprox(self.x,self.dFx)
    
    def simulate(self, return_all = True):
        self.accepted = 0
        if return_all:
            x_all = np.zeros((self.d, self.n_samples, self.max_iter+1))
            x_all[:,:,0] = self.x0
        while self.iter < self.max_iter:
            self.update()
            if return_all:
                x_all[:,:,self.iter] = self.x
        x_return = x_all if return_all else self.x
        return x_return, self.accepted/(self.max_iter*self.n_samples)
    
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
        xi = self.rng.normal(loc=0, scale=1, size=(self.d, self.n_samples))
        x_proposal = self.proxVal + np.sqrt(2*self.tau) * xi
        dFx_proposal = self.dF(x_proposal)
        proxVals_proposal = self.proxApprox(x_proposal, dFx_proposal)
        s1 = - np.sum((self.x-proxVals_proposal)**2,axis=0)/(4*self.tau)
        s2 = + np.sum((x_proposal-self.proxVal)**2, axis=0)/(4*self.tau)
        s3 = - np.reshape(self.pd.F(x_proposal) + self.pd.G(x_proposal),(-1,))
        s4 = + np.reshape(self.pd.F(self.x) + self.pd.G(self.x),(-1,))
        s = s1+s2+s3+s4
        q = np.exp(s)
        p = np.minimum(1,q)
        p[q==float('inf')] = 1  # if the sum in exp was very large then p should be 1
        
        #p = np.ones((1,self.n_samples))
        r = self.rng.binomial(1, p)
        I = r==1
        self.accepted += np.sum(I)
        self.x[:,I] = x_proposal[:,I]
        self.proxVal[:,I] = proxVals_proposal[:,I]
        self.dFx[:,I] = dFx_proposal[:,I]
        #self.x = x_proposal
        #self.proxVal = proxVals_proposal
        #self.dFx = dFx_proposal
        
    def proxApprox(self,u,dFu):
        w = u - self.tau * dFu
        if self.proxExact:
            y = self.proxG(w, gamma=self.tau)
        else:
            y = self.proxG(w, gamma=self.tau, epsilon=None, maxiter=1e3, verbose=False)
        return y
                