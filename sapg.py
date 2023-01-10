#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:24:56 2023

@author: lorenzkuger
"""

import numpy as np
import sys
from numpy.random import default_rng

import distributions as pd

class SAPG():
    def __init__(self, iter_wu, iter_outer, iter_inner, tau, delta, x0, theta0, thetamin, thetamax, epsilon_prox, pd):
        """Important: Some of the classes in potentials.py  like l1loss or TV
        allow a scaling by a parameter theta (regularization parameter).
        Here in SAPG we assume that this is set to 1, i.e. pd.G computes the 
        non-differentiable part of the potential without a scaling.
        """
        self.iter_wu = iter_wu
        self.iter_outer = iter_outer
        self.iter_inner = iter_inner
        self.i_wu = 0
        self.i_out = 0
        self.tau = tau
        self.delta = delta  # lambda expression mapping n to delta_n
        self.theta, self.mean_theta = np.zeros((iter_outer+1,)), np.zeros((iter_outer+1,))
        self.theta[0], self.mean_theta[0], self.thetamin, self.thetamax = theta0, theta0, thetamin, thetamax
        self.eta = np.zeros((iter_outer+1,))
        self.eta[0], self.etamin, self.etamax = np.log(theta0), np.log(thetamin), np.log(thetamax)
        self.eps_prox = epsilon_prox
        self.d = x0.shape[0]
        self.x0 = np.copy(x0)
        self.x = np.copy(x0)
        self.pd = pd
        self.F = self.pd.F
        self.dF = self.F.grad
        self.dFx = self.dF(self.x)
        self.G = self.pd.G
        self.meansG = np.zeros((iter_outer+1,))
        self.meansG[0] = self.G(x0)
        try:
            self.proxExact = True
            self.proxG = self.G.prox
        except AttributeError:
            self.proxExact = False
            self.proxG = self.G.inexact_prox
        self.rng = default_rng()
    
    def simulate(self, return_all=False, verbose=1):
        if verbose:
            print('Running SAPG to determine optimal regularization parameter')
            sys.stdout.write('Warming up Markov chain: {:3d}%'.format(0))
        self.warmup(verbose)
        if verbose:
            sys.stdout.write('\nRun SAPG: {:3d}%'.format(0))
        while self.i_out < self.iter_outer:
            self.i_out += 1
            self.outer()
            if verbose:
                sys.stdout.write('\b'*4+'{:3d}%'.format(int(self.i_out/self.iter_outer*100)))
    
    def warmup(self, verbose):
        """ warm up Markov chain using (inexact) PLA """
        self.logpi_wu = np.zeros((self.iter_wu,))
        while self.i_wu < self.iter_wu:
            self.i_wu += 1
            xi = self.rng.normal(loc=0, scale=1, size=(self.d,1))
            y = self.x - self.tau * self.dFx + np.sqrt(2*self.tau) * xi
            if self.proxExact:
                self.x = self.proxG(y, gamma=self.tau*self.theta[0])
            else:
                self.x = self.proxG(y, gamma=self.tau*self.theta[0], epsilon = self.eps_prox)
            self.dFx = self.dF(self.x)
            # monitor likelihood during warm-up to estimate burn-in time
            self.logpi_wu[self.i_wu-1] = - self.F(self.x) - self.theta[0]*self.G(self.x)
            if verbose:
                sys.stdout.write('\b'*4+'{:3d}%'.format(int(self.i_wu/self.iter_wu*100)))
    
    def outer(self):
        self.i_in = 0
        values_G = np.zeros((self.iter_inner,))
        while self.i_in < self.iter_inner:
            self.inner()
            values_G[self.i_in-1] = self.G(self.x)
        mean_valsG = np.mean(values_G)
        self.eta[self.i_out] = self.eta[self.i_out-1] + self.delta(self.i_out)*(self.d/self.theta[self.i_out-1]- mean_valsG)*self.theta[self.i_out-1]
        self.eta[self.i_out] = np.minimum(np.maximum(self.etamin,self.eta[self.i_out]),self.etamax)
        self.theta[self.i_out] = np.exp(self.eta[self.i_out])
        self.mean_theta[self.i_out] = np.mean(self.theta[:self.i_out+1])
        self.meansG[self.i_out] = mean_valsG
        
        #print(self.theta[self.i_out])
            
    def inner(self):
        """ this is essentially one step of inexact PLA, contrarily to the paper
        version by Vidal where they used MYULA, this difference should not be a 
        problem. """
        self.i_in += 1
        xi = self.rng.normal(loc=0, scale=1, size=(self.d,1))
        y = self.x - self.tau * self.dFx + np.sqrt(2*self.tau) * xi
        if self.proxExact:
            self.x = self.proxG(y, gamma=self.tau*self.theta[self.i_out-1])
        else:
            self.x = self.proxG(y, gamma=self.tau*self.theta[self.i_out-1], epsilon=self.eps_prox)
        self.dFx = self.dF(self.x)
        