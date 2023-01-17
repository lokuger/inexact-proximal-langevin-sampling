#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:24:56 2023

@author: lorenzkuger
"""

import numpy as np
import sys
from numpy.random import default_rng

class sapg():
    """
    Implements the SAPG algorithm introduced by [1]. For the numerical
    tests of inexact PLA, we need distributions with densities of the form 
        d mu(x) / d Leb(x) ~ exp( - f(x) - mu * g(x) )
    where 
        - f is smooth, gradient-Lipschitz and convex
        - g is convex and potentially non-smooth
    SAPG is an empirical Bayesian strategy to choose the parameter mu by 
    computing the marginal MLE of the parameter.
    
    Important: Some of the classes in potentials.py like l1loss or TV
    allow a scaling by a parameter 'scale' (which is implicitly the 
    regularization parameter mu), because we need this for the later 
    sampling algorithms.
    This class assumes that this is set to 1, i.e. pd.g computes the 
    non-differentiable part of the potential without a scaling.
    
    Parameters:
        - iter_wu:      Muber of warm-up sampling steps
        - iter_outer:   Number of steps calibrating mu
        - iter_inner:   Number of sampling steps in each outer loop
        - iter_burnin:  Number of burn-in steps used for the estimates of theta
        - tau:          Step size of the sampling algorithm, here (inexact) PLA
        - delta:        lambda expression, Step size for mu, will be called as delta(n)
        - x0:           Initialization for warm-up. Choose as noisy image or MAP
        - theta0:       Initialization of regularization parameter
        - theta_min     lower bound for parameter
        - theta_max     upper bound for parameter
        - epsilon_prox  accuracy for evaluation of proximal mapping in inexact PLA
        - pd            posterior distribution, class instance from distributions.pyi
        
    [1]: "Maximum likelihood estimation of regularisation parameters in
    high-dimensional inverse problems: an empirical Bayesian approach Part I: 
        Methodology and Experiments", 2020. Vidal, De Bortoli, Pereyra, Durmus
    """
    
    def __init__(self, iter_wu, iter_outer, iter_inner, iter_burnin, tau, delta, x0, theta0, theta_min, theta_max, epsilon_prox, pd):
        self.iter_wu = iter_wu
        self.iter_outer = iter_outer
        self.iter_inner = iter_inner
        self.iter_burnin = iter_burnin
        self.i_wu = 0
        self.i_out = 0
        self.tau = tau
        self.delta = delta  # lambda expression mapping n to delta_n
        self.theta, self.mean_theta = np.zeros((iter_outer+1,)), np.zeros((iter_outer-self.iter_burnin,))
        self.theta[0], self.mean_theta[0], self.theta_min, self.theta_max = theta0, theta0, theta_min, theta_max
        self.eta = np.zeros((iter_outer+1,))
        self.eta[0], self.eta_min, self.eta_max = np.log(theta0), np.log(theta_min), np.log(theta_max)
        self.eps_prox = epsilon_prox
        self.x0 = np.copy(x0)
        self.x = np.copy(x0)
        self.x_shape = self.x.shape
        self.d = np.prod(self.x_shape)
        self.pd = pd
        self.f = pd.f
        self.df = pd.f.grad
        self.dfx = self.df(self.x)
        self.g = pd.g
        self.mean_g = np.zeros((iter_outer+1,))
        self.mean_g[0] = self.g(x0)
        try:
            self.prox_is_exact = True
            self.prox_g = self.g.prox
        except AttributeError:
            self.prox_is_exact = False
            self.prox_g = self.g.inexact_prox
        self.rng = default_rng()
    
    def simulate(self, return_all=False, verbose=1):
        if verbose:
            print('Running SAPG to determine optimal regularization parameter')
            sys.stdout.write('Warming up Markov chain: {:3d}%'.format(0))
            sys.stdout.flush()
        self.warmup(verbose)
        if verbose:
            sys.stdout.write('\nRun SAPG: {:3d}%'.format(0))
            sys.stdout.flush()
        while self.i_out < self.iter_outer:
            self.i_out += 1
            self.outer()
            if verbose:
                sys.stdout.write('\b'*4+'{:3d}%'.format(int(self.i_out/self.iter_outer*100)))
                sys.stdout.flush()
        if verbose:
            print('\nFinal estimate of regularization parameter: {:.4f}\n'.format(self.mean_theta[-1]))
    
    def warmup(self, verbose):
        self.logpi_wu = np.zeros((self.iter_wu,))
        while self.i_wu < self.iter_wu:
            self.i_wu += 1
            xi = self.rng.normal(loc=0, scale=1, size=self.x_shape)
            y = self.x - self.tau * self.dfx + np.sqrt(2*self.tau) * xi
            if self.prox_is_exact:
                self.x = self.prox_g(y, gamma=self.tau*self.theta[0])
            else:
                self.x, _ = self.prox_g(y, gamma=self.tau*self.theta[0], epsilon=self.eps_prox)
            self.dfx = self.df(self.x)
            # monitor likelihood during warm-up to estimate burn-in time
            self.logpi_wu[self.i_wu-1] = - self.f(self.x) - self.theta[0]*self.g(self.x)
            if verbose:
                sys.stdout.write('\b'*4+'{:3d}%'.format(int(self.i_wu/self.iter_wu*100)))
                sys.stdout.flush()
    
    def outer(self):
        self.i_in = 0
        values_g = np.zeros((self.iter_inner,))
        while self.i_in < self.iter_inner:
            self.inner()
            values_g[self.i_in-1] = self.g(self.x)
        self.mean_g[self.i_out] = np.mean(values_g)
        self.eta[self.i_out] = self.eta[self.i_out-1] + self.delta(self.i_out)*(self.d/self.theta[self.i_out-1]- self.mean_g[self.i_out])*self.theta[self.i_out-1]
        self.eta[self.i_out] = np.minimum(np.maximum(self.eta_min,self.eta[self.i_out]),self.eta_max)
        self.theta[self.i_out] = np.exp(self.eta[self.i_out])
        if self.i_out > self.iter_burnin:
            self.mean_theta[self.i_out-1-self.iter_burnin] = np.mean(self.theta[self.iter_burnin:self.i_out+1])
        
        #print(self.theta[self.i_out])
            
    def inner(self):
        self.i_in += 1
        xi = self.rng.normal(loc=0, scale=1, size=self.x_shape)
        y = self.x - self.tau * self.dfx + np.sqrt(2*self.tau) * xi
        if self.prox_is_exact:
            self.x = self.prox_g(y, gamma=self.tau*self.theta[self.i_out-1])
        else:
            self.x, _ = self.prox_g(y, gamma=self.tau*self.theta[self.i_out-1], epsilon=self.eps_prox)
        self.dfx = self.df(self.x)
        