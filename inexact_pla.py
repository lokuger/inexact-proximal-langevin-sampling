import numpy as np
from numpy.random import default_rng
import sys
import matplotlib.pyplot as plt

class inexact_pla():
    """
    Instance of the inexact proximal Langevin algorithm
    __init__ parameters:
        - max_iter  : maximum number of iterations
        - tau       : step-size, either a scalar or a handle depending on iteration number n as in tau(n)
        - x0        : chain initialization, assumed to be an image, shape n1,n2
        - epsilon   : prox accuracy, either a scalar or a handle of n as epsilon(n)
        - pd        : probability distribution, object of distributions.py
    """
    def __init__(self, x0, tau, epsilon_prox, iter_prox, n_iter, burnin, pd, rng=None, efficient=False):
        self.n_iter = n_iter
        self.burnin = burnin
        self.iter = 0
        
        self.shape_x = x0.shape
        self.eff = efficient
        if self.eff:        # save only running sum and sum of squares to compute mean & std estimates
            self.x = np.copy(x0)
            self.sum = np.zeros(self.shape_x)
            self.sum_sq = np.zeros(self.shape_x)
        else:
            self.x = np.zeros(self.shape_x+(self.n_iter+1,))
            self.x[...,0] = x0
        
        # iteration parameters
        self.f = pd.f
        self.df = pd.f.grad
        self.dfx = self.df(self.x) if self.eff else self.df(self.x[...,0])
        self.g = pd.g
        self.inexact_prox_g = pd.g.inexact_prox
        self.rng = rng if rng is not None else default_rng()    # for reproducibility allow to pass rng
        self.tau = lambda n : tau if np.isscalar(tau) else tau
        self.epsilon_prox = lambda n : epsilon_prox if np.isscalar(epsilon_prox) else epsilon_prox
        self.iter_prox = iter_prox
        
        # diagnostic checks
        self.logpi_vals = np.zeros((self.n_iter,))
        self.num_prox_its_total = 0
    
    def simulate(self, verbose=False):
        if verbose: sys.stdout.write('run inexact PLA: {:3d}% '.format(0)); sys.stdout.flush()
        while self.iter < self.n_iter:
            self.update()
            if verbose and self.iter%20==0: sys.stdout.write('\b'*5 + '{:3d}% '.format(int(self.iter/self.n_iter*100))); sys.stdout.flush()
        if verbose > 0: sys.stdout.write('\n'); sys.stdout.flush()
        
        if self.eff:
            # once loop is done, compute mean and variance point estimates
            N = self.n_iter-self.burnin
            self.mean = self.sum/N
            self.var = (self.sum_sq - (self.sum**2)/N)/(N-1)
            self.std = np.sqrt(self.var)
        else:
            self.mean = np.mean(self.x[...,self.burnin+1:],axis=-1)
            self.std = np.std(self.x[...,self.burnin+1:],axis=-1)
            self.var = self.std**2
        
    def update(self):
        self.iter = self.iter + 1
        xi = self.rng.normal(size=self.shape_x)
        tau = self.tau(self.iter)
        epsilon_prox = self.epsilon_prox(self.iter) if self.epsilon_prox is not None else None
        iter_prox = self.iter_prox
        
        if self.eff:
            self.x, num_prox_its = self.inexact_prox_g(self.x-tau*self.dfx+np.sqrt(2*tau)*xi, tau, epsilon=epsilon_prox, max_iter=iter_prox)
            self.dfx = self.df(self.x)
            self.logpi_vals[self.iter-1] = self.f(self.x) + self.g(self.x)
            if self.iter > self.burnin:
                self.sum = self.sum + self.x
                self.sum_sq = self.sum_sq + self.x**2
        else:
            self.x[...,self.iter], num_prox_its = self.inexact_prox_g(self.x[...,self.iter-1]-tau*self.dfx+np.sqrt(2*tau)*xi, tau, epsilon=epsilon_prox, max_iter=iter_prox)
            self.dfx = self.df(self.x[...,self.iter])
            self.logpi_vals[self.iter-1] = self.f(self.x[...,self.iter]) + self.g(self.x[...,self.iter])
        
        self.num_prox_its_total += num_prox_its
        