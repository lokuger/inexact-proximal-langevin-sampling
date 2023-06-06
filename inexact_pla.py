import numpy as np
from numpy.random import default_rng
import sys
import matplotlib.pyplot as plt

class inexact_pla():
    """
    Instance of the inexact proximal Langevin algorithm
    __init__ parameters:
        - x0            : chain initialization, assumed to be an image, shape n1,n2
        - n_iter        : number of samples to be drawn
        - burnin        : number of burnin samples to throw away
        - pd            : probability distribution, object of distributions.py3q
        Optional parameters
        - step-size             : iPGLA step-size (tau), either a scalar or a handle for iteration number input
        - rng (default_rng())   : random number generator for reproducibility, init new one if None
        - epsilon_prox (1e-2)   : prox accuracy, either a scalar or a handle of n as epsilon(n)
        - iter_prox (np.Inf)    : number of iterations for prox, can be given as alternative to epsilon
        - efficient (True)      : if True, do not save iterates but only current iterate, running mean and std of samples
        - exact (False)         : if pd.g has an exact proximal operator, can choose True and run exact PGLA
        
    """
    def __init__(self, x0, n_iter, burnin, pd, step_size=None, rng=None, epsilon_prox=1e-2, iter_prox=np.Inf, efficient=False, output_iterates=None, exact=False):
        self.n_iter = n_iter
        self.burnin = burnin
        self.iter = 0
        self.rng = rng if rng is not None else default_rng()    # for reproducibility allow to pass rng
        step_size = step_size if step_size is not None else 1/pd.f.L
        self.step_size = (lambda n : step_size) if np.isscalar(step_size) else step_size
        
        self.shape_x = x0.shape
        self.eff = efficient
        if self.eff:        # save only running sum and sum of squares to compute mean & std estimates
            self.x = np.copy(x0)
            self.sum = np.zeros(self.shape_x)
            self.sum_sq = np.zeros(self.shape_x)
            # in efficient mode, there is the option to output a selected number of iterates at given indices
            if output_iterates is not None:
                self.I = output_iterates
            else:
                self.I = np.reshape(self.n_iter,(1,))   # output last sample if nothing specified
            n_outputs = np.size(self.I)
            self.output_iterates = np.zeros(self.shape_x+(n_outputs,))
        else:
            self.x = np.zeros(self.shape_x+(self.n_iter+1,))
            self.x[...,0] = x0
        
        # iteration parameters
        self.f = pd.f
        self.df = pd.f.grad
        self.dfx = self.df(self.x) if self.eff else self.df(self.x[...,0])
        self.g = pd.g
        self.exact = exact
        if self.exact:
            self.prox_g = pd.g.prox
        else:
            self.inexact_prox_g = pd.g.inexact_prox
            self.epsilon_prox = (lambda n : epsilon_prox) if np.isscalar(epsilon_prox) else epsilon_prox
            self.iter_prox = iter_prox
        
        # diagnostic checks
        self.logpi_vals = np.zeros((self.n_iter,))
        self.num_prox_its_total = 0
    
    def simulate(self, verbose=False):
        if verbose: sys.stdout.write('run inexact PLA: {:3d}% '.format(0)); sys.stdout.flush()
        i = 0
        while self.iter < self.n_iter:
            self.update()
            if self.eff and self.iter in self.I:
                self.output_iterates[...,i] = self.x
                i+=1
            if verbose and self.iter%20==0: 
                progress = int(self.iter/self.n_iter*100)
                sys.stdout.write('\b'*5 + '{:3d}% '.format(progress))
                sys.stdout.flush()
        if verbose > 0: sys.stdout.write('\n'); sys.stdout.flush()
        
        if self.eff:
            # once loop is done, compute mean and variance point estimates
            N = self.n_iter-self.burnin
            self.mean = self.sum/N
            self.var = (self.sum_sq - (self.sum**2)/N)/(N-1) if N>1 else np.NAN
            self.std = np.sqrt(self.var)
        else:
            self.x = self.x[...,self.burnin+1:]
            self.mean = np.mean(self.x,axis=-1)
            self.std = np.std(self.x,axis=-1)
            self.var = self.std**2
        
    def update(self):
        self.iter = self.iter + 1
        xi = self.rng.normal(size=self.shape_x)
        step_size = self.step_size(self.iter)
        
        if not self.exact:
            epsilon_prox = self.epsilon_prox(self.iter) if self.epsilon_prox is not None else None
            iter_prox = self.iter_prox
        
        if self.eff:
            if self.exact:
                self.x, num_prox_its = self.prox_g(self.x-step_size*self.dfx+np.sqrt(2*step_size)*xi, step_size), 0
            else:
                self.x, num_prox_its = self.inexact_prox_g(self.x-step_size*self.dfx+np.sqrt(2*step_size)*xi, step_size, epsilon=epsilon_prox, max_iter=iter_prox)
            self.dfx = self.df(self.x)
            self.logpi_vals[self.iter-1] = self.f(self.x) + self.g(self.x)
            if self.iter > self.burnin:
                self.sum = self.sum + self.x
                self.sum_sq = self.sum_sq + self.x**2
        else:
            if self.exact:
                self.x[...,self.iter], num_prox_its = self.prox_g(self.x[...,self.iter-1]-step_size*self.dfx+np.sqrt(2*step_size)*xi, step_size), 0
            else:
                self.x[...,self.iter], num_prox_its = self.inexact_prox_g(self.x[...,self.iter-1]-step_size*self.dfx+np.sqrt(2*step_size)*xi, step_size, epsilon=epsilon_prox, max_iter=iter_prox)
            self.dfx = self.df(self.x[...,self.iter])
            self.logpi_vals[self.iter-1] = self.f(self.x[...,self.iter]) + self.g(self.x[...,self.iter])
        
        self.num_prox_its_total += num_prox_its
        