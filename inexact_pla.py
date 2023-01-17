import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import ot
import sys

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
    def __init__(self, n_iter, tau, x0, epsilon, pd):
        self.n_iter = n_iter
        self.iter = 0
        self.shape_x = x0.shape
        self.x0 = np.copy(x0)
        self.x = np.zeros((self.shape_x[0],self.shape_x[1],n_iter+1))
        self.x[:,:,0] = x0
        self.f = pd.f
        self.df = pd.f.grad
        self.dfx = self.df(self.x[:,:,0])
        self.g = pd.g
        self.inexact_prox_g = pd.g.inexact_prox
        self.rng = default_rng()
        self.tau = lambda n : tau if np.isscalar(tau) else tau
        self.epsilon = lambda n : epsilon if np.isscalar(epsilon) else epsilon
        self.logpi_iterates = np.zeros((n_iter,))
        self.num_prox_iterations_total = 0
    
    def simulate(self, verbose=1):
        # from wasserstein comparisons:
        # W2dist = np.zeros((self.max_iter+1,))
        # W2dist[0] = ot.emd2_1d(np.reshape(self.x0,(-1,)), np.reshape(x_comp,(-1,)))
        if verbose:
            print('Running inexact PLA')
            sys.stdout.write('Sampling progress: {:3d}%'.format(0))
            sys.stdout.flush()
        while self.iter < self.n_iter:
            self.update()
            if verbose > 0:
                sys.stdout.write('\b'*4+'{:3d}%'.format(int(self.iter/self.n_iter*100)))
                sys.stdout.flush()
            #W2dist[self.iter] = ot.emd2_1d(np.reshape(self.x,(-1,)), np.reshape(x_comp,(-1,)))
        if verbose > 0:
            sys.stdout.write('\n')
        
    def update(self):
        self.iter = self.iter + 1
        xi = self.rng.normal(size=self.shape_x)
        tau = self.tau(self.iter)
        epsilon = self.epsilon(self.iter)
        
        y = self.x[:,:,self.iter-1] - tau * self.dfx + np.sqrt(2*tau) * xi
        x, num_its = self.inexact_prox_g(y, gamma=tau, epsilon=epsilon, verbose=False)
        self.x[:,:,self.iter] = x
        self.dfx = self.df(x)
        self.logpi_iterates[self.iter-1] = - self.f(x) - self.g(x)
        
        self.num_prox_iterations_total += num_its
        