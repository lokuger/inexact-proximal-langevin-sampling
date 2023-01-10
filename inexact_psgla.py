import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import ot

class IPSGLA():
    def __init__(self, max_iter, tau, x0, epsilon, pd):
        # we're computing samples in self.x with shape (d, n_samples)
        # infer shape from input x0
        # step size tau is held constant for now
        # prox accuracies epsilon can be a constant or a sequence of length max_iters
        self.max_iter = max_iter
        self.iter = 0
        self.d, self.n_samples = x0.shape
        self.x0 = np.copy(x0)
        self.x = np.copy(x0)
        self.dF = pd.F.grad
        self.dFx = self.dF(self.x)
        self.F = pd.F
        self.inexact_proxG = pd.G.inexact_prox
        self.rng = default_rng()
        if not np.isscalar(tau):
            self.tau = np.reshape(tau,(-1,))
        else:
            self.tau = tau * np.ones((self.max_iter,))
        if not np.isscalar(epsilon):
            self.epsilon = np.reshape(epsilon,(-1,))
        else:
            self.epsilon = epsilon * np.ones((self.max_iter,))
    
    def simulate(self, x_comp = None, return_all = False):
        return_errs = False
        if x_comp is not None:
            assert x_comp.shape[0]==self.x.shape[0],'Comparison samples must have the same dimension as running samples'
            return_errs = True
            W2dist = np.zeros((self.max_iter+1,))
            W2dist[0] = ot.emd2_1d(np.reshape(self.x0,(-1,)), np.reshape(x_comp,(-1,)))
        if return_all:
            x_all = np.zeros((self.d, self.n_samples, self.max_iter+1))
            x_all[:,:,0] = self.x0
        
        while self.iter < self.max_iter:
            self.update()
            if return_all:
                x_all[:,:,self.iter] = self.x
            if return_errs:
                W2dist[self.iter] = ot.emd2_1d(np.reshape(self.x,(-1,)), np.reshape(x_comp,(-1,)))
        
        x_return = x_all if return_all else self.x
        if return_errs:
            return x_return, W2dist
        else:
            return x_return
    
    def update(self):
        self.iter = self.iter + 1
        xi = self.rng.normal(loc=0, scale=1, size=(self.d, self.n_samples))
        tau = self.tau[self.iter-1]
        epsilon = self.epsilon[self.iter-1]
        y = self.x - tau * self.dFx + np.sqrt(2*tau) * xi
        self.x = self.inexact_proxG(y, gamma = tau, epsilon = epsilon, maxiter = 1e2, verbose=False)
        self.dFx = self.dF(self.x)
