import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import ot

class PDLA():
    def __init__(self, max_iter, sigma, tau, x0, y0, F, G, D, K):
        self.max_iter = max_iter
        self.iter = 0
        self.sigma = sigma
        self.tau = tau
        self.x0 = np.copy(x0)
        self.y0 = np.copy(y0)
        self.x = np.copy(x0)
        self.xold = np.copy(x0)
        self.y = np.copy(y0)
        assert x0.shape[1]==y0.shape[1], 'Primal and dual variables should have equal number of samples'
        self.d, self.n_samples = x0.shape
        self.m = y0.shape[0]
        self.F = F
        self.G = G
        self.D = D
        self.K = K
        # very careful here, depends on splitting which term should be F and which G
        self.proxDconj = self.D.conjProx
        self.proxG = self.G.prox
        self.gradF = self.F.grad
        self.rng = default_rng()

    def simulate(self, x_comp = None, return_all = False):
        return_errs = False
        if x_comp is not None:
            assert(x_comp.shape[0]==self.x.shape[0],'Comparison samples must have the same dimension as running samples')
            return_errs = True
            W2dist = np.zeros((self.max_iter+1,))
            W2dist[0] = ot.emd2_1d(np.reshape(self.x0,(-1,)), np.reshape(x_comp,(-1,)))
        if return_all:
            x_all = np.zeros((self.d, self.n_samples, self.max_iter+1))
            y_all = np.zeros((self.m, self.n_samples, self.max_iter+1))
            x_all[:,:,0] = self.x0
            y_all[:,:,0] = self.y0
            
        while self.iter < self.max_iter:
            self.update()
            if return_all:
                x_all[:,:,self.iter] = self.x
                y_all[:,:,self.iter] = self.y
            # compute W2 distance between x and samples from the a comparison distribution x_comp
            if return_errs:
                W2dist[self.iter] = ot.emd2_1d(np.reshape(self.x,(-1,)), np.reshape(x_comp,(-1,)))
        
        x_return = x_all if return_all else self.x
        y_return = y_all if return_all else self.y
        if return_errs:
            return x_return, y_return, W2dist
        else:
            return x_return, y_return

    def update(self):
        self.iter = self.iter + 1
        xi = self.rng.normal(loc=0, scale=1, size=(self.d, self.n_samples))
        
        x_relax = 2*self.x - self.xold
        self.xold = self.x
        self.y = self.proxDconj(self.y + self.sigma*(self.K @ x_relax), self.sigma)
        
        y_relax = self.y
        self.x = self.proxG(self.x - self.tau*(self.K.T @ y_relax) - self.tau*self.gradF(self.x), self.tau) + np.sqrt(2*self.tau)*xi
        
        
        