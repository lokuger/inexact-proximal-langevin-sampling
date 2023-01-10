import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import ot

class PSGLA():
    def __init__(self, max_iter, tau, x0, pd):
        # we're computing samples in self.x with shape (d, n_samples)
        # expect input x0 to be (d, n_samples) as well
        # step size tau is held constant if backtracking = None, otherwise this is only the initial step size
        self.max_iter = max_iter
        self.iter = 0
        self.T = 0
        self.d, self.n_samples = x0.shape
        self.x0 = np.copy(x0)
        self.x = np.copy(x0)
        self.dF = pd.F.grad
        self.dFx = self.dF(self.x)
        self.F = pd.F
        self.proxG = pd.G.prox
        self.rng = default_rng()
        if tau > 0:
            self.tau = tau
            self.bt = None
        else:
            # negative step size is used to indicate backtracking
            # compute and store function value/norm of grad to speed up the backtracking
            self.Fx = self.F(self.x)
            self.dFx_squared_norm = np.sum(self.dFx ** 2, axis = 0)
            self.beta = 0.8
            if tau == -1:
                self.bt = 'risky'
            elif tau == -2: 
                self.bt = 'safe'
            else:
                raise(ValueError('Set step size tau to positive value or -1/-2 for backtracking choices.'))
        self.taus = tau * np.ones((self.max_iter,))
        self.sum_taus = np.zeros((self.max_iter,))
    
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
        if self.bt is not None:
            self.tau = self.computeStepsize()
            self.taus[self.iter-1] = self.tau
        y = self.x - self.tau * self.dFx + np.sqrt(2*self.tau) * xi
        self.x = self.proxG(y, gamma=self.tau)
        self.dFx = self.dF(self.x)
        self.T = self.T + self.tau
        self.sum_taus[self.iter-1] = self.T
    
    def computeStepsize(self):
        self.Fx = self.F(self.x)
        self.dFx_squared_norm = np.sum(self.dFx ** 2, axis = 0)
        tau_attempt = 1
        while True:
            x_attempt = self.x - tau_attempt * self.dFx
            D = self.F(x_attempt) - self.Fx + tau_attempt/2 * self.dFx_squared_norm
            is_valid_tau = (np.mean(D) <= 0) if self.bt == 'risky' else np.max(D) <= 0
            if is_valid_tau:
                return tau_attempt
            else:
                tau_attempt = self.beta * tau_attempt
                
