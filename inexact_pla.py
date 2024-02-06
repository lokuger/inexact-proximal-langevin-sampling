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
        - pd            : probability distribution, object of distributions
        Optional parameters
        - step-size             : has to be None (then we try to use 1/L if L is specified) or a tuple declaring the step size choice. step_size[0] must be one of 
                                    - 'fxd' (then step_size[1] must be the fixed value)
                                    - 'fun' (then step_size[1] must be a function of the iteration number, used for decreasing choices of step size)
                                    - 'bt' (for backtracking, then step_size[1] is used as an initial guess)
        - rng (default_rng())   : random number generator for reproducibility, init new one if None
        - epsilon_prox (1e-2)   : prox accuracy, either a scalar or a handle of n as epsilon(n). If epsilon == 0, equivalent to setting exact=True
        - iter_prox (np.Inf)    : number of iterations for prox, can be given as alternative to epsilon
        - efficient (True)      : if True, do not save iterates but only current iterate, running mean and std of samples. If False, save all iterates
        - exact (False)         : if pd.g has an exact proximal operator, can choose True and run exact PGLA
        
    """
    def __init__(self, x0, n_iter, burnin, pd, step_size=None, rng=None, epsilon_prox=1e-2, iter_prox=np.Inf, callback=None, exact=False, stop_crit=None):
        self.n_iter = n_iter
        self.burnin = burnin
        self.iter = 0
        self.rng = rng if rng is not None else default_rng()    # for reproducibility allow to pass rng
        if step_size is None:
            self.step_type = 'fxd'
            self.step_size = 1/pd.f.l
        else:
            self.step_type = step_size[0]
            if self.step_type == 'fxd':
                self.step_size = step_size[1]
            elif self.step_type == 'fun':
                self.step_fun = step_size[1]
            elif self.step_type == 'bt':
                self.tau_old = step_size[1]
        
        self.shape_x = x0.shape
        self.x = np.copy(x0)
        
        self.stop_crit = stop_crit          # should be an executable with inexact_pgla object as only argument

        self.callback = callback
        
        # iteration parameters
        self.f = pd.f
        self.df = pd.f.grad
        self.dfx = self.df(self.x) if self.eff else self.df(self.x[...,0])
        self.g = pd.g
        self.exact = exact
        if self.exact or epsilon_prox == 0:
            self.exact = True
            self.prox_g = pd.g.prox
        else:
            self.inexact_prox_g = pd.g.inexact_prox
            self.epsilon_prox = (lambda n : epsilon_prox) if np.isscalar(epsilon_prox) else epsilon_prox
            self.iter_prox = iter_prox
        
        self.num_prox_its = 0
    
    def simulate(self, verbose=False):
        if verbose: sys.stdout.write('run inexact PLA: {:3d}% '.format(0)); sys.stdout.flush()
        while not stop:
            # update step
            self.update()

            # callback actions, e.g. plotting the iterate, updating a running mean, etc
            if self.callback is not None:
                self.callback(self)

            if verbose and self.iter%100==0: 
                progress = int(self.iter/self.n_iter*100)
                sys.stdout.write('\b'*5 + '{:3d}% '.format(progress))
                sys.stdout.flush()
            stop = (self.iter == self.n_iter) if (self.stop_crit is None) else ((self.iter == self.n_iter) or (self.stop_crit(self)))
        
        # adjust a few variables if stopping criterion was met before maximum number of iterations
        if self.iter < self.n_iter :
            self.n_iter = self.iter
            self.logpi_vals = self.logpi_vals[:self.iter]
            self.num_prox_its = self.num_prox_its[:self.iter]
            if verbose > 0: sys.stdout.write('\b'*5 + 'Stopping criterion satisfied at iteration {}. 100%\n'.format(self.iter)); sys.stdout.flush()
        else:
            if verbose > 0: sys.stdout.write('\b'*5 + '100%\n'); sys.stdout.flush()
        
    def update(self):
        self.iter = self.iter + 1
        xi = self.rng.normal(size=self.shape_x)

        # set step size
        if self.step_type == 'fxd':
            tau = self.step_size
        elif self.step_type == 'fun':
            tau = self.step_size(self.iter-1)
        elif self.step_type == 'bt':
            tau = 2*self.tau_old
            m = -0.5*np.sum(self.dfx**2)
            fx = self.f(self.x)
            gamma = 0.7
            while True:
                z = self.x-tau*self.dfx
                if self.f(z) > fx + tau*m:  # armijo-goldstein cond.
                    tau *= gamma
                else:                       # accept step size
                    self.tau_old = tau
                    break

        # set inexactness level
        if not self.exact:
            epsilon_prox = self.epsilon_prox(self.iter) if self.epsilon_prox is not None else None
            iter_prox = self.iter_prox

        prox_g = lambda u, tau: (self.prox_g(u,tau) if self.exact else self.inexact_prox_g(u, tau, epsilon=epsilon_prox, max_iter=iter_prox))

        # central update here:
        # note that res has variable number of elements, since the inexact prox routines should also output the number of iterations for diagnostics
        z = z if (self.step_type == 'bt') else self.x-tau*self.dfx   # gradient step
        res = prox_g(z+np.sqrt(2*tau)*xi, tau)

        # assign output correctly, compute log-density at sample
        (self.x,num_prox_its) = (res,0) if self.exact else res
        self.dfx = self.df(self.x)
        self.logpi_vals[self.iter-1] = self.f(self.x) + self.g(self.x)
        
        self.num_prox_its[self.iter-1] = num_prox_its
        if self.iter > self.burnin:
            self.num_prox_its_total += num_prox_its
        