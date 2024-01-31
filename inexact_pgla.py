import numpy as np
from numpy.random import default_rng
import sys
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean

class inexact_pgla():
    """
    Instance of the inexact proximal Langevin algorithm
    __init__ parameters:
        - x0            : chain initialization, assumed to be an image, shape n1,n2
        - n_iter        : number of samples to be drawn
        - burnin        : number of burnin samples to throw away
        - pd            : probability distribution, object of distributions.py3q
        Optional parameters
        - step-size             : has to be None (then we try to use 1/L if L is specified) or a tuple declaring the step size choice. step_size[0] must be one of 
                                    - 'fxd' (then step_size[1] must be the fixed value)
                                    - 'fun' (then step_size[1] must be a function of the iteration number, used for decreasing choices of step size)
                                    - 'bt' (for backtracking, then step_size[1] is used as an initial guess)
        - rng (default_rng())   : random number generator for reproducibility, init new one if None
        - epsilon_prox (1e-2)   : prox accuracy, either a scalar or a handle of n as epsilon(n). If epsilon == 0, equivalent to setting exact=True
        - iter_prox (np.Inf)    : number of iterations for prox, can be given as alternative to epsilon
        - efficient (True)      : if True, do not save iterates but only current iterate, running mean and std of samples
        - exact (False)         : if pd.g has an exact proximal operator, can choose True and run exact PGLA
        
    """
    def __init__(self, x0, n_iter, burnin, pd, step_size=None, rng=None, epsilon_prox=1e-2, iter_prox=np.Inf, efficient=False, output_iterates=None, output_means=None, exact=False, stop_crit=None, downsampling_scales=None):
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
                self.tau_all = np.zeros((n_iter,))
        
        self.shape_x = x0.shape
        self.eff = efficient
        if self.eff:
            # sample at current step
            self.x = np.copy(x0)

            # running sum and sum of squares to compute mean & std estimates
            self.sum = np.zeros(self.shape_x)
            self.sum_sq = np.zeros(self.shape_x)

            # running sum and sum of squares of downsampled images. downsampling_scales contains the scaling factors
            self.downsampling_scales = downsampling_scales
            if self.downsampling_scales is not None:
                sz = lambda s : tuple([int(np.ceil(n/s)) for n in self.x.shape])
                self.sum_scaled = [np.zeros(sz(s)) for s in self.downsampling_scales]
                self.sum_sq_scaled = [np.zeros(sz(s)) for s in self.downsampling_scales]
                self.mean_scaled = [np.zeros(sz(s)) for s in self.downsampling_scales]
                self.var_scaled = [np.zeros(sz(s)) for s in self.downsampling_scales]
                self.std_scaled = [np.zeros(sz(s)) for s in self.downsampling_scales]

            # in efficient mode, there is the option to output a selected number of iterates or running means at given indices
            # self.I contains the indices of returned iterates
            if output_iterates is not None:
                self.I_output_iterates = output_iterates
            else:
                self.I_output_iterates = np.reshape(self.n_iter,(1,))   # output last sample if nothing specified
            n_outputs = np.size(self.I_output_iterates)
            self.output_iterates = np.zeros(self.shape_x+(n_outputs,))
            if output_means is not None:
                self.I_output_means = output_means + burnin
            else:
                self.I_output_means = np.reshape(self.n_iter,(1,))   # output last running mean if nothing specified
            n_means = np.size(self.I_output_means)
            self.output_means = np.zeros(self.shape_x+(n_means,))
            
            self.stop_crit = stop_crit          # should be an executable with inexact_pgla object as only argument
        else:
            self.x = np.zeros(self.shape_x+(self.n_iter+1,))
            self.x[...,0] = x0
            self.stop_crit = stop_crit
        
        # iteration parameters
        self.f = pd.f
        self.df = pd.f.grad
        self.dfx = self.df(self.x) if self.eff else self.df(self.x[...,0])
        self.g = pd.g
        self.exact = exact
        if self.exact:
            self.prox_g = pd.g.prox
        else:
            if epsilon_prox == 0:
                self.exact = True
                self.prox_g = pd.g.prox
            else:
                self.inexact_prox_g = pd.g.inexact_prox
                self.epsilon_prox = (lambda n : epsilon_prox) if np.isscalar(epsilon_prox) else epsilon_prox
                self.iter_prox = iter_prox
        
        # diagnostic checks
        self.logpi_vals = np.zeros((self.n_iter,))
        self.num_prox_its = np.zeros((self.n_iter,))
        self.num_prox_its_total = 0
    
    def simulate(self, verbose=False):
        if verbose: sys.stdout.write('run inexact PLA: {:3d}% '.format(0)); sys.stdout.flush()
        i,j,stop = 0,0,False
        while not stop:
            # update step
            self.update()
            # potentially save iterate
            if self.eff and self.iter in self.I_output_iterates:
                self.output_iterates[...,i] = self.x
                i+=1
            # potentially save running mean
            if self.eff and self.iter in self.I_output_means:
                self.output_means[...,j] = self.sum/(self.iter-self.burnin)
                j+=1
            if verbose and self.iter%1==0: 
                progress = int(self.iter/self.n_iter*100)
                sys.stdout.write('\b'*5 + '{:3d}% '.format(progress))
                sys.stdout.flush()
            stop = (self.iter == self.n_iter) if (self.stop_crit is None) else ((self.iter == self.n_iter) or (self.stop_crit(self)))
        
        # adjust a few variables if stopping criterion was met before maximum number of iterations
        if self.iter < self.n_iter :
            self.n_iter = self.iter
            self.logpi_vals = self.logpi_vals[:self.iter]
            self.num_prox_its = self.num_prox_its[:self.iter]
            if self.eff:
                # remove output samples and means at iterations after the stopping criterion
                self.I_output_iterates = self.I_output_iterates[self.I_output_iterates < self.n_iter+1]
                self.output_iterates = self.output_iterates[...,:i]
                self.I_output_means = self.I_output_means[self.I_output_means < self.n_iter+1]
                self.output_means = self.output_means[...,:j]
            else:
                self.x = self.x[...,0:self.n_iter+1]
            if verbose > 0: sys.stdout.write('\b'*5 + 'Stopping criterion satisfied at iteration {}. 100%\n'.format(self.iter)); sys.stdout.flush()
        else:
            if verbose > 0: sys.stdout.write('\b'*5 + '100%\n'); sys.stdout.flush()
        
        # compute running mean and standard deviation of Markov chain after iteration
        if self.eff:
            N = self.n_iter-self.burnin
            self.mean = self.sum/N
            self.var = (self.sum_sq - (self.sum**2)/N)/(N-1) if N>1 else np.NAN
            self.std = np.sqrt(self.var)
            for i,_ in enumerate(self.downsampling_scales):
                self.mean_scaled[i] = self.sum_scaled[i]/N
                self.var_scaled[i] = (self.sum_sq_scaled[i] - (self.sum_scaled[i]**2)/N)/(N-1) if N>1 else np.NAN
                self.std_scaled[i] = np.sqrt(self.var_scaled[i])
        else:
            self.x = self.x[...,self.burnin+1:]
            self.mean = np.mean(self.x,axis=-1)
            self.std = np.std(self.x,axis=-1)
            self.var = self.std**2
        
    def update(self):
        self.iter = self.iter + 1
        xi = self.rng.normal(size=self.shape_x)
        x = self.x if self.eff else self.x[...,self.iter-1]

        # set step size
        if self.step_type == 'fxd':
            tau = self.step_size
        elif self.step_type == 'fun':
            tau = self.step_size(self.iter-1)
        elif self.step_type == 'bt':
            tau = 2*self.tau_old
            m = -0.5*np.sum(self.dfx**2)
            fx = self.f(x)
            gamma = 0.7
            while True:
                z = x-tau*self.dfx
                if self.f(z) > fx + tau*m:  # armijo-goldstein cond.
                    tau *= gamma
                else:                       # accept step size
                    self.tau_old,self.tau_all[self.iter-1] = tau,tau
                    break

        # set inexactness level
        if not self.exact:
            epsilon_prox = self.epsilon_prox(self.iter) if self.epsilon_prox is not None else None
            iter_prox = self.iter_prox

        prox_g = (lambda u, tau: self.prox_g(u,tau)) if self.exact else (lambda u, tau: self.inexact_prox_g(u, tau, epsilon=epsilon_prox, max_iter=iter_prox))

        # central update here:
        # note that res has variable number of elements, since the inexact prox routines should also output the number of iterations for diagnostics
        z = z if (self.step_type == 'bt') else x-tau*self.dfx   # gradient step
        res = prox_g(z+np.sqrt(2*tau)*xi, tau)

        # assign output correctly, compute log-density at sample and running mean & std diagnostics
        (x_new,num_prox_its) = (res,0) if self.exact else res
        self.dfx = self.df(x_new)
        self.logpi_vals[self.iter-1] = self.f(x_new) + self.g(x_new)
        if self.eff:
            self.x = x_new
            if self.iter > self.burnin:
                # running mean and sum of squares
                self.sum = self.sum + self.x
                self.sum_sq = self.sum_sq + self.x**2

                # downsampled running sum and sum of squares
                if self.downsampling_scales is not None:
                    for i,scale in enumerate(self.downsampling_scales):
                        if len(self.x.shape) == 2:
                            xds = downscale_local_mean(self.x, scale)
                            self.sum_scaled[i] = self.sum_scaled[i] + xds
                            self.sum_sq_scaled[i] = self.sum_sq_scaled[i] + xds**2
                        else:
                            raise NotImplementedError('Downscaling only supported for images')
        else:
            self.x[...,self.iter] = x_new
        
        self.num_prox_its[self.iter-1] = num_prox_its
        if self.iter > self.burnin:
            self.num_prox_its_total += num_prox_its
        