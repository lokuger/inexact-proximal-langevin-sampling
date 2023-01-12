import numpy as np
import potentials as pot
from scipy.special import erf
    
class L2Loss_SparsityReg():
    """
    Represents posterior for an L2/Gaussian data loss together with an L1 
    regularization term.
    l2shift is the mean, l2scale the standard deviation in the l2 loss term
    l1reg is the L1-regularization parameter
    """
    def __init__(self, d, l2shift, l2scale, mu_l1):
        if d is None and l2shift is None:
            raise ValueError("Please supply dimension or mean from which dimension can be inferred!")
        self.d = d if d is not None else l2shift.shape[0]
        self.l2shift = l2shift if l2shift is not None else np.zeros((d,1))
        self.l2scale = l2scale if l2scale is not None else 1
        self.mu_l1 = mu_l1 if mu_l1 is not None else 1
        
        self.f = pot.L2loss_homoschedastic(d=self.d, mu=self.l2shift, sigma=self.l2scale) 
        self.g = pot.L1loss_scaled(d=self.d, mu=np.zeros((self.d,1)), scale=self.mu_l1) if self.mu_l1 > 0 else pot.Zero()
    
    def unscaled_pdf(self, x):
        return np.exp(-self.f(x)-self.g(x))
    
class l2_denoise_tv():
    """
    Represents posterior for an L2/Gaussian data loss together with a TV 
    regularization term scaled by a parameter mu_TV
    Potential:
        
        V(u) = F(u) + G(u) with
        
        F(u) = 1/(2*sigma^2) * ||u - y||_2^2,
        G(u) = mu * TV(u)
    
    __init__ input parameters:
    n1, n2:     dimensions of image
    y:          shift y in the l2-loss, shape (n1,n2)
    noise_std:  standard deviation sigma of the l2-loss (noise model: 
                homoschedastic errors with variance sigma^2)
    mu_tv:      TV regularization parameter
    """
    def __init__(self, n1, n2, y, noise_std=1, mu_tv=1):
        self.d = n1*n2
        self.n1 = n1
        self.n2 = n2
        self.y = y
        self.noise_std = noise_std
        self.mu_tv = mu_tv
        
        self.f = pot.l2_loss_homoschedastic(d=self.d, y=self.y, sigma=noise_std)
        self.g = pot.total_variation(self.n1, self.n2, mu_tv) if self.mu_tv > 0 else pot.Zero()
    
    def pdf(self, x):
        raise NotImplementedError("Cannot compute the correct pdf because normalization constant is unknown. Please use L2Loss_TVReg.unscaled_pdf()")
        
    def unscaled_pdf(self, x):
        return np.exp(-self.f(x)-self.g(x))
    
class l2_deblur_tv():
    """
    Represents posterior for an L2/Gaussian deblurring data loss together with 
    a TV regularization term scaled by a parameter mu_TV
    Potential:
        
        V(u) = F(u) + G(u) with
        
        F(u) = 1/(2*sigma^2) * ||A*u - v||_2^2,
        G(u) = mu * TV(u)
    
    Instantiation input:
    n1, n2:     dimensions of image
    a, at:      blur operator and its transpose
    y:          right-hand-side data term in the l2-loss
    noise_std:  standard deviation sigma of the l2-loss (noise model: 
                homoschedastic errors with variance sigma^2)
    mu_tv:      TV regularization parameter
    """
    def __init__(self, n1, n2, a, at, y, noise_std=1, mu_tv=1):
        self.d = n1*n2
        self.n1 = n1
        self.a = a
        self.at = at
        self.n2 = n2
        self.y = y
        self.noise_std = noise_std
        self.mu_tv = mu_tv
        
        self.f = pot.l2_loss_reconstruction_homoschedastic(im_shape=(n1,n2), y=y, sigma2=noise_std**2, a=a,at=at)
        self.g = pot.total_variation(n1,n2,mu_tv) if mu_tv > 0 else pot.zero()
    
    def pdf(self, x):
        raise NotImplementedError("Cannot compute the correct pdf because normalization constant is unknown. Please use .unscaled_pdf(x)")
        
    def unscaled_pdf(self, x):
        """ compute the unscaled density, assuming x is n x n """
        return np.exp(-self.f(x)-self.g(x))
    
    
class MY_L2Loss_SparsityReg():
    """
    Represents posterior for an L2/Gaussian data loss together with an L1 regularization term.
    l2shift is the mean, l2scale the standard deviation of the l2 loss term
    l1reg is the L1-regularization parameter
    """
    def __init__(self, d, l2shift=None, l2scale=1, l1reg=1, gamma=1):
        if d is None and l2shift is None:
            raise ValueError("Please supply dimension or mean from which dimension can be inferred!")
        self.d = d if d is not None else l2shift.shape[0]
        self.l2shift = l2shift if l2shift is not None else np.zeros((self.d,1))
        self.l2scale = l2scale
        self.l1reg = l1reg
        self.gamma = gamma
        
        self.f = pot.L2norm(d=self.d, mu=self.l2shift, Var=self.l2scale**2*np.eye(d))
        if self.l1reg==0:
            self.g = pot.Zero()
        else:
            self.g = pot.MY_L1norm(d=self.d, gamma=self.gamma, mu=np.zeros((self.d,1)), b=1/self.l1reg)
        # self.z cannot be computed in higher-dimensional settings here
    
    def pdf(self, x):
        raise NotImplementedError("Cannot compute the correct pdf because normalization constant is unknown. Please use L2Loss_SparsityReg.unscaled_pdf()")
        
    def unscaled_pdf(self, x):
        return np.exp(-self.f(x)-self.g(x))
    

class Normal():
    """
    d-dimensional normal distribution with mean mu and covariance matrix Var.
    """
    def __init__(self, d = None, mu = None, Var = None):
        if d is None and mu is None and Var is None:
            raise ValueError("Please supply dimension or parameters from which dimension can be inferred!")
        
        # infer dimension by parameters
        self.d = d if d is not None else [mu.shape[0] if mu is not None else Var.shape[0]]
                
        # set distribution parameters
        self.mu = mu if mu is not None else np.zeros((self.d,1))
        self.Var = Var if Var is not None else np.eye(self.d)
        self.Prec = np.linalg.inv(self.Var)
        
        # potential and density normalizing constant
        self.z = np.sqrt(np.linalg.det(2*np.pi*self.Var))
        self.f = pot.L2norm(d=self.d, mu=self.mu, Var=self.Var)
        self.g = pot.Zero()
    
    def pdf(self, x):
        return 1/self.z * np.exp(-self.f(x))


class Laplace():
    """ 
    note that this is not really a Laplace distribution,
    but rather the distribution defined by the density proportional to
    exp(- ||x||_1 / b)
    Its marginals are zero-mean Laplace distributions with parameter b
    """
    def __init__(self, d = None, mu = None, b = None):
        if d is None and mu is None:
            raise ValueError("Please supply dimension or mean from which dimension can be inferred!")
        
        # infer dimension by parameters
        self.d = d if d is not None else mu.shape[0]
        
        # set distribution parameters
        self.mu = mu if mu is not None else np.zeros((self.d,1))
        self.b = b if b is not None else 1
        
        # potential and density normalizing constant
        self.z = (2*self.b)**self.d
        self.f = pot.Zero()
        self.g = pot.L1norm(d=self.d, mu=self.mu, b=self.b)
    
    def pdf(self, x):
        return 1/self.z * np.exp(-self.g(x))
    
class MY_Laplace1D():
    """ 
    MY-regularization of the 1D Laplace-distribution. 
    The subgradient of L1norm is approximated by soft shrinkage here. Is used 
    to visualize the effect of smoothing the potential in algorithms like MYULA.
    """
    def __init__(self, gamma = 1, mu = None, b = None):
        self.d = 1
        self.mu = mu if mu is not None else 0
        self.b = b if b is not None else 1
        self.gamma = gamma
        
        #potential and density normalizing constant
        self.z = 2*np.exp(-self.gamma/2) + np.sqrt(2*np.pi*self.gamma)*erf(np.sqrt(self.gamma/2))
        self.f = pot.MY_L1norm(d=self.d, gamma = self.gamma, mu = self.mu, b = self.b)
        self.g = pot.Zero()
    
    def pdf(self, x):
        return 1/self.z * np.exp(-self.f(x))
    
    
class Gamma_Gauss1D_Posterior_Salim21():
    """
    This is a numerical example from Salim, Richtárik; 2021. 
    The distribution is the posterior for mean estimation in a set of 
    Gaussian points with a Gamma distribution as the prior (constrains the 
    samples to positive numbers)
    data must be an array containing  Gaussian data points for the likelihood function
    nu > 2 is the parameter steering the shape of the Gamma distribution (prior)
    """
    def __init__(self, data, scale, nu):
        self.d = 1
        self.data = np.reshape(data,(-1,1))
        self.n = self.data.shape[0]
        self.scale = scale if scale is not None else 1
        self.nu = nu
        
        self.f = pot.L2norm_linear_transform(d = self.n, mu = self.data, sigma2 = self.scale**2, K = np.ones((self.n,1)))
        self.g = pot.Log_Gamma1D(alpha = self.nu/2, beta = 0.5)
        t = np.reshape(np.linspace(1e-10,5*self.nu+np.mean(self.data),5000),(1,-1))
        unscaled_dens_vals = np.exp(-self.f(t)-self.g(t))
        self.z = np.trapz(unscaled_dens_vals, x=t)
    
    def pdf(self, x):
        return 1/self.z * np.exp(-self.f(x)-self.g(x))
    
class MY_Gamma_Gauss1D_Posterior_Salim21():
    """
    MY-regularized version of the above. 
    Is only used in order to plot the target density of MYULA applied to the above.
    """
    def __init__(self, data, scale, nu, gamma=1):
        self.d = 1
        self.data = np.reshape(data,(-1,1))
        self.n = self.data.shape[0]
        self.nu = nu
        self.gamma = gamma
        self.scale = scale
        
        self.f = pot.L2norm_linear_transform(d = self.n, mu = self.data, sigma2 = self.scale**2, K = np.ones((self.n,1)))
        self.g = pot.MY_Log_Gamma1D(alpha=self.nu/2, beta=0.5, gamma=gamma)
        t = np.reshape(np.linspace(-5*self.nu-np.mean(self.data),5*self.nu+np.mean(self.data),5000),(1,-1))
        unscaled_dens_vals = np.exp(-self.f(t)-self.g(t))
        self.z = np.trapz(unscaled_dens_vals, x=t)
    
    def pdf(self, x):
        return 1/self.z * np.exp(-self.f(x)-self.g(x))
    
    
class Poisson():
    """
    An extended Poisson distribution in the sense that we allow a linearly transformed input 
    (this is an instance of a generalized linear model)
    The observations g are drawn from Pois(Ku+b), so the potential terms are
    F(u) = KL(Ku+b, v) with KL the Kullback-Leibler distance
    G(u) a positivity-enforcing indicator
    """
    def __init__(self, data = None, background = None, K = None):
        self.v = data
        self.m = self.v.shape[0]
        self.d = K.shape[1] if K is not None else self.v.shape[0]
        self.b = background if background is not None else np.zeros((self.m,1))
        self.K = K if K is not None else np.eye(self.d)
        
        self.f = pot.KLDistance(data=self.v, background=self.b, K=self.K)
        self.g = pot.nonNegIndicator(d=self.d)
        
    def pmf(self, x):
        raise NotImplementedError('Too lazy until now, sorry')

class Custom_distribution():
    """
    This class can be used to create an arbitrary distribution by directly inserting the terms of the potential.
    
    Additionally, give parameters a and b so that 'most' of the pdf lies inside [a,b]. 
    This is used to compute an approximation to the normalization constant 
    Z = \int_R exp(-F(x)-G(x)) dx,
    so that the pdf can be plotted.
    """
    def __init__(self, F, G):
        self.f = F
        self.g = G