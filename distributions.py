import numpy as np
import potentials as pot

    
class l2_denoise_tv():
    """
    Represents posterior for an L2/Gaussian data loss together with a TV 
    regularization term scaled by a parameter mu_TV potential:
        
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
        
        self.f = pot.l2_loss_homoschedastic(y=self.y, sigma2=noise_std**2)
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
    def __init__(self, n1, n2, a, at, max_ev_ata, y, noise_std=1, mu_tv=1):
        self.d = n1*n2
        self.n1 = n1
        self.n2 = n2
        self.a = a
        self.at = at
        self.max_ev_ata = max_ev_ata
        self.y = y
        self.noise_std = noise_std
        self.mu_tv = mu_tv
        
        self.f = pot.l2_loss_reconstruction_homoschedastic(y=self.y, sigma2=self.noise_std**2, a=self.a, at=self.at, max_ev_ata=self.max_ev_ata)
        self.g = pot.total_variation(n1,n2,self.mu_tv) if self.mu_tv > 0 else pot.zero()
    
    def pdf(self, x):
        raise NotImplementedError("Cannot compute the correct pdf because normalization constant is unknown. Please use .unscaled_pdf(x)")
        
    def unscaled_pdf(self, x):
        """ compute the unscaled density, assuming x is n x n """
        return np.exp(-self.f(x)-self.g(x))
    
class kl_deblur_tvnonneg_prior():
    """
    Represents posterior for a Kullback-Leibler as negative log-likelihood
    and a TV regularization term scaled by a parameter mu_TV.
    Potential:
        
        V(u) = F(u) + G(u) with
        
        F(u) = KL(A*u + b, y)
        G(u) = mu * TV(u) + 1_{R+}(u)
    
    Instantiation input:
    n1, n2:     dimensions of image
    a, at:      blur operator and its transpose
    y:          observed data in the Kullback-Leibler loss
    b:          estimated background parameter, of same size as y
    mu_tv:      TV regularization parameter
    """
    def __init__(self, n1, n2, a, at, max_ev_ata, y, b, mu_tv=1):
        self.d = n1*n2
        self.n1 = n1
        self.n2 = n2
        self.a = a
        self.at = at
        self.max_ev_ata = max_ev_ata
        self.y = y
        self.b = b
        assert(np.all(self.b.shape == self.y.shape))
        self.mu_tv = mu_tv
        
        self.f = pot.kl_divergence(self.y,self.b,self.a,self.at)
        self.g = pot.total_variation_nonneg(self.n1,self.n2,self.mu_tv)
    
    def pdf(self, x):
        raise NotImplementedError("Cannot compute the correct pdf because normalization constant is unknown. Please use .unscaled_pdf(x)")
        
    def unscaled_pdf(self, x):
        """ compute the unscaled density, assuming x is n x n """
        return np.exp(-self.f(x)-self.g(x))

class l2_l1prior():
    """
    Represents posterior for an L2/Gaussian data loss together with a 
    sparsity/l1 prior:
        
        V(u) = F(u) + G(u) with
        
        F(u) = 1/(2*sigma^2) * ||u - y||_2^2,
        G(u) = mu * ||u||_1
    
    __init__ input parameters:
    n1, n2:     dimensions of image
    y:          shift y in the l2-loss, shape (n1,n2)
    noise_std:  standard deviation sigma of the l2-loss (noise model: 
                homoschedastic errors with variance sigma^2)
    mu_l1:      l1 regularization parameter
    """
    def __init__(self, y, noise_std=1, mu_l1=1):
        self.y = y
        self.noise_std = noise_std
        self.mu_l1 = mu_l1
        
        self.f = pot.l2_loss_homoschedastic(y=self.y, sigma2=noise_std**2)
        self.g = pot.l1_loss_unshifted_homoschedastic(mu_l1) if self.mu_l1 > 0 else pot.Zero()
    
    def pdf(self, x):
        raise NotImplementedError("Cannot compute the correct pdf because normalization constant is unknown. Please use L2Loss_TVReg.unscaled_pdf()")
        
    def unscaled_pdf(self, x):
        return np.exp(-self.f(x)-self.g(x))
    
class l2_tikhprior():
    """
    Represents posterior for an L2/Gaussian data loss together with a 
    sparsity/l1 prior:
        
        V(u) = F(u) + G(u) with
        
        F(u) = 1/(2*sigma1^2) * ||u - mu1||_2^2,
        G(u) = 1/(2*sigma2^2) * ||u - mu2||_2^2
    
    __init__ input parameters:
    n1, n2:     dimensions of image
    mu1:        shift parameter of likelihood, shape (n1,n2)
    mu2:        shift parameter of prior, shape (n1,n2)
    sigma1:     standard deviation of likelihood
    sigma2:     standard deviation of prior
    """
    def __init__(self, mu1=0, mu2=0, sigma1=1, sigma2=1):
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        self.f = pot.l2_loss_homoschedastic(y=self.mu1, sigma2=self.sigma1**2)
        self.g = pot.l2_loss_homoschedastic(y=self.mu2, sigma2=self.sigma2**2)
    
    def pdf(self, x):
        raise NotImplementedError("Cannot compute the correct pdf because normalization constant is unknown. Please use L2Loss_TVReg.unscaled_pdf()")
        
    def unscaled_pdf(self, x):
        return np.exp(-self.f(x)-self.g(x))
    
class l2_deblur_l1prior():
    """
    Represents posterior for an L2/Gaussian data loss, where the data is 
    transformed by some operator a together with sparsity/l1 prior:
        
        V(x) = F(x) + G(x) with
        
        F(x) = 1/(2*sigma^2) * ||A*x - y||_2^2,
        G(x) = mu * ||x||_1
    
    __init__ input parameters:
    a, at:      blur operator A and its transpose
    y:          shift y in the l2-loss, shape (n1,n2)
    noise_std:  standard deviation sigma of the l2-loss (noise model: 
                homoschedastic errors with variance sigma^2)
    mu_l1:      l1 regularization parameter
    """
    def __init__(self, a, at, max_ev_ata, y, noise_std=1, mu_l1=1):
        self.a, self.at, self.max_ev_ata = a, at, max_ev_ata
        self.y = y
        self.noise_std = noise_std
        self.mu_l1 = mu_l1
        
        self.f = pot.l2_loss_reconstruction_homoschedastic(y=y, sigma2=noise_std**2, a=a, at=at, max_ev_ata=max_ev_ata)
        self.g = pot.l1_loss_unshifted_homoschedastic(mu_l1) if self.mu_l1 > 0 else pot.Zero()
    
    def pdf(self, x):
        raise NotImplementedError("Cannot compute the correct pdf because normalization constant is unknown. Please use L2Loss_TVReg.unscaled_pdf()")
        
    def unscaled_pdf(self, x):
        return np.exp(-self.f(x)-self.g(x))
