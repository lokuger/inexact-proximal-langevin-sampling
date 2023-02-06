import numpy as np
import warnings
import sys
    
class l2_loss_homoschedastic():
    """
    a simplified form of the l2_loss_heteroschedastic class that is faster for 
    the (very common) special case that the variance matrix is sigma^2*I
    
    F(x) = 1/(2*sigma^2) * ||x - y||_2^2
    
    __init__ input parameters:
        - y:        right hand side data
        - sigma2:   variance of Gaussian noise
    
    __call__ input parameter:
        - x:        image of shape self.im_shape
    """
    def __init__(self, y, sigma2):
        self.y = y
        self.sigma2 = sigma2
    
    def __call__(self, x):
        return 1/(2*self.sigma2) * np.sum((x-self.y)**2)
    
    def grad(self, x):
        return 1/self.sigma2 * (x-self.y)
    
    def prox(self, x, gamma):
        return 1/(self.sigma2+gamma)*(self.sigma2*x+gamma*self.y)
    
    def conj(self, x):
        return self.sigma2/2 * np.sum(x**2) + np.sum(x * self.y)
    
    def conj_prox(self, x, gamma = 1):
        return 1/(1+gamma*self.sigma2)*(x-gamma*self.y)
    
class l2_loss_reconstruction_homoschedastic():
    """
    A variant of l2_loss_homoschedastic which allows a linear transformation 
    of the input image
    Necessary for all indirect models, like deblurring, tomography etc. K is
    the forward operator of the associated inverse problem.
    
    F(x) = 1/(2*sigma^2) * ||Ax - y||_2^2
    
    Inputs:
        - y:        right hand side data
        - sigma2:   variance of Gaussian noise
        - a:        forward operator A, given as callable a(x)
        - at:       transpose/adjoint of A. Given as callable at(y)
    """
    def __init__(self, y, sigma2, a, at):
        self.y = y
        self.sigma2 = sigma2
        self.a = a
        self.at = at
    
    def __call__(self, x):
        """ 
        Computes 
            1/(2*sigma^2) ||a(x) - y||_2^2
        Make sure that shape of input x is correct for the application of x
        """
        return 1/(2*self.sigma2) * np.sum((self.a(x)-self.y)**2)
    
    def grad(self, x):
        return 1/self.sigma2 * self.at(self.a(x)-self.y)

class l1_loss_homoschedastic():
    """
    An l1 loss term corresponding to the likelihood of 
    
    F(x) = 1/b * ||x - y||_1
    
    Inputs: 
        - y:        data
        - b:        scale parameter. 
    The correspnding Laplace distribution Laplace(y,b) has variance 2*b^2
    """
    def __init__(self, y, b):
        self.y = y
        self.b = b
    
    def __call__(self, x):
        """
        Accepts only inputs of shape self.y.shape
        """
        return 1/self.b * np.sum(np.abs(x-self.y))
    
    def grad(self, x):
        raise NotImplementedError('L1 norm has no gradient')
        
    def prox(self, x, gamma):
        return self.y + np.maximum(0, np.abs(x-self.y)-gamma/self.b)*np.sign(x-self.y)

# class L1loss_scaled():
#     """
#     symbolizes the L1-norm, weighted by a parameter scale. Used for 
#     L1-regularization or the Laplace distribution
#     G(u) = scale * ||u - mu||_1
#     """
#     def __init__(self, d = None, mu = None, scale = 1):
#         if d is None and mu is None:
#             raise ValueError("Please supply dimension or parameters from which dimension can be inferred!")
        
#         # infer dimension by parameters
#         self.d = d if d is not None else mu.shape[0]
        
#         # set distribution parameters
#         self.mu = mu if mu is not None else np.zeros((self.d,1))
#         self.scale = scale
    
#     def __call__(self, x):
#         return np.sum(np.abs(x-self.mu),axis=0,keepdims=True)*self.scale
    
#     def grad(self, x):
#         raise NotImplementedError("L1norm does not have a gradient, use prox operator instead!")
    
#     def prox(self, x, gamma = 1):
#         return np.sign(x-self.mu) * np.maximum(0, np.abs(x-self.mu)-gamma*self.scale) + self.mu
    
#     def inexact_prox(self, x, gamma = 1, epsilon=None, maxiter=1e2, verbose=False):
#         """Careful, noticed that this implementation is only correct if 
#         self.mu = 0. Of course, this is the interesting case """
#         gamma *= self.scale
#         checkAccuracy = epsilon is not None
#         # iterative scheme to minimize the dual objective
#         y = np.zeros_like(x)  # solve for solution y of the dual using proximal gradient descent
#         stopcrit = False
#         tauprime = 0.99
#         tau = 1  # tau = 1 would be the most efficient but we want to make the routine artificially bad! Recheck this later.
#         i = 0
#         if checkAccuracy and verbose:
#             print('Run (backward) gradient descent on the dual Lasso with gamma*mu = {:.3e}'.format(gamma))
#             print('|{:^11s}|{:^31s}|'.format('Iterate','D-Gap (stop if < {:.3e})'.format(epsilon)))
#         while i < maxiter and not stopcrit:
#             i = i + 1
#             v = (1-tauprime)*y + tauprime*x
#             n = np.abs(v)
#             w = v/np.maximum(1,1/gamma * n)
#             # explicit gradient descent step on the Moreau-Yosida regularization: Gradient is y-w, see Chambolle, Pock 2016
#             y = y - tau*(y - w)
#             if checkAccuracy:
#                 halfsquarednormy = 1/2 * np.sum(y**2)
#                 # compute primal dual gap here and check if smaller than epsilon
#                 P = gamma * self(x-y)[0,0] + halfsquarednormy # primal value
#                 ndual = np.sqrt(y**2)
#                 dualInadmissible = np.any(ndual > gamma+1e-15)
#                 Pconj = np.Inf if dualInadmissible else halfsquarednormy - np.sum(y * x)
#                 dgap = P+Pconj
#                 stopcrit = dgap < epsilon
#                 if verbose and (i%25 == 0 or stopcrit):
#                     print('|{:^11d}|{:^31.3e}|'.format(i,dgap))
#         return x - y
        
# class KLDistance():
#     """
#     Kullback Leibler distance KL(Ku+b, v)
#     with background b,
#     data observation v,
#     linear transformation K.
#     """
#     def __init__(self, data = None, background = None, K = None):
#         self.v = data
#         self.m = self.v.shape[0]
#         self.d = K.shape[1] if K is not None else self.v.shape[0]
#         self.b = background if background is not None else np.zeros((self.m,1))
#         self.K = K if K is not None else np.eye(self.d)
        
#     def __call__(self, u):
#         Ku = self.K @ u
#         # be careful with this next line - it is here since the stepsize 
#         # backtracking naturally jumps too far and considers points u where 
#         # Ku < 0. We set these to the corresponding axis to prevent warnings.
#         Ku[Ku<0] = 0
#         # from here everything's fine again
#         Kub = Ku + self.b
#         return np.sum(Kub - self.v + self.v*np.log(self.v/Kub), axis=0)
    
#     def grad(self, u):
#         """
#         The KL distance is gradient Lipschitz with constant L; where the best (smallest)
#         L can be bounded by the easily computable constant
#         v_1/(b_1^2) * norm(K_1)^2 + ... + v_m/(b_m^2) * norm(K_m)^2
#         where K_i are the lines of the matrix K. See Obsidian notes for details
#         """
#         Kub = self.K @ u + self.b
#         return self.K.T @ (np.ones_like(self.v) - self.v/Kub)
    
class total_variation():
    """
    total variation of 2D image u with shape (n1, n2). Scaled by a constant
    regularization parameter scale. Corresponds to the functional 
        scale * TV(u)
    with u in R^{n1 x n2}
    
    __init__ input:
        - n1, n2:   shape of u
        - scale:    scaling factor, usually a regularization parameter
        
    __call__ input:
        - u:        image of shape n1,n2 or n1*n2,
                    
    TV is computed on a grid via finite differences, assuming equidistant 
    spacing of the grid. The gradient of this potential does not exist since 
    TV is not smooth.
    The proximal mapping is approximated using the dual problem. Pass either 
    a maximum number of steps, an accuracy (in the primal-dual gap), or both 
    to the prox evaluation, for more details see 
        total_variation.inexact_prox
        and
        total_variation._inexact_prox_singleImage_vanillaGD
        total_variation._inexact_prox_singleImage_acceleratedGD
    """
    def __init__(self, n1, n2, scale=1):
        self.n1 = n1
        self.n2 = n2
        self.scale = scale
        
    def _imgrad(self, u):
        """
        applies a 2D image gradient to the image u of shape (n1,n2)
        
        Parameters
        ----------
        u : numpy 2D array, shape n1, n2
            Image

        Returns
        -------
        (px,py) image gradients in x- and y-directions.

        """
        px = np.concatenate((u[1:,:]-u[0:-1,:], np.zeros((1,self.n2))),axis=0)
        py = np.concatenate((u[:,1:]-u[:,0:-1], np.zeros((self.n1,1))),axis=1)
        return np.concatenate((px[np.newaxis,:,:],py[np.newaxis,:,:]), axis=0)
    
    def _imdiv(self, p):
        """
        Computes the negative divergence of the 2D vector field px,py.
        can also be seen as a tensor from R^(n1xn2x2) to R^(n1xn2)

        Parameters
        ----------
            - p : 2 x n1 x n2 np.array

        Returns
        -------
            - divergence, n1 x n2 np.array
        """
        u1 = np.concatenate((-p[0,0:1,:], -(p[0,1:-1,:]-p[0,0:-2,:]), p[0,-2:-1,:]), axis = 0)
        u2 = np.concatenate((-p[1,:,0:1], -(p[1,:,1:-1]-p[1,:,0:-2]), p[1,:,-2:-1]), axis = 1)
        return u1+u2
    
    def __call__(self, u):
        """
        Computes the TV-seminorm of u
        
        Parameters 
        ----------
        u : numpy array of shape n1, n2
        
        Returns
        -------
        TV(u) (scalar)
        """
        return self.scale * np.sum(np.sqrt(np.sum(self._imgrad(u)**2,axis=0)))
    
    def inexact_prox(self, u, gamma, epsilon=None, max_iter=np.Inf, verbose=False):
        """
        Computing the prox of TV is solving the ROF model. See Chambolle, Pock 2016,
        Example 4.8 for a precise description of what is done here. We are 
        using accelerated proximal gradient descent on the dual ROF problem.
        -- gamma is the prox parameter, hence if the TV scaling parameter mu is 
        set to a value other than 1, solving this problem is equivalent to 
        computing a backward gradient step (=solving ROF) with step
            gamma * mu
            
        inexact_prox(self, u, gamma=1, epsilon=None, maxiter=np.Inf, verbose=False)
        parameters:
            - u:        image to be denoised, shape self.n1, self.n2
            - gamma:    prox step size
            - epsilon:  accuracy for duality gap stopping criterion
            - maxiter:  maximum number of iterations
            - verbose:  verbosity
        """
        if epsilon is None and max_iter is np.Inf:
            raise ValueError('provide either an accuracy or a maximum number of iterations to the tv prox please')
        checkAccuracy = True if epsilon is not None else False
        # iterative scheme to minimize the dual objective
        p = np.zeros((2,self.n1,self.n2))
        stopcrit = False
        tauprime = 0.99 * 1/8
        # tau_gd = 1
        i = 0
        tau_agd = 1
        t_agd = 0
        p_prev = np.copy(p)
        if verbose: sys.stdout.write('run AGD on dual ROF model: {:3d}% '.format(0)); sys.stdout.flush()
        # if checkAccuracy and verbose:
        #     print('Run (backward) accelerated gradient descent on the dual ROF problem with gamma = {:.3e}'.format(gamma*self.scale))
        #     print('|{:^11s}|{:^31s}|'.format('Iterate','D-Gap (stop if < {:.3e})'.format(epsilon)))
        while i < max_iter and not stopcrit:
            i = i + 1
            t_agd_new = (1+np.sqrt(1+4*t_agd**2))/2
            q = p + (t_agd-1)/t_agd_new * (p - p_prev)
            
            # compute the gradient of Moreau-Yosida regularization at q
            v = q - tauprime*self._imgrad(self._imdiv(q) - u)
            s = 1/np.maximum(1, np.sqrt(np.sum(v**2,axis=0))/(gamma*self.scale))[np.newaxis,:,:]
            w = v * s     # projection in dual norm
            # The gradient is q-w, see Chambolle, Pock 2016
            
            # updates
            t_agd = t_agd_new
            p_prev = np.copy(p)
            p = (1-tau_agd)*q + tau_agd*w
            
            # stopping criterion: check if primal-dual gap < epsilon
            if checkAccuracy:
                div_p = self._imdiv(p)
                h = 1/2 * np.sum(div_p**2)
                primal = gamma * self(u-div_p) + h
                norm_dual_iterate = np.sqrt(np.sum(p**2,axis=0))
                dual_inadmissible = np.any(norm_dual_iterate > gamma*self.scale+1e-12)
                dual = -np.Inf if dual_inadmissible else - h + np.sum(div_p * u) # dual value. dual iterate should never be inadmissible since we project in the end
                dgap = primal-dual
                stopcrit = dgap < epsilon
                if dgap < 0: # for debugging purpose
                    raise ValueError('Duality gap was negative (which should never happen), please check the prox computation routine!')
                if verbose: sys.stdout.write('\b'*5 + '{:3d}% '.format(int(i/max_iter*100))); sys.stdout.flush()
                # if verbose and (i%10 == 0 or stopcrit or i==max_iter):
                #     print('|{:^11d}|{:^31.3e}|'.format(i,dgap))
        return (u - self._imdiv(p)), i
        
    
class l2_l1_norm():
    """This class implements the norm
            ||p||_{2,1} = || sqrt(p1^2 + p2^2) ||_1
        where both p1 and p2 are of size n1,n2. With the horizontal and 
        vertical finite differences in D, this forms the TV norm as
            TV(u) = ||Du||_{2,1}
        The norm can be scaled by a parameter 'scale'
        
        __init__ input parameter:
            - n1,n2: image sizes
            - scale: scaling parameter
    """
    def __init__(self, n1, n2, scale=1):
        self.n1, self.n2 = n1, n2
        self.scale = scale
        
    def __call__(self, p):
        return self.scale * np.sum(np.sqrt(np.sum(p**2,axis=0)))
    
    def conj(self, p):
        if np.any(np.sqrt(np.sum(p**2,axis=0)) > self.scale+1e-12):
            return np.Inf
        else:
            return 0
        
    def conj_prox(self, p, tau): # does not depend on tau at all..?
        n = np.sqrt(np.sum(p**2,axis=0))[np.newaxis,...]
        return p/np.maximum(1,1/self.scale*n)

    
# class nonNegIndicator():
#     """ 
#     Indicator function of R_++^d, 0 if all entries are >= 0, otherwise infinity
#     """
#     def __init__(self, d):
#         self.d = d
    
#     def __call__(self, x):
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
#             return np.nan_to_num(np.infty * (1-(np.min(x, axis=0) >= 0)), 0)
        
#     def prox(self, x, gamma):
#         return np.maximum(x,0)
    
class zero():
    """ a dummy class if we want to set one of the potential terms to zero"""
    def __call__(self, x):
        return 0
    
    def grad(self, x):
        return np.zeros_like(x)
    
    def prox(self, x):
        return np.copy(x)
    
# class l2_loss_heteroschedastic():
#     """
#     symbolizes the weighted l2-norm for d-dimensional vectors. Is used to generate 
#     normal distributions and the data-fidelity/log-likelihood in problems with 
#     Gaussian noise.
#     """
#     def __init__(self, d = None, mu = None, Var = None):
#         if d is None and mu is None and Var is None:
#             raise ValueError("Please supply dimension or parameters from which dimension can be inferred!")
#         # infer dimension from parameters
#         self.d = d if d is not None else (mu.shape[0] if mu is not None else Var.shape[0])
               
#         # set distribution parameters
#         self.mu = mu if mu is not None else np.zeros((self.d,1))
#         if Var is None:
#             self.Var = np.eye(self.d)
#             self.Prec = np.eye(self.d)
#         elif np.isscalar(Var):
#             self.Var = Var*np.eye(self.d)
#             self.Prec = 1/Var*np.eye(self.d)
#         else:
#             self.Var = Var
#             self.Prec = np.linalg.inv(self.Var)
    
#     def __call__(self, x):
#         return 1/2 * np.sum((x-self.mu) * (self.Prec @ (x-self.mu)), axis=0)
    
#     def grad(self, x):
#         return self.Prec @ (x-self.mu)
    
#     def prox(self, x, gamma = 1):
#         return np.linalg.solve(self.Var+gamma*np.eye(self.d), self.Var@x + gamma*self.mu)
    
#     def conj(self, y):
#         return 1/2 * np.sum(y * self.Var @ y, axis=0) + np.sum(y * self.mu, axis=0)
    
#     def conjProx(self, x, gamma = 1):
#         return np.linalg.solve(np.eye(self.d)+gamma*self.Var, x-gamma*self.mu)
    
# class Log_Gamma1D():
#     """
#     negative Log-density of a 1D Gamma distribution with parameters
#     - shape: alpha > 0
#     - rate: beta > 0
#     """
#     def __init__(self, alpha = 1, beta = 0.5):
#         self.alpha = alpha
#         self.beta = beta
    
#     def __call__(self, x):
#         admissible_idcs = x>0
#         r = np.inf*np.ones_like(x)
#         r[admissible_idcs] = (1-self.alpha)*np.log(x[admissible_idcs]) + self.beta * x[admissible_idcs]
#         return r
    
#     def prox(self, x, gamma = 1):
#         return x/2 - self.beta*gamma/2 + np.sqrt(
#             1/4* (x**2) - x*self.beta*gamma/2 + (self.beta**2)*(gamma**2)/4 - gamma*(1-self.alpha))
    
#     def inexact_prox(self, x, gamma = 1, epsilon=None, maxiter=1e3):
#         """comment: we do actually know the exact prox (see above), 
#         this is only to compare inexact psgla and psgla
#         The inexact prox computes the prox by iteratively solving the dual
#         problem using gradient descent.
#         The dual problem is given by
#         - min_v {(gamma*v - x)^2/(2*gamma) - x^2/(2*gamma) + (alpha-1)*(log((alpha-1)/(beta-v)) - 1)}
#             =: - min_v Psi_gamma(v)
#         Once the dual solution v is approximated, the approximation to the 
#         prox evaluation is   x - gamma*v
#         Check the accuracy by bounding the duality gap Phi(y)+Psi(v)
#         """
#         checkAccuracy = True if epsilon is not None else False
#         # iterative scheme to minimize the dual objective
#         v = np.zeros_like(x)    # function is only defined on [-Inf, beta], beta > 0, hence initilizing at 0 might be safe
#         stopcrit = np.full(x.shape, False)
#         tau = 1/(gamma + (self.alpha-1)/(self.beta**2)) # since optimal v < 0 and initialize at 0, this is lower bound for 1/L
#         i = 0
#         while i < maxiter and not np.all(stopcrit):
#             i = i + 1
#             # update v
#             v[~stopcrit] = v[~stopcrit] - tau * (gamma*v[~stopcrit] - x[~stopcrit] + (1-self.alpha)/(v[~stopcrit]-self.beta))
#             if checkAccuracy:
#                 # compute primal dual gap here and check if smaller than epsilon
#                 P = self(x-gamma*v) + gamma/2 * v**2  # primal value
#                 Pconj = ((gamma*v-x)**2)/(2*gamma) - (x**2)/(2*gamma) + (self.alpha-1)*(np.log((self.alpha-1)/(self.beta-v)) - 1)
#                 stopcrit = P + Pconj < epsilon
#         return x - gamma*v
        

# class MY_Log_Gamma1D():
#     """
#     negative Log-density of a 1D Gamma distribution with parameters
#     - shape: alpha > 0
#     - rate: beta > 0
#     Only need this class to check whether the MYULA implementation works and 
#     converges to its (biased) target.
#     """
#     def __init__(self, alpha = 1, beta = 0.5, gamma = 1):
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.unsmoothed_pot = Log_Gamma1D(alpha, beta)
    
#     def __call__(self, x):
#         prox_vals = self.unsmoothed_pot.prox(x, self.gamma)
#         return self.unsmoothed_pot(prox_vals) + 1/(2*self.gamma)*np.sum((x-prox_vals)**2, axis=0)
    
#     def prox(self, x, gamma = 1):
#         raise NotImplementedError("Please don't try to compute the prox of an already MY-regularized function")
        
# class MY_L1loss_scaled():
#     """
#     Moreau-Yosida regularization of L1-loss with parameter gamma 
#     This equals the Huber loss. Only need this class to check whether the 
#     MYULA implementation works and converges to its (biased) target
#     """
#     def __init__(self, gamma = 1, d=None, mu = None, b = None):
#         if d is None and mu is None:
#             raise ValueError("Please supply dimension or parameters from which dimension can be inferred!")
#         self.d = d if d is not None else mu.shape[0]
#         self.mu = mu if mu is not None else np.zeros((self.d,1))
#         self.b = b if b is not None else 1
#         self.gamma = gamma
#         self.unsmoothed_pot = L1loss_scaled(d=self.d,mu=self.mu,b=self.b)
    
#     def __call__(self, x):
#         prox_vals = self.unsmoothed_pot.prox(x, self.gamma)
#         return self.unsmoothed_pot(prox_vals) + 1/(2*self.gamma)*np.sum((x-prox_vals)**2, axis=0)
#         #return np.sum(
#         #    (np.abs(x-self.mu) > self.gamma/self.b)*(np.abs(x-self.mu) - self.gamma/(2*self.b))
#         #        + (np.abs(x-self.mu) <= self.gamma/self.b)*(self.b/(2*self.gamma) * (x-self.mu)**2)
#         #    ,axis=0)
    
#     def grad(self, x):
#         return ( (np.abs(x-self.mu) > self.gamma/self.b)*np.sign(x-self.mu)
#                 + (np.abs(x-self.mu) <= self.gamma/self.b)*self.b/self.gamma * (x-self.mu) )
    
#     def prox(self, x, gamma = 1):
#         raise NotImplementedError("Please don't try to compute the prox of an already MY-regularized function")