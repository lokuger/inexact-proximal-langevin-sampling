import numpy as np
import sys
import warnings
    
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
        self.L = 1/self.sigma2
    
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
    def __init__(self, y, sigma2, a, at, max_ev_ata):
        self.y = y
        self.sigma2 = sigma2
        self.a = a
        self.at = at
        self.L = max_ev_ata/self.sigma2
    
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

class l1_loss_unshifted_homoschedastic():
    """
    An l1 loss term that can be used as a prior
    
    F(x) = mu * ||x||_1
    
    Inputs: 
        - mu:        scale parameter. 
    The correspnding Laplace distribution Laplace(0,b) has variance 2*b^2
    """
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, x):
        return self.scale*np.sum(np.abs(x))
    
    def grad(self, x):
        raise NotImplementedError('L1 norm has no gradient')
        
    def prox(self, x, gamma):
        return np.maximum(0, np.abs(x)-gamma*self.scale) * np.sign(x)
    
    def inexact_prox(self, u, gamma, epsilon, max_iter=np.Inf):
        """
        deliberately compute the prox inexactly here. We want to use this to
        compare the inexact version of the PGLA algorithm with the exact one.
        Prox problem - primal form:
            min_x {scale*||x||_1 + 1/(2*gamma)*||x-u||^2}
        Dual problem:
            max_{y : ||y||_infty <= scale}{-1/(2*gamma)*||gamma*y-u||^2} + 1/(2*gamma)*||u||^2
        Hence solve argmin_{y : ||y||_infty <= scale}{1/(2*gamma)*||gamma*y-u||^2}:
        The solution is the projection of u/gamma onto infty-ball of radius scale
        Just approximate this from below (inside the ball) by a sequence 
            (1 - q^k) * sol
        whick converges in norm as q^k, for some 0<q<1.
        """
        i = 0
        stopcrit = False
        
        # this is the true solution of the dual problem
        v = u / gamma
        sol = v / np.maximum(1,np.abs(v)/self.scale)
        q = 0.5
        while not stopcrit:
            x = (1 - q**i)*sol
            
            r = gamma/2*np.sum(x**2)
            primal = r + self(u-gamma*x)
            dual = - r + np.sum(x * u) # dual value. dual iterate should never be inadmissible since we project in the end
            dgap = primal - dual
            
            stopcrit = dgap <= epsilon
            i += 1
        return u-gamma*x, i
        
    def conj(self, z):
        if np.any(np.abs(z) > self.scale+1e-12):
            return np.Inf
        else:
            return 0
        
    def conj_prox(self, p, gamma):
        return p/np.maximum(1,np.abs(p)/self.scale)
    
class kl_divergence():
    """
    Kullback-Leibler divergence 
        KL(Ax+b; y) = sum_i (Ax)_i + b_i - y_i + y_i*log(y_i/(Ax+b)_i)
    Inputs:
        - y:        right hand side data
        - b:        estimated background in data space
        - a:        forward operator A, given as callable a(x)
        - at:       transpose/adjoint of A. Given as callable at(y)
    """
    def __init__(self, y, b, a, at):
        self.y = y      # data
        self.b = b      # background
        self.a = a      # forward model
        self.at = at    # adjoint of forward model
    
    def __call__(self, x):
        axb = self.a(x) + self.b
        if np.any(axb<0) or np.any(self.y<0):
            return np.Infinity
        else:
            I = self.y > 0
            return np.sum(axb[~I]) + np.sum(axb[I]-self.y[I]+self.y[I]*np.log(self.y[I]/axb[I]))
    
    def grad(self, x):
        return self.at(1 - self.y/(self.a(x) + self.b))
    
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
        Computing the prox of TV is solving the ROF model with FISTA
            
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
        q = np.copy(p)
        
        stopcrit = False
        t, t_prev = 1, 1
        
        i = 0
        if verbose: sys.stdout.write('run FISTA on dual ROF model: {:3d}% '.format(0)); sys.stdout.flush()
        
        while i < max_iter and not stopcrit:
            i = i + 1
            p_prev = np.copy(p)
            
            v = q - 1/(8*gamma) * self._imgrad(gamma*self._imdiv(q) - u)
            p = v/np.maximum(1, np.sqrt(np.sum(v**2,axis=0))/self.scale)[np.newaxis,:,:]
            
            t_new = (1+np.sqrt(1+4*t**2))/2
            t, t_prev = t_new, t
            q = p + (t_prev-1)/t * (p - p_prev)
            
            # stopping criterion: check if primal-dual gap < epsilon
            if checkAccuracy:
                div_p = self._imdiv(p)
                h = gamma/2 * np.sum(div_p**2)
                primal = self(u-gamma*div_p) + h
                conj_TV_p = np.max(np.sqrt(np.sum(p**2,axis=0)))
                dual_inadmissible = conj_TV_p > self.scale+1e-12
                dual = np.Inf if dual_inadmissible else h - np.sum(div_p * u) # dual value. dual iterate should never be inadmissible since we project in the end
                dgap = primal+dual
                stopcrit = dgap <= epsilon
                if dgap < -5e-15: # for debugging purpose
                    raise ValueError('Duality gap was negative (which should never happen), please check the prox computation routine!')
                if verbose: sys.stdout.write('\b'*5 + '{:3d}% '.format(int(i/max_iter*100))); sys.stdout.flush()
        if verbose: sys.stdout.write('\b'*5 + '100% '); sys.stdout.flush()
        return (u - gamma*self._imdiv(p)), i
        

class total_variation_nonneg():
    """
    total variation of 2D image u with shape (n1, n2), if u is in the 
    positive orthant. Otherwise, infinity. Scaled by a constant
    regularization parameter scale. Corresponds to the functional 
        scale * TV(u) + 1_{R+}(u)
    with u in R^{n1 x n2}, where 1_{R+} is indicator of R^n_+
    
    __init__ input:
        - n1, n2:   shape of u
        - scale:    scaling factor, usually a regularization parameter
        
    __call__ input:
        - u:        image of shape n1,n2 or n1*n2,
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
        return self.scale * np.sum(np.sqrt(np.sum(self._imgrad(u)**2,axis=0))) if np.all(u>=0) else np.infty
    
    def inexact_prox(self, u, gamma, epsilon=None, max_iter=np.Inf, verbose=False):
        """
        Compute an approximation of the inexact prox. Ineactness level epsilon is not quite clear, we tried
        to generalize Proposition 2.3 from Villa et al. 2013 to the case when not g(x) = w(Bx) but rather 
            g(x) = w(Bx) + 1_C(x)
        with an indicator of a convex set C (in this case positive orthant). It did not work, however!
        Strong duality holds as before and we can relate primal and dual iterates, but the implication 
            [duality gap at (p(v),v) < epsilon) => p(v) is epsilon-inexact prox point]
        is not true anymore since the projection messes up the computations. We will use it anyway, however, 
        since it is clear that primal and dual problems are convex and similarly easy to solve and the 
        duality gap is not too far off once we have entirely positive samples and sufficiently small steps..
            
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
        nusq = np.sum(u**2)

        # iterative scheme to minimize the dual objective
        p = np.zeros((2,self.n1,self.n2))
        q = np.copy(p)
        
        stopcrit = False
        t, t_prev = 1, 1
        
        i = 0
        if verbose: sys.stdout.write('run FISTA on dual ROF model with nonnegativity constraint: {:3d}% '.format(0)); sys.stdout.flush()
        
        while i < max_iter and not stopcrit:
            i = i + 1
            p_prev = np.copy(p)
            
            v = q + 1/8 * self._imgrad(np.maximum(0, u/gamma-self._imdiv(q)))
            p = v/np.maximum(1, np.sqrt(np.sum(v**2,axis=0))/self.scale)[np.newaxis,:,:]
            
            t_new = (1+np.sqrt(1+4*t**2))/2
            t, t_prev = t_new, t
            q = p + (t_prev-1)/t * (p - p_prev)
            
            # stopping criterion: check if primal-dual gap < epsilon
            if checkAccuracy:
                x_primal = np.maximum(0,u - gamma*self._imdiv(p))   # same as before but project onto R^n_+ orthant
                primal = self(x_primal) + 1/(2*gamma) * np.sum((x_primal-u)**2)
                conj_TV_p = np.max(np.sqrt(np.sum(p**2,axis=0)))
                dual = conj_TV_p + 1/(2*gamma)*(np.sum(x_primal**2) - nusq) # dual value. dual iterate should never be inadmissible since we project in the end
                dgap = primal+dual
                stopcrit = dgap <= epsilon
                if dgap < -5e-15: # for debugging purpose
                    raise ValueError('Duality gap was negative (which should never happen), please check the prox computation routine!')
                if verbose: sys.stdout.write('\b'*5 + '{:3d}% '.format(int(i/max_iter*100))); sys.stdout.flush()
        if verbose: sys.stdout.write('\b'*5 + '100% '); sys.stdout.flush()
        return np.maximum(0,u - gamma*self._imdiv(p)), i
    
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

    
class nonNegIndicator():
    """ 
    Indicator function of R_++^d, 0 if all entries are >= 0, otherwise infinity
    """
    def __init__(self, d):
        self.d = d
    
    def __call__(self, x):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            return np.nan_to_num(np.infty * (1-(np.min(x, axis=0) >= 0)), 0)
        
    def prox(self, x, gamma):
        return np.maximum(x,0)
    
class zero():
    """ a dummy class if we want to set one of the potential terms to zero"""
    def __call__(self, x):
        return 0
    
    def grad(self, x):
        return np.zeros_like(x)
    
    def prox(self, x, gamma):
        return x
    