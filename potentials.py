import numpy as np
import warnings
    
class L2loss_homoschedastic():
    """
    a simplified form of the L2loss_heteroschedastic class that is faster for 
    the (very common) special case that the variance matrix is sigma^2*I
    
    F(u) = 1/(2*sigma^2) * ||u - mu||_2^2
    
    d:      dimension of u, can be inferred from mu if mu is given\n
    mu:     shift parameter/data, e.g. noisy image in denoising\n
    sigma:  standard deviation sigma
    """
    def __init__(self, d = None, mu = None, sigma = 1):
        if d is None and mu is None:
            raise ValueError("Please supply dimension or parameters from which dimension can be inferred!")
        # infer dimension from parameters
        self.d = d if d is not None else mu.shape[0]
               
        # set distribution parameters
        self.mu = mu if mu is not None else np.zeros((self.d,1))
        self.sigma = sigma
        self.sigma2 = self.sigma**2
    
    def __call__(self, x):
        return 1/(2*self.sigma2) * np.sum((x-self.mu)**2, axis=0)
    
    def grad(self, x):
        return 1/self.sigma2 * (x-self.mu)
    
    def prox(self, x, gamma = 1):
        return 1/(self.sigma2+gamma)*(self.sigma2*x+gamma*self.mu)
    
    def conj(self, y):
        return self.sigma2/2 * np.sum(y**2, axis=0) + np.sum(y * self.mu, axis=0)
    
    def conjProx(self, x, gamma = 1):
        return 1/(1+gamma*self.sigma2)*(x-gamma*self.mu)
    
class L2loss_reconstruction_homoschedastic():
    """
    A variant of L2norm which allows a linear transformation of the input, i.e.
    1/(2*sigma2) ||Kx-mu||_2^2
    where K is a d x m real matrix, mu is in R^d, x in R^m
    
    A variant of L2loss_homoschedastic which allows a linear transformation 
    of the input
    Necessary for all indirect models, hence everything except denoising. K is
    the forward operator of the associated inverse problem
    
    F(u) = 1/(2*sigma^2) * ||K*u - mu||_2^2
    
    d:      dimension of u, can be inferred from mu or K if they are given\n
    mu:     shift parameter/data, e.g. noisy image in denoising\n
    sigma:  standard deviation sigma
    K:      linear operator, given as an array of shape dxm
    """
    def __init__(self, d = None, mu = None, sigma2 = None, K = None):
        if d is None and mu is None and K is None:
            raise ValueError("Please supply dimension, mean or operator from which dimension can be inferred!")
        # infer dimension from parameters
        self.d = d if d is not None else (mu.shape[0] if mu is not None else K.shape[0])
               
        # set distribution parameters
        self.mu = mu if mu is not None else np.zeros((self.d,1))
        self.sigma2 = sigma2 if sigma2 is not None else 1
        self.K, self.m = (K, K.shape[1]) if K is not None else (np.eye(self.d), self.d)
        self.KTK = self.K.T @ self.K
    
    def __call__(self, x):
        """
        Accepts only inputs in R^(self.m x n_samples)
        """
        return 1/(2*self.sigma2) * np.sum((self.K@x-self.mu)**2, axis=0)
    
    def grad(self, x):
        return 1/self.sigma2 * self.K.T@(np.reshape(self.K@x,(self.d,-1))-self.mu)
    
    def prox(self, x, gamma):
        return np.linalg.solve(self.KTK + self.sigma2/gamma * np.eye(self.m), self.K.T@self.mu + self.sigma2/gamma * x)
    
    def conjProx(self, x, gamma):
        return np.linalg.solve(self.KTK + gamma*self.sigma2*np.eye(self.m), self.KTK@x - gamma*(self.K.T@self.mu))


class L1loss_scaled():
    """
    symbolizes the L1-norm, weighted by a parameter scale. Used for 
    L1-regularization or the Laplace distribution
    G(u) = scale * ||u - mu||_1
    """
    def __init__(self, d = None, mu = None, scale = 1):
        if d is None and mu is None:
            raise ValueError("Please supply dimension or parameters from which dimension can be inferred!")
        
        # infer dimension by parameters
        self.d = d if d is not None else mu.shape[0]
        
        # set distribution parameters
        self.mu = mu if mu is not None else np.zeros((self.d,1))
        self.scale = scale
    
    def __call__(self, x):
        return np.sum(np.abs(x-self.mu),axis=0,keepdims=True)*self.scale
    
    def grad(self, x):
        raise NotImplementedError("L1norm does not have a gradient, use prox operator instead!")
    
    def prox(self, x, gamma = 1):
        return np.sign(x-self.mu) * np.maximum(0, np.abs(x-self.mu)-gamma*self.scale) + self.mu
    
    def inexact_prox(self, x, gamma = 1, epsilon=None, maxiter=1e2, verbose=False):
        """Careful, noticed that this implementation is only correct if 
        self.mu = 0. Of course, this is the interesting case """
        gamma *= self.scale
        checkAccuracy = epsilon is not None
        # iterative scheme to minimize the dual objective
        y = np.zeros_like(x)  # solve for solution y of the dual using proximal gradient descent
        stopcrit = False
        tauprime = 0.99
        tau = 1  # tau = 1 would be the most efficient but we want to make the routine artificially bad! Recheck this later.
        i = 0
        if checkAccuracy and verbose:
            print('Run (backward) gradient descent on the dual Lasso with gamma*mu = {:.3e}'.format(gamma))
            print('|{:^11s}|{:^31s}|'.format('Iterate','D-Gap (stop if < {:.3e})'.format(epsilon)))
        while i < maxiter and not stopcrit:
            i = i + 1
            v = (1-tauprime)*y + tauprime*x
            n = np.abs(v)
            w = v/np.maximum(1,1/gamma * n)
            # explicit gradient descent step on the Moreau-Yosida regularization: Gradient is y-w, see Chambolle, Pock 2016
            y = y - tau*(y - w)
            if checkAccuracy:
                halfsquarednormy = 1/2 * np.sum(y**2)
                # compute primal dual gap here and check if smaller than epsilon
                P = gamma * self(x-y)[0,0] + halfsquarednormy # primal value
                ndual = np.sqrt(y**2)
                dualInadmissible = np.any(ndual > gamma+1e-15)
                Pconj = np.Inf if dualInadmissible else halfsquarednormy - np.sum(y * x)
                dgap = P+Pconj
                stopcrit = dgap < epsilon
                if verbose and (i%25 == 0 or stopcrit):
                    print('|{:^11d}|{:^31.3e}|'.format(i,dgap))
        return x - y
        
class KLDistance():
    """
    Kullback Leibler distance KL(Ku+b, v)
    with background b,
    data observation v,
    linear transformation K.
    """
    def __init__(self, data = None, background = None, K = None):
        self.v = data
        self.m = self.v.shape[0]
        self.d = K.shape[1] if K is not None else self.v.shape[0]
        self.b = background if background is not None else np.zeros((self.m,1))
        self.K = K if K is not None else np.eye(self.d)
        
    def __call__(self, u):
        Ku = self.K @ u
        # be careful with this next line - it is here since the stepsize 
        # backtracking naturally jumps too far and considers points u where 
        # Ku < 0. We set these to the corresponding axis to prevent warnings.
        Ku[Ku<0] = 0
        # from here everything's fine again
        Kub = Ku + self.b
        return np.sum(Kub - self.v + self.v*np.log(self.v/Kub), axis=0)
    
    def grad(self, u):
        """
        The KL distance is gradient Lipschitz with constant L; where the best (smallest)
        L can be bounded by the easily computable constant
        v_1/(b_1^2) * norm(K_1)^2 + ... + v_m/(b_m^2) * norm(K_m)^2
        where K_i are the lines of the matrix K. See Obsidian notes for details
        """
        Kub = self.K @ u + self.b
        return self.K.T @ (np.ones_like(self.v) - self.v/Kub)
    
class TotalVariation_scaled():
    """
    total variation of 2D image u with shape n1, n2, scaled by a constant
    regularization parameter scale.
    
    Computed on a grid via finite differences, assuming equidistant spacing of 
    the grid. The gradient of this potential does not exist since TV is not 
    smooth.
    The proximal mapping is approximated using the dual problem. Pass either 
    a maximum number of steps, an accuracy (in the primal-dual gap), or both 
    to the prox evaluation, for more details see 
        TotalVariation_scaled.inexact_prox
        and
        TotalVariation_scaled.inexact_prox_singleImage_vanillaGD
        TotalVariation_scaled.inexact_prox_singleImage_acceleratedGD
    """
    def __init__(self, n1, n2, scale=1):
        self.n1 = n1
        self.n2 = n2
        self.scale = scale
        
    def D(self, u):
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
        px = np.concatenate((u[1:,:] - u[0:-1,:], np.zeros((1,self.n2))),axis=0)
        py = np.concatenate((u[:,1:] - u[:,0:-1], np.zeros((self.n1,1))),axis=1)
        return self.scale*px, self.scale*py
    
    def D_adj(self, px, py):
        """
        Computes the negative divergence of the 2D vector field px,py.
        could als be seen as a tensor from R^(mxnx2) to R^(mxn) (for each input)

        Parameters
        ----------
        px : numpy array of shape n1, n2
            x-component of vector field.
        py : numpy array of shape n1, n2
            y-component of vector field.

        Returns
        -------
        divergence of shape n1, n2
        """
        u1 = np.concatenate((-px[0,:][np.newaxis,:], -(px[1:-1,:]-px[0:-2,:]), px[-2,:][np.newaxis,:]), axis = 0)
        u2 = np.concatenate((-py[:,0][:,np.newaxis], -(py[:,1:-1]-py[:,0:-2]), py[:,-2][:,np.newaxis]), axis = 1)
        return self.scale*(u1+u2)
    
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
        if u.shape[0] == self.n1*self.n2:
            dx_u, dy_u = self.D(np.reshape(u,(self.n1,self.n2)))
        else:
            dx_u, dy_u = self.D(u)
        return self.scale*np.sum(np.sqrt(dx_u**2 + dy_u**2))
    
    def __inexact_prox_singleImage_vanillaGD__(self, u, gamma=1, epsilon=None, maxiter=1e3, verbose=False):
        """
        Computing the prox of TV is solving the ROF model. See Chambolle, Pock 2016,
        Example 4.8 for a precise description of what is done here
        -- gamma is the prox parameter, hence if the TV scaling parameter mu is 
        set to a value other than 1, solving this problem is equivalent to 
        computing a backward gradient step (=solving ROF) with step
            gamma * mu
        """
        checkAccuracy = True if epsilon is not None else False
        # iterative scheme to minimize the dual objective
        px = np.zeros((self.n1,self.n2))  # solve for solution of the dual p using proximal gradient descent
        py = np.zeros((self.n1,self.n2))
        Dadjp = self.D_adj(px, py)
        stopcrit = False
        tauprime = 0.99 * 1/8
        tau = 1
        i = 0
        if checkAccuracy and verbose:
            print('Run (backward) gradient descent on the dual ROF problem with gamma*mu = {:.3e}'.format(gamma*self.scale))
            print('|{:^11s}|{:^31s}|'.format('Iterate','D-Gap (stop if < {:.3e})'.format(epsilon)))
        while i < maxiter and not stopcrit:
            i = i + 1
            # gradient of Moreau-Yosida regularization
            zx,zy = self.D(Dadjp - u)
            vx = px - tauprime * zx
            vy = py - tauprime * zy
            # project 
            s = 1/np.maximum(1,1/(gamma*self.scale) * np.sqrt(vx**2 + vy**2))
            wx, wy = vx*s, vy*s     # projection in dual norm
            # explicit gradient descent step on the Moreau-Yosida regularization: Gradient is p-w, see Chambolle, Pock 2016
            px = px - tau*(px - wx)
            py = py - tau*(py - wy)
            Dadjp = self.D_adj(px, py)
            if checkAccuracy:
                halfnormDadjp = 1/2 * np.sum(Dadjp**2)
                # compute primal dual gap here and check if smaller than epsilon
                P = gamma * self(u-Dadjp) + halfnormDadjp # primal value
                ndual = np.sqrt(px**2+py**2)
                dualInadmissible = np.any(ndual > gamma*self.scale+1e-15)
                Pconj = np.Inf if dualInadmissible else halfnormDadjp - np.sum(Dadjp * u)
                dgap = P+Pconj
                stopcrit = dgap < epsilon
                if dgap < 0:
                    raise ValueError('Bad, Bad, Bad! Duality gap was negative, please check your prox computation routine!')
                if verbose and (i%10 == 0 or stopcrit):
                    print('|{:^11d}|{:^31.3e}|'.format(i,dgap))
                
        return u - self.D_adj(px, py)
    
    def __inexact_prox_singleImage_acceleratedGD__(self, u, gamma=1, epsilon=None, maxiter=1e3, verbose=False):
        """
        Same as "_vanillaGD but using accelerated GD, see Chambolle & Pock 2016
        """
        checkAccuracy = True if epsilon is not None else False
        # iterative scheme to minimize the dual objective
        px = np.zeros((self.n1,self.n2))  # solve for solution of the dual p using proximal gradient descent
        py = np.zeros((self.n1,self.n2))
        stopcrit = False
        tauprime = 0.99 * 1/8
        # tau_gd = 1
        i = 0
        tau_agd = 1
        t_agd = 0
        px_curr, px_prev, py_curr, py_prev = np.copy(px), np.copy(px), np.copy(py), np.copy(py)
        qx, qy = np.copy(px), np.copy(py)
        if checkAccuracy and verbose:
            print('Run (backward) accelerated gradient descent on the dual ROF problem with gamma = {:.3e}'.format(gamma*self.scale))
            print('|{:^11s}|{:^31s}|'.format('Iterate','D-Gap (stop if < {:.3e})'.format(epsilon)))
        while i < maxiter and not stopcrit:
            i = i + 1
            
            t_agd_new = (1+np.sqrt(1+4*t_agd**2))/2
            qx = px_curr + (t_agd-1)/t_agd_new * (px_curr - px_prev)
            qy = py_curr + (t_agd-1)/t_agd_new * (py_curr - py_prev)
            
            # compute the gradient of Moreau-Yosida regularization at qx,qy
            zx,zy = self.D(self.D_adj(qx, qy) - u)
            vx = qx - tauprime * zx
            vy = qy - tauprime * zy
            s = 1/np.maximum(1,1/(gamma*self.scale) * np.sqrt(vx**2 + vy**2))
            wx, wy = vx*s, vy*s     # projection in dual norm
            # The gradient is q-w, see Chambolle, Pock 2016
            
            # updates
            t_agd = t_agd_new
            px_prev = np.copy(px_curr)
            py_prev = np.copy(py_curr)
            px_curr = qx - tau_agd * (qx - wx)
            py_curr = qy - tau_agd * (qy - wy)
            
            # stopping criterion
            if checkAccuracy:
                Dadjp = self.D_adj(px_curr, py_curr)
                halfnormDadjp = 1/2 * np.sum(Dadjp**2)
                # compute primal dual gap here and check if smaller than epsilon
                P = gamma * self(u-Dadjp) + halfnormDadjp # primal value
                ndual = np.sqrt(px_curr**2 + py_curr**2)
                dualInadmissible = np.any(ndual > gamma*self.scale+1e-12)
                Pconj = np.Inf if dualInadmissible else halfnormDadjp - np.sum(Dadjp * u)
                dgap = P+Pconj
                stopcrit = dgap < epsilon
                if dgap < 0:
                    raise ValueError('Bad, Bad, Bad! Duality gap was negative, please check your prox computation routine!')
                if verbose and (i%10 == 0 or stopcrit or i==maxiter):
                    print('|{:^11d}|{:^31.3e}|'.format(i,dgap))
                
        return u - self.D_adj(px_curr, py_curr)
    
    def inexact_prox(self, u, gamma=1, epsilon=None, maxiter=1e3, verbose=False, gd_type='accelerated'):
        """
        Calls the inexact_prox routines in a loop for all samples.
        Maybe to be parallelized later...
        """
        n_samples = u.shape[1]
        r = np.zeros_like(u)
        for i in np.arange(n_samples):
            v = np.reshape(u[:,i],(self.n1,self.n2))
            if gd_type == 'accelerated':
                proxv = self.__inexact_prox_singleImage_acceleratedGD__(v, gamma, epsilon, maxiter, verbose)
            elif gd_type == 'vanilla':
                proxv = self.__inexact_prox_singleImage_vanillaGD__(v, gamma, epsilon, maxiter, verbose)
            else:
                raise ValueError("Invalid gradient descent type, use 'accelerated' (default) or 'vanilla'")
            r[:,i] = np.reshape(proxv,(-1,))
        return r
    
    def rescale(self, scale_new):
        self.scale = self.scale_new
        
    
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
    
# a dummy class if we want to set F or G to zero
class Zero():
    def __call__(self, x):
        return np.zeros((1,x.shape[1])) if len(x.shape) > 1 else 0
    
    def grad(self, x):
        return np.zeros_like(x)
    
    def prox(self, x):
        return np.copy(x)
    
class L2loss_heteroschedastic():
    """
    symbolizes the weighted l2-norm for d-dimensional vectors. Is used to generate 
    normal distributions and the data-fidelity/log-likelihood in problems with 
    Gaussian noise.
    """
    def __init__(self, d = None, mu = None, Var = None):
        if d is None and mu is None and Var is None:
            raise ValueError("Please supply dimension or parameters from which dimension can be inferred!")
        # infer dimension from parameters
        self.d = d if d is not None else (mu.shape[0] if mu is not None else Var.shape[0])
               
        # set distribution parameters
        self.mu = mu if mu is not None else np.zeros((self.d,1))
        if Var is None:
            self.Var = np.eye(self.d)
            self.Prec = np.eye(self.d)
        elif np.isscalar(Var):
            self.Var = Var*np.eye(self.d)
            self.Prec = 1/Var*np.eye(self.d)
        else:
            self.Var = Var
            self.Prec = np.linalg.inv(self.Var)
    
    def __call__(self, x):
        return 1/2 * np.sum((x-self.mu) * (self.Prec @ (x-self.mu)), axis=0)
    
    def grad(self, x):
        return self.Prec @ (x-self.mu)
    
    def prox(self, x, gamma = 1):
        return np.linalg.solve(self.Var+gamma*np.eye(self.d), self.Var@x + gamma*self.mu)
    
    def conj(self, y):
        return 1/2 * np.sum(y * self.Var @ y, axis=0) + np.sum(y * self.mu, axis=0)
    
    def conjProx(self, x, gamma = 1):
        return np.linalg.solve(np.eye(self.d)+gamma*self.Var, x-gamma*self.mu)
    
class Log_Gamma1D():
    """
    negative Log-density of a 1D Gamma distribution with parameters
    - shape: alpha > 0
    - rate: beta > 0
    """
    def __init__(self, alpha = 1, beta = 0.5):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, x):
        admissible_idcs = x>0
        r = np.inf*np.ones_like(x)
        r[admissible_idcs] = (1-self.alpha)*np.log(x[admissible_idcs]) + self.beta * x[admissible_idcs]
        return r
    
    def prox(self, x, gamma = 1):
        return x/2 - self.beta*gamma/2 + np.sqrt(
            1/4* (x**2) - x*self.beta*gamma/2 + (self.beta**2)*(gamma**2)/4 - gamma*(1-self.alpha))
    
    def inexact_prox(self, x, gamma = 1, epsilon=None, maxiter=1e3):
        """comment: we do actually know the exact prox (see above), 
        this is only to compare inexact psgla and psgla
        The inexact prox computes the prox by iteratively solving the dual
        problem using gradient descent.
        The dual problem is given by
        - min_v {(gamma*v - x)^2/(2*gamma) - x^2/(2*gamma) + (alpha-1)*(log((alpha-1)/(beta-v)) - 1)}
            =: - min_v Psi_gamma(v)
        Once the dual solution v is approximated, the approximation to the 
        prox evaluation is   x - gamma*v
        Check the accuracy by bounding the duality gap Phi(y)+Psi(v)
        """
        checkAccuracy = True if epsilon is not None else False
        # iterative scheme to minimize the dual objective
        v = np.zeros_like(x)    # function is only defined on [-Inf, beta], beta > 0, hence initilizing at 0 might be safe
        stopcrit = np.full(x.shape, False)
        tau = 1/(gamma + (self.alpha-1)/(self.beta**2)) # since optimal v < 0 and initialize at 0, this is lower bound for 1/L
        i = 0
        while i < maxiter and not np.all(stopcrit):
            i = i + 1
            # update v
            v[~stopcrit] = v[~stopcrit] - tau * (gamma*v[~stopcrit] - x[~stopcrit] + (1-self.alpha)/(v[~stopcrit]-self.beta))
            if checkAccuracy:
                # compute primal dual gap here and check if smaller than epsilon
                P = self(x-gamma*v) + gamma/2 * v**2  # primal value
                Pconj = ((gamma*v-x)**2)/(2*gamma) - (x**2)/(2*gamma) + (self.alpha-1)*(np.log((self.alpha-1)/(self.beta-v)) - 1)
                stopcrit = P + Pconj < epsilon
        return x - gamma*v
        

class MY_Log_Gamma1D():
    """
    negative Log-density of a 1D Gamma distribution with parameters
    - shape: alpha > 0
    - rate: beta > 0
    Only need this class to check whether the MYULA implementation works and 
    converges to its (biased) target.
    """
    def __init__(self, alpha = 1, beta = 0.5, gamma = 1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.unsmoothed_pot = Log_Gamma1D(alpha, beta)
    
    def __call__(self, x):
        prox_vals = self.unsmoothed_pot.prox(x, self.gamma)
        return self.unsmoothed_pot(prox_vals) + 1/(2*self.gamma)*np.sum((x-prox_vals)**2, axis=0)
    
    def prox(self, x, gamma = 1):
        raise NotImplementedError("Please don't try to compute the prox of an already MY-regularized function")
        
class MY_L1loss_scaled():
    """
    Moreau-Yosida regularization of L1-loss with parameter gamma 
    This equals the Huber loss. Only need this class to check whether the 
    MYULA implementation works and converges to its (biased) target
    """
    def __init__(self, gamma = 1, d=None, mu = None, b = None):
        if d is None and mu is None:
            raise ValueError("Please supply dimension or parameters from which dimension can be inferred!")
        self.d = d if d is not None else mu.shape[0]
        self.mu = mu if mu is not None else np.zeros((self.d,1))
        self.b = b if b is not None else 1
        self.gamma = gamma
        self.unsmoothed_pot = L1loss_scaled(d=self.d,mu=self.mu,b=self.b)
    
    def __call__(self, x):
        prox_vals = self.unsmoothed_pot.prox(x, self.gamma)
        return self.unsmoothed_pot(prox_vals) + 1/(2*self.gamma)*np.sum((x-prox_vals)**2, axis=0)
        #return np.sum(
        #    (np.abs(x-self.mu) > self.gamma/self.b)*(np.abs(x-self.mu) - self.gamma/(2*self.b))
        #        + (np.abs(x-self.mu) <= self.gamma/self.b)*(self.b/(2*self.gamma) * (x-self.mu)**2)
        #    ,axis=0)
    
    def grad(self, x):
        return ( (np.abs(x-self.mu) > self.gamma/self.b)*np.sign(x-self.mu)
                + (np.abs(x-self.mu) <= self.gamma/self.b)*self.b/self.gamma * (x-self.mu) )
    
    def prox(self, x, gamma = 1):
        raise NotImplementedError("Please don't try to compute the prox of an already MY-regularized function")