#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:58:25 2022

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

#from inexact_psgla import *
#from psgla import *
from pxmala import PxMALA
import distributions as pds
#import potentials as pot


def main():
    """ generate data """
    mu_true = 0.8
    sigma2_true = 0.5
    n_data = 1
    nu = 4
    rng = default_rng(34523)
    D = rng.normal(loc=mu_true,scale=np.sqrt(sigma2_true),size=(n_data,))
    
    """ define posterior """
    posterior = pds.Gamma_Gauss1D_Posterior_Salim21(data=D, scale=np.sqrt(sigma2_true), nu=nu)
    # metaparameter of the distribution: L = Lipschitz constant of nabla F
    L = np.linalg.norm(posterior.F.K)**2 / sigma2_true
    t = np.reshape(np.linspace(1e-4, 5*nu+np.mean(D),500),(1,-1))
    dens_vals = posterior.pdf(t)
    
    plt.figure(figsize=(5,5))
    plt.plot(np.reshape(t,(-1,)),np.reshape(dens_vals,(-1,)))
    plt.xlim(-0.5, 4.0)
    plt.ylim(0,2.2)
    mu_map = sigma2_true/(2*n_data)*(n_data/sigma2_true*np.mean(D)-1/2 + np.sqrt((1/2-n_data/sigma2_true*np.mean(D))**2 - 4*n_data/sigma2_true*(1-nu/2)))
    plt.plot([mu_map, mu_map],[0,2])
    #plt.show()
    
    """ sample using px-mala """
    # one run of Px-MALA for Wasserstein comparison
    tau_pxmala = 0.9/L*1.5
    max_iter_pxmala = 5000
    n_samples_pxmala = 500
    # initialize at the MAP to minimize burn-in time
    x0_pxmala = mu_map*np.ones((1,n_samples_pxmala))
    pxmala = PxMALA(max_iter_pxmala, tau_pxmala, x0_pxmala, pd = posterior)
    
    x_pxmala,acceptance = pxmala.simulate(return_all=True)
    print('Acceptance rate after {} iterations on {} samples: {}%'.format(max_iter_pxmala,n_samples_pxmala,100*acceptance))
    x_pxmala = np.reshape(x_pxmala, (-1,))
    plt.hist(x_pxmala,100,density=True);

if __name__ == '__main__':
    main()
    
    