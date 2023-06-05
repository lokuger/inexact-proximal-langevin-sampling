#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:39:11 2023

@author: lorenzkuger
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import sys, getopt, os
#from time import time
from skimage import io, transform

from inexact_pla import inexact_pla
#from sapg import sapg
from pdhg import pdhg#, acc_pdhg
import potentials as pot
import distributions as pds

#%% initial parameters: test image, computation settings etc.
params = {
    'iterations': 10000,
    'testfile_path': 'test_images/flintstones.png',
    'blurtype': 'gaussian',
    'bandwidth': 1,
    'noise_std': 0.20,
    'log_epsilon': None,
    'iter_prox': 10,
    'step': 'large',
    'efficient': True,
    'verbose': True
    }

#%% auxiliary functions
def blur_unif(n, b):
    """compute the blur operator a, its transpose a.t and the maximum eigenvalue 
    of ata.
    Carfeul, this assumes a quadratic n x n image
    blur length b must be integer and << n to prevent severe ill-posedness"""
    h = np.ones((1, b))
    lh = h.shape[1]
    h = h / np.sum(h)
    h = np.pad(h, ((0,0), (0, n-b)), mode='constant')
    h = np.roll(h, -int((lh-1)/2))
    h = np.matmul(h.T, h)
    H_FFT = np.fft.fft2(h)
    HC_FFT = np.conj(H_FFT)
    a = lambda x : np.real(np.fft.ifft2(H_FFT * np.fft.fft2(x)))
    at = lambda x : np.real(np.fft.ifft2(HC_FFT * np.fft.fft2(x)))
    ata = lambda x : np.real(np.fft.ifft2(H_FFT * HC_FFT * np.fft.fft2(x)))
    max_eigval = power_method(ata, n, 1e-4, int(1e3))
    return a,at,max_eigval

def blur_gauss(n, sigma):
    """compute the blur operator a, its transpose a.t and the maximum eigenvalue 
    of ata.
    Carfeul, this assumes a quadratic n x n image, with n even
    blur standard dev is assumed to be given in #pixels"""
    t = np.arange(-n/2+1,n/2+1)
    h = np.exp(-t**2/(2*sigma**2))
    h = h / np.sum(h)
    h = np.roll(h, -int(n/2)+1)
    h = h[np.newaxis,:] * h[:,np.newaxis]
    H_FFT = np.fft.fft2(h)
    HC_FFT = np.conj(H_FFT)
    a = lambda x : np.real(np.fft.ifft2(H_FFT * np.fft.fft2(x)))
    at = lambda x : np.real(np.fft.ifft2(HC_FFT * np.fft.fft2(x)))
    ata = lambda x : np.real(np.fft.ifft2(H_FFT * HC_FFT * np.fft.fft2(x)))
    max_eigval = power_method(ata, n, 1e-4, int(1e3))
    return a,at,max_eigval
    
def power_method(ata, n, tol, max_iter, verbose=False):
    """power method to compute the maximum eigenvalue of the linear op at*a"""
    x = np.random.normal(size=(n,n))
    x = x/np.linalg.norm(x.ravel())
    val, val_old = 1, 1
    for k in range(max_iter):
        x = ata(x)
        val = np.linalg.norm(x.ravel())
        rel_var = np.abs(val-val_old)/val_old
        val_old = val
        x = x/val
        if rel_var < tol:
            break
    return val

def my_imsave(im, filename, vmin=-0.02, vmax=1.02):
    im = np.clip(im,vmin,vmax)
    im = np.clip((im-vmin)/(vmax-vmin) * 256,0,255).astype('uint8')
    io.imsave(filename, im)
    
def my_imshow(im, label, vmin=-0.02, vmax=1.02, cbar=False):
    fig = plt.figure()
    plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
    q = plt.imshow(im, cmap='Greys_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    if cbar: fig.colorbar(q)
    plt.show()
    
    # # draw a new figure and replot the colorbar there
    # fig,ax = plt.subplots(figsize=(2,3))
    # plt.colorbar(q,ax=ax)
    # ax.remove()
    # plt.savefig(cbarfile,bbox_inches='tight')

#%% Main method - generate results directories
def main():
    # if not os.path.exists('./results'): os.makedirs('./results')
    # if not os.path.exists('./results/deblur_tv'): os.makedirs('./results/deblur_tv')
    # accuracy_dir = './results/deblur_tv/{}prox_iters'.format(params['iter_prox'])
    # if not os.path.exists(accuracy_dir): os.makedirs(accuracy_dir)
    # step_dir = accuracy_dir + '/{}_steps'.format(params['step'])
    # if not os.path.exists(step_dir): os.makedirs(step_dir)
    # sample_dir = step_dir + '/{}samples'.format(params['iterations'])
    # if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    # results_dir = sample_dir + '/{}'.format(params['testfile_path'].split('/')[-1].split('.')[0])
    # if not os.path.exists(results_dir): os.makedirs(results_dir)
        
    #%% Ground truth
    # results_file = results_dir+'/result_images.npy'
    if True: #not os.path.exists(results_file): 
        rng = default_rng(13928696)
        verb = params['verbose']
        # try:
        #     x = io.imread(params['testfile_path'],as_gray=True).astype(float)
        # except FileNotFoundError:
        #     print('Provided test image did not exist under that path, aborting.')
        #     sys.exit()
        # # handle images that are too large
        # Nmax = 128
        # if x.shape[0] > Nmax or x.shape[1] > Nmax: x = transform.resize(x, (Nmax,Nmax))
        
        # chess test image
        n,t = 32, 8
        x = np.zeros((n,n))
        for i in np.arange(32):
            for j in np.arange(32):
                if (i//t)%2 == (j//t)%2:
                    x[i,j] = 1
        
        x = x-np.min(x)
        x = x/np.max(x)

        # assume quadratic images
        n = x.shape[0]
        
        tv = pot.total_variation(n, n, scale=1)
        #tv_groundtruth = tv(x)
        
        #%% Forward model & corrupted data
        blur_width = params['bandwidth']
        if params['blurtype'] == 'gaussian':
            a,at,max_ev_ata = blur_gauss(n,blur_width) 
        elif params['blurtype'] == 'uniform':
            a,at,max_ev_ata = blur_unif(n,blur_width)
        elif params['blurtype'] == 'none':
            a,at,max_ev_ata = lambda x : x, lambda x : x, 1
        else:
            print('Unknown blur type, aborting')
            sys.exit()
        
        noise_std = params['noise_std']
        y = a(x) + noise_std*rng.normal(size=x.shape)
        L = max_ev_ata/noise_std**2
        
        # show ground truth and corrupted image
        # my_imshow(x,'ground truth')
        # my_imshow(y,'noisy image')
        
        #%% SAPG - compute the optimal regularization parameter
        # unscaled_posterior = pds.l2_deblur_tv(n, n, a, at, y, noise_std=noise_std, mu_tv=1)
        # # metaparameter of the posterior: L = Lipschitz constant of nabla F, necessary for stepsize
        
        # theta0 = 1
        # # empirically, for blur b=10 we need ~1000 warm up iterations with tau = 0.9/L. 
        # # for blur=5 roughly 500 warm up iterations
        # # For b=0 almost immediate warm-up since the noisy image seems to be in a region of high probability
        # s = sapg(iter_wu=25,iter_outer=60,iter_burnin=10,iter_inner=1,
        #           tau=0.9/L,delta=lambda k: 0.2/(theta0*n**2)*(k+1)**(-0.8),
        #           x0=x,theta0=theta0,theta_min=0.01,theta_max=1e2,
        #           epsilon_prox=3e-2,pd=unscaled_posterior)
        # ###################### change initialization later : iter_wu back to 500, x0 back to y
        # s.simulate()
        # mu_tv = s.mean_theta[-1]
        
        ##### -- plots to check that SAPG converged --
        # # log pi values during warm-up Markov chain
        # plt.plot(s.logpi_wu, label='log-likelihood warm-up samples')
        # plt.legend()
        # plt.show()
        
        # # thetas
        # plt.plot(s.theta,label='theta_n')
        # plt.plot(n**2/s.mean_g, label='dim/g(u_n)', color='orange')
        # plt.plot(np.arange(s.iter_burnin+1,s.iter_outer+1), s.mean_theta, label='theta_bar',color='green')
        # plt.legend()
        # plt.show()
        
        # # values g(X_n)
        # plt.plot(s.mean_g, label='g(u_n)')
        # plt.hlines(tv_groundtruth,0,len(s.mean_g)+1, label='g(u_true)')
        # plt.legend()
        # plt.show()
        
        #%% regularization parameter
        # mu_tv = s.mean_theta[-1]          # computed by SAPG
        mu_tv = 2.1                       # set by hand, tuned for best MAP PSNR
            
        #%% MAP computation - L2-TV deblurring
        # deblur using PDHG in the version f(Kx) + g(x) + h(x) with smooth h
        # splitting is f(Kx) = TV(x) and h(x) = smooth L2-data term, g = 0
        # this matches example 5.7 - PD-explicit in Chambolle+Pock 2016
        x0_pd, y0_pd = np.zeros(x.shape), np.zeros((2,)+x.shape)
        tau_pd, sigma_pd = 1/(np.sqrt(8)+L), 1/np.sqrt(8)
        n_iter_pd = 2000
        f = pot.l2_l1_norm(n, n, scale=mu_tv)
        k,kt = tv._imgrad, tv._imdiv
        g = pot.zero()
        h = pot.l2_loss_reconstruction_homoschedastic(y, noise_std**2, a, at, max_ev_ata)
        pd = pdhg(x0_pd, y0_pd, tau_pd, sigma_pd, n_iter_pd, f, k, kt, g, h)
        
        if verb: sys.stdout.write('Compute MAP - '); sys.stdout.flush()
        u = pd.compute(verbose=True)
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
        
        # my_imshow(u,'MAP estimate')
        print('MAP: mu_TV = {:.1f};\tPSNR: {:.2f}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((u-x)**2))))
            
        #%% sample using inexact PLA
        x0 = np.copy(u) # np.zeros_like(x)
        if params['step'] == 'large':
            tau = 1/L
        elif params['step'] == 'small':
            tau = 0.5/L
        iter_prox = params['iter_prox']
        epsilon_prox = 10**params['log_epsilon'] if params['log_epsilon'] is not None else None
        n_samples = params['iterations']
        burnin = 1000
        posterior = pds.l2_deblur_tv(n, n, a, at, max_ev_ata, y, noise_std=noise_std, mu_tv=mu_tv)
        eff = params['efficient']
        
        #x0, n_iter, burnin, pd, step_size=None, rng=None, epsilon_prox=1e-2, iter_prox=np.Inf, efficient=False, exact=False
        ipla = inexact_pla(x0, n_samples, burnin, posterior, step_size=tau, rng=rng, epsilon_prox=epsilon_prox, iter_prox=iter_prox, efficient=eff)
        if verb: sys.stdout.write('Sample from posterior - '); sys.stdout.flush()
        ipla.simulate(verbose=verb)
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()
        
        #%% plots
        # diagnostic plot, making sure the sampler looks plausible
        # plt.plot(np.arange(1,n_samples+1), ipla.logpi_vals)
        # plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [All]')
        # plt.show()
        # plt.plot(np.arange(50+1,n_samples+1), ipla.logpi_vals[50:])
        # plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [after burn-in]')
        # plt.show()
        # plt.plot(np.arange(burnin+1,n_samples+1), ipla.logpi_vals[burnin:])
        # plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [after burn-in]')
        # plt.show()
        
        my_imshow(ipla.mean, 'Sample Mean, log10(epsilon)={}'.format(params['log_epsilon']))
        logstd = np.log10(ipla.std)
        my_imshow(logstd, 'Sample standard deviation (log10)', np.min(logstd), np.max(logstd))
        print('Total no. iterations to compute proximal mappings: {}'.format(ipla.num_prox_its_total))
        print('No. iterations per sampling step: {:.1f}'.format(ipla.num_prox_its_total/n_samples))
        
        #%% saving
        # np.save(results_file,(x,y,u,ipla.mean,ipla.std))
        
    # else:
    #     #%% results were already computed, show images
    #     x,y,u,mn,std = np.load(results_file)
    #     logstd = np.log10(std)
    #     # my_imsave(x, results_dir+'/ground_truth.png')
    #     # my_imsave(y, results_dir+'/noisy.png')
    #     # my_imsave(u, results_dir+'/map.png')
    #     # my_imsave(mn, results_dir+'/posterior_mean.png')
    #     # my_imsave(logstd, results_dir+'/posterior_logstd.png',-1.33,-0.83)
        
    #     # my_imshow(x, 'ground truth')
    #     # my_imshow(y, 'blurred & noisy')
    #     # my_imshow(u, 'map estimate')
    #     my_imshow(mn, 'post. mean / mmse estimate', results_dir+'/cbar.pdf')
    #     my_imshow(logstd, 'posterior log std',results_dir+'/cbar_std.pdf',-1.33,-0.83)
    #     print('Posterior mean PSNR: {:.7f}'.format(10*np.log10(np.max(x)**2/np.mean((mn-x)**2))))
        
        
#%% help function for calling from command line
def print_help():
    print('<>'*39)
    print('Run primal dual Langevin algorithm to generate samples from ROF posterior.')
    print('<>'*39)
    print(' ')
    print('Options:')
    print('    -h (--help): Print help.')
    print('    -i (--iterations=): Number of iterations of the Markov chain')
    print('    -f (--testfile_path=): Path to test image file')
    print('    -e (--efficientOff): Turn off storage-efficient mode, where we dont save samples but only compute a runnning mean and standard deviation during the algorithm. This can be used if we need the samples for some other reason (diagnostics etc). Then modify the code first')
    print('    -l (--log_epsilon=): log-10 of the accuracy parameter epsilon. The method will report the total number of iterations in the proximal computations for this epsilon = 10**log_epsilon in verbose mode')
    print('    -p (--iter_prox=): log-10 of the accuracy parameter epsilon. The method will report the total number of iterations in the proximal computations for this epsilon = 10**log_epsilon in verbose mode')
    print('    -s (--step=): \'large\' for 1/L or \'small\' for 0.5/L')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:f:el:p:s:v",
                                   ["help","iterations=","testfile_path=",
                                    "efficientOff","log_epsilon=",
                                    "iter_prox=","step=","verbose"])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-i", "--iterations"):
            params['iterations'] = int(arg)
        elif opt in ("-f", "--testfile_path"):
            params['testfile_path'] = arg
        elif opt in ("-e","--efficientOff"):
            params['efficient'] = False
        elif opt in ("-l", "--log_epsilon"):
            params['log_epsilon'] = float(arg)
        elif opt in ("-p", "--iter_prox"):
            params['iter_prox'] = int(arg)
        elif opt in ["-s", "--step="]:
            if arg in ['large','small']:
                params['step'] = arg
            else:
                print('Unknown step size specifier')
                sys.exit()
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    