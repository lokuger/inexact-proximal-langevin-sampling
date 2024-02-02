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
from skimage import io, transform

from inexact_pgla import inexact_pgla
from ista import ista
import potentials as pot
import distributions as pds

#%% initial parameters: test image, computation settings etc.
params = {
    'iterations': int(float('1e6')),
    'testfile_path': 'test-images/phantom256.png',
    'mu_tv': 2e00,
    'bandwidth': 5,
    'mean_intensity': 2,
    'mean_bg':  0.02,
    'iter_prox': 10,
    'step_type': 'bt',       # 'bt' or 'fixed'
    'efficient': True,
    'verbose': True,
    'result_root': './results/poisson-deblur-tv',
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
    max_ev_ata = power_method(ata, n, 1e-4, int(1e3))
    return a,at,max_ev_ata
    
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
    
def my_imshow(im, label, vmin=-0.02, vmax=1.02, cbarfile=None, cbarfigsize=None, cbarticks=None):
    fig = plt.figure()
    plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
    q = plt.imshow(im, cmap='Greys_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.show()

    # draw a new figure and replot the colorbar there
    if cbarfile is not None:
        fig = plt.figure(figsize=cbarfigsize)
        ax = fig.add_axes((0.05,0.05,0.2,0.9))
        cbar = plt.colorbar(q,cax=ax)
        if cbarticks is not None: cbar.set_ticks(cbarticks)
        plt.tick_params(axis='y', labelsize=30)
        plt.tight_layout()
        plt.savefig(cbarfile, bbox_inches='tight')
        print(fig.get_size_inches())
        #plt.show()

#%% Main method - generate results directories
def main():
    result_root = params['result_root']
    
    test_image_name = params['testfile_path'].split('/')[-1].split('.')[0]
    meanintensity = '{:.0e}miv'.format(params['mean_intensity'])
    accuracy = '{}proxiters'.format(params['iter_prox'])
    regparam = '{:.0e}regparam'.format(params['mu_tv'])
    steptype = 'btsteps' if params['step_type'] == 'bt' else 'fixsteps'
    nsamples = '{:.0e}samples'.format(params['iterations'])
    file_specifier = '{}_{}_{}_{}_{}_{}'.format(test_image_name,meanintensity,accuracy,regparam,steptype,nsamples)
    results_file = result_root+'/'+file_specifier+'.npz'

    # file names for result images
    mmse_file = result_root+'/mmse_'+file_specifier+'.png'
    logstd_file = lambda s: result_root+'/logstd_scale{}_'.format(s)+file_specifier+'.png'
    cbargt_file = result_root+'/'+test_image_name+'_colorbar.pdf'
    cbarlogstd_file = lambda s: result_root+'/'+file_specifier+'_logstd_scale{}_colorbar.pdf'.format(s)
        
    #%% Ground truth
    # results_file = results_dir+'/result_images.npy'
    if not os.path.exists(results_file): 
        rng = default_rng(1392)
        verb = params['verbose']
        try:
            x = io.imread(params['testfile_path']).astype(float)
        except FileNotFoundError:
            print('Provided test image did not exist under that path, aborting.')
            sys.exit()
        # handle images that are too large
        Nmax = 256
        if x.shape[0] > Nmax or x.shape[1] > Nmax: x = transform.resize(x, (Nmax,Nmax))
        x = x-np.min(x)
        x /= np.mean(x)

        n = x.shape[0] # assume quadratic images
        tv = pot.total_variation_nonneg(n, n, scale=1)
        
        ########## Forward model & noisy observation ##########
        # blur operator
        blur_width = params['bandwidth']
        a,at,max_ev_ata = blur_unif(n,blur_width)
        
        # scale mean intensity of ground truth, background and data
        scale = params['mean_intensity']
        scale_bg = params['mean_bg']
        x *= scale
        max_intensity = np.max(x)

        y = rng.poisson(np.maximum(0,a(x)))
        b = np.ones_like(y)*scale_bg
        L = max_ev_ata * np.max(y/(b**2))
        
        # show ground truth and corrupted image
        my_imshow(x,'ground truth',vmin=0,vmax=max_intensity)
        my_imshow(y,'noisy image',vmin=0,vmax=max_intensity)
        
        # regularization parameter
        mu_tv = params['mu_tv']
            
        ########## MAP computation ##########
        # deblur using ISTA on the composite functional f(x)+g(x). The splitting is 
        #       f(x) = KL(Ax+sigma,y) and g(x) = TV(x) + i_{R+}(x)
        # where KL is Kullback-Leibler, TV total variation and i_{R+} indicator of positive orthant.
        # For step size choice, we use backtracking since the Lipschitz constant of gradient of KL
        # is very heterogeneous in the admissible set
        x0 = np.zeros(x.shape)
        tau_ista = ('bt',1)
        n_iter_ista = 500
        f = pot.kl_divergence(y,b,a,at)
        g = pot.total_variation_nonneg(n1=n,n2=n,scale=mu_tv)
        opt_ista = ista(x0, tau_ista, n_iter_ista, f, g, efficient=True)
        
        if verb: sys.stdout.write('Compute MAP - '); sys.stdout.flush()
        u = opt_ista.compute(verbose=True)
        if verb: sys.stdout.write('Done.\n'); sys.stdout.flush()

        my_imshow(u,'MAP estimate',vmin=0,vmax=np.max(u))
        print('MAP: mu_TV = {:.1f};\tPSNR: {:.2f}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((u-x)**2))))

        # ########## sample using inexact PLA ##########
        # x0 = np.copy(u) # initialize chain at map to minimize burn-in
        # tau = ('bt',1) if params['step_type'] == 'bt' else ('fxd',1/L)
        # iter_prox = params['iter_prox']
        # epsilon_prox = None
        # burnin = 1000 if params['step_type'] == 'bt' else 10000
        # n_samples = params['iterations']+burnin
        # posterior = pds.kl_deblur_tvnonneg_prior(n,n,a,at,y,b,mu_tv)
        # eff = params['efficient']
        # downsampling_scales = [2,4,8]
        
        # ipla = inexact_pgla(x0, n_samples, burnin, posterior, step_size=tau, rng=rng, epsilon_prox=epsilon_prox, iter_prox=iter_prox, efficient=eff, downsampling_scales=downsampling_scales)
        # if verb: sys.stdout.write('Sample from posterior - '); sys.stdout.flush()
        # ipla.simulate(verbose=verb)
        
        # ########## plots ##########
        # plt.plot(np.arange(1,n_samples+1), ipla.logpi_vals)
        # plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [All]')
        # plt.show()
        # plt.plot(np.arange(burnin+1,n_samples+1), ipla.logpi_vals[burnin:])
        # plt.title('- log(pi(X_n)) = F(K*X_n) + G(X_n) [after burn-in]')
        # plt.show()
        # if params['step_type'] == 'bt':
        #     tau_cum = np.cumsum(ipla.tau_all)
        #     plt.plot(np.arange(n_samples),ipla.tau_all)
        #     plt.title('Chosen step sizes')
        #     plt.show()
        #     print('Average chosen step size (using two-way backtracking) was {:.2e}.'.format(np.mean(ipla.tau_all)))
        #     print('Naive estimate 1/L would have chosen {:.2e}'.format(1/L))
        #     plt.plot(np.arange(n_samples),tau_cum)
        #     plt.title('Cumulative steps')
        #     plt.show()
        
        # # show means
        # my_imshow(ipla.mean, 'Sample Mean', vmin=0, vmax=max_intensity)
        # for i,scale in enumerate(downsampling_scales):
        #     my_imshow(ipla.mean_scaled[i], 'Sample mean, downsampling = {}'.format(scale), vmin=0, vmax=max_intensity)

        # # show and save standard deviations
        # my_imshow(np.log10(ipla.std), 'Sample standard deviation (log10)', vmin=np.log10(np.min(ipla.std)), vmax=np.log10(np.max(ipla.std)))
        # std_scaled = {}
        # for i,scale in enumerate(downsampling_scales):
        #     std_scaled['std_scale{}'.format(scale)] = ipla.std_scaled[i]
        #     my_imshow(np.log10(ipla.std_scaled[i]), 'Sample mean, downsampling = {}'.format(scale), vmin=np.log10(np.min(ipla.std_scaled[i])), vmax=np.log10(np.max(ipla.std_scaled[i])))
        # print('Total no. iterations to compute proximal mappings: {}'.format(ipla.num_prox_its_total))
        # print('No. iterations per sampling step: {:.1f}'.format(ipla.num_prox_its_total/(n_samples-burnin)))
        # print('MMSE: mu_TV = {:.1f};\tPSNR: {:.2f}'.format(mu_tv,10*np.log10(np.max(x)**2/np.mean((ipla.mean-x)**2))))
        
        # # saving
        # np.savez(results_file,x=x,y=y,u=u,mn=ipla.mean,std=ipla.std,**std_scaled)
    else:
        #%% results were already computed, show images
        R = np.load(results_file)
        logstd = np.log(R['std'])

        gt = R['x']
        vmax = np.max(gt)
        vmin = -0.02*vmax
        
        # ground truth
        my_imshow(gt, 'ground truth', vmin, vmax, cbarfile=cbargt_file,cbarfigsize=(0.5*0.6*6.15, 0.5*3.1*6.15))
        my_imsave(gt, result_root+'/ground_truth.png', vmin, vmax)
        # noisy image
        my_imshow(R['y'], 'blurred & noisy', vmin, vmax)
        my_imsave(R['y'], result_root+'/noisy.png', vmin, vmax)
        # map
        my_imshow(R['u'], 'map estimate', vmin, vmax)
        my_imsave(R['u'], result_root+'/map.png', vmin, vmax)
        # mean
        my_imshow(R['mn'], 'post. mean / mmse estimate', vmin, vmax)
        my_imsave(R['mn'], mmse_file, vmin, vmax)
        # log(std)
        wmin, wmax = np.min(logstd), np.max(logstd)
        my_imshow(logstd, 'posterior log std',wmin,wmax,cbarfile=cbarlogstd_file(0),cbarfigsize=(0.5*0.4*6.15, 0.5*2.1*6.15), cbarticks=np.arange(np.ceil(wmin),np.ceil(wmax)))
        my_imsave(logstd, logstd_file(0), np.min(logstd), np.max(logstd))
        for s,scale in [(R[k],k[-1]) for k in R if k.startswith('std_scale')]:
            logstd_scaled = np.log(s)
            wmin, wmax = np.min(logstd_scaled), np.max(logstd_scaled)
            my_imshow(logstd_scaled, 'posterior log std at scale', wmin, wmax, cbarlogstd_file(int(scale)),cbarfigsize=(0.5*0.4*6.15, 0.5*2.1*6.15),cbarticks=np.arange(np.ceil(wmin),np.ceil(wmax)))
            my_imsave(logstd_scaled, logstd_file(int(scale)), np.min(logstd_scaled), np.max(logstd_scaled))
        print('MAP PSNR: {:.7f}'.format(10*np.log10(vmax**2/np.mean((R['u']-gt)**2))))
        print('Posterior mean PSNR: {:.7f}'.format(10*np.log10(vmax**2/np.mean((R['mn']-gt)**2))))
        
        
#%% help function for calling from command line
def print_help():
    print('<>'*39)
    print('Run Langevin algorithm to generate samples from Poisson data TV deblurring posterior.')
    print('<>'*39)
    print(' ')
    print('Options:')
    print('    -h (--help): Print help.')
    print('    -i (--iterations=): Number of iterations of the Markov chain')
    print('    -f (--testfile_path=): Path to test image file')
    print('    -e (--efficient_off): Turn off storage-efficient mode, where we dont save samples but only compute a runnning mean and standard deviation during the algorithm. This can be used if we need the samples for some other reason (diagnostics etc). Then modify the code first')
    print('    -m (--mu_tv=): TV regularization parameter')
    print('    -p (--iter_prox=): log-10 of the accuracy parameter epsilon. The method will report the total number of iterations in the proximal computations for this epsilon = 10**log_epsilon in verbose mode')
    print('    -s (--step_type=): \'bt\' for backtracking, \'fixed\' for fixed step size (use estimate of Lipschitz constant when background is uniformly positive).')
    print('    -d (--result_dir=): root directory for results. Default: ./results/deblur-wavelets')
    print('    -v (--verbose): Verbose mode.')
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:f:em:p:s:d:v",
                                   ["help","iterations=","testfile_path=",
                                    "efficient_off","mu_tv=","iter_prox=",
                                    "step_type=","result_dir=","verbose"])
    except getopt.GetoptError as E:
        print(E.msg)
        print_help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-i", "--iterations"):
            params['iterations'] = int(float(arg))
        elif opt in ("-f", "--testfile_path"):
            params['testfile_path'] = arg
        elif opt in ("-e","--efficient_off"):
            params['efficient'] = False
        elif opt in ("-m", "--mu_tv"):
            params['mu_tv'] = float(arg)
        elif opt in ("-p", "--iter_prox"):
            params['iter_prox'] = int(arg)
        elif opt in ("-s", "--step_type"):
            if arg in ['bt','fixed']:
                params['step_type'] = arg
            else:
                print('Unknown step type choice. Choose either \'bt\' or \'fixed\'')
                sys.exit(2)
        elif opt in ("-d","--result_dir"):
            params['result_root'] = arg
        elif opt in ("-v", "--verbose"):
            params['verbose'] = True
    main()
    