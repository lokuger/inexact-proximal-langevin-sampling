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

from inexact_pla import inexact_pla
from ista import ista
import potentials as pot
import distributions as pds
import running_moments

#%% initial parameters: test image, computation settings etc.
params = {
    'iterations': int(float('1e5')),
    'testfile_path': 'test-images/phantom256.png',
    'mu_tv': 1e00,
    'mean_intensity': 2,        # for MIV 2, mutv=1e00 has best mmse psnr on phantom. For MIV 10, mutv=0.5
    'iter_prox': 10,
    'step_type': 'fixed',       # 'bt' or 'fixed'
    'result_root': './results/poisson-deblur-tv',
    }

#%% auxiliary functions

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
    config = [(10,'fixsteps'),(1,'btsteps'),(10,'btsteps')]
    file_specifiers = ['{}_{:.0e}miv_{}proxiters_{:.0e}regparam_{}_{:.0e}samples'.format(\
        test_image_name,params['mean_intensity'],p,params['mu_tv'],s,params['iterations']\
        ) for p,s in config]
    results_files = [result_root+'/'+f+'.npz' for f in file_specifiers]
        
    #%% Ground truth
    # results_file = results_dir+'/result_images.npy'
    if not all([os.path.exists(f) for f in results_files]):
        print('Some results file(s) did not exist yet')
    else:
        #%% results were already computed, show images
        if len(results_files) == 1:
            R = np.load(results_files[0])
            logstd = np.log(R['std'])

            gt = R['x']
            vmax = np.max(gt)
            vmin = -0.02*vmax
            
            # ground truth
            my_imshow(gt, 'ground truth', vmin, vmax)
            # my_imsave(gt, result_root+'/ground_truth.png', vmin, vmax)
            # noisy image
            my_imshow(R['y'], 'blurred & noisy', vmin, vmax)
            # my_imsave(R['y'], result_root+'/noisy.png', vmin, vmax)
            # map
            my_imshow(R['u'], 'map estimate', vmin, vmax)
            # my_imsave(R['u'], result_root+'/map.png', vmin, vmax)
            # mean
            my_imshow(R['mn'], 'post. mean / mmse estimate', vmin, vmax)
            # my_imsave(R['mn'], mmse_file, vmin, vmax)
            # log(std)
            wmin, wmax = np.min(logstd), np.max(logstd)
            my_imshow(logstd, 'posterior log std',wmin,wmax)
            # my_imsave(logstd, logstd_file(0), np.min(logstd), np.max(logstd))
            for s,scale in [(R[k],k[-1]) for k in R if k.startswith('std_scale')]:
                logstd_scaled = np.log(s)
                wmin, wmax = np.min(logstd_scaled), np.max(logstd_scaled)
                my_imshow(logstd_scaled, 'posterior log std at scale', wmin, wmax)
                # my_imsave(logstd_scaled, logstd_file(int(scale)), np.min(logstd_scaled), np.max(logstd_scaled))
            print('MAP PSNR: {:.7f}'.format(10*np.log10(vmax**2/np.mean((R['u']-gt)**2))))
            print('Posterior mean PSNR: {:.7f}'.format(10*np.log10(vmax**2/np.mean((R['mn']-gt)**2))))
        else:
            acfs_fast,acfs_slow,acfs_med = [],[],[]
            fig,axs = plt.subplots(1,3,figsize=(10,3.6))
            linestyles = ['-xr','-+b','-ok','-vg','-hy']
            for f,ls,conf in zip(results_files,linestyles,config):
                R = np.load(f)
                acfs_fast.append(R['acf_fast'])
                acfs_slow.append(R['acf_slow'])
                acfs_med.append(R['acf_med'])
                legend_name = '{} prox its., '.format(conf[0]) + (r'BT $\gamma_k$' if conf[1] == 'btsteps' else r'$\gamma = 1/\tilde L$')
                axs[0].plot(range(101),R['acf_fast'],ls,label=legend_name,markersize=5,markevery=5)
                axs[1].plot(range(101),R['acf_med'],ls,label=legend_name,markersize=5,markevery=5)
                axs[2].plot(range(101),R['acf_slow'],ls,label=legend_name,markersize=5,markevery=5)
            # acf plots
            axs[0].set_title('Fastest component')
            axs[1].set_title('Median component')
            axs[2].set_title('Slowest component')
            for i in range(3): 
                axs[i].legend()#(loc='center right')
                axs[i].axis([-0.5, 100.5, -0.08, 1.02])
                axs[i].set_box_aspect(1)
                axs[i].set_xlabel('lag',labelpad=0)
                axs[i].set_ylabel('ACF',labelpad=0)
            axs[0].legend(loc='center right')
            axs[1].legend(loc='center right')
            axs[2].legend(loc='lower left')
            fig.tight_layout()
            # plt.show()
            plt.savefig(result_root+'/ACFs_{}_{}miv.pdf'.format(test_image_name,params['mean_intensity']),bbox_inches=None,pad_inches=0)
    
#%% gather parameters from shell and call main
if __name__ == '__main__':
    main()
    