# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:12:08 2023

@author: kugerlor
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

params = {
    'testfile_path': 'test-images/wheel.png',
    'result_root': './results/denoise-tv',
    }

def main():
    res_dir = params['result_root']
    test_image_name = params['testfile_path'].split('/')[-1].split('.')[0]
    
    fig1,ax1 = plt.subplots(1,1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r'$\eta$')
    ax1.set_ylabel('Total inner iterations')
    ax1.set_ylim([5e1,1e5])
    
    fig2,ax2 = plt.subplots(1,1)
    ax2.set_xscale("log")
    ax2.set_xlabel(r'$\eta$')
    ax2.set_ylabel('Number of samples '+r'$k^{\ast}$')
    ax2.set_ylim([0,300])
    colors = ['b','r','g','y','m','c','k','r']
    markers = ['^','v','o','s','*','H','X','D']
    
    mmse_errs = np.array([ 0.02, 0.015, 0.01, 0.005]) # 10.0**(-np.arange(0.0,3.0))   #tbc
    for i,mmse_err in enumerate(mmse_errs):
        res_file = res_dir+'/'+test_image_name+'_mmse-accuracy{}'.format(mmse_err)+'.npy'
        with open(res_file,'rb') as f:
            epsilons = np.load(f)
            n_samples = np.load(f)
            prox_its = np.load(f)
            prox_its_per_sample = np.load(f)
            
            eta = 10**np.arange(-2,-4.1,-0.2)
            I = n_samples<1e4
            
            s = '{:g}'.format(mmse_err)
            # ax1.plot(epsilons,prox_its,colors[i]+'-'+markers[i],label=r'$\mathrm{err}^{\mathrm{MMSE}}_{\mathrm{rel}}=$'+'{:s}'.format(s))
            ax1.plot(eta[I],prox_its[I],colors[i]+'-'+markers[i],label=r'$\kappa=$'+'{:s}'.format(s))
            # ax2.plot(epsilons,n_samples,colors[i]+'-'+markers[i],label=r'$\mathrm{err}^{\mathrm{MMSE}}_{\mathrm{rel}}=$'+'{:s}'.format(s))
            ax2.plot(eta[I],n_samples[I],colors[i]+'-'+markers[i],label=r'$\kappa=$'+'{:s}'.format(s))
            
    ax1.legend(loc='lower left')
    ax2.legend(loc='upper left')
    # ax.set_xlim(1,xmax+0.2)
    
    fig1.savefig(res_dir+'/prox_its_MMSE_accuracy.pdf')
    fig2.savefig(res_dir+'/n_samples_MMSE_accuracy.pdf')
        

if __name__ == '__main__':
    main()