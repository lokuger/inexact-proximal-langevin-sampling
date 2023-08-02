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
    
    plt.figure()
    ax1 = plt.axes()
    ax1.set_title('#Prox iterations until given MMSE accuracy is reached')
    ax1.set_xscale("log")
    ax1.set_xlabel(r'$\varepsilon$')
    ax1.set_ylabel('Total inner iterations')
    plt.figure()
    ax2 = plt.axes()
    ax2.set_title('#Samples until given MMSE accuracy is reached')
    ax2.set_xscale("log")
    ax2.set_xlabel(r'$\varepsilon$')
    ax2.set_ylabel('Number of samples')
    colors = ['b','r','g','y','m','c','k','r']
    markers = ['^','v','o','s','*','H','X','D']
    
    mmse_errs = np.array([0.025, 0.02, 0.015]) # 10.0**(-np.arange(0.0,3.0))   #tbc
    for i,mmse_err in enumerate(mmse_errs):
        res_file = res_dir+'/'+test_image_name+'_mmse-accuracy{}'.format(mmse_err)+'.npy'
        with open(res_file,'rb') as f:
            epsilons = np.load(f)
            n_samples = np.load(f)
            prox_its = np.load(f)
            prox_its_per_sample = np.load(f)
            
            s = '{:g}'.format(mmse_err)
            ax1.plot(epsilons,prox_its,colors[i]+'-'+markers[i],label=r'$\mathrm{err}^{\mathrm{MMSE}}_{\mathrm{rel}}=$'+'{:s}'.format(s))
            ax2.plot(epsilons,n_samples,colors[i]+'-'+markers[i],label=r'$\mathrm{err}^{\mathrm{MMSE}}_{\mathrm{rel}}=$'+'{:s}'.format(s))
            
    ax1.legend(loc='lower left')
    # ax.set_xlim(1,xmax+0.2)
    
    plt.savefig(res_dir+'/MMSE_err_plots.pdf')
        

if __name__ == '__main__':
    main()