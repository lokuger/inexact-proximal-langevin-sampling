# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:31:37 2023

@author: kugerlor
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

params = {
    'step_type': 'fixed', # 'decay' or 'fixed'
    'inexactness_type': 'decay', # 'decay' or 'fixed'
    }

def main():
    ax = plt.axes()
    # ax.set_title('Squared Wasserstein-2 distances')
    # ax.set_xscale("log")
    ax.set_xlabel(r'$k$')
    ax.set_yscale("log")
    ax.set_ylabel(r'$\mathcal{W}_2^2(\tilde\mu^{k},\tilde\mu^{\ast})$')
    colors = ['b','r','g','y','m','c','k','r']
    markers = ['^','v','o','s','*','H','X','D']
    
    res_dir = './results/wasserstein-dists-validation/steps_'+params['step_type']+'_inexactness_'+params['inexactness_type']+'/'
    if params['inexactness_type'] == 'fixed':
        if params['step_type'] == 'fixed':
            epsilons = np.array([0.5, 0.25, 0.1, 0.05])#10.0**(-np.arange(0.0,3.0))   #tbc
            xmax = 0
            for ie,epsilon in enumerate(epsilons):
                res_file = res_dir+'W2dists_epsilon'+str(epsilon)+'.npy'
                res_file_ub = res_dir+'W2dists_ub_epsilon'+str(epsilon)+'.npy'
                steps_file = res_dir+'steps'+str(epsilon)+'.npy'
                W2sq = np.load(res_file)
                W2sq_ub = np.load(res_file_ub)
                K = np.load(steps_file)
                if np.max(K) > xmax: xmax = np.max(K)
                
                s = '{:g}'.format(epsilon) if epsilon > 0 else '0'
                ax.plot(K,W2sq,colors[ie]+'-'+markers[ie],label=r'$\mathcal{W}_2^2(\tilde\mu^k,\tilde\mu^\ast), \epsilon = $'+'{:s}'.format(s))
                ax.plot(K,W2sq_ub,colors[ie]+'--'+markers[ie])
                # ax.plot(K,W2sq_ub,colors[ie]+'--',label=r'upper bound $\epsilon = ${:s}'.format(s))
    else:
        if params['step_type'] == 'fixed':
            rates = np.array([-0.1, -0.2, -0.4, -0.6])       #tbc
            xmax = 0                       #tbc
            for ir,rate in enumerate(rates):
                res_file = res_dir+'W2dists_rate'+str(rate)+'.npy'
                res_file_ub = res_dir+'W2dists_ub_rate'+str(rate)+'.npy'
                steps_file = res_dir+'steps'+str(rate)+'.npy'
                W2sq = np.load(res_file)
                W2sq_ub = np.load(res_file_ub)
                K = np.load(steps_file)
                if np.max(K) > xmax: xmax = np.max(K)
                
                s = r'$k^{'+'{:.1f}'.format(rate)+'}$'
                ax.plot(K,W2sq,colors[ir]+'-'+markers[ir],label=r'$\mathcal{W}_2^2(\tilde\mu^k,\tilde\mu^\ast), \epsilon_k = $'+'{:s}'.format(s))
                ax.plot(K,W2sq_ub,colors[ir]+'--'+markers[ir])
                # ax.plot(K,W2sq_ub,colors[ir]+'--',label=r'upper bound $\epsilon_k \propto ${:s}'.format(s))
        else:
            rate = np.array([-0.2,-0.4,-0.6])       #tbc
    ax.legend(loc='lower left')
    ax.set_xlim(1,xmax+0.2)
    
    plt.savefig(res_dir+'/W2_plots.pdf')
        

if __name__ == '__main__':
    main()