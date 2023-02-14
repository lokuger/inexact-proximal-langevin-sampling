#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 23:32:31 2023

@author: lorenzkuger
"""
import potentials as pot
import numpy as np
import matplotlib.pyplot as plt

def main():
    t = np.linspace(-5,5)
    l1norm = pot.l1_loss_unshifted_homoschedastic(scale=1)
    y = l1norm.prox(t, gamma=2)
    z = np.zeros_like(t)
    i = 0
    s = 0
    for tt in t:
        z[i],r = l1norm.inexact_prox(tt,gamma=2,epsilon=0.1)
        s += r
        i += 1
    print(s/t.size)
    plt.plot(t,z)
    plt.plot(t,y)
    plt.show()

if __name__ == '__main__':
    main()