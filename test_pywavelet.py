#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:28:22 2023

@author: lorenzkuger
"""

import pywt as wt
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio

def main():
    im_true = iio.imread('test_images/barbara.png').astype(float)
    n1,n2 = im_true.shape
    
    plt.imshow(im_true,cmap='Greys_r',vmin=0,vmax=255)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()