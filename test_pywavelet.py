#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:28:22 2023

@author: lorenzkuger
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import sys

#%% auxiliary functions
def wavelet_operators(slices, wav_type, wav_level):
    a = lambda x : pywt.coeffs_to_array(pywt.wavedec2(x, wav_type, level=wav_level))[0]
    at = lambda c : pywt.waverec2(pywt.array_to_coeffs(c, slices, output_format='wavedec2'), wav_type)
    return at, a #rec, dec

def my_imshow(im, label, vmin=-0.02, vmax=1.02, cbar=False):
    fig = plt.figure()
    plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
    q = plt.imshow(im, cmap='Greys_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(label)
    if cbar: fig.colorbar(q)
    plt.show()

#%% main
def main():
    try:
        x = io.imread('test_images/fibo.jpeg',as_gray=True).astype(float)
    except FileNotFoundError:
        print('Provided test image did not exist under that path, aborting.')
        sys.exit()
    # handle images that are too large or colored
    if x.shape[0] > 512 or x.shape[1] > 512: x = transform.resize(x, (512,512))
    x = x-np.min(x)
    x = x/np.max(x)
    # assume quadratic images
    # n = x.shape[0]
    
    wav_type = 'haar'
    wav_level = 3
    _,slices = pywt.coeffs_to_array(pywt.wavedec2(x, wav_type, level=wav_level))
    a,at = wavelet_operators(slices, wav_type, wav_level) # daubechies is orthogonal wavelet so at = inverse of a
    
    my_imshow(x, 'Ground truth')
    my_imshow(at(a(x)), 'Recon from Wav coeff')

if __name__ == '__main__':
    main()