# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import LoadSig
import torch
import torch.nn as nn
from utils.torch_random_seed import seed_torch
from scipy import interpolate
import seaborn as sns
from paper_model.FMBWN import FMWN
from utils.utils_ae import lr_scheduler_choose
from matplotlib.colors import ListedColormap

def fft_fignal_plot(signal, fs, window, plot=True):
    sampling_rate = fs
    fft_size = window
    # t = np.arange(0, fft_size/sampling_rate, 1.0 / sampling_rate)
    t = np.arange(0, fft_size, 1)
    mean=signal.mean()
    xs = signal[:fft_size]-mean
    xf = np.fft.rfft(xs) / fft_size
    freqs = np.linspace(0.0, sampling_rate/2.0, int(fft_size / 2 + 1))
    xfp = np.abs(xf)
    # xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    if plot:
        plt.figure(figsize=(8, 4))
        plt.subplot(211)
        plt.plot(t[:fft_size], xs)
        plt.xlabel("Time/s")
        plt.ylabel('Amplitude')
        # plt.title("signal")

        plt.subplot(212)
        plt.plot(freqs, xfp)
        plt.xlabel("Frequency/Hz")
        # 字体FangSong
        plt.ylabel('Amplitude')
        plt.subplots_adjust(hspace=0.4)
        plt.title("Wavelet Filter")
        '''subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        有六个可选参数来控制子图布局。值均为0~1之间。其中left、bottom、right、top围成的区域就是子图的区域。
        wspace、hspace分别表示子图之间左右、上下的间距。实际的默认值由matplotlibrc文件控制的。
        '''
        plt.show()
m=[2, 5, 4]
k=[4, 4, 4]
len =[8,20,16]
'''
画滤波器
'''
signal=np.array([ ])
signal
signal = np.array(signal)
fs = 2
fft_fignal_plot(signal, fs, 8, plot=True)
'''
05-08 21:54:30 ******showing threshold******
DE 0 filter in 0the layer
[-0.24815607 -0.21851793 -0.4961758  -0.42893037  0.02545123  0.6401252 ]
DE 1 filter in 0the layer
[-0.5113824   0.5597847   0.71146053  0.18125066  0.22920062  0.22071534]
DE 2 filter in 0the layer
[ 0.01112822 -0.07029354 -0.56555235  0.77672565 -0.16065131 -0.12574951]
DE 0 filter in 1the layer
[-0.6608978  -0.06741503  0.43809927 -0.67156446  0.9670765   0.6502153 ]
DE 1 filter in 1the layer
[ 0.21920985  0.3348632   0.24077898 -0.06581198  0.31243315 -0.0797171 ]
DE 2 filter in 1the layer
[-0.14395787  0.70325047  0.5604545  -0.18545309  0.06064515  0.15568991]
DE 0 filter in 2the layer
[-0.0708092   0.19299702  0.34297755 -0.23440492 -0.28172767  0.3732569 ]
DE 1 filter in 2the layer
[ 0.2619728   0.8446085  -0.14999893  0.58814293  0.34187025 -1.001827  ]
DE 2 filter in 2the layer
[-0.26235     0.7904672  -0.2203485   0.35567698  0.04481345 -0.2781854 ]
05-08 21:54:30 ******showing threshold******
threshold in 0th layer of band!!******
Parameter containing:
tensor([[[[[0.1207]]]],



        [[[[0.1038]]]]], device='cuda:0', requires_grad=True)
Parameter containing:
tensor([[[[[0.0946]]]],



        [[[[0.1056]]]]], device='cuda:0', requires_grad=True)
threshold in 1th layer of band!!******
Parameter containing:
tensor([[[[[0.0987]]]],



        [[[[0.0956]]]]], device='cuda:0', requires_grad=True)
Parameter containing:
tensor([[[[[0.1004]]]],



        [[[[0.0962]]]]], device='cuda:0', requires_grad=True)
threshold in 2th layer of band!!******
Parameter containing:
tensor([[[[[0.1136]]]],



        [[[[0.0979]]]]], device='cuda:0', requires_grad=True)
Parameter containing:
tensor([[[[[0.1122]]]],



        [[[[0.0969]]]]], device='cuda:0', requires_grad=True)
'''
'''
**************************************************************
**************************************************************
**************************************************************
**************************************************************
**************************************************************
**************************************************************
**************************************************************
**************************************************************
**************************************************************
**************************************************************


05-08 21:55:39 ******showing threshold******
threshold in 0th layer of band!!******
Parameter containing:
tensor([[[[[0.5828]]]],



        [[[[0.4165]]]]], device='cuda:0', requires_grad=True)
Parameter containing:
tensor([[[[[0.4730]]]],



        [[[[0.3682]]]]], device='cuda:0', requires_grad=True)
threshold in 1th layer of band!!******
Parameter containing:
tensor([[[[[0.9780]]]],



        [[[[0.9652]]]]], device='cuda:0', requires_grad=True)
Parameter containing:
tensor([[[[[1.0693]]]],



        [[[[1.3294]]]]], device='cuda:0', requires_grad=True)
threshold in 2th layer of band!!******
Parameter containing:
tensor([[[[[1.3712]]]],



        [[[[1.3817]]]]], device='cuda:0', requires_grad=True)
Parameter containing:
tensor([[[[[1.4510]]]],



        [[[[1.2731]]]]], device='cuda:0', requires_grad=True)
05-08 21:55:39 ******showing threshold******
DE 0 filter in 0the layer
[ 0.267656    0.2799867  -0.07107019  0.09390606  0.37005377  0.74912673]
DE 1 filter in 0the layer
[-0.8529036   0.36444095  0.27968183 -0.0321711   0.10596506  0.0974424 ]
DE 2 filter in 0the layer
[-0.26767072 -0.7453283   0.24172947  0.26366773  0.08197679  0.42199355]
DE 0 filter in 1the layer
[-0.05344805  0.28328398  0.49691287 -0.15092446  0.6452249   0.41742948]
DE 1 filter in 1the layer
[ 0.4038598   0.01657198 -0.16329421 -0.21092956  0.58392674 -0.637635  ]
DE 2 filter in 1the layer
[ 0.42869157  0.3031246   0.16180003 -0.50334173 -0.37792158  0.00444915]
DE 0 filter in 2the layer
[ 0.4780581   0.61130494  0.30988145 -0.05540115 -0.0102992   0.39865974]
DE 1 filter in 2the layer
[ 0.08082478  0.4986522  -0.12442306  0.27058718  0.05500758 -0.7588104 ]
DE 2 filter in 2the layer
[-0.76242524  0.5391126  -0.1363213   0.02363831  0.04975597  0.26644725]
05-08 21:55:39 No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
05-08 21:55:39 last epoch's model performance
100%|████████████████████████████| 2000/2000 [00:02<00:00, 694.63it/s]
05-08 21:55:45 searching th: min_score: 0.25540602642210186, max_score: 0.3262808875925045, best_f1: 1.0, best th:0.28150102283101447 
05-08 21:55:45 [[126   0]
 [  0 126]]
05-08 21:55:45 Acc score is [1.00000],  F1 score is [1.00000] , Pre score is [1.00000], Re(FDR) score is [1.00000], auc score is [1.00000], FAR is [0.00000].

Process finished with exit code 0

'''