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
'''
(default: cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', 
'#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
'#bcbd22', '#17becf'])).

'''

indPlot = 0

fs = 500
n = np.arange(1, 0.5 * fs)# 0.5时间
t = n / fs
signal_1 = np.sin(20 * 2 * np.pi * t)
signal_2 = np.sin(50* 2 * np.pi * t)

lTrain = 249

fig = plt.figure(num=1)
fig.clf()
# fig.suptitle('SUP Title')
ax = fig.add_subplot(1, 1, 1)
# ax.title.set_text('Signal A')
ax.plot(t,signal_1,'#ff7f0e',linewidth=2)#'#ff7f0e' '#1f77b4'
fig.show()
# fig = plt.figure(num=1)
# fig.clf()
# # fig.suptitle('SUP Title')
# ax = fig.add_subplot(1, 1, 1)
# # ax.title.set_text('Signal B')
# ax.plot(t, signal_2,'#2ca02c',linewidth=1)
# fig.show()

freq = 20  # 频率为100Hz
sampling_rate = fs  # 采样率为1000Hz
time = t
sin_wave = np.sin(2 * np.pi * freq * time)
sin_wave = np.where(sin_wave >= 0, 1, 0)
fig = plt.figure(num=1)
fig.clf()
ax = fig.add_subplot(1, 1, 1)
ax.plot(time,sin_wave,'#ff7f0e',linewidth=2) #'#2ca02c'
fig.show()


# fig = plt.figure(num=1)
# fig.clf()
# # fig.suptitle('SUP Title')
# ax = fig.add_subplot(1, 1, 1)
# ax.title.set_text('')
# ax.plot(np.arange(signal.shape[2]), signal[indPlot, 0, :, 0], 'b')
# ax.plot(np.arange(signal.shape[2]), out[0][indPlot, 0, :, 0].detach().cpu().numpy(), 'r')
# x = np.arange(signal.shape[2], signalT2.shape[2])
#
# y = signal_test[indPlot, 0, :, 0].squeeze()
# ax.plot(x, y, 'g')  # signal B
#
# x = np.arange(signal.shape[2], signalT2.shape[2])
# y = outTe[0][indPlot, 0, :, 0].detach().cpu().numpy()
# ax.plot(x, y, 'g')
# ax.legend(['Signal A', 'Reconstructed signal A', 'Signal B', 'Reconstructed Signal B'])