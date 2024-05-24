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
indPlot = 0

fs = 500
n = np.arange(1, 4 * fs)  # 2s时间，1s训练，1s测试
t = n / fs
signal_1 = np.sin(10 * 2 * np.pi * t)
signal_2 = np.where(signal_1 >= 0, 1, 0)

lTrain = 1000

signalT1 = signal_1[np.newaxis, np.newaxis, :, np.newaxis]
signalT2 = signal_2[np.newaxis, np.newaxis, :, np.newaxis]

signal = signalT1[:, :, :lTrain, :]
signal_A = signalT1[:, :, lTrain:, :]
signal_test = signalT2[:, :, :lTrain, :]
signal_B = signalT2[:, :, lTrain:, :]

fig = plt.figure(num=1)
fig.clf()
# fig.suptitle('SUP Title')
ax = fig.add_subplot(1, 1, 1)
# ax.title.set_text('Signal A')
ax.plot(np.arange(signal_A.shape[2]), signal_A[indPlot, 0, :, 0],'#1f77b4',linewidth=2)
fig.show()
fig = plt.figure(num=1)
fig.clf()
# fig.suptitle('SUP Title')
ax = fig.add_subplot(1, 1, 1)
# ax.title.set_text('Signal B')
ax.plot(np.arange(signal_B.shape[2]), signal_B[indPlot, 0, :, 0],'#2ca02c',linewidth=2)
fig.show()

ax = fig.add_subplot(1, 1, 1)
# ax.title.set_text('Reconstruction')
ax.plot(np.arange(signal_B.shape[2]), 0*np.arange(signal_B.shape[2]),'#ff7f0e',linewidth=2)
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