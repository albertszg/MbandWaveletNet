# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import random

cmap2 = sns.dark_palette("green", as_cmap=True)

def kdeplot(normal,abnormal):
    # sns.set()#切换到seaborn的默认运行配置
    sns.distplot(normal,kde=True,hist=False,kde_kws={"shade":True,"linewidth":4},color='deepskyblue')#'#1f77b4'
    sns.distplot(abnormal,kde=True,hist=False,kde_kws={"shade":True,"linewidth":4},color='limegreen')#'#2ca02c' color='green'

    plt.legend(labels=['Normal','Abnormal'],fontsize=24)
    plt.show()

normal = np.random.normal(size=200,scale=0.03)+0.2
abnormal = np.random.normal(size=200,scale=0.03)+0.6
kdeplot(normal,abnormal)
