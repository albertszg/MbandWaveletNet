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
def list_generator(mean, dis, number):  # 封装一下这个函数，用来后面生成数据
    return np.random.normal(mean, dis * dis, number)  # normal分布，输入的参数是均值、标准差以及生成的数量

list1 = list_generator(0.8531, 0.0956, 70)
list2 = list_generator(0.8631, 0.0656, 80)
list3 = list_generator(0.8731, 0.1056, 90)
list4 = list_generator(0.8831, 0.0756, 100)
s1 = pd.Series(np.array(list1))
s2 = pd.Series(np.array(list2))
s3 = pd.Series(np.array(list3))
s4 = pd.Series(np.array(list4))
# 把四个list导入到pandas的数据结构中，dataframe
data = pd.DataFrame({"1-2": s1, "2": s2, "3": s3, "4": s4})
data.boxplot(grid=False)  # 这里，pandas自己有处理的过程，很方便哦。
plt.xlabel("level-band")  # 我们设置横纵坐标的标题。
plt.show()


# 数据
# 数据
data = [800, 600, 500]
x_labels = ["1-1", "1-2", "1-3"]

# 画图
plt.bar(range(len(data)), data,color='b')

# 指定横坐标刻度
plt.xticks(range(len(data)), x_labels)

# 显示图形
plt.show()

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100
b = [1, 2, 3, 4, 5] * 40
c = [1, 2, 3, 4, 5] * 100

# 绘制箱形图
data = [a, b, c]
fig, ax = plt.subplots()
ax.boxplot(data,patch_artist=True,vert=True,notch=True)

# 显示图形
plt.show()