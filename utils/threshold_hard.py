# !/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch as t
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import torch.nn as nn
'''
对比MAE，MSE，DM+区别
loss幅值
对x求梯度
'''

def exp_calculcation(x,t,b):
    # exp_x = torch.exp(-t*F.relu(x-b))# x较大时使用
    exp_x = torch.exp(-t * F.relu(x/b - 1))# x和b较小时使用
    return exp_x

def wavelet_threshold(x,t,left,right):
    return F.relu(x - torch.mul(F.relu(torch.mul(torch.sign(x),right)), exp_calculcation(x,t,right))) - F.relu(-x - torch.mul(F.relu(torch.mul(torch.sign(-x),left)), exp_calculcation(-x,t,left)))

def wavelet_threshold_sigmoid(x,t,left,right):
    return Relu(x-torch.mul(Relu(torch.sign(x)*right),2.0-2.0*torch.sigmoid(t*(x-right))))\
           -Relu(-x-torch.mul(Relu(torch.sign(-x)*left),2.0-2.0*torch.sigmoid(t*(-x-left))))

x = torch.range(-4.0,4.0,0.0001,requires_grad=True)
Relu=nn.ReLU()


left=torch.tensor(1.,requires_grad=True)
right=torch.tensor(1.,requires_grad=True)




thresholded_x= torch.ge(x,right)*x + torch.le(x,-left)*x

plt.plot(x.detach().numpy(),thresholded_x.detach().numpy(),linewidth=3.0)

plt.plot(x.detach().numpy(),x.detach().numpy(),linestyle='--',color='0.5',alpha=0.5)
plt.axvline(0,linestyle='--',color='0.5',alpha=0.5)
plt.axhline(0,linestyle='--',color='0.5',alpha=0.5)
# plt.title('Thresholded x')
# plt.xlabel('x')
# plt.title('Hard threshold')
# plt.savefig('threshold_design.png',dpi=1000,bbox_inches='tight')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
## 画x梯度
plt.figure()
loss = torch.sum(torch.abs(thresholded_x))
loss.backward()
gradient = np.squeeze(x.grad.numpy())
data=np.squeeze(x.detach().numpy())
indices=np.argsort(data)
plt.plot(data[indices],gradient[indices],linewidth=3.0)
# plt.title('Gradient of x')
plt.axvline(0,linestyle='--',color='0.5',alpha=0.5)
plt.axhline(0,linestyle='--',color='0.5',alpha=0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
## 画阈值梯度
plt.figure()

plt.plot(x.detach().numpy(),x.detach().numpy()*0,linewidth=3.0)
plt.axvline(0,linestyle='--',color='0.5',alpha=0.5)
plt.axhline(0,linestyle='--',color='0.5',alpha=0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.title('Gradient of threshold')
plt.show()