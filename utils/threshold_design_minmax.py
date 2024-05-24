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

def exp_calculcation_a(x,t,b):
    # exp_x = torch.exp(-t*F.relu(x-b))# x较大时使用
    exp_x = torch.exp(-t * F.relu(1- x/b))# x和b较小时使用
    return exp_x

def wavelet_threshold(x,t,left,right):
    return F.relu(x - torch.mul(F.relu(torch.mul(torch.sign(x),right)), exp_calculcation(x,t,right))) - F.relu(-x - torch.mul(F.relu(torch.mul(torch.sign(-x),left)), exp_calculcation(-x,t,left)))

def wavelet_threshold_sigmoid(x,t,left,right):
    return Relu(x-torch.mul(Relu(torch.sign(x)*right),2.0-2.0*torch.sigmoid(t*(x-right))))\
           -Relu(-x-torch.mul(Relu(torch.sign(-x)*left),2.0-2.0*torch.sigmoid(t*(-x-left))))

# olga fink组的
def HardThresholdAssym(x,t,left,right):
    return torch.multiply(x, torch.sigmoid(t * (x - right)) + torch.sigmoid(-t * (x + left)))

def wavelet_threshold_normal_min(x,t,left,right,left1,right1):
    maximization = torch.max(torch.abs(x))
    x = torch.div(x,maximization)
    # thresholded_x = F.relu(x - torch.mul(F.relu(torch.mul(torch.sign(x),right)), exp_calculcation(x,t,right))) - F.relu(-x - torch.mul(F.relu(torch.mul(torch.sign(-x),left)), exp_calculcation(-x,t,left)))
    thresholded_x_r = x
    thresholded_x_r = F.relu(x - torch.mul(F.relu(torch.mul(torch.sign(x),right)), exp_calculcation(x,t,right)))
    # thresholded_x_r = F.relu(torch.mul(F.relu(torch.mul(torch.sign(thresholded_x_r), right1)),
    #                                  exp_calculcation(thresholded_x_r , 100.0, right1)))
    # thresholded_x_l = - F.relu(-x - torch.mul(F.relu(torch.mul(torch.sign(-x),left)), exp_calculcation(-x,t,left)))

    # thresholded_x = F.relu(torch.mul(F.relu(torch.mul(torch.sign(thresholded_x),right1)), exp_calculcation(thresholded_x,t,right1)))
    return torch.mul(thresholded_x_r,maximization)

def HardThresholdAssym(x,t,left,right):
    return torch.multiply(x, torch.sigmoid(t * (x - right)) + torch.sigmoid(-t * (x + left)))

def HardThresholdAssymplus(x,t,l,r,l1=0.0,r1=0.0):
    return torch.multiply(x, (torch.sigmoid(t * (x - r))-torch.sigmoid(t * (x - r1))) + (torch.sigmoid(-t * (x + l))-torch.sigmoid(-t * (x + l1))))
#1.需要对系数进行归一化处理
# 需要注意是用batch的maximization近似替代整个数据集的maximization?
# (后期需要对比验证有用性，幅值差异变化)
#2.需要保留中间部分

def wavelet_threshold_normal_min_max(x,t,left,right,left1,right1,norm=False):
    if norm:
        maximization = torch.max(torch.abs(x))
        x = torch.div(x,maximization)
    # thresholded_x = HardThresholdAssym(x,t,left,right)
    thresholded_x = HardThresholdAssymplus(x,t,left,right,left1,right1)
    if norm:
        thresholded_x =torch.mul(thresholded_x,maximization)
    return thresholded_x

x = torch.range(-2.0,2.0,0.0001,requires_grad=True)

Relu=nn.ReLU()
ifNorm = False
t=[1.0,10.,20.,100.0]
left=torch.tensor(.5,requires_grad=True)
left1=torch.tensor(1.5,requires_grad=True)
right=torch.tensor(.5,requires_grad=True)
right1=torch.tensor(1.5,requires_grad=True)
for i in range(len(t)):
    thresholded_x=wavelet_threshold_normal_min_max(x,t[i],left,right,left1, right1,norm=ifNorm)
    # thresholded_x = wavelet_threshold_sigmoid(x, t[i], left, right)
    plt.plot(x.detach().numpy(),thresholded_x.detach().numpy(),linewidth=3.0)

plt.plot(x.detach().numpy(),x.detach().numpy(),linestyle='--',color='0.5',alpha=0.5)
plt.axvline(0,linestyle='--',color='0.5',alpha=0.5)
plt.axhline(0,linestyle='--',color='0.5',alpha=0.5)
# plt.title('Thresholded x')
# plt.xlabel('x')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(labels=['t=0.01','t=0.3','t=3.0','t=10.0'],fontsize=18)
# plt.savefig('threshold_design.png',dpi=1000,bbox_inches='tight')
plt.show()

# 
# plt.figure()
# ## 画输入系数w梯度
# for i in t:
#     x = torch.range(-4.0, 4.0, 0.1, requires_grad=True)
#     thresholded_x = wavelet_threshold_normal_min(x, i, left, right,left1, right1)
#     loss = torch.sum(torch.abs(thresholded_x))
#     loss.backward()
#     gradient = np.squeeze(x.grad.numpy())
#     data = np.squeeze(x.detach().numpy())
#     indices = np.argsort(data)
#     plt.plot(data[indices], gradient[indices],linewidth=3.0)
#     plt.legend(labels=['t=0.01','t=0.3','t=3.0','t=10.0'],fontsize=18)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# # plt.title('Gradient of x')
# plt.axvline(0,linestyle='--',color='0.5',alpha=0.5)
# plt.axhline(0,linestyle='--',color='0.5',alpha=0.5)
# 
# plt.show()
# ## 画阈值梯度
# plt.figure()
# # t_left = torch.range(-4.0,0.0,0.0001,requires_grad=True)
# t_right = np.arange(0.0,4.1,0.1)
# t_left = np.arange(4.0,-0.1,-0.1)
# # t=[1.,3.0,10.]
# t=[0.01,1.,3.,10.0]
# color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
# for it in range(len(t)):
#     gradient_list = []
#     data_list = []
#     gradient_list_left = []
#     data_list_left = []
#     for i in range(np.size(t_right)):
#         x = torch.range(-4.0, 4.0, 0.1, requires_grad=True)
#         right = torch.tensor(t_right[i], requires_grad=True)
#         left = torch.tensor(t_left[i], requires_grad=True)
#         thresholded_x = wavelet_threshold(x, t[it], left, right)
#         loss = torch.sum(torch.abs(thresholded_x))
#         loss.backward()
#         gradient_list.append(right.grad.item())
#         data_list.append(right.detach().item())
#         gradient_list_left.append(-left.grad.item())
#         data_list_left.append(-left.detach().item())
# 
#     #拼接一下
#     final_data = data_list_left + data_list
#     final_gradient =gradient_list_left + gradient_list
#     # plt.plot(data_list,gradient_list,linewidth=3.0,color=color[it])
#     # plt.plot(data_list_left,gradient_list_left,linewidth=3.0,color=color[it])
#     data_temp=np.array(final_data)
#     data_temp[np.isnan(data_temp)] = 0
#     gradient_temp=np.array(final_gradient)
#     gradient_temp[np.isnan(gradient_temp)] = 0
#     plt.plot(data_temp,gradient_temp,linewidth=3.0,color=color[it])
# plt.legend(labels=['t=0.01','t=0.3','t=3.0','t=10.0'],fontsize=18)
# # plt.legend(labels=['right','left'])
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.axvline(0,linestyle='--',color='0.5',alpha=0.5)
# plt.axhline(0,linestyle='--',color='0.5',alpha=0.5)
# # plt.title('Gradient of threshold')
# plt.show()