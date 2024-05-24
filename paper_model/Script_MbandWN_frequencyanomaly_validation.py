# -*- coding: utf-8 -*-
"""

"""
# Usual packages
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
import pywt


def Kernel_selector(band=3):
    kernelselection = band
    if kernelselection == 1:
        # Initialise wavelet kernel (here db-4)
        KI = [np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                        -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965]),
              np.array(
                  [-0.230377813308855, 0.714846570552542, -0.630880767929590, -0.027983769416984, 0.187034811718881,
                   0.030841381835987, -0.032883011666983, -0.010597401784997])]
    elif kernelselection == 2:
        KI = [np.array([-0.129409522550921, 0.224143868041857, 0.836516303737469, 0.482962913144690]),
              np.array([-0.482962913144690, 0.836516303737469, -0.224143868041857, -0.129409522550921])]
    elif kernelselection == 4:
        KI = [np.array(
            [-0.0190928308000000, 0.0145382757000000, 0.0229906779000000, 0.0140770701000000, 0.0719795354000000,
             -0.0827496793000000, -0.130694890900000, -0.0952930728000000, -0.114536126100000, 0.219030893900000,
             0.414564773700000, 0.495502982800000, 0.561649421500000, 0.349180509700000, 0.193139439300000,
             0.0857130200000000]),
            np.array([-0.115281343300000, 0.0877812188000000, 0.138816305600000, 0.0849964877000000,
                      -0.443703932000000, 0.169154971800000, 0.268493699200000, 0.0722022649000000,
                      -0.0827398180000000, -0.426427736100000, -0.255040161600000, 0.600591382300000,
                      -0.0115563891000000, -0.101106504400000, 0.118328206900000, -0.104508652500000]),
            np.array([-0.0280987676000000, 0.0213958651000000, 0.0338351983000000, 0.0207171125000000,
                      0.220295183000000, -0.208864350300000, -0.330053682700000, -0.224561804100000,
                      0.556231311800000, -0.0621881917000000, 0.00102740000000000, 0.447749675200000,
                      -0.248427727200000, -0.250343323000000, -0.204808915700000, 0.256095016300000]),
            np.array([-0.0174753464000000, 0.0133066389000000, 0.0210429802000000, 0.0128845052000000,
                      -0.0918374833000000, 0.0443561794000000, 0.0702950474000000, 0.0290655661000000,
                      -0.0233349758000000, -0.0923899104000000, -0.0823301969000000, 0.0446493766000000,
                      -0.137950244700000, 0.688008574600000, -0.662289313000000, 0.183998602200000]),
        ]
    elif kernelselection == 3:  # 相位偏移
        KI = [np.array(
            [-0.145936007553990, 0.0465140821758900, 0.238964171905760, 0.723286276743610, 0.530836187013740,
             0.338386097283860]),
            np.array([0.426954037816980, -0.136082763487960, -0.699119564792890, -0.0187057473531300,
                      0.544331053951810, -0.117377016134830]),
            np.array([0.246502028665230, -0.0785674201318500, -0.403636868928920, 0.460604752521310,
                      -0.628539361054710, 0.403636868928920]),
        ]
    elif kernelselection == 5:
        KI = [np.array(
            [0.0272993452180114, 0.0265840663469593, 0, -0.0428593893779949, -0.0810269414612686, -0.117023723786349,
             -0.102653539494725, -2.73839349132101e-17, 0.165500189580178, 0.347337063987899, 0.478068455182483,
             0.509357208870298, 0.447213595499958, 0.315935108535685, 0.161069337685743, 0.0588695188858126,
             0.0139258597774260, 1.28985929088956e-32, 0.00863768676208944, 0.0198341352875850]),
            np.array([0.0104274220026634, -0.0164298565616031, 0, 0.0264885593726670, -0.0309495376337566,
                      -0.189348362576392, -0.268750455462670, -2.73839349132101e-17, 0.433285121465458,
                      0.562003175085017, 0.182605900930553, -0.314800067496623, -0.447213595499958,
                      -0.195258635314441, 0.0615230124505194, 0.0952528824585986, 0.0364583742198662,
                      1.54826577665454e-31, 0.0226137575273252, 0.0320923050327763]),
            np.array([0.0337438464306959, -5.13472509819086e-18, 0, -2.75943716886759e-18, -0.100154807655024,
                      -1.21909088566198e-17, 0.332193831935890, 8.21518047396304e-17, -0.535569863770559,
                      -1.80918635666264e-16, 0.590925108503860, 1.18092256227073e-15, -0.447213595499958,
                      4.07025734526344e-16, 0.199092650470447, -1.10450832039150e-16, -0.0450650288848805,
                      -1.80623763483245e-31, -0.0279521415304715, 2.89478922531347e-17]),
            np.array([-0.0104274220026635, -0.0164298565616031, 0, 0.0264885593726670, 0.0309495376337565,
                      -0.189348362576392, 0.268750455462670, 8.21518047396304e-17, -0.433285121465458,
                      0.562003175085017, -0.182605900930552, -0.314800067496625, 0.447213595499958,
                      -0.195258635314440, -0.0615230124505196, 0.0952528824585985, -0.0364583742198662,
                      -1.97821887361772e-31, -0.0226137575273252, 0.0320923050327762]),
            np.array([0.0272993452180114, -0.0265840663469593, 0, 0.0428593893779949, -0.0810269414612686,
                      0.117023723786349, -0.102653539494725, -1.36919674566051e-16, 0.165500189580178,
                      -0.347337063987900, 0.478068455182483, -0.509357208870298, 0.447213595499958,
                      -0.315935108535685, 0.161069337685743, -0.0588695188858126, 0.0139258597774258,
                      2.23619073179563e-31, 0.00863768676208948, -0.0198341352875852]),
        ]
    return KI


# Set sparsity (dummy) loss:
# the sparsity term has no ground truth => just input an empty numpy array as ground truth (anything would do, in coeffLoss, yTrue is not called)
class CoeffLoss(nn.Module):
    def __init__(self, loss=True, mode='DWT'):
        super(CoeffLoss, self).__init__()
        self.loss = loss
        self.mode = mode

    def forward(self, all_coefficients):
        if self.loss:
            if self.mode == 'WPT':
                node_number = len(all_coefficients)  # 分解出的节点小波系数
                for i in range(node_number):
                    if i == 0:
                        vLossCoeff = torch.mean(torch.abs(all_coefficients[i]))
                    else:
                        vLossCoeff = vLossCoeff + torch.mean(torch.abs(all_coefficients[i]))
                # vLossCoeff = torch.div(vLossCoeff, node_number)
                return torch.div(vLossCoeff, node_number)
            else:
                list_len = len(all_coefficients)
                node_number = 0
                # 原版程序
                for i in range(list_len):
                    lev_coefficients = all_coefficients[i]
                    lev_len = len(lev_coefficients)
                    for j in range(lev_len):
                        if i == 0 and j == 0:
                            vLossCoeff = torch.mean(torch.abs(lev_coefficients[j]))
                        else:
                            vLossCoeff = vLossCoeff + torch.mean(torch.abs(lev_coefficients[j]))
                        node_number += 1
                return torch.div(vLossCoeff, node_number)

                # for i in range(list_len):
                #     lev_coefficients = all_coefficients[i]
                #     lev_len = len(lev_coefficients)
                #     for j in range(lev_len):
                #         if i==0 and j==0:
                #             vLossCoeff = torch.sum(torch.abs(lev_coefficients[j]))
                #         else:
                #             vLossCoeff = vLossCoeff + torch.sum(torch.abs(lev_coefficients[j]))
                #         # node_number += 1
                #         node_number += lev_coefficients[j].numel()

                return torch.div(vLossCoeff, node_number)
        else:
            # return all_coefficients
            return 0


# Set residual loss:
class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()

    def forward(self, yTrue, yPred):
        return torch.mean(torch.abs(yTrue - yPred))


def Two_interplate(coeff, now_len, target_len):
    '''
    :param coeff: numpy
    :param len: int
    :param target_len: int
    :return: coeff with length target_len
    '''

    interplated_signal = coeff
    len_list = []
    len_temp = 2 * now_len
    while len_temp < target_len:
        len_list.append(len_temp)
        len_temp = 2 * len_temp
    len_list.append(target_len)
    x = np.linspace(1, target_len, now_len)  # 初始数目

    for i in range(len(len_list)):
        f = interpolate.interp1d(x, interplated_signal, kind='nearest')
        x = np.linspace(1, target_len, len_list[i])
        interplated_signal = f(x)
    return interplated_signal


def get_alpha_blend_cmap(cmap, alpha):
    cls = plt.get_cmap(cmap)(np.linspace(0, 1, 256))
    cls = (1 - alpha) + alpha * cls
    return ListedColormap(cls)



def coefficient_visul_batch(all_coeff, mode='DWT', m=None, sort='freq', prefined_sort=None,title='coeffifient',batch_idx=0):
    '''
    :param all_coeff: list consis of tensor
    prefined_sort:list [0,2,3,1] dwt后小波的顺序
    :param mode:
        WPT：低频到高频 按节点来排
        DWT：从低频到高频，最后一个是低频，倒数（m-1）个共同组成最后一个M带分解。
            以后每一个m-1
            长度不同的需要复制插值
    :param m:
    :return: 从上至下：高频至低频的时频图
    '''
    if mode == 'DWT':
        if isinstance(prefined_sort, list):
            decomposition_level = len(m)
            coeff = all_coeff[0][0].detach().cpu().numpy()
            target_len = coeff.shape[2]
            coeff = coeff.reshape(1, target_len)
            for lev in range(decomposition_level):  # 分解层级
                overlap_height = 1
                coeff_lev = all_coeff[lev]
                for height_level in range(lev, decomposition_level - 1):  # 最后一级高度为1 不用乘
                    overlap_height = overlap_height * m[height_level + 1]
                for m_band in prefined_sort:  # 根据顺序组装
                    coeff_temp = coeff_lev[m_band].detach().cpu().numpy()
                    length = coeff_temp.shape[2]
                    coeff_temp = coeff_temp.reshape(1, length)
                    if length != target_len:
                        coeff_temp = Two_interplate(coeff_temp, length, target_len)
                    for i in range(overlap_height):
                        coeff = np.concatenate([coeff, coeff_temp])
            # 最后一个组装
            last_lowpass_ceff = all_coeff[decomposition_level][0].detach().cpu().numpy()
            length = last_lowpass_ceff.shape[2]
            coeff_temp = last_lowpass_ceff.reshape(1, length)
            if length != target_len:
                coeff_temp = Two_interplate(coeff_temp, length, target_len)

            coeff = np.concatenate([coeff, coeff_temp])
            coeff = coeff[1:]
        else:
            decomposition_level = len(m)
            coeff = all_coeff[0][0].detach().cpu().numpy()
            target_len = coeff.shape[2]
            batch_number=coeff.shape[0]

            coeff = coeff.reshape(batch_number, target_len)
            coeff = coeff[batch_idx,:]
            coeff = coeff.reshape(1, target_len)
            for lev in range(decomposition_level):  # 分解层级
                overlap_height = 1
                coeff_lev = all_coeff[lev]
                for height_level in range(lev, decomposition_level - 1):  # 最后一级高度为1 不用乘
                    overlap_height = overlap_height * m[height_level + 1]
                for m_band in range(m[lev] - 2, -1, -1):  # 带内组装(DWT分解后当前带是M-1个) 逆向
                    coeff_temp = coeff_lev[m_band].detach().cpu().numpy()
                    length = coeff_temp.shape[2]
                    # coeff_temp = coeff_temp.reshape(1, length)
                    coeff_temp = coeff_temp.reshape(batch_number, length)
                    coeff_temp = coeff_temp[batch_idx, :]
                    coeff_temp = coeff_temp.reshape(1, length)
                    if length != target_len:
                        coeff_temp = Two_interplate(coeff_temp, length, target_len)
                    for i in range(overlap_height):
                        coeff = np.concatenate([coeff, coeff_temp])
            # 最后一个组装
            last_lowpass_ceff = all_coeff[decomposition_level][0].detach().cpu().numpy()
            length = last_lowpass_ceff.shape[2]
            # coeff_temp = last_lowpass_ceff.reshape(1, length)
            coeff_temp = last_lowpass_ceff.reshape(batch_number, length)
            coeff_temp = coeff_temp[batch_idx, :].reshape(1, length)

            if length != target_len:
                coeff_temp = Two_interplate(coeff_temp, length, target_len)

            coeff = np.concatenate([coeff, coeff_temp])
            coeff = coeff[1:]
        plt.show()
        # ax = sns.heatmap(coeff)
        ax = sns.heatmap(coeff, vmin=-2.0, vmax=2.0)  # cmap='YlGnBu_r' 也还可以 蓝绿'crest' ,vmin=-10,vmax=50
        # ax = sns.heatmap(coeff, cmap="rocket_r", linewidths=0.0, edgecolor="none", alpha=0.5)
        # ax = sns.heatmap(coeff,cmap=get_alpha_blend_cmap("rocket_r", 0.5),linewidths=0.0)  # cmap='YlGnBu_r' 也还可以
        # plt.show()
        # ax = sns.heatmap(coeff, cmap='YlGnBu_r')
        plt.title(title)
        plt.show()

    else:
        node_number = len(all_coeff)
        if sort == 'freq':
            # https://zhuanlan.zhihu.com/p/528435064
            for lev in range(len(m)):  # 递归推算
                if lev == 0:
                    idx = [i for i in range(m[lev])]
                else:
                    idx_temp = []
                    for i in range(len(idx)):
                        m_lev = m[lev]
                        # 从0开始计数序号则偶不变奇变
                        # if i == len(idx)-1:
                        if i % 2 == 0:  # 偶不变
                            temp = [t for t in range(idx[i] * m_lev, (idx[i] + 1) * m_lev)]
                        else:  # 奇变
                            temp = [t for t in range((idx[i] + 1) * m_lev - 1, idx[i] * m_lev - 1, -1)]
                        idx_temp.extend(temp)
                    idx = idx_temp
            assert len(idx) == node_number
            coeff = all_coeff[idx[node_number - 1]].detach().cpu().numpy()
            length = coeff.shape[2]
            coeff = coeff.reshape(1, length)
            for i in range(node_number - 2, -1, -1):
                coeff_temp = all_coeff[idx[i]].detach().cpu().numpy().reshape(1, length)
                coeff = np.concatenate([coeff, coeff_temp])
        else:
            node_number = len(all_coeff)
            coeff = all_coeff[node_number - 1].detach().cpu().numpy()
            length = coeff.shape[2]
            coeff = coeff.reshape(1, length)
            for i in range(node_number - 2, -1, -1):
                coeff_temp = all_coeff[i].detach().cpu().numpy().reshape(1, length)
                coeff = np.concatenate([coeff, coeff_temp])

        # ax = sns.heatmap(coeff,cmap='YlGnBu_r')
        # plt.show()
        # ax = sns.heatmap(coeff, cmap=sns.cubehelix_palette(as_cmap=True))#渐变色盘：sns.cubehelix_palette()使用)
        # plt.show()
        # cmap = plt.get_cmap('Set3')
        # ax = sns.heatmap(coeff, cmap=cmap)
        # plt.show()
        ax = sns.heatmap(coeff)  # cmap='YlGnBu_r' 也还可以
        plt.show()

def coefficient_visul(all_coeff, mode='DWT', m=None, sort='freq', prefined_sort=None,title='coeffifient'):
    '''
    :param all_coeff: list consis of tensor
    prefined_sort:list [0,2,3,1] dwt后小波的顺序
    :param mode:
        WPT：低频到高频 按节点来排
        DWT：从低频到高频，最后一个是低频，倒数（m-1）个共同组成最后一个M带分解。
            以后每一个m-1
            长度不同的需要复制插值
    :param m:
    :return: 从上至下：高频至低频的时频图
    '''
    if mode == 'DWT':
        if isinstance(prefined_sort, list):
            decomposition_level = len(m)
            coeff = all_coeff[0][0].detach().cpu().numpy()
            target_len = coeff.shape[2]
            coeff = coeff.reshape(1, target_len)
            for lev in range(decomposition_level):  # 分解层级
                overlap_height = 1
                coeff_lev = all_coeff[lev]
                for height_level in range(lev, decomposition_level - 1):  # 最后一级高度为1 不用乘
                    overlap_height = overlap_height * m[height_level + 1]
                for m_band in prefined_sort:  # 根据顺序组装
                    coeff_temp = coeff_lev[m_band].detach().cpu().numpy()
                    length = coeff_temp.shape[2]
                    coeff_temp = coeff_temp.reshape(1, length)
                    if length != target_len:
                        coeff_temp = Two_interplate(coeff_temp, length, target_len)
                    for i in range(overlap_height):
                        coeff = np.concatenate([coeff, coeff_temp])
            # 最后一个组装
            last_lowpass_ceff = all_coeff[decomposition_level][0].detach().cpu().numpy()
            length = last_lowpass_ceff.shape[2]
            coeff_temp = last_lowpass_ceff.reshape(1, length)
            if length != target_len:
                coeff_temp = Two_interplate(coeff_temp, length, target_len)

            coeff = np.concatenate([coeff, coeff_temp])
            coeff = coeff[1:]
        else:
            decomposition_level = len(m)
            coeff = all_coeff[0][0].detach().cpu().numpy()
            target_len = coeff.shape[2]
            coeff = coeff.reshape(1, target_len)
            for lev in range(decomposition_level):  # 分解层级
                overlap_height = 1
                coeff_lev = all_coeff[lev]
                for height_level in range(lev, decomposition_level - 1):  # 最后一级高度为1 不用乘
                    overlap_height = overlap_height * m[height_level + 1]
                for m_band in range(m[lev] - 2, -1, -1):  # 带内组装(DWT分解后当前带是M-1个) 逆向
                    coeff_temp = coeff_lev[m_band].detach().cpu().numpy()
                    length = coeff_temp.shape[2]
                    coeff_temp = coeff_temp.reshape(1, length)
                    if length != target_len:
                        coeff_temp = Two_interplate(coeff_temp, length, target_len)
                    for i in range(overlap_height):
                        coeff = np.concatenate([coeff, coeff_temp])
            # 最后一个组装
            last_lowpass_ceff = all_coeff[decomposition_level][0].detach().cpu().numpy()
            length = last_lowpass_ceff.shape[2]
            coeff_temp = last_lowpass_ceff.reshape(1, length)
            if length != target_len:
                coeff_temp = Two_interplate(coeff_temp, length, target_len)

            coeff = np.concatenate([coeff, coeff_temp])
            coeff = coeff[1:]
        plt.show()
        # ax = sns.heatmap(coeff)
        ax = sns.heatmap(coeff, vmin=-1.5, vmax=1.5)  # cmap='YlGnBu_r' 也还可以 蓝绿'crest' ,vmin=-10,vmax=50
        # ax = sns.heatmap(coeff, cmap="rocket_r", linewidths=0.0, edgecolor="none", alpha=0.5)
        # ax = sns.heatmap(coeff,cmap=get_alpha_blend_cmap("rocket_r", 0.5),linewidths=0.0)  # cmap='YlGnBu_r' 也还可以
        # plt.show()
        # ax = sns.heatmap(coeff, cmap='YlGnBu_r')
        plt.title(title)
        plt.show()

    else:
        node_number = len(all_coeff)
        if sort == 'freq':
            # https://zhuanlan.zhihu.com/p/528435064
            for lev in range(len(m)):  # 递归推算
                if lev == 0:
                    idx = [i for i in range(m[lev])]
                else:
                    idx_temp = []
                    for i in range(len(idx)):
                        m_lev = m[lev]
                        # 从0开始计数序号则偶不变奇变
                        # if i == len(idx)-1:
                        if i % 2 == 0:  # 偶不变
                            temp = [t for t in range(idx[i] * m_lev, (idx[i] + 1) * m_lev)]
                        else:  # 奇变
                            temp = [t for t in range((idx[i] + 1) * m_lev - 1, idx[i] * m_lev - 1, -1)]
                        idx_temp.extend(temp)
                    idx = idx_temp
            assert len(idx) == node_number
            coeff = all_coeff[idx[node_number - 1]].detach().cpu().numpy()
            length = coeff.shape[2]
            coeff = coeff.reshape(1, length)
            for i in range(node_number - 2, -1, -1):
                coeff_temp = all_coeff[idx[i]].detach().cpu().numpy().reshape(1, length)
                coeff = np.concatenate([coeff, coeff_temp])
        else:
            node_number = len(all_coeff)
            coeff = all_coeff[node_number - 1].detach().cpu().numpy()
            length = coeff.shape[2]
            coeff = coeff.reshape(1, length)
            for i in range(node_number - 2, -1, -1):
                coeff_temp = all_coeff[i].detach().cpu().numpy().reshape(1, length)
                coeff = np.concatenate([coeff, coeff_temp])

        # ax = sns.heatmap(coeff,cmap='YlGnBu_r')
        # plt.show()
        # ax = sns.heatmap(coeff, cmap=sns.cubehelix_palette(as_cmap=True))#渐变色盘：sns.cubehelix_palette()使用)
        # plt.show()
        # cmap = plt.get_cmap('Set3')
        # ax = sns.heatmap(coeff, cmap=cmap)
        # plt.show()
        ax = sns.heatmap(coeff)  # cmap='YlGnBu_r' 也还可以
        plt.show()


def coefficient_visul_pywt(all_coeff, mode='DWT'):
    '''
    :param all_coeff: list consis of tensor
    :param mode:
        WPT：低频到高频 按节点来排
        DWT：从低频到高频，最后一个是低频，倒数（m-1）个共同组成最后一个M带分解。
            以后每一个m-1
            长度不同的需要复制插值
    :param m:
    :return: 从上至下：高频至低频的时频图
    '''

    node_number = len(all_coeff)
    coeff = all_coeff[node_number - 1]
    length = coeff.shape[0]
    coeff = coeff.reshape(1, length)
    for i in range(node_number - 2, -1, -1):
        coeff_temp = all_coeff[i].reshape(1, length)
        coeff = np.concatenate([coeff, coeff_temp])

    # ax = sns.heatmap(coeff,cmap='YlGnBu_r')
    # plt.show()
    # ax = sns.heatmap(coeff, cmap=sns.cubehelix_palette(as_cmap=True))#渐变色盘：sns.cubehelix_palette()使用)
    # plt.show()
    # cmap = plt.get_cmap('Set3')
    # ax = sns.heatmap(coeff, cmap=cmap)
    # plt.show()
    ax = sns.heatmap(coeff)  # cmap='YlGnBu_r' 也还可以
    plt.show()


def fft_fignal_plot(signal, fs, window, plot=True):
    sampling_rate = fs
    fft_size = window
    # t = np.arange(0, fft_size/sampling_rate, 1.0 / sampling_rate)
    t = np.arange(0, fft_size, 1)
    mean = signal.mean()
    xs = signal[:fft_size] - mean
    xf = np.fft.rfft(xs) / fft_size
    freqs = np.linspace(0.0, sampling_rate / 2.0, int(fft_size / 2 + 1))
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


# def kernel_visual(kernel):
#     t=1
if __name__ == '__main__':
    seed_torch(666)
    generating_signal = 3

    # Load a toy time series data to run DeSPAWN
    # signal: batch_size*channel*length*1

    if generating_signal == 1:
        # 生成信号
        fs = np.power(2, 13)
        n = np.arange(1, 13 * fs)
        FS_varying = 0.05 * n
        t = n / fs
        # generation_signal = np.cos(np.pi*FS_varying*t)
        generation_signal = np.cos(np.pi * FS_varying * n)

        length_noise = n.shape[0]
        noise = np.random.randn(length_noise)

        signalT = generation_signal
        lTrain = 81920
        signalT = signalT[np.newaxis, np.newaxis, :, np.newaxis]
        signal = signalT[:, :, :lTrain, :]
        signal_test = signalT[:, :, lTrain:, :]
    elif generating_signal == 0:
        signal = pd.read_csv("monthly-sunspots.csv")
        lTrain = 2000  # length of the training section
        signalT = ((signal['Sunspots'] - signal['Sunspots'].mean()) / signal['Sunspots'].std()).values[np.newaxis,
                  np.newaxis, :, np.newaxis]
        signal = signalT[:, :, :lTrain, :]
        signal_test = signalT[:, :, lTrain:, :]
    elif generating_signal == 2:
        fs = np.power(2, 13)
        n = np.arange(1, 13 * fs)
        t = n / fs
        generation_signal = np.sin(-400 * np.cos(2 * np.pi * t) + 4000 * np.pi * t)

        length_noise = n.shape[0]
        noise = np.random.normal(loc=0, scale=0.2, size=length_noise)

        signalT = generation_signal  # + noise
        lTrain = 81920
        signalT = signalT[np.newaxis, np.newaxis, :, np.newaxis]
        signal = signalT[:, :, :lTrain, :]
        signal_test = signalT[:, :, lTrain:, :]
    elif generating_signal == 3:
        #共分2个通道，一个低频0-1000Hz，一个高频1000-2000Hz，500Hz和1500Hz，采频4000Hz，
        fs = 500
        n = np.arange(1, 4 * fs)# 2s时间，1s训练，1s测试
        t = n / fs
        signal_1 = np.sin(10 * 2 * np.pi * t)
        signal_2 = np.sin(240* 2 * np.pi * t)

        lTrain = 1000

        signalT1 = signal_1[np.newaxis, np.newaxis, :, np.newaxis]
        signalT2 = signal_2[np.newaxis, np.newaxis, :, np.newaxis]

        signal = signalT1[:, :, :lTrain, :]
        signal_A = signalT1[:, :, lTrain:, :]
        signal_test = signalT2[:, :, :lTrain, :]
        signal_B = signalT2[:, :, lTrain:, :]

    else:
        raise Exception('Not implemented')

    # plt.figure(figsize=(50,8))
    # plt.plot(generation_signal[8000:8000+3000])
    # plt.show()

    # Number of decomposition level is max log2 of input TS
    level = np.floor(np.log2(signal.shape[2])).astype(int)
    # Train hard thresholding (HT) coefficient?
    trainHT = True
    # Initialise HT value
    initHT = 1.0
    # Which loss to consider for wavelet coeffs ('l1' or None)
    lossCoeff = 'l1'
    # Weight for sparsity loss versus residual?
    lossFactor = 1.0
    # Train wavelets? (Trainable kernels)
    kernTrainable = True
    # Which training mode?
    # cf (https://arxiv.org/pdf/2105.00899.pdf -- https://doi.org/10.1073/pnas.2106598119) [Section 4.4 Ablation Study]
    #   CQF => learn wavelet 0 infer all other kernels from the network
    #   PerLayer => learn one wavelet per level, infer others
    #   PerFilter => learn wavelet + scaling function per level + infer other
    #   Free => learn everything
    lossCoeff_setting = True
    coeffLoss = CoeffLoss(lossCoeff_setting)
    recLoss = RecLoss()

    epochs = 2000
    # generates model: model outputs the reconstructed signals, the loss on the wavelet coefficients and wavelet coefficients
    # m=[5,5,5]
    # k=[4,4,4]
    m = [2]
    k = [2]
    level = len(m)
    mode = 'DWT'
    visualization_kernel = True
    device = 'cuda'
    # Kernel_I = Kernel_selector(band=3)
    model = FMWN(kernelInit=2, m=m, k=k, level=level, kernelsConstraint='PL', mode=mode, device=device,
                 coefffilter=5.0, realconv=True, initHT=0.01, kernTrainable=True, threshold=True,t=30.0)
    model.cuda()

    opt = torch.optim.Adam(params=model.parameters(), lr=0.03, betas=(0.9, 0.999), eps=1e-07)
    lr_scheduler = lr_scheduler_choose(lr_scheduler_way='step', optimizer=opt, steps='800,1300,1400', gamma=0.1)

    kernels = model.MBAND
    kernellen = m[0] * k[0]

    threshold_list = model.thre  # 每一层有一个
    for i in range(m[0]-1):
        print('Before'+'*' * 6 + 'threshold in first layer of band!!'+str(i) + '*' * 6)
        print(threshold_list[i].left)
        print(threshold_list[i].right)
    print('low pass')
    print(model.thre_a.left)
    print(model.thre_a.right)


    for i in range(m[0]):
        kerneltemp = kernels[0].DeFilter[i].kernel  # 方便的获取滤波器系数
        print('DE filter')
        print(torch.sum(kerneltemp))
        print(kerneltemp.detach().cpu().numpy().reshape(kernellen))
        # kerneltemp = kernels[0].ReFilter[i].kernel  # 方便的获取滤波器系数
        # print('Corresponding RE filter')
        # print(torch.sum(kerneltemp))
        # print(kerneltemp.detach().cpu().numpy().reshape(kernellen))
    # for name,param in model.named_parameters():
    #     print(name)
    print("TRAINING STAGE")
    train_iter = LoadSig(signal)  # batch_size * c * h * w
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        # train
        for i, sig in enumerate(train_iter):
            print('current lr: {}'.format(lr_scheduler.get_lr()))
            opt.zero_grad()
            # sigPred,vLossCoeff,g_last,hl = model(sig)
            # r=recLoss(sig,sigPred)
            # c=coeffLoss(sigPred)
            # loss=r+c
            # loss.backward(retain_graph=True)
            sig = sig.cuda()
            # sigPred, coeff, loss2, _ = model(sig)
            sigPred, coeff, loss2, hard_threshold_max = model(sig)
            loss1 = coeffLoss(coeff)
            r = recLoss(sig, sigPred)
            loss = r
            # loss = 8.0*r +20.0*loss1 +10.0*loss2
            # loss = r + 0.3 * loss1 + 0.3 * loss2
            # loss = r + 0.3 * loss1 + 0.3 * loss2- 0.1*torch.log(hard_threshold_max+1) # 重构+稀疏+正交+阈值最大化
            # loss = r + 10.0*loss2
            # loss =  loss1
            # loss =  0.01 * loss2
            loss.backward()
            opt.step()
            running_loss += r.item()
            lr_scheduler.step()
        print('[epoch:%d/%d] loss: %.3f'
              % (epoch + 1, epochs, running_loss / len(train_iter)))

    coefficient_visul(coeff, mode=mode, m=m,title='train')  # ,prefined_sort=[1,2,0,3]
    # print('*'*6+'pywt 可视化'+'*'*6)
    # max_level=5
    # wp = pywt.WaveletPacket(data=signal.reshape(lTrain), wavelet='db4', mode='symmetric',maxlevel=max_level)
    # node_list = [node.path for node in wp.get_level(max_level, 'freq')]#'natural' 'freq'
    # coeff_pywt =[wp[i].data for i in node_list]
    # coefficient_visul_pywt(coeff_pywt)

    # Examples for plotting the model outputs and learnings
    indPlot = 0
    out = []
    outTe = []
    print("TESTING STAGE")
    model.eval()
    for i in range(m[0]-1):
        print('*' * 6 + 'threshold in first layer of band!!'+str(i) + '*' * 6)
        print(threshold_list[i].left)
        print(threshold_list[i].right)
    print('low pass')
    print(model.thre_a.left)
    print(model.thre_a.right)
    print('*' * 6 + 'After traning' + '*' * 6)
    for i in range(m[0]):
        kerneltemp = kernels[0].DeFilter[i].kernel  # 方便的获取滤波器系数
        print('DE filter')
        print(torch.sum(kerneltemp))
        print(kerneltemp.detach().cpu().numpy().reshape(kernellen))
        # kerneltemp = kernels[0].ReFilter[i].kernel  # 方便的获取滤波器系数
        # print('Corresponding RE filter')
        # print(torch.sum(kerneltemp))
        # print(kerneltemp.detach().cpu().numpy().reshape(kernellen))

        if visualization_kernel:
            fft_fignal_plot(kerneltemp.detach().cpu().numpy().reshape(kernellen), fs=2, window=kernellen)

    test_iter_A = LoadSig(signal_A)
    for i, sig in enumerate(test_iter_A):
        sig = sig.cuda()
        out = model(sig)  # [sigPred, vLossCoeff, g_last, hl]
    coefficient_visul(out[1], mode=mode, m=m, title=' ')  # ,prefined_sort=[1,2,0,3]
    # Test part of the signal
    test_iter = LoadSig(signal_B)

    for i, sig in enumerate(test_iter):
        sig = sig.cuda()
        outTe = model(sig)
        # print(outTe[0].squeeze())
    coefficient_visul(outTe[1], mode=mode, m=m,title=' ')  # ,prefined_sort=[1,2,0,3]


    fig = plt.figure(num=1)
    fig.clf()
    # fig.suptitle('SUP Title')
    ax = fig.add_subplot(1, 1, 1)
    # ax.title.set_text('Test')
    # ax.plot(np.arange(signal.shape[2]), signal[indPlot, 0, :, 0],'#1f77b4')
    ax.plot(np.arange(signal_A.shape[2]), out[0][indPlot, 0, :, 0].detach().cpu().numpy(),'#ff7f0e')
    fig.show()

    # ax.legend(['Signal A', 'Reconstructed signal A'])
    # x =np.arange(signal_A.shape[2], signalT2.shape[2])
    # y =signal_test[indPlot, 0,:, 0].squeeze()
    # ax.plot(x,y, '#2ca02c')#signal B
    fig = plt.figure(num=1)
    fig.clf()
    x = np.arange(signal.shape[2], signalT2.shape[2])
    y = outTe[0][indPlot, 0, :, 0].detach().cpu().numpy()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x,y,'#ff7f0e')
    # ax.legend(['Signal B', 'Reconstructed Signal B'])
    fig.show()

    # ax = fig.add_subplot(2,2,3)
    # idpl = 0
    # # for e,o in enumerate(out[2:]):
    # #     ax.boxplot(np.abs(np.squeeze(o[indPlot,:,:,:].detach().numpy())), positions=[e], widths=0.8)
    # # ax.set_xlabel('Decomposition Level')
    # # ax.set_ylabel('Coefficient Distribution')
    # trainYLim = ax.get_ylim()
    # trainXLim = ax.get_xlim()
    # ax = fig.add_subplot(2,2,4)
    # idpl = 0
    # for e,o in enumerate(outTe[2:]):
    #     # print(o.shape[2])
    #     if o.shape[2]>1:
    #         ax.boxplot(np.abs(np.squeeze(o[indPlot,:,:,:])), positions=[e], widths=0.8)
    #     else:
    #         ax.plot(e,np.abs(np.squeeze(o[indPlot,:,:,:])),'o',color='k')
    # ax.set_xlabel('Decomposition Level')
    # ax.set_ylabel('Coefficient Distribution')
    # ax.set_ylim(trainYLim)
    # ax.set_xlim(trainXLim)
