#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pickle import load,dump
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch import optim
import random
import torch.nn as nn
from anomaly_Datasets.RY_data import RY_local,RY #original dataset
from anomaly_Datasets.HYFJ_data import HYFJ_local,HYFJ #original dataset
from anomaly_Datasets.MIMI_data import MIMI_local,MIMI #original dataset
from anomaly_Datasets.IMS_data_polluted import IMS_local,IMS
from anomaly_Datasets.gear_data import PG,PG_local
import seaborn as sns
import logging
from datetime import datetime

def seed_all(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)# 为CPU设置随机种子
    torch.cuda.manual_seed(seed)# 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    print('Setting the seed_torch done!')

def normalize(seq):
    return (seq - np.min(seq)) / (np.max(seq) - np.min(seq))

def anomaly_scoring_mband(values, reconstruction_values,coefficients_list=None,EMA_coefficient=None):
    creterion = nn.MSELoss(reduction='none')
    values = np.squeeze(values)
    reconstruction_values =np.squeeze(reconstruction_values)
    # coefficients_list = np.squeeze(coefficients_list)
    # EMA_coefficient = np.squeeze(EMA_coefficient)

    if values.ndim == 3:  # fc,各通道分离重构
        n = np.size(values, 2) * np.size(values, 1)
        score = (creterion(input=torch.from_numpy(reconstruction_values), target=torch.from_numpy(values)) / n).sum(
            (-1, -2)).numpy()
    elif values.ndim == 2:  # conv ae
        n = np.size(values, 1)
        score = (creterion(input=torch.from_numpy(reconstruction_values), target=torch.from_numpy(values)) / n).sum(
            -1).numpy()
    else:
        raise Exception('data dim is not right! in anomaly socoring')
    # score_z = np.sum((coefficients_list-EMA_coefficient),axis=(-2,-1))
    return score  # (11154,)

def anomaly_scoring(values, reconstruction_values):
    creterion = nn.MSELoss(reduction='none')
    if values.ndim==3:#fc,各通道分离重构
        n=np.size(values,2)*np.size(values,1)
        score = (creterion(input=torch.from_numpy(reconstruction_values), target=torch.from_numpy(values)) / n).sum(
            (-1, -2)).numpy()
    elif values.ndim==2:#conv ae
        n=np.size(values,1)
        score = (creterion(input=torch.from_numpy(reconstruction_values), target=torch.from_numpy(values)) / n).sum(
            -1).numpy()
    elif values.ndim == 4:  # mband coefficients
        n = np.size(values, 2)
        score = (creterion(input=torch.from_numpy(reconstruction_values), target=torch.from_numpy(values)) / n).sum(
            (-1, -2)).numpy()
        score = np.squeeze(score)
    else:
        raise Exception('data dim is not right! in anomaly socoring')
    return score#(11154,)

# def plot_training(values, re_values, labels, epoch=0):#每一次训练画一次图,并返回偏度和峭度
#     scores = anomaly_scoring(values, re_values)  # 计算每个样本的mse
#     skewness,kurtosis=calculate_normalindex(scores)
#     kdeplot(labels,scores,title='training: ecpoh'+str(epoch))
#     return skewness,kurtosis

# def calculate_normalindex(R):
#     if isinstance(R,np.ndarray):
#         data = list(R)
#         return pd.Series(data).skew(), pd.Series(data).kurt()
#     else:
#         raise Exception("not numpy")

def metrics_calculate(values, re_values, labels,plot_confusion_matrix=False, title='none',savedir=None):#plot kde or normalized histograms
    scores = anomaly_scoring(values, re_values)#计算每个样本的mse，并
    preds, _ = evaluate(labels, scores, title=title,savedir=savedir)#基于阈值得到预测输出，根据得分搜索最佳F1对应的阈值
    #阳性-正例-1
    #隐形-负例-0
    acc = accuracy_score(y_true=labels,y_pred=preds)#准确率
    f1 = f1_score(y_true=labels, y_pred=preds)#F指标
    pre = precision_score(y_true=labels, y_pred=preds)#精准率，TP/(TP+FP), 鉴别为异常的样本里面实际为异常的概率（检出无误率），越大越好
    re = recall_score(y_true=labels, y_pred=preds)#召回率，故障检出率（FDR），TP/(TP+FN), 异常样本被检出的概率（异常检出率），越大越好
    auc = roc_auc_score(y_true=labels, y_score=normalize(scores))
    C=confusion_matrix(y_true=labels,y_pred=preds)
    logging.info(C)
    TN=float(C[0][0])
    FN=float(C[1][0])
    TP=float(C[1][1])
    FP=float(C[0][1])

    FAR= FP/(FP+TN)#虚警率（false alarm rate）=False positive rate， FP/(TN+FP), 正常样本被误报为异常的概率，误报警概率
    FDR=TP/(TP+FN)
    # logging.info('Re score is [%.5f], FDR score is [%.5f]'%(re,FDR))
    if plot_confusion_matrix:
        disp=ConfusionMatrixDisplay(confusion_matrix=C,display_labels=['Noraml','Abnormal'])
        disp.plot(cmap='Blues')
        plt.show()

    logging.info('Acc score is [%.5f],  F1 score is [%.5f] , Pre score is [%.5f], Re(FDR) score is [%.5f], auc score is [%.5f], FAR is [%.5f].' % (acc, f1, pre, re, auc,FAR))

def metrics_calculate_mband(values, re_values, labels,coefficients_list,EMA_coefficient,plot_confusion_matrix=False, title='none',savedir=None):#plot kde or normalized histograms
    scores = anomaly_scoring_mband(values, re_values,coefficients_list,EMA_coefficient)#计算每个样本的mse，并
    preds, _ = evaluate(labels, scores, title=title,savedir=savedir)#基于阈值得到预测输出，根据得分搜索最佳F1对应的阈值
    #阳性-正例-1
    #隐形-负例-0
    acc = accuracy_score(y_true=labels,y_pred=preds)#准确率
    f1 = f1_score(y_true=labels, y_pred=preds)#F指标
    pre = precision_score(y_true=labels, y_pred=preds)#精准率，TP/(TP+FP), 鉴别为异常的样本里面实际为异常的概率（检出无误率），越大越好
    re = recall_score(y_true=labels, y_pred=preds)#召回率，故障检出率（FDR），TP/(TP+FN), 异常样本被检出的概率（异常检出率），越大越好
    auc = roc_auc_score(y_true=labels, y_score=normalize(scores))
    C=confusion_matrix(y_true=labels,y_pred=preds)
    logging.info(C)
    TN=float(C[0][0])
    FN=float(C[1][0])
    TP=float(C[1][1])
    FP=float(C[0][1])

    FAR= FP/(FP+TN)#虚警率（false alarm rate）=False positive rate， FP/(TN+FP), 正常样本被误报为异常的概率，误报警概率
    FDR=TP/(TP+FN)
    # logging.info('Re score is [%.5f], FDR score is [%.5f]'%(re,FDR))
    if plot_confusion_matrix:
        disp=ConfusionMatrixDisplay(confusion_matrix=C,display_labels=['Noraml','Abnormal'])
        disp.plot(cmap='Blues')
        plt.show()

    logging.info('Acc score is [%.5f],  F1 score is [%.5f] , Pre score is [%.5f], Re(FDR) score is [%.5f], auc score is [%.5f], FAR is [%.5f].' % (acc, f1, pre, re, auc,FAR))



def metrics_calculate_gmm(scores, labels,plot_confusion_matrix=False, title='none',savedir=None):#plot kde or normalized histograms

    preds, _ = evaluate(labels, scores, title=title,savedir=savedir)#基于阈值得到预测输出，根据得分搜索最佳F1对应的阈值
    #阳性-正例-1
    #隐形-负例-0
    acc = accuracy_score(y_true=labels,y_pred=preds)#准确率
    f1 = f1_score(y_true=labels, y_pred=preds)#F指标
    pre = precision_score(y_true=labels, y_pred=preds)#精准率，TP/(TP+FP), 鉴别为异常的样本里面实际为异常的概率（检出无误率），越大越好
    re = recall_score(y_true=labels, y_pred=preds)#召回率，故障检出率（FDR），TP/(TP+FN), 异常样本被检出的概率（异常检出率），越大越好
    auc = roc_auc_score(y_true=labels, y_score=normalize(scores))
    C=confusion_matrix(y_true=labels,y_pred=preds)
    logging.info(C)
    TN=float(C[0][0])
    FN=float(C[1][0])
    TP=float(C[1][1])
    FP=float(C[0][1])

    FAR= FP/(FP+TN)#虚警率（false alarm rate）=False positive rate， FP/(TN+FP), 正常样本被误报为异常的概率，误报警概率
    FDR=TP/(TP+FN)
    # logging.info('Re score is [%.5f], FDR score is [%.5f]'%(re,FDR))
    if plot_confusion_matrix:
        disp=ConfusionMatrixDisplay(confusion_matrix=C,display_labels=['Noraml','Abnormal'])
        disp.plot(cmap='Blues')
        plt.show()

    logging.info('Acc score is [%.5f],  F1 score is [%.5f] , Pre score is [%.5f], Re(FDR) score is [%.5f], auc score is [%.5f], FAR is [%.5f].' % (acc, f1, pre, re, auc,FAR))



def metrics_calculate_dsvdd(scores, labels,plot_confusion_matrix=False, title='none',savedir=None):#plot kde or normalized histograms

    preds, _ = evaluate(labels, scores, title=title,savedir=savedir)#基于阈值得到预测输出，根据得分搜索最佳F1对应的阈值
    #阳性-正例-1
    #隐形-负例-0
    acc = accuracy_score(y_true=labels,y_pred=preds)#准确率
    f1 = f1_score(y_true=labels, y_pred=preds)#F指标
    pre = precision_score(y_true=labels, y_pred=preds)#精准率，TP/(TP+FP), 鉴别为异常的样本里面实际为异常的概率（检出无误率），越大越好
    re = recall_score(y_true=labels, y_pred=preds)#召回率，故障检出率（FDR），TP/(TP+FN), 异常样本被检出的概率（异常检出率），越大越好
    auc = roc_auc_score(y_true=labels, y_score=normalize(scores))
    C=confusion_matrix(y_true=labels,y_pred=preds)
    logging.info(C)
    TN=float(C[0][0])
    FN=float(C[1][0])
    TP=float(C[1][1])
    FP=float(C[0][1])

    FAR= FP/(FP+TN)#虚警率（false alarm rate）=False positive rate， FP/(TN+FP), 正常样本被误报为异常的概率，误报警概率
    FDR=TP/(TP+FN)
    # logging.info('Re score is [%.5f], FDR score is [%.5f]'%(re,FDR))
    if plot_confusion_matrix:
        disp=ConfusionMatrixDisplay(confusion_matrix=C,display_labels=['Noraml','Abnormal'])
        disp.plot(cmap='Blues')
        plt.show()

    logging.info('Acc score is [%.5f],  F1 score is [%.5f] , Pre score is [%.5f], Re(FDR) score is [%.5f], auc score is [%.5f], FAR is [%.5f].' % (acc, f1, pre, re, auc,FAR))


def kdeplot(labels,scores,title,savedir=None):
    # sns.set()#切换到seaborn的默认运行配置
    # index_normal = np.where(labels == 0)
    # index_abnormal = np.where(labels == 1)
    try:
        index_normal,_ = np.where(labels == 0)
        index_abnormal,_ = np.where(labels == 1)
    except:
        index_normal = np.where(labels == 0)
        index_abnormal = np.where(labels == 1)
    # sns.distplot(scores[index_normal],kde=True,hist=False,kde_kws={"shade":True},color='g')
    # sns.distplot(scores[index_abnormal],kde=True,hist=False,kde_kws={"shade":True},color='r')
    sns.kdeplot(scores[index_normal],fill=True,color='green')
    sns.kdeplot(scores[index_abnormal],fill=True,color='red')

    plt.title(title)
    plt.legend(labels=['Normal','Abnormal'])
    if savedir is not None:
        plt.savefig(os.path.join(savedir,datetime.strftime(
        datetime.now(), '%m%d-%H%M%S')+'.png'))
    plt.show()

def evaluate(labels, scores, step=2000, title='none',savedir=None):
    # best f1
    min_score = min(scores)
    max_score = max(scores)
    #plot the kde
    kdeplot(labels,scores,title,savedir)
    best_f1 = 0.0
    best_preds = None
    best_th = None
    for th in tqdm(np.linspace(min_score, max_score, step), ncols=70):#ncols:宽度
        preds = (scores > th).astype(int)
        f1 = f1_score(y_true=labels, y_pred=preds)
        if f1 > best_f1:
            best_f1 = f1
            best_preds = preds
            best_th =th
    logging.info('searching th: min_score: {}, max_score: {}, best_f1: {}, best th:{} '.format(min_score, max_score, best_f1, best_th))
    return best_preds, best_f1


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = load(f)
        return data

def save_pickle(data,file_path):
    with open(file_path, 'wb') as f:
        dump(data,f)


def get_from_one(ts, window_size, stride):
    ts_length = ts.shape[0]#lenth*channel
    samples = []
    for start in np.arange(0, ts_length, stride):
        if start + window_size > ts_length:
            break
        samples.append(ts[start:start+window_size])
    return np.array(samples)


def remove_all_same(train_x, test_x):
    # 去除样本中的异常值 最大等于最小
    remove_idx = []
    for col in range(train_x.shape[1]):
        if max(train_x[:, col]) == min(train_x[:, col]):
            remove_idx.append(col)
        else:
            train_x[:, col] = normalize(train_x[:, col])

        if max(test_x[:, col]) == min(test_x[:, col]):
            remove_idx.append(col)
        else:
            test_x[:, col] = normalize(test_x[:, col])

    all_idx = set(range(train_x.shape[1]))
    remain_idx = list(all_idx - set(remove_idx))
    return train_x[:, remain_idx], test_x[:, remain_idx]

def lr_scheduler_choose(lr_scheduler_way='fix', optimizer=None, steps='2', gamma=0.1):
    # 选择学习率更新策略
    if lr_scheduler_way == 'step':
        steps = [int(step) for step in steps.split(',')]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=gamma)
    elif lr_scheduler_way == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)# lr = base_lr * gamma^epoch
    elif lr_scheduler_way == 'stepLR':
        steps = int(steps)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, gamma)#lr*gamma
    elif lr_scheduler_way == 'cos':
        steps = int(steps)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 0)
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.001)
    elif lr_scheduler_way == 'fix':
        lr_scheduler = None
    else:
        raise Exception("lr schedule not implement")
    return lr_scheduler

def load_data_v(val_size, window_size=160000, stride=1, batch_size=64,dim=1, normalization=None,dataloder=False, snr=None, typelist=None, idlist=None):
    # MIMI data 数据集
    # 最大长度160000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datadir = r'/usr/data_disk1/szg/data/anomaly_detection/MIMI'
    savedir = './local_data/' + 'MIMI' + '.pkl'
    normlizetype = normalization

    if dataloder:
        Dataset = MIMI_local(savedir, normlizetype)

    else:
        Dataset = MIMI(datadir, normlizetype, snr, typelist, idlist, window_size=window_size, stride=stride, save=True,
                       savedir=savedir, dim=dim, test_size=val_size)
    datasets = {}
    batch_size = batch_size
    num_workers = 8
    datasets['train'], datasets['val'], datasets['test'] = Dataset.data_preprare()
    logging.info('MIMI数据集：')
    logging.info('训练集样本个数：{}'.format(datasets['train'].__len__()))
    logging.info('验证集样本个数：{}'.format(datasets['val'].__len__()))
    logging.info('测试集样本个数：{}'.format(datasets['test'].__len__()))


    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=(True if x == 'train' else False),
                                                  num_workers=num_workers,
                                                  pin_memory=(True if device == 'cuda' else False))
                   for x in ['train', 'val', 'test']}
    tmp = dataloaders['val'].__iter__().__next__()
    nc=tmp[0].size()[1]
    logging.info('sample dimension：{}'.format(tmp[0].size()))
    logging.info('channel number=：{}'.format(nc))

    data_loader = {"train": dataloaders['train'],
        "val": dataloaders['val'],
        "test": dataloaders['test'],
        "nc": nc,
        "len":window_size
    }

    return data_loader
def load_data_HYFJ(val_size, window_size=1024, stride=1, batch_size=64, normalization=None,dataloder=False,  speed=None,  snr=0.0,sigma=0.1,ifsnr=False):
    ## 20480#信号长度一分钟 20480*(50~60)=
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    savedir = './local_data/' + 'HYFJ' + '.pkl'
    normlizetype = normalization

    if dataloder:
        Dataset = HYFJ_local(savedir, normlizetype)

    else:
        Dataset = HYFJ(speed=speed,normlizetype=normlizetype,  window_size=window_size, stride=stride, save=True,
                     savedir=savedir,  test_size=val_size, snr=snr, sigma=sigma, ifsnr=ifsnr)
    datasets = {}
    batch_size = batch_size
    num_workers = 16
    datasets['train'], datasets['val'], datasets['test'] = Dataset.data_preprare()
    logging.info('HYFJ数据集：')
    logging.info('训练集样本个数：{}'.format(datasets['train'].__len__()))
    logging.info('验证集样本个数：{}'.format(datasets['val'].__len__()))
    logging.info('测试集样本个数：{}'.format(datasets['test'].__len__()))

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=(True if x == 'train' else False),
                                                  num_workers=num_workers,
                                                  pin_memory=(True if device == 'cuda' else False))
                   for x in ['train', 'val', 'test']}
    tmp = dataloaders['val'].__iter__().__next__()
    nc = tmp[0].size()[1]
    logging.info('sample dimension：{}'.format(tmp[0].size()))
    logging.info('channel number=：{}'.format(nc))

    data_loader = {"train": dataloaders['train'],
                   "val": dataloaders['val'],
                   "test": dataloaders['test'],
                   "nc": nc,
                   "len": window_size
                   }
    return data_loader

def load_data_RY( val_size, window_size=1024, stride=1, batch_size=64,dim=2, normalization=None,dataloder=False, fault_list=None, speed=None, load=None, snr=0.0,sigma=0.1,ifsnr=False):
    ## 20480#信号长度一分钟 20480*(50~60)=
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # datadir ='/data1/szg/data/燃油控制系统-ZL'
    datadir ='/usr/data_disk1/szg/data/燃油控制系统-ZL'

    savedir = './local_data/' + 'RY' + '.pkl'
    normlizetype = normalization

    if dataloder:
        Dataset = RY_local(savedir, normlizetype)

    else:
        Dataset = RY(datadir, normlizetype, fault_list, speed, load, window_size=window_size, stride=stride, save=True,
                       savedir=savedir, dim=dim, test_size=val_size, snr=snr,sigma=sigma,ifsnr=ifsnr)
    datasets = {}
    batch_size = batch_size
    num_workers = 16
    datasets['train'], datasets['val'], datasets['test'] = Dataset.data_preprare()
    logging.info('RY数据集：')
    logging.info('训练集样本个数：{}'.format(datasets['train'].__len__()))
    logging.info('验证集样本个数：{}'.format(datasets['val'].__len__()))
    logging.info('测试集样本个数：{}'.format(datasets['test'].__len__()))


    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=(True if x == 'train' else False),
                                                  num_workers=num_workers,
                                                  pin_memory=(True if device == 'cuda' else False))
                   for x in ['train', 'val', 'test']}
    tmp = dataloaders['val'].__iter__().__next__()
    nc=tmp[0].size()[1]
    logging.info('sample dimension：{}'.format(tmp[0].size()))
    logging.info('channel number=：{}'.format(nc))

    data_loader = {"train": dataloaders['train'],
        "val": dataloaders['val'],
        "test": dataloaders['test'],
        "nc": nc,
        "len":window_size
    }
    return data_loader



def load_data_IMS( file_set, val_size, window_size=1024, stride=1, batch_size=64, normalization=None,dataloder=False, polluted_ratio=0.0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    savedir = './local_data/' + 'IMS' + '.pkl'
    normlizetype = normalization

    if dataloder:
        Dataset = IMS_local(file_set, normlizetype)

    else:
        Dataset = IMS(file_set=file_set, normlizetype=normlizetype,window_size=window_size, stride=stride, polluted_ratio=polluted_ratio,test_size=val_size)
    datasets = {}
    batch_size = batch_size
    num_workers = 16
    datasets['train'], datasets['val'], datasets['test'] = Dataset.data_preprare()
    logging.info('MIMI数据集：')
    logging.info('训练集样本个数：{}'.format(datasets['train'].__len__()))
    logging.info('验证集样本个数：{}'.format(datasets['val'].__len__()))
    logging.info('测试集样本个数：{}'.format(datasets['test'].__len__()))

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=(True if x == 'train' else False),
                                                  num_workers=num_workers,
                                                  pin_memory=(True if device == 'cuda' else False))
                   for x in ['train', 'val','test']}
    tmp = dataloaders['val'].__iter__().__next__()
    nc=tmp[0].size()[1]
    logging.info('sample dimension：{}'.format(tmp[0].size()))
    logging.info('channel number=：{}'.format(nc))

    data_loader = {"train": dataloaders['train'],
        "val": dataloaders['val'],
        "test": dataloaders['test'],
        "nc": nc,
        "len":window_size
    }
    return data_loader


def load_data_PG( work_condition ,fault_type, val_size=0.1, window_size=1024, stride=1, batch_size=64, normalization=None,dataloder=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    savedir = './local_data/' + 'PG' + '.pkl'
    normlizetype = normalization

    if dataloder:
        Dataset = PG_local(savedir, normlizetype)

    else:
        Dataset = PG(work_condition =work_condition,fault_type = fault_type, normlizetype=normlizetype,window_size=window_size, stride=stride, test_size=val_size)
    datasets = {}
    batch_size = batch_size
    num_workers = 16
    datasets['train'], datasets['val'], datasets['test'] = Dataset.data_preprare()
    logging.info('PG数据集：')
    logging.info('训练集样本个数：{}'.format(datasets['train'].__len__()))
    logging.info('验证集样本个数：{}'.format(datasets['val'].__len__()))
    logging.info('测试集样本个数：{}'.format(datasets['test'].__len__()))

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=(True if x == 'train' else False),
                                                  num_workers=num_workers,
                                                  pin_memory=(True if device == 'cuda' else False))
                   for x in ['train', 'val','test']}
    tmp = dataloaders['val'].__iter__().__next__()
    nc=tmp[0].size()[1]
    logging.info('sample dimension：{}'.format(tmp[0].size()))
    logging.info('channel number=：{}'.format(nc))

    data_loader = {"train": dataloaders['train'],
        "val": dataloaders['val'],
        "test": dataloaders['test'],
        "nc": nc,
        "len":window_size
    }
    return data_loader