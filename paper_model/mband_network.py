#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import os
from time import time
from copy import deepcopy
import scipy.io
import torch
from torch import optim
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
import logging
import matplotlib.pyplot as plt
from utils.utils_ae import metrics_calculate_mband, metrics_calculate, lr_scheduler_choose
from utils.t_sne import T_SNE
from paper_model.FMBWN import FMWN
from paper_model.Script_MbandWN import Kernel_selector,CoeffLoss
from paper_model.Script_MbandWN_frequencyanomaly_validation import coefficient_visul_batch as coefficient_visul
import math
import pandas as pd
'''
FMWN(kernelInit=None, m=[3], k=[2],level=None, mode='DWT',kernTrainable=True,  kernelsConstraint='CF',
                 coefffilter=1.0,device='cuda',realconv=True,t=20.0,initHT=1.0, trainHT=True,threshold=True)
Kernel_selector:1,2,3,4,5
coefficient_visul_pywt(all_coeff,mode='DWT')
coefficient_visul(all_coeff,mode='DWT', m=None,sort='freq',prefined_sort=None)
'''
def concate_coeff(z):#z: list=[[c1,c2],[c3,c4],[]]
        for i in range(len(z)):
            for j in range(len(z[i])):
                if i==0 and j==0:
                    temp=z[i][j].detach()
                else:
                    temp=torch.cat((temp,z[i][j].detach()),dim=2)
        return temp
def plot_box(z,batchidx=0,title=' '):
    # s12=pd.Series(np.squeeze(z[0][0][batchidx,:,:,:].detach().cpu().numpy()))
    # s13=pd.Series(np.squeeze(z[0][1][batchidx,:,:,:].detach().cpu().numpy()))
    # s22=pd.Series(np.squeeze(z[1][0][batchidx,:,:,:].detach().cpu().numpy()))
    # s23=pd.Series(np.squeeze(z[1][1][batchidx,:,:,:].detach().cpu().numpy()))
    # s31=pd.Series(np.squeeze(z[3][0][batchidx,:,:,:].detach().cpu().numpy()))
    # s32=pd.Series(np.squeeze(z[2][0][batchidx,:,:,:].detach().cpu().numpy()))
    # s33=pd.Series(np.squeeze(z[2][1][batchidx,:,:,:].detach().cpu().numpy()))
    # data = pd.DataFrame({"1-2": s12, "1-3": s13, "2-2": s22,"2-3":s23,"3-1":s31,"3-2":s32,"3-3":s33})
    # data.boxplot(grid=False)  # 这里，pandas自己有处理的过程，很方便哦。
    # plt.xlabel("level-band")  # 我们设置横纵坐标的标题。
    # plt.title('coefficient'+title)
    # plt.show()

    data = []
    for i in range(len(z)):
        for j in range(len(z[i])):
            # data.append(np.squeeze(z[i][j][batchidx,:,:,:].detach().cpu().numpy()))
            data.append(np.abs(np.squeeze(z[i][j][batchidx,:,:,:].detach().cpu().numpy())))

    fig, ax = plt.subplots()
    ax.boxplot(data,patch_artist=True,vert=True,notch=True)
    plt.title('coefficient'+title)
    # 显示图形
    plt.show()

def fft_fignal_plot(signal, fs, window, plot=True):
    sampling_rate = fs
    fft_size = window
    t = np.arange(0, fs, 1.0 / sampling_rate)
    mean=signal.mean()
    xs = signal[:fft_size]-mean
    xf = np.fft.rfft(xs) / fft_size
    freqs = np.linspace(0.0, sampling_rate/2.0, int(fft_size / 2 + 1))
    xfp = np.abs(xf)
    # xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.plot(freqs, xfp)
    plt.xlabel("Frequency/Hz")
    # 字体FangSong
    plt.ylabel('Amplitude')
    # plt.subplots_adjust(hspace=0.4)
    '''subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    有六个可选参数来控制子图布局。值均为0~1之间。其中left、bottom、right、top围成的区域就是子图的区域。
    wspace、hspace分别表示子图之间左右、上下的间距。实际的默认值由matplotlibrc文件控制的。
    '''
    if plot:
        plt.show()

def plot_energydistribution(z, batchidx=0,title=' '):# 数据

    # x_labels = ["1-2", "1-3", "2-2","2-3","3-1","3-2","3-3"]
    #
    # s12=np.sum(z[0][0][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s13=np.sum(z[0][1][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s22=np.sum(z[1][0][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s23=np.sum(z[1][1][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s31=np.sum(z[3][0][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s32=np.sum(z[2][0][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s33=np.sum(z[2][1][batchidx,:,:,:].detach().cpu().numpy()**2)
    #
    # data = np.array([s12,s13,s22,s23,s31,s32,s33])
    # data =data / np.max(data)
    # data = list(data)
    # plt.bar(range(len(data)), data, color='b')
    # # 指定横坐标刻度
    # plt.xticks(range(len(data)), x_labels)
    # plt.xlabel("level-band")  # 我们设置横纵坐标的标题。
    # plt.title(title + 'energy distribution')
    # plt.show()
    data = []
    for i in range(len(z)):
        for j in range(len(z[i])):
            data.append(np.sum(z[i][j][batchidx, :, :, :].detach().cpu().numpy() ** 2))
            # data.append(np.average(z[i][j][batchidx, :, :, :].detach().cpu().numpy() ** 2))

    data = np.array(data)
    # data =data / np.max(data)
    data = list(data)
    plt.bar(range(len(data)), data, color='b')
    # 指定横坐标刻度
    # plt.xticks(range(len(data)), x_labels)
    plt.xlabel("level-band")  # 我们设置横纵坐标的标题。
    plt.title(title + 'energy distribution')
    plt.show()




class Mband_AE(object):
    def __init__(self, data_loader, savedir, args):
        # Consider the gpu or cpu condition
        self.args = args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))
        #inp_dim=data_loader['nc'], z_dim=args.z_dim,seqlen=data_loader['len']

        # kernel_init = Kernel_selector(args.KernelInit)
        ae = FMWN(kernelInit=args.KernelInit, m=args.m, k=args.k,level=args.level, mode=args.mband_mode,kernTrainable=args.kernTrainable, kernelsConstraint=args.kernelsConstraint,
                 coefffilter=args.coeff_filter,device=self.device,realconv=args.realconv,t=args.threshold_t,initHT=args.ThreInit, trainHT=args.trainHT,threshold=args.threshold,seqlen=args.window_size)

        logging.info(ae)

        self.savedir = savedir
        self.polluted_ratio = args.snr# SNR
        self.data_prefix = args.key_information
        self.filterlosscoeff=args.filterlosscoeff
        self.sparsitylosscoeff=args.sparsitylosscoeff
        self.hardcoeff = args.hardlosscoeff
        self.lr = args.lr
        self.epoch = args.epoch

        self.early_stop = args.early_stop
        self.early_stop_tol = args.early_stop_tol

        self.plot_confusion_matrix = args.plot_confusion_matrix

        self.ae = ae.to(self.device)
        if self.device_count > 1:
            self.ae = torch.nn.DataParallel(self.ae)
        self.data_loader = data_loader  # 加载数据器：train, val, test, nc

        if args.loss_function=='MSE':
            self.mse = MSELoss()  # 训练ae
        elif args.loss_function=='MAE':
            self.mse = nn.L1Loss()  # 训练ae
        else:
            raise Exception('Other loss funciton is not supported')
        self.coff_sparsity = CoeffLoss(loss=args.coeff_sparsity,mode = args.mband_mode)


        if args.opt == 'adam':
            self.ae_optimizer = optim.Adam(self.ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            self.ae_optimizer = optim.SGD(self.ae.parameters(), lr=args.lr, momentum=args.momentum,
                                          weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        self.lr_scheduler = lr_scheduler_choose(lr_scheduler_way=args.lr_scheduler, optimizer=self.ae_optimizer,
                                                steps=args.scheduler_step_size, gamma=args.scheduler_gamma)

        self.cur_step = 0
        self.cur_epoch = 0
        # self.best_ae = 0  # None

        self.early_stop_count = 0

        #loss
        self.re_loss = 0  # None
        self.traning_mse = 0
        self.traning_sparse=0
        self.traning_filter=0
        self.hard_threshold=0

        self.best_val_loss = np.inf
        self.val_loss = 0  # None

        self.time_per_epoch = 0  # None
        # loss list
        self.loss_normal_list = []
        # self.loss_abnormal_list = []
        self.EMA_coefficient=0
        self.show_filter()
        self.show_threshold()

    def train(self):  # 主程序
        logging.info('*' * 20 + 'Start training' + '*' * 20)
        for i in range(self.epoch):
            self.cur_epoch += 1
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(self.args.lr))
            self.train_epoch()  # 训练一次
            self.validate()  # 验证一次

            if self.val_loss < self.best_val_loss and self.best_val_loss - self.val_loss >= 1e-4:
                self.best_val_loss = self.val_loss
                # self.best_ae = deepcopy(self.ae)
                self.best_epoch = i + 1
                # if self.args.save_model:
                self.save_best_model(self.ae)
                self.early_stop_count = 0
            elif self.early_stop:  # 早停, return结束程序：验证loss比最好loss高或者几乎一样连续累计n次后结束程序。只看重构误差
                self.early_stop_count += 1
                if self.early_stop_count > self.early_stop_tol:
                    logging.info('*' * 20 + 'Early stop' + '*' * 20)
                    return
            else:
                pass

            logging.info('[Epoch %d/%d] current training loss is Totoal: %.5f, MSE: %.5f, Sparse: %.5f, Filter: %.5f, hard threshold: %.5f, Val MSE: %.5f'
                         ' time per epoch is %.5f' % (i + 1, self.epoch, self.re_loss,self.traning_mse,self.traning_sparse,self.traning_filter, self.hard_threshold, self.val_loss,
                                                     self.time_per_epoch))
        self.show_threshold()
        self.show_filter()
        self.plot()

    def train_epoch(self):  # 主程序训练
        start_time = time()
        number_normal = 0.0

        normal_sum = 0.0
        mse_sum =0.0
        sparse_sum = 0.0
        filter_sum = 0.0
        hard_threshold = 0.0
        for batch_idx, (x, label) in enumerate(self.data_loader['train']):
            self.cur_step += 1
            x = torch.unsqueeze(x.to(self.device),dim=3)
            # label = label.to(self.device)

            loss_tmp, traning_mse, traning_sparse, traning_filter, hard_threshold_max= self.ae_train(x)  # 主程序训练生成器
            normal_sum += loss_tmp.item()*x.size(0)
            mse_sum += traning_mse.item()*x.size(0)
            if self.args.coeff_sparsity:
                sparse_sum += traning_sparse.item()*x.size(0)
            filter_sum += traning_filter.item()*x.size(0)
            hard_threshold += hard_threshold_max.item()*x.size(0)
            number_normal += x.size(0)  # 样本数量
        # one epoch add loss to list
        self.re_loss = normal_sum / number_normal

        self.traning_mse = mse_sum/ number_normal
        self.traning_sparse = sparse_sum / number_normal
        self.traning_filter = filter_sum / number_normal
        self.hard_threshold = hard_threshold/ number_normal

        self.loss_normal_list.append(self.re_loss)
        end_time = time()
        self.time_per_epoch = end_time - start_time
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()



    def ae_train(self, x):  # 主程序训练生成器：对抗loss加上根据重构误差自适应的加权MSE的loss
        self.ae_optimizer.zero_grad()
        # sigPred, coeff, lossFilter
        re_x, z, lossFilters, hard_threshold_max,_ = self.ae(x)  # 重构变量和中间变量

        traning_mse = self.mse(re_x, x)
        traning_sparse = self.coff_sparsity(z)
        traning_filter = lossFilters

        p = self.cur_epoch / (0.33 * self.epoch) #0.33 控制甚么时候到0， 1/3 max epoch即到了
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        progress_1 = math.sin(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        # loss =  2.0*traning_mse + self.sparsitylosscoeff*traning_sparse + self.filterlosscoeff*traning_filter*progress - 0.1* hard_threshold_max

        p_2 = self.cur_epoch / (0.8 * self.epoch)  # 系数控制什么时候到中点 x% 的时候到达最大值,而后逐渐变小
        progress_2 = math.cos(min(max(p_2-1.0, -1), 1) * (math.pi / 2)) ** 2
        #1.常规 ln(max+1)
        # loss = traning_mse - self.hardcoeff*torch.log(hard_threshold_max+1) + self.sparsitylosscoeff * traning_sparse + self.filterlosscoeff*lossFilters
        #
        #先变大后变小 全局
        # loss = traning_mse - self.hardcoeff*progress_2*hard_threshold_max + self.sparsitylosscoeff * traning_sparse + self.filterlosscoeff*lossFilters
        #2.逐渐变大
        loss = traning_mse - self.hardcoeff*progress_1*hard_threshold_max + self.sparsitylosscoeff * traning_sparse + self.filterlosscoeff*lossFilters
        #3.逐渐变大+log
        # loss = traning_mse - self.hardcoeff * progress_1 * torch.log(hard_threshold_max+1) + self.sparsitylosscoeff * traning_sparse + self.filterlosscoeff * lossFilters
        #渐进 全局+log
        # loss = traning_mse - self.hardcoeff*progress_2*torch.log(hard_threshold_max+1) + self.sparsitylosscoeff * traning_sparse + self.filterlosscoeff*lossFilters

        # loss = traning_mse - self.hardcoeff*hard_threshold_max + self.sparsitylosscoeff * traning_sparse + self.filterlosscoeff*lossFilters
        #渐进(log内部) 常规
        # loss = traning_mse - self.hardcoeff*torch.log(progress_2*hard_threshold_max+1) + self.sparsitylosscoeff * traning_sparse + self.filterlosscoeff*lossFilters
        # loss =  traning_mse - 0.1*progress_2 * hard_threshold_max
        # loss = traning_mse - 0.1 * progress_2 * hard_threshold_max+ self.sparsitylosscoeff*traning_sparse
        # loss = traning_mse - hard_threshold_max + self.sparsitylosscoeff * traning_sparse
        # loss = traning_mse - torch.log(hard_threshold_max+1) + self.sparsitylosscoeff*traning_sparse
        # loss = traning_mse - torch.log(hard_threshold_max + 1)
        # loss = traning_mse
        # loss = traning_mse+ self.sparsitylosscoeff*traning_sparse
        # backpropagation
        # self.EMA_coefficient = 0.9*self.EMA_coefficient+0.1*torch.mean(concate_coeff(z),dim=0).detach()#移动平均获取系数特征
        loss.backward()
        self.ae_optimizer.step()

        return loss, traning_mse, traning_sparse, traning_filter, hard_threshold_max

    def validate(self):
        self.ae.eval()  # 验证模式
        self.val_loss = 0
        num_batch = 0
        for batch_idx, (x, _) in enumerate(self.data_loader['val']):
            x = torch.unsqueeze(x.to(self.device),dim=3)
            # re_values, z, lossFilters, hard_threshold_max = self.ae(x)  # 重构变量和中间变量
            re_values = self.value_reconstruction_val(x)
            self.val_loss += self.mse(x, re_values).item() * x.size(0)
            num_batch += x.size(0)
        self.val_loss = self.val_loss / num_batch
        self.ae.train()  # 训练模式

    def test(self, load_from_file=False, last_epoch=False,titleadd=''):  # 测试集测试
        if load_from_file:
            logging.info('load from file, the best model in the epoch: {}'.format(self.best_epoch))
            model = self.load_best_model()
        else:
            logging.info('last epoch\'s model performance')
            model = self.ae
        model.eval()
        # else:
        #     logging.info('the best model in the epoch: {}'.format(self.best_epoch))
        #     model = self.best_ae.eval()

        values_list = []
        labels_list = []
        re_values_list = []
        coefficients_list=[]

        for batch_idx, (test_x, test_y) in enumerate(self.data_loader['test']):
            test_x =  torch.unsqueeze(test_x.to(self.device),dim=3)
            re_values, z, loss_Filters, hard_threshold_max,_ = model(test_x)
            labels_list += test_y.numpy().tolist()
            values_list += test_x.cpu().numpy().tolist()
            re_values_list += re_values.detach().cpu().numpy().tolist()
            coefficients_list += concate_coeff(z).detach().cpu().numpy().tolist()

        values_a = np.array(values_list)
        re_values_a = np.array(re_values_list)
        labels_a = np.array(labels_list)
        coefficients_a = np.array(coefficients_list)
        # EMA_coefficient_a = self.EMA_coefficient.detach().cpu().numpy()
        # metrics_calculate_mband(values_a, re_values_a, labels_a, coefficients_a, EMA_coefficient_a, self.plot_confusion_matrix,
        #                   title=str(self.polluted_ratio) + '_' + self.data_prefix+titleadd,savedir=self.savedir)# 加上根据小波系数计算异常分数
        #self.EMA_coefficient
        if self.args.save_reconstructed_data:
            self.save_result(values_a, re_values_a, labels_a)
        metrics_calculate(values_a, re_values_a, labels_a, self.plot_confusion_matrix,
                          title=self.data_prefix+titleadd,savedir=self.savedir)




    def value_reconstruction_val(self, raw_values, val=True):
        '''
        if train(val): reconstruct with current model
        if validate:   reconstruct with best model
        '''
        if val:
            reconstructed_value_, z, filterloss, hard_threshold,_ = self.ae(raw_values)
        else:
            ae = self.load_best_model()
            reconstructed_value_, z, filterloss, hard_threshold,_  = ae(raw_values)
        return reconstructed_value_

    def test_hidden(self, load_from_file=False, metric_districution=False, feature_tsne=True, last_epoch=False):
        '''
        test hidden features
        plot tsne for hidden features
        '''
        if load_from_file:
            logging.info('load from file, the best model in the epoch: {}'.format(self.best_epoch))
            model = self.load_best_model()
            model.eval()
        elif last_epoch:
            logging.info('last epoch\'s model performance')
            model = self.ae.eval()
        values_list = []
        labels_list = []
        re_values_list = []

        for batch_idx, (test_x, test_y) in enumerate(self.data_loader['test']):
            test_x = torch.unsqueeze(test_x.to(self.device),dim=3)
            _, re_values, filterloss, hard_threshold_max,_ = model(test_x)  # 隐层特征
            re_values = concate_coeff(re_values)
            zero = torch.zeros(re_values.size())
            labels_list += test_y.numpy().tolist()
            values_list += zero.numpy().tolist()
            re_values_list += re_values.detach().cpu().numpy().tolist()

        values_a = np.array(values_list)  # 0
        re_values_a = np.array(re_values_list)  # 隐层特征
        labels_a = np.array(labels_list)
        if metric_districution:
            metrics_calculate(values_a, re_values_a, labels_a, self.plot_confusion_matrix,
                              title='hidden' + '_' + self.data_prefix)
        if feature_tsne:
            T_SNE(np.squeeze(re_values_a), labels_a, dim=2)# dim=1画分布图需要先求和，取消第二维度

    def save_best_model(self, model):
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), os.path.join(self.savedir,
                                                               self.args.model_prefix + '.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(self.savedir,
                                                        self.args.model_prefix + '.pth'))

    def load_best_model(self):
        kernel_init = Kernel_selector(self.args.KernelInit)
        # ae = FMWN(kernelInit=kernel_init, m=self.args.m, k=self.args.k,level=self.args.level, mode=self.args.mband_mode,kernTrainable=self.args.kernTrainable, kernelsConstraint=self.args.kernTrainable,
        #          coefffilter=self.args.coeff_filter,device=self.device,realconv=self.args.realconv,t=self.args.threshold_t,initHT=self.args.ThreInit, trainHT=self.args.trainHT,threshold=self.args.threshold)

        ae = FMWN(kernelInit=kernel_init, m=self.args.m, k=self.args.k, level=self.args.level, mode=self.args.mband_mode,
             kernTrainable=self.args.kernTrainable, kernelsConstraint=self.args.kernelsConstraint,
             coefffilter=self.args.coeff_filter, device=self.device, realconv=self.args.realconv, t=self.args.threshold_t,
             initHT=self.args.ThreInit, trainHT=self.args.trainHT, threshold=self.args.threshold)
        ae.load_state_dict(
            torch.load(os.path.join(self.savedir, self.args.model_prefix + '.pth')))
        ae = ae.to(self.device)
        if self.device_count > 1:
            ae = torch.nn.DataParallel(ae)
        return ae

    def save_result(self, values, re_values, labels):
        filename = os.path.join(self.savedir, 'reconstruction_data.mat')
        scipy.io.savemat(filename, {'values': values, 're_values': re_values, 'labels': labels})


    def plot(self):
        x = range(0, self.epoch)
        plt.title('loss in traning stage_' + '_' + self.data_prefix)
        plt.plot(x, self.loss_normal_list, color='black')#, label='normal_loss'
        # plt.plot(x, self.loss_abnormal_list, color='red', label='abnormal_loss')
        plt.ylabel('Loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()
        scipy.io.savemat(os.path.join(self.savedir, 'loss.mat'),
                         {'normal': self.loss_normal_list})

    def plot_reconstruction_trainset(self, figure_number=1, channel=0, test=False):  # 可视化 训练集 重构效果图 只有正常
        self.ae.eval()
        first_batch = self.data_loader['train'].__iter__().__next__()
        sample = first_batch[0]
        reconstruct_sample, _,_,_,_ = self.ae(torch.unsqueeze(sample.to(self.device),dim=3))
        reconstruct_sample = reconstruct_sample.detach().cpu().numpy()

        for i in range(figure_number):
            plt.title('train set Normal:'+ '_' + self.data_prefix)
            plt.plot(sample[i, channel, :], color='black', label='original')
            plt.plot(reconstruct_sample[i, channel, :,0], color='red', label='reconstructed')
            plt.legend()
            plt.show()

    def show_filter(self):
        logging.info('*' * 6 + 'showing threshold' + '*' * 6)
        m = self.args.m
        k = self.args.k
        level = len(m)
        kernels = self.ae.MBAND
        for lev in range(level):
            for i in range(m[lev]):
                kernellen = m[lev] * k[lev]
                kerneltemp = kernels[lev].DeFilter[i].kernel  # 方便的获取滤波器系数
                print('DE '+str(i)+' filter in '+str(lev)+'the layer')
                # print(torch.sum(kerneltemp))
                print(kerneltemp.detach().cpu().numpy().reshape(kernellen))


    def show_threshold(self):
        logging.info('*'*6+'showing threshold'+'*'*6)
        m=self.args.m
        level =len(m)
        threshold_list = self.ae.thre  # 每一层有一个
        for lev in range(level):
            print('threshold in '+str(lev)+ 'th layer of band!!'+ '*' * 6)
            print(threshold_list[lev].left)
            print(threshold_list[lev].right)
        print('threshold in the last low-pass layer of band!!' + '*' * 6)
        print(self.ae.thre_a.left)
        print(self.ae.thre_a.right)

    def plot_reconstruction_testset(self, figure_number=1, channel=0):  # 可视化 测试集 重构效果图 头部正常 尾部异常
        batch_number = len(self.data_loader['test'])
        Sample_frequency=20480
        if batch_number>1:
            for batch_idex, (x, y) in enumerate(self.data_loader['test']):
                if batch_idex == 0:
                    first_batch_label = y
                    sample = x.detach().numpy()
                    reconstruct_sample, coeff,_,_,ori_coeff = self.ae(torch.unsqueeze(x.to(self.device),dim=3))
                    reconstruct_sample = reconstruct_sample.detach().cpu().numpy()
                elif batch_idex == batch_number - 1:
                    sample_last = x.detach().numpy()
                    last_batch_label = y
                    reconstruct_sample_last, coeff,_,_,ori_coeff = self.ae(torch.unsqueeze(x.to(self.device),dim=3))
                    reconstruct_sample_last = reconstruct_sample_last.detach().cpu().numpy()

            for i in range(figure_number):  # 头部的重构效果
                if first_batch_label[i] == 0:
                    plt.title('test set Normal:' + '_' + self.data_prefix)
                    title = 'test set Normal:' + '_' + self.data_prefix
                else:
                    plt.title('test set Abnormal:' + '_' + self.data_prefix)
                    title = 'test set Abnormal:' + '_' + self.data_prefix
                plt.plot(sample[i, channel, :], color='black', label='original signal')
                plt.plot(reconstruct_sample[i, channel, :,0], color='red', label='reconstructed')
                plt.legend()
                plt.show()

                # fft_fignal_plot(sample[i, channel, :], Sample_frequency, 2048, plot=False)
                # fft_fignal_plot(reconstruct_sample[i, channel, :,0], Sample_frequency, 2048, plot=False)
                # plt.show()
                # plt.clf()
                # coefficient_visul(ori_coeff, mode='DWT', m=self.args.m, title='ori'+title, batch_idx=i)
                # coefficient_visul(coeff, mode='DWT', m=self.args.m,title='thre'+title,batch_idx=i)  # ,prefined_sort=[1,2,0,3]
                # plot_box(coeff,batchidx=i,title=title)
                # plot_energydistribution(ori_coeff, batchidx=i,title='ori'+title)
                # plot_energydistribution(coeff, batchidx=i,title='thre'+title)

            if figure_number > last_batch_label.numel():
                figure_number = last_batch_label.numel()
            else:
                figure_number = figure_number
            for i in range(figure_number):  # 尾部的重构效果
                if last_batch_label[i] == 0:
                    plt.title('test set Normal:' + '_' + self.data_prefix)
                    title = 'test set Normal:' + '_' + self.data_prefix
                else:
                    plt.title('test set Abnormal:' + '_' + self.data_prefix)
                    title = 'test set Abnormal:' + '_' + self.data_prefix
                plt.plot(sample_last[i, channel, :], color='black', label='original signal')
                plt.plot(reconstruct_sample_last[i, channel, :,0], color='red', label='reconstructed')
                plt.legend()
                plt.show()

                # plt.figure()
                # fft_fignal_plot(sample_last[i, channel, :], Sample_frequency, 2048, plot=False)
                # fft_fignal_plot(reconstruct_sample_last[i, channel, :,0], Sample_frequency, 2048, plot=False)
                # plt.show()
                # plt.clf()
                # coefficient_visul(ori_coeff, mode='DWT', m=self.args.m, title='ori' + title,
                #                   batch_idx=i)  # ,prefined_sort=[1,2,0,3]
                # coefficient_visul(coeff, mode='DWT', m=self.args.m, title='thre'+title,batch_idx=i)  # ,prefined_sort=[1,2,0,3]
                # plot_box(coeff, batchidx=i,title=title)
                # plot_energydistribution(ori_coeff, batchidx=i,title='ori' +title)
                # plot_energydistribution(coeff, batchidx=i,title='thre'+title)
        else:
            #只有一个batch时候
            for batch_idex, (x, y) in enumerate(self.data_loader['test']):
                first_batch_label = y
                sample = x.detach().numpy()
                reconstruct_sample, coeff, _,_,ori_coeff= self.ae(torch.unsqueeze(x.to(self.device), dim=3))
                reconstruct_sample = reconstruct_sample.detach().cpu().numpy()

            for i in range(figure_number):  # 头部的重构效果
                if first_batch_label[i] == 0:
                    plt.title('test set Normal:' + '_' + self.data_prefix)
                    title = 'test set Normal:' + '_' + self.data_prefix
                else:
                    plt.title('test set Abnormal:' + '_' + self.data_prefix)
                    title = 'test set Abnormal:' + '_' + self.data_prefix
                plt.plot(sample[i, channel, :], color='black', label='original signal')
                plt.plot(reconstruct_sample[i, channel, :, 0], color='red', label='reconstructed')
                plt.legend()
                plt.show()

                # fft_fignal_plot(sample[i, channel, :], Sample_frequency, 2048, plot=False)
                # fft_fignal_plot(reconstruct_sample[i, channel, :, 0], Sample_frequency, 2048, plot=False)
                # plt.show()
                # plt.clf()
                # coefficient_visul(ori_coeff, mode='DWT', m=self.args.m, title='ori'+title,batch_idx=i)  # ,prefined_sort=[1,2,0,3]
                # coefficient_visul(coeff, mode='DWT', m=self.args.m, title='thre'+title,batch_idx=i)  # ,prefined_sort=[1,2,0,3]
                # plot_box(coeff, batchidx=i,title=title)
                # plot_energydistribution(ori_coeff, batchidx=i,title='ori'+title)
                # plot_energydistribution(coeff, batchidx=i,title='thre'+title)

            for i in range(figure_number):  # 尾部的重构效果
                if first_batch_label[-i] == 0:
                    plt.title('test set Normal:' + '_' + self.data_prefix)
                    title = 'test set Normal:' + '_' + self.data_prefix
                else:
                    plt.title('test set Abnormal:' + '_' + self.data_prefix)
                    title = 'test set Abnormal:' + '_' + self.data_prefix
                plt.plot(sample[-i, channel, :], color='black', label='original signal')
                plt.plot(reconstruct_sample[-i, channel, :, 0], color='red', label='reconstructed')
                plt.legend()
                plt.show()


                # fft_fignal_plot(sample[-i, channel, :], Sample_frequency, 2048, plot=False)
                # fft_fignal_plot(reconstruct_sample[-i, channel, :, 0], Sample_frequency, 2048, plot=False)
                # plt.show()
                # coefficient_visul(ori_coeff, mode='DWT', m=self.args.m, title='ori' + title,
                #                   batch_idx=i)  # ,prefined_sort=[1,2,0,3]
                # coefficient_visul(coeff, mode='DWT', m=self.args.m, title='thre'+title,
                #                   batch_idx=i)  # ,prefined_sort=[1,2,0,3]
                # plot_box(coeff, batchidx=i,title=title)
                # plot_energydistribution(ori_coeff, batchidx=i, title='ori' +title)
                # plot_energydistribution(coeff, batchidx=i,title='thre'+title)