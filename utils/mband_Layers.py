# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from utils.base_convolution import Conv2d,ConvTranspose2d
import torch.nn.functional as F
'''
Problem1: 输入数据奇数偶数不同时，会出现重构的偏移现象（tensorflow 补0方式）
滤波器总长度=M*K 
M：band 
K：regularity,当K=1时为Haar小波. 文中也多记为L
'''



class Kernel(nn.Module):
    def __init__(self, bandM=3, regularity=2,  trainKern=True, device='cpu',**kwargs):
        super(Kernel, self).__init__()
        self.trainKern  = trainKern
        self.kernel=None
        if isinstance(bandM,int):
            self.kernelSize = regularity*bandM# filter length = M * K
            self.kernel = Parameter(torch.randn(1,1,self.kernelSize, 1,dtype=torch.float32,device=device),requires_grad=self.trainKern,)# 生成 M
            # nn.init.normal_(self.kernel, mean=0,std=1)#normal 可以替换为uniform
            nn.init.xavier_normal(self.kernel, gain=1.0)#无ReLU时候使用 可以替换为uniform
            # torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

        else:
            self.kernelSize = len(bandM)
            self.kernel = Parameter(torch.tensor(bandM,dtype=torch.float32,device=device).reshape(1,1,self.kernelSize,1),requires_grad=self.trainKern)
    def forward(self):
        return self.kernel
#decomposition mode:'DWT' 'WPT' 'arbitrary' # 数据是分组的
#优化  c,l, = input
class Mband(nn.Module):
    """
        mband class include analysis and its corresponding synthesis module
        M: band number
        k: regularity
        给滤波器组
        给分解器组
    """
    def __init__(self, De=None, Re=None, M=4,k=2, realconv=False, trainable=False,device = 'cuda',seqlen=1000,dropout_L = 0.1, dropout_M=0.3):
        super(Mband, self).__init__()
        self.realconv=realconv
        self.M=len(De)
        self.regularity=k
        self.trainable=trainable
        self.signal_size=0
        self.de_kernel_list = De
        #
        # self.dropout_L = nn.Dropout(dropout_L)
        # self.dropout_M = nn.Dropout(dropout_M)

        assert len(De)==len(Re)
        assert len(De)==M
        # assert De[0].size()[2]==M*k

        DeFilter=[]
        ReFilter=[]

        for i in range(self.M):
            # kernl_temp = Kernel(bandM=kernel[i], regularity=self.regularity, trainKern=self.trainable)
            DeFilter.append(Analysis(kernel=De[i], M=self.M, realconv=self.realconv, device=device))
            ReFilter.append(Synthesis(kernel=Re[i], M=self.M, realconv=self.realconv, device=device))
        self.DeFilter = nn.ModuleList(DeFilter)
        self.ReFilter = nn.ModuleList(ReFilter)


    def filter(self):
        return self.de_kernel_list

    def forward(self, input, decomposition=1):
        '''
        :param input:x1 (decompostion) [x1,x2,...,xm] (reconstruction)
        :param decomposition: decomposition or reconstruction
        :return:[x1,x2,x3,x4,x5](decompostion)   x(reconstruction)
        '''

        if decomposition==1:
            x_de=[]
            self.signal_size=input.size()
            for i in range(self.M):
                x_de.append(self.DeFilter[i](input))
            return x_de
        else:#重构
            x_re = self.ReFilter[0](input[0],self.signal_size)
            # x_re = self.ReFilter[0](self.dropout_L(input[0]),self.signal_size)
            for i in range(1,self.M):
                x_re = x_re + self.ReFilter[i](input[i],self.signal_size)
                # x_re = x_re + self.ReFilter[i](self.dropout_M(input[i]), self.signal_size)
                # 其实可以分通道丢弃
            return x_re

class Analysis(nn.Module):
    """
    Layer that performs a convolution between its two inputs with stride related to M-band (stride,1)
    modified
    """

    def __init__(self, kernel, M, realconv=False, device ='cuda', **kwargs):
        super(Analysis, self).__init__()
        self.M = M
        self.realconv =realconv
        self.device =device
        # if realconv:
        #     self.kernel = torch.flip(kernel, [2])  # real convolution
        # else:
        #     self.kernel = kernel  # Compared to real convolution, in real calculation, correlation has a better edge without per extension
        self.kernel = kernel
        self.ConvLayer = Conv2d(in_channels=1, out_channels=1, kernel=self.kernel, stride=(M, 1),device=self.device)

    def forward(self, input):

        return self.ConvLayer(input,self.realconv)


class Synthesis(nn.Module):
    """
    Layer that performs a convolution transpose between its two inputs with stride (2,1).
    The third input specifies the size of the reconstructed signal (to make sure it matches the decomposed one)
    """
    def __init__(self, kernel, M, realconv=False, device ='cuda', **kwargs):
        super(Synthesis, self).__init__()
        # if realconv:
        #     self.kernel = torch.flip(kernel, [2])  # real convolution
        # else:
        #     self.kernel = kernel #kernel
        self.device = device
        self.kernel = kernel  # kernel
        self.realconv = realconv
        self.ConvTransLayer = ConvTranspose2d(in_channels=1, out_channels=1, kernel=self.kernel, stride=(M, 1),device=self.device)

    def forward(self, input, out_size):
        return self.ConvTransLayer(input,out_size,self.realconv)

def exp_calculcation(x,t,b):
    # exp_x = torch.exp(-t*F.relu(x-b))# x较大时使用
    exp_x = torch.exp(-t * F.relu(x/b - 1))# x和b较小时使用
    return exp_x

def wavelet_threshold(x,t,left,right):
    return F.relu(x - torch.mul(F.relu(torch.mul(torch.sign(x),right)), exp_calculcation(x,t,right))) - F.relu(-x - torch.mul(F.relu(torch.mul(torch.sign(-x),left)), exp_calculcation(-x,t,left)))

def wavelet_threshold_normal(x,t,left,right):
    maximization,_ = torch.max(torch.abs(x),dim=2,keepdim=True)#x = batchsize*1*length*1
    x = torch.div(x,maximization)#maximization = batchsize*1*1*1
    thresholded_x = F.relu(x - torch.mul(F.relu(torch.mul(torch.sign(x),right)), exp_calculcation(x,t,right))) - F.relu(-x - torch.mul(F.relu(torch.mul(torch.sign(-x),left)), exp_calculcation(-x,t,left)))
    return torch.mul(thresholded_x,maximization)
    # with torch.no_grad():
    #     maximization,_ = torch.max(torch.abs(x),dim=2,keepdim=True)
    #     x = torch.div(x,maximization)
    # thresholded_x = F.relu(x - torch.mul(F.relu(torch.mul(torch.sign(x),right)), exp_calculcation(x,t,right))) - F.relu(-x - torch.mul(F.relu(torch.mul(torch.sign(-x),left)), exp_calculcation(-x,t,left)))
    # with torch.no_grad():
    #     thresholded_x = torch.mul(thresholded_x,maximization)
    # return thresholded_x

def wavelet_threshold_normal_min(x,t,left,right,left1,right1):
    maximization,_ = torch.max(torch.abs(x),dim=2,keepdim=True)
    x = torch.div(x,maximization)
    thresholded_x = F.relu(x - torch.mul(F.relu(torch.mul(torch.sign(x),right)), exp_calculcation(x,t,right))) - F.relu(-x - torch.mul(F.relu(torch.mul(torch.sign(-x),left)), exp_calculcation(-x,t,left)))

    thresholded_x = F.relu(torch.mul(F.relu(torch.mul(torch.sign(thresholded_x),right1)), exp_calculcation(thresholded_x,t,right1)))-F.relu(- torch.mul(F.relu(torch.mul(torch.sign(-thresholded_x),left1)), exp_calculcation(-thresholded_x,t,left1)))
    return torch.mul(thresholded_x,maximization)
#
#
class WaveletThreshold(nn.Module):
    """
    with normalization
    Learnable Hard-thresholding layers = biased Relu
    :param
    t: 温度控制与硬阈值的近似程度
    m：band-1，只进行高频去噪
    """
    def __init__(self, init=None, trainBias=True, t=100.0, m=1, threshold=True,**kwargs):
        super(WaveletThreshold, self).__init__()
        if isinstance(init,float) or isinstance(init,int):
            # self.left = Parameter(torch.ones(m, 1, 1, 1, 1)*init,requires_grad=trainBias)
            # self.right = Parameter(torch.ones(m, 1, 1, 1, 1)*init,requires_grad=trainBias)
            self.left = Parameter(torch.ones(m, 1, 1, 1, 1) * init+torch.normal(mean=0,std=0.01*torch.torch.ones(m, 1, 1, 1, 1)), requires_grad=trainBias)
            self.right = Parameter(torch.ones(m, 1, 1, 1, 1) * init+torch.normal(mean=0,std=0.01*torch.torch.ones(m, 1, 1, 1, 1)), requires_grad=trainBias)
        else:
            self.left  = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
            self.right = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
        self.trainBias = trainBias
        self.t=torch.tensor(t)
        self.threshold=threshold
        self.m=m

    def forward(self, input, NodefineT=True,HT=1.0):
        # for i in range(len(input)):
        # return torch.multiply(input,torch.sigmoid(10*(input-self.thrP))+torch.sigmoid(-10*(input+self.thrN)))
        assert len(input)==self.m
        if NodefineT:
            if self.threshold:
                return [wavelet_threshold_normal(input[m_in_level],self.t,left=self.left[m_in_level],right=self.right[m_in_level]) for m_in_level in range(len(input))]
            else:
                return input
        else:
            return [wavelet_threshold_normal(input[m_in_level],self.t,left=HT,right=HT) for m_in_level in range(len(input))]

# #
# class WaveletThreshold(nn.Module):
#     """
#     Learnable Hard-thresholding layers = biased Relu
#     :param
#     t: 温度控制与硬阈值的近似程度
#     m：band-1，只进行高频去噪
#     """
#     def __init__(self, init=None, trainBias=True, t=100.0, m=1, threshold=True,**kwargs):
#         super(WaveletThreshold, self).__init__()
#         if isinstance(init,float) or isinstance(init,int):
#             # self.left = Parameter(torch.ones(m, 1, 1, 1, 1)*init,requires_grad=trainBias)
#             # self.right = Parameter(torch.ones(m, 1, 1, 1, 1)*init,requires_grad=trainBias)
#             self.left = Parameter(torch.ones(m, 1, 1, 1, 1) * init+torch.normal(mean=0,std=0.01*torch.torch.ones(m, 1, 1, 1, 1)), requires_grad=trainBias)
#             self.right = Parameter(torch.ones(m, 1, 1, 1, 1) * init+torch.normal(mean=0,std=0.01*torch.torch.ones(m, 1, 1, 1, 1)), requires_grad=trainBias)
#         else:
#             self.left  = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
#             self.right = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
#         self.trainBias = trainBias
#         self.t=torch.tensor(t)
#         self.threshold=threshold
#         self.m=m
#
#     def forward(self, input, NodefineT=True,HT=1.0):
#         # for i in range(len(input)):
#         # return torch.multiply(input,torch.sigmoid(10*(input-self.thrP))+torch.sigmoid(-10*(input+self.thrN)))
#         assert len(input)==self.m
#         if NodefineT:
#             if self.threshold:
#                 return [wavelet_threshold(input[m_in_level],self.t,left=self.left[m_in_level],right=self.right[m_in_level]) for m_in_level in range(len(input))]
#             else:
#                 return input
#         else:
#             return [wavelet_threshold(input[m_in_level],self.t,left=HT,right=HT) for m_in_level in range(len(input))]
#

class HardThresholdAssym(nn.Module):
    """
    Learnable Hard-thresholding layers
    """
    def __init__(self, init=None, trainBias=True,m=5, **kwargs):
        super(HardThresholdAssym, self).__init__()
        if isinstance(init,float) or isinstance(init,int):
            self.left  = Parameter(torch.ones(m, 1, 1, 1, 1)*init,requires_grad=trainBias)
            self.right = Parameter(torch.ones(m, 1, 1, 1, 1)*init,requires_grad=trainBias)
        else:
            self.left = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
            self.right = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
        self.trainBias = trainBias

        self.m = m
    def forward(self, input):
        assert len(input) == self.m
        return [torch.multiply(input[m_in_level],torch.sigmoid(10*(input[m_in_level]-self.left[m_in_level]))+torch.sigmoid(-10*(input[m_in_level]+self.right[m_in_level])))for
            m_in_level in range(len(input))]

        # return [wavelet_threshold(input[m_in_level], self.t, left=self.left[m_in_level], right=self.right[m_in_level]) for
        #     m_in_level in range(len(input))]

'''
双向截断的硬阈值函数
选择初值会有问题，因此理想是系数有个归一化的过程，则
'''
def HardThresholdAssymplus(x,t,l,r,l1=100.0,r1=100.0):
    return torch.multiply(x, (torch.sigmoid(t * (x - r))-torch.sigmoid(t * (x - r1))) + (torch.sigmoid(-t * (x + l))-torch.sigmoid(-t * (x + l1))))

def wavelet_threshold_normal_min_max(x,t,left,right,left1,right1,norm=False):
    if norm:
        maximization = torch.max(torch.abs(x))
        x = torch.div(x,maximization)
    # thresholded_x = HardThresholdAssym(x,t,left,right)
    thresholded_x = HardThresholdAssymplus(x,t,left,right,left1,right1)
    if norm:
        thresholded_x =torch.mul(thresholded_x,maximization)
    return thresholded_x
class WaveletThreshold_plus(nn.Module):
    """
    with normalization
    Learnable Hard-thresholding layers = biased Relu
    :param
    t: 温度控制与硬阈值的近似程度
    m：band数，此处进行所有频带都进行阈值处理，与DWT不同
    """
    def __init__(self, init=None,init1=None, trainBias=True, t=10.0, m=1, threshold=True,norm=False,**kwargs):
        super(WaveletThreshold_plus, self).__init__()
        if isinstance(init,float) or isinstance(init,int):
            #根据给定的init初始化左右的阈值
            #加入一些随机数绕动
            self.l = Parameter(torch.ones(m, 1, 1, 1, 1) * init+torch.normal(mean=0,std=0.01*torch.torch.ones(m, 1, 1, 1, 1)), requires_grad=trainBias)
            self.l1 = Parameter(torch.ones(m, 1, 1, 1, 1) * init1+torch.normal(mean=0,std=0.01*torch.torch.ones(m, 1, 1, 1, 1)), requires_grad=trainBias)
            self.r= Parameter(torch.ones(m, 1, 1, 1, 1) * init+torch.normal(mean=0,std=0.01*torch.torch.ones(m, 1, 1, 1, 1)), requires_grad=trainBias)
            self.r1= Parameter(torch.ones(m, 1, 1, 1, 1) * init1+torch.normal(mean=0,std=0.01*torch.torch.ones(m, 1, 1, 1, 1)), requires_grad=trainBias)
        else:
            self.l = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
            self.l1 = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
            self.r = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
            self.r1 = Parameter(torch.ones(m, 1, 1, 1, 1), requires_grad=trainBias)
        self.trainBias = trainBias # 是否训练阈值
        self.t=torch.tensor(t) #温度t
        self.threshold=threshold #是否使用阈值操作
        self.m=m # Mband的带数
        self.norm=norm #是否对系数归一化后操作，若使用此模式，则阈值在0-1范围则可操作数据.

    def forward(self, input, NodefineT=True,HT=1.0,HT1=100.0):
        '''
        :param input: 输入x
        :param NodefineT: 是否有定义的阈值t，若有则使用指定的阈值，否则适应可学习的阈值
        :param HT:指定的硬阈值
        :return:阈值处理后的输入
        '''
        # for i in range(len(input)):
        # return torch.multiply(input,torch.sigmoid(10*(input-self.thrP))+torch.sigmoid(-10*(input+self.thrN)))
        assert len(input)==self.m
        if NodefineT:
            if self.threshold:
                # return [wavelet_threshold_normal_min_max(input[m_in_level], self.t, left=self.l[m_in_level], right=self.r[m_in_level],
                #                                          left1=self.l1[m_in_level], right1=self.r1[m_in_level], norm=self.norm) for m_in_level in range(len(input))]
                return [wavelet_threshold_normal_min_max(input[m_in_level], self.t, left=self.l[m_in_level],
                                                     right=self.r[m_in_level],
                                                     left1=self.l[m_in_level]+torch.abs(self.l1[m_in_level]), right1=self.r[m_in_level]+torch.abs(self.r1[m_in_level]),
                                                     norm=self.norm) for m_in_level in range(len(input))]

            else:
                return input
        else:
            return [wavelet_threshold_normal_min_max(input[m_in_level],self.t,left=HT,right=HT,left1=HT1,right1=HT1,norm=self.norm) for m_in_level in range(len(input))]


if __name__ == '__main__':
    # inputSize=16
    #代数M-band小波
    band_num = 4
    if band_num==4:
        kernelInit = [np.array([-0.0190928308000000,0.0145382757000000,0.0229906779000000,0.0140770701000000,0.0719795354000000,-0.0827496793000000,-0.130694890900000,-0.0952930728000000,-0.114536126100000,0.219030893900000,0.414564773700000,0.495502982800000,0.561649421500000,0.349180509700000,0.193139439300000,0.0857130200000000]),
                  np.array([-0.115281343300000,0.0877812188000000,0.138816305600000,0.0849964877000000,-0.443703932000000,0.169154971800000,0.268493699200000,0.0722022649000000,-0.0827398180000000,-0.426427736100000,-0.255040161600000,0.600591382300000,-0.0115563891000000,-0.101106504400000,0.118328206900000,-0.104508652500000]),
                  np.array([-0.0280987676000000,0.0213958651000000,0.0338351983000000,0.0207171125000000,0.220295183000000,-0.208864350300000,-0.330053682700000,-0.224561804100000,0.556231311800000,-0.0621881917000000,0.00102740000000000,0.447749675200000,-0.248427727200000,-0.250343323000000,-0.204808915700000,0.256095016300000]),
                  np.array([-0.0174753464000000,0.0133066389000000,0.0210429802000000,0.0128845052000000,-0.0918374833000000,0.0443561794000000,0.0702950474000000,0.0290655661000000,-0.0233349758000000,-0.0923899104000000,-0.0823301969000000,0.0446493766000000,-0.137950244700000,0.688008574600000,-0.662289313000000,0.183998602200000]),
                  ]

    elif band_num==3:#相位偏移
        kernelInit = [np.array([-0.145936007553990,0.0465140821758900,0.238964171905760,0.723286276743610,0.530836187013740,0.338386097283860]),
                  np.array([0.426954037816980,-0.136082763487960,-0.699119564792890,-0.0187057473531300,0.544331053951810,-0.117377016134830]),
                  np.array([0.246502028665230,-0.0785674201318500,-0.403636868928920,0.460604752521310,-0.628539361054710,0.403636868928920]),
                  ]
    elif band_num==2:
        # kernelInit = [np.array([-0.129409522550921,0.224143868041857,0.836516303737469,0.482962913144690]),
        #               np.array([-0.482962913144690,0.836516303737469,-0.224143868041857,-0.129409522550921])]
        kernelInit = [np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                           -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778]),
                      np.array([0.230377813308855,0.714846570552542,0.630880767929590,-0.027983769416984,-0.187034811718881,0.030841381835987,0.032883011666983,-0.010597401784997])]
    elif band_num==5:
        kernelInit = [np.array([0.0272993452180114,0.0265840663469593,0,-0.0428593893779949,-0.0810269414612686,-0.117023723786349,-0.102653539494725,-2.73839349132101e-17,0.165500189580178,0.347337063987899,0.478068455182483,0.509357208870298,0.447213595499958,0.315935108535685,0.161069337685743,0.0588695188858126,0.0139258597774260,1.28985929088956e-32,0.00863768676208944,0.0198341352875850]),
                      np.array([0.0104274220026634	,-0.0164298565616031	,0	,0.0264885593726670,-0.0309495376337566,	-0.189348362576392	,-0.268750455462670	,-2.73839349132101e-17,	0.433285121465458,	0.562003175085017,	0.182605900930553,	-0.314800067496623	,-0.447213595499958,-0.195258635314441,0.0615230124505194,0.0952528824585986,0.0364583742198662,1.54826577665454e-31,0.0226137575273252,0.0320923050327763]),
                      np.array([0.0337438464306959,	-5.13472509819086e-18,	0	,-2.75943716886759e-18,	-0.100154807655024,	-1.21909088566198e-17,	0.332193831935890	,8.21518047396304e-17	,-0.535569863770559	,-1.80918635666264e-16,	0.590925108503860,	1.18092256227073e-15,	-0.447213595499958,	4.07025734526344e-16	,0.199092650470447	,-1.10450832039150e-16	,-0.0450650288848805	,-1.80623763483245e-31,	-0.0279521415304715,	2.89478922531347e-17]),
                      np.array([-0.0104274220026635	,-0.0164298565616031	,0	,0.0264885593726670	,0.0309495376337565	,-0.189348362576392,	0.268750455462670	,8.21518047396304e-17	,-0.433285121465458,	0.562003175085017,	-0.182605900930552	,-0.314800067496625,	0.447213595499958,	-0.195258635314440,	-0.0615230124505196,	0.0952528824585985,	-0.0364583742198662	,-1.97821887361772e-31	,-0.0226137575273252,	0.0320923050327762]),
                      np.array([0.0272993452180114	,-0.0265840663469593	,0,	0.0428593893779949,	-0.0810269414612686	,0.117023723786349	,-0.102653539494725	,-1.36919674566051e-16	,0.165500189580178	,-0.347337063987900,	0.478068455182483	,-0.509357208870298	,0.447213595499958	,-0.315935108535685,	0.161069337685743	,-0.0588695188858126,	0.0139258597774258,	2.23619073179563e-31	,0.00863768676208948	,-0.0198341352875852]),
                      ]
    else:
        raise Exception('other band not implement')
    signal = torch.Tensor(range(256))-128.0
    # signal = torch.Tensor(range(1024)).reshape(1, 1, 1024, 1) - 512.0
    signallen = signal.size(0)
    signal=signal.view(1, 1, signallen, 1)


    De=[]
    Re=[]
    De = [Kernel(bandM=kernelInit[i], regularity=4, trainKern=True)() for i in range(band_num)]
    # for i in range(band_num):
    #     kernel_temp = Kernel(bandM=kernelInit[i], regularity=4, trainKern=True)
    #     De.append(kernel_temp())
    Re=De
    ##
    print('------------New function--------------')
    mbandt=Mband(De = De, Re = Re,M=band_num,k = 4)
    #decomposition
    d_coeff=mbandt(input=signal)
    #reconstruction
    re_signal=mbandt(input=d_coeff,decomposition=0)
    print('------------signal--------------')
    signal = signal.view(signallen).numpy()
    print(signal)
    print('------------reconstruction--------------')
    re_signal = re_signal.view(signallen).detach().numpy()
    print(re_signal)
    print('------------abs_residual--------------')
    residual = np.abs(signal - re_signal)
    print(np.sum(residual[6:-6]))

    # for name,param in mbandt.named_parameters():
    #         print(name,param)



