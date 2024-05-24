# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import utils.mband_Layers as impLay


def Kernel_selector(band=3):
    kernelselection=band
    if kernelselection ==1:
        # Initialise wavelet kernel (here db-4)
        KI = [np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                               -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965]),
                      np.array([-0.230377813308855,0.714846570552542,-0.630880767929590,-0.027983769416984,0.187034811718881,0.030841381835987,-0.032883011666983,-0.010597401784997])]
    elif kernelselection==2:
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
        KI= [np.array(
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

class FMWN(nn.Module):
    """
        Function that generates a torch FMWN network

        Parameters
        ----------
        inputSize : INT, optional
            Length of the time series. Network is more efficient if set.
            Can be set to None to allow various input size time series.
            The default is None.

        kernelInit: LIST consist of M numpy array
                    the kernelInit is the kernel
                    only used for CF constrains

        M: LIST consist of INT, optional
            Initialisation of the kernel.
            the length of FIR filter = M[i]*K[i]
            The default is [2]. one layer decomposition

        Regularity (k): LIST consist of INT, optional
            the length of FIR filter = M[i]*K[i]
            The default is [3]. one layer decomposition

        Mode : String, optional
            'WPT'
            'DWT': Only scaling filter is used to decompose signal
            'Cus': Specify the coefficients you want to decompose. Not implemented

        kernTrainable : BOOL, optional
            Whether the kernels are trainable. Set to FALSE to compare to traditional wavelet decomposition.
            The default is True.

        lossCoeff : STRING, optional
            To specify which loss on the wavelet coefficient to compute.
            Can be None (no loss computed) or 'l1'' for the L1-norm of the coefficients.
            The default is 'l1'.

        kernelsConstraint : STRING, optional
                ''

        initHT : FLOAT, optional
            Value to initialise the Hard-thresholding coefficient.
            The default is 1.0.
        trainHT : BOOL, optional
            Whether the hard-thresholding coefficient is trainable or not.
            Set FALSE to compare to traditional wavelet decomposition.
            The default is True.
        t: temperature for hard thresholding default is 20

        Returns
        -------
        model1: a torch neural network with outputs the :
        1.reconstructed signals
        2.the loss on the wavelet coefficients: L1 loss
        3.the loss on filter
        4.wavelet coefficients
        """
    def __init__(self, kernelInit=None, m=[3], k=[2],level=None, mode='DWT',kernTrainable=True,  kernelsConstraint='CF',
                 coefffilter=1.0,device='cuda',realconv=True,t=100.0,initHT=1.0, trainHT=True,threshold=True,seqlen=1000):
        super(FMWN, self).__init__()

        self.m = m
        self.k = k
        assert len(self.m)==len(self.k)
        self.level = len(m)
        self.mode=mode
        self.FilterCoeff = coefffilter

        if kernelsConstraint == 'CF':
            '''
            
            此模式下所有层 带数和正则数 一样，设置为m和k的第一项!!
            一组滤波器生成DWT所有滤波器，正交滤波器组
            限定长度可学习时就是网络
            '''
            ''' kernel setting'''
            if isinstance(level, int):
                self.level = level
            self.multilayer_constraints=False #多层滤波器约束，只有单层
            self.lossFilters = 'ON'

            De = [impLay.Kernel(bandM=self.m[0], regularity=self.k[0], trainKern=kernTrainable,device=device)() for _ in
                  range(self.m[0])]
            ''' mband function'''
            mband = [impLay.Mband(De=De, Re=De, M=self.m[0], k=self.k[0],realconv=realconv, device=device) for _ in range(self.level)]
            ''' list to torch module'''
            self.MBAND = nn.ModuleList(mband)


        elif kernelsConstraint == 'KI':
            '''
            kernelInit
            利用滤波器系数建立M-般的分解，不可学习时就是M-band分解，
            可学习就是利用已有的小波系数初始化 + CF约束学习模式？：合成滤波器由分解滤波器确定，每一层都是一个，
            '''
            self.multilayer_constraints = False
            self.lossFilters = 'ON'
            if isinstance(level, int):
                self.level = level
            kernelInit = Kernel_selector(kernelInit)
            De = [impLay.Kernel(bandM=kernelInit[band], trainKern=kernTrainable,device=device)() for band in range(self.m[0])]
            mband = [impLay.Mband(De=De, Re=De, M=self.m[0], k=self.k[0],realconv=realconv, device=device) for _ in range(self.level)]
            self.MBAND = nn.ModuleList(mband)

        elif kernelsConstraint == 'KIPL':
            '''
            kernelInit + PerLayer 
            利用滤波器系数建立M-般的分解，不可学习时就是M-band分解，
            可学习就是利用已有的小波系数初始化 + CF约束学习模式？：合成滤波器由分解滤波器确定，每一层都是一个，
            '''
            self.multilayer_constraints = True
            self.lossFilters = 'ON'
            if isinstance(level, int):
                self.level = level
            # De = [impLay.Kernel(bandM=kernelInit[band], trainKern=kernTrainable,device=device)() for band in range(self.m[0])]
            De = []
            for lev in range(self.level):# 层
                if self.m[lev]==2:
                    kernel_temp = Kernel_selector(1)
                else:
                    kernel_temp = Kernel_selector(self.m[lev])
                De.append([impLay.Kernel(bandM=kernel_temp[band], trainKern=kernTrainable, device=device)() for band in
                  range(self.m[lev])]) # 每层的kernel
            # mband = [impLay.Mband(De=De, Re=De, M=self.m[0], k=self.k[0],realconv=realconv, device=device) for _ in range(self.level)]
            mband = [impLay.Mband(De=De[lev], Re=De[lev], M=self.m[lev], k=self.k[lev], realconv=realconv, device=device) for lev in range(self.level)]
            self.MBAND = nn.ModuleList(mband)

        elif kernelsConstraint == 'PL':
            '''
            PerLayer
            满足M-band小波条件
            每一层mband不一样: m不一样 k不一样
            '''
            self.multilayer_constraints = True
            self.lossFilters = 'ON'

            De=[]
            for lev in range(self.level):# 层
                De.append([impLay.Kernel(bandM=self.m[lev], regularity=self.k[lev], trainKern=kernTrainable,device=device)() for _ in
                  range(self.m[lev])]) # 每层的kernel
            mband = [impLay.Mband(De=De[lev], Re=De[lev], M=self.m[lev], k=self.k[lev],realconv=realconv, device=device) for lev in range(self.level)]
            self.MBAND = nn.ModuleList(mband)


        elif kernelsConstraint == 'Free':
            '''
            所有高低通以及重构滤波器都有自己的核，及无约束
            '''
            self.lossFilters = 'OFF'
            De = []
            Re = []
            for lev in range(self.level):
                De.append([impLay.Kernel(bandM=self.m[lev], regularity=self.k[lev], trainKern=kernTrainable,device=device)() for _ in range(self.m[lev])])
                Re.append([impLay.Kernel(bandM=self.m[lev], regularity=self.k[lev], trainKern=kernTrainable,device=device)() for _ in range(self.m[lev])])
            mband = [impLay.Mband(De=De[lev], Re=Re[lev], M=self.m[lev], k=self.k[lev],realconv=realconv, device=device) for lev in range(self.level)]
            self.MBAND = nn.ModuleList(mband)


        elif kernelsConstraint == 'PerLayer_wavelet':
            '''
            每一层mband不一样, 但低通滤波器一样？
            '''
            raise Exception('kernelsConstraint not implemented')
        else:
            raise Exception('kernelsConstraint not found!')

        '''设置阈值滤波激活函数'''
        # print(self.MBAND[0].M)
        thre = []
        if mode == 'DWT':
            thre = [impLay.WaveletThreshold(init=initHT, trainBias=trainHT, t=t, m=i-1,threshold=threshold)  for i in self.m]
            # thre = [impLay.HardThresholdAssym(init=initHT, trainBias=trainHT, m=i-1) for i in self.m]
        else:
            #计算得到每个层的带数
            tmp = 1
            for m_band in self.m:
                tmp = tmp * m_band
            thre = [impLay.WaveletThreshold(init=initHT, trainBias=trainHT, t=t, m=tmp,threshold=threshold)]#我设计的threshold
            #fink组设计的

        self.thre = nn.ModuleList(thre)
        self.thre_a = impLay.WaveletThreshold(init=initHT, trainBias=trainHT, t=t, m=1,threshold=threshold)
        self.dropout = nn.Dropout(0.3)
        # self.dropout1d = nn.Dropout1d(0.3)#对通道进行整体置零

    def forward(self, inputSig,noDHT=True,ht=None):#inputSig size: batch*channel*length*hight   hight = 1
        assert inputSig.size()[-1]==1
        device=inputSig.device
        if self.mode=='DWT':
            '''分解第一个系数'''
            low_passed_signal=inputSig
            coefficients=[]#由列表组成的
            all_coefficients=[]
            for lev in range(self.level):
                #back up
                # c_tmp=self.MBAND[lev](low_passed_signal)
                # coefficients.append(self.thre[lev](c_tmp[1:], NodefineT=noDHT,HT=ht))#自己设计的threshold
                # # coefficients.append(self.thre[lev](c_tmp[1:]))#fink设计的threshold
                # all_coefficients.append(c_tmp[1:])#原来的，阈值并没有加入系数统计中，也就没有参与进稀疏的度量里面去
                # low_passed_signal=c_tmp[0]#低通逼近系数

                #NEW 阈值直接发挥作用
                c_tmp=self.MBAND[lev](low_passed_signal)
                c_tmp[1:] = self.thre[lev](c_tmp[1:], NodefineT=noDHT,HT=ht)
                coefficients.append(c_tmp[1:] )#自己设计的threshold
                # coefficients.append(self.thre[lev](c_tmp[1:]))#fink设计的threshold
                all_coefficients.append(c_tmp[1:])#new，阈值的影响被考虑进系数统计中，参与进稀疏的度量里面去
                low_passed_signal=c_tmp[0]#低通逼近系数
            #不对低频处理
            # all_coefficients.append([low_passed_signal])
            # re_signal = low_passed_signal
            #对低频也处理
            low_passed_signal =self.thre_a([low_passed_signal], NodefineT=noDHT,HT=ht)
            all_coefficients.append(low_passed_signal)
            re_signal = low_passed_signal[0]
            low_pass_t=True

            for lev in range(self.level-1,-1,-1):
                temp=[]
                # #未加入dropout版
                temp.append(re_signal)
                temp.extend(coefficients[lev])
                re_signal = self.MBAND[lev](input=temp, decomposition=0)

        elif self.mode=='WPT':
            '''分解所有系数
            需要考虑小波系数重排的问题
            奇数不变，偶数变
            '''
            low_passed_signal=[inputSig]
            for lev in range(self.level):
                tmp=[]
                for Mcoefficients in low_passed_signal:
                    tmp.extend(self.MBAND[lev](Mcoefficients))
                low_passed_signal = tmp
            low_passed_signal = self.thre[0](low_passed_signal, NodefineT=noDHT,HT=ht)
            re_signal = low_passed_signal# node 节点个数组成的 tensor
            all_coefficients = low_passed_signal
            for lev in range(self.level - 1, -1, -1):
                temp = []
                m_temp=self.MBAND[lev].M
                len_temp = len(re_signal)
                for i in range(0,len_temp,m_temp):
                    temp.append(self.MBAND[lev](input=re_signal[i:i+m_temp], decomposition=0))
                re_signal=temp
            re_signal=re_signal[0]
        else:
            raise Exception('Decomposition mode not found!')
        ### calculating hard threshold

        for lev in range(self.level):
            if lev == 0:
                # temp = self.thre[lev].left
                # hard_threshold = torch.sum(torch.abs(self.thre[lev].left))
                hard_threshold = torch.sum(self.thre[lev].left)
                # temp = self.thre[lev].right
                # hard_threshold += torch.sum(torch.abs(self.thre[lev].right))
                hard_threshold += torch.sum(self.thre[lev].right)
            else:
                # temp = self.thre[lev].left
                # hard_threshold += torch.sum(torch.abs(self.thre[lev].left))
                hard_threshold += torch.sum(self.thre[lev].left)
                # temp = self.thre[lev].right
                # hard_threshold += torch.sum(torch.abs(self.thre[lev].right))
                hard_threshold += torch.sum(self.thre[lev].right)
                #
            if low_pass_t:
                hard_threshold += torch.sum(self.thre_a.left)
                hard_threshold += torch.sum(self.thre_a.right)
        # hard_threshold = 0
        #compute specified loss on filters
        #kernelsConstraint == 'PerFilter'
        # k = self.MBAND[0].filter()

        # Compute specified loss on coefficients  系数稀疏的损失
        # if self.lossCoeff == 'None':
        #     vLossCoeff = torch.tensor(0.0,device=device)
        #     # print(vLossCoeff)
        # elif self.lossCoeff == 'l1':# L1-Sum
        #     node_number = len(all_coefficients)  # 分解出的节点小波系数
        #     for i in range(node_number):
        #         if i == 0:
        #             vLossCoeff = torch.mean(torch.abs(all_coefficients[i]))
        #         else:
        #             vLossCoeff = vLossCoeff + torch.mean(torch.abs(all_coefficients[i]))
        #     vLossCoeff = torch.div(vLossCoeff, node_number)
        # else:
        #     raise ValueError(
        #         'Could not understand value in \'lossCoeff\'. It should be either \'l1\' or \'None\'')

        if self.lossFilters == 'ON':
            #约束平方和，好优化 MSE
            if self.multilayer_constraints:
                for lev in range(self.level):
                    filter_list = self.MBAND[lev].filter()
                    stacked_filter = torch.stack(filter_list).squeeze()
                    m = self.m[lev]
                    k = self.k[lev]
                    ''' the low-pass and high-pass filter condition '''
                    for i in range(0, m):
                        if i == 0:
                            filter_condition = (torch.sum(filter_list[i]) - torch.sqrt(
                                torch.tensor(m, device=device, dtype=float))) ** 2
                        else:
                            filter_condition = filter_condition + (torch.sum(filter_list[i])) ** 2
                    ''' the orthonormal_condition '''
                    eye_target = torch.eye(k * m, device=device)  # 单位阵
                    P = torch.zeros([k * m, k * m + (k - 1) * m], dtype=torch.float, device=device)
                    Q = torch.zeros([k * m, k * m + (k - 1) * m], dtype=torch.float, device=device)

                    for i in range(k):  # 块的行数
                        P[i * m:m * (i + 1), i * m:(i * m + m * k)] = stacked_filter[:, :]

                    for i in range(k):  # 块的行数,第i行块
                        for j in range(k):  # 第j列块
                            Q[i * m:(i + 1) * m, (i * m + j * m):(i * m + (j + 1) * m)] = stacked_filter[:,
                                                                                          j * m:(j + 1) * m].T

                    orthonormal_condition = torch.sum(torch.square(torch.mm(P, P.T) - eye_target)) + torch.sum(
                        torch.square(torch.mm(Q, Q.T) - eye_target))  # 元素平方
                    if lev ==0:
                        vLossFilters = self.FilterCoeff * filter_condition + orthonormal_condition
                    else:
                        vLossFilters = self.FilterCoeff * filter_condition + orthonormal_condition + vLossFilters
            else:
                filter_list = self.MBAND[0].filter()
                stacked_filter = torch.stack(filter_list).squeeze()
                m=self.m[0]
                k=self.k[0]
                ''' the low-pass and high-pass filter condition '''
                for i in range(0,m):
                    if i == 0:
                        filter_condition = (torch.sum(filter_list[i]) - torch.sqrt(torch.tensor(m,device=device,dtype=float)))**2
                    else:
                        filter_condition = filter_condition + (torch.sum(filter_list[i]))**2
                ''' the orthonormal_condition '''
                eye_target = torch.eye(k*m,device=device) #单位阵
                P = torch.zeros([k*m,k*m+(k-1)*m],dtype=torch.float,device=device)
                Q = torch.zeros([k*m,k*m+(k-1)*m],dtype=torch.float,device=device)

                for i in range(k):#块的行数
                    P[i*m:m*(i+1),i*m:(i*m+m*k)]=stacked_filter[:,:]

                for i in range(k):#块的行数,第i行块
                    for j in range(k):#第j列块
                        Q[i * m:(i + 1) * m, (i * m + j * m):(i * m + (j + 1) * m)] = stacked_filter[:,j*m:(j+1)*m].T

                orthonormal_condition = torch.sum(torch.square(torch.mm(P,P.T)-eye_target)) + torch.sum(torch.square(torch.mm(Q,Q.T)-eye_target))#元素平方

                vLossFilters = self.FilterCoeff*filter_condition + orthonormal_condition

        elif self.lossFilters == 'OFF':
            vLossFilters = torch.tensor(0.0,device=device)
        else:
            raise ValueError(
                'Could not understand value in \'lossCoeff\'. It should be either \'ON\' or \'OFF\'')

        # 需要改成返回小波系数，在主程序计算，并用于画图，filter loss可以在这里计算，special 版本需求在改成返回filter
        return re_signal,all_coefficients,vLossFilters,hard_threshold

if __name__ == '__main__':
    # signal=torch.randn(1,1,64,1,dtype=torch.float32)
    signal = torch.Tensor(range(256)) - 128.0
    signallen = signal.size(0)
    signal = signal.view(1, 1, signallen, 1)
    band_num = 2
    if band_num == 4:
        kernelInit = [np.array(
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

    elif band_num == 3:  # 相位偏移
        kernelInit = [np.array(
            [-0.145936007553990, 0.0465140821758900, 0.238964171905760, 0.723286276743610, 0.530836187013740,
             0.338386097283860]),
                      np.array([0.426954037816980, -0.136082763487960, -0.699119564792890, -0.0187057473531300,
                                0.544331053951810, -0.117377016134830]),
                      np.array([0.246502028665230, -0.0785674201318500, -0.403636868928920, 0.460604752521310,
                                -0.628539361054710, 0.403636868928920]),
                      ]
    elif band_num == 2:
        kernelInit = [np.array([-0.129409522550921, 0.224143868041857, 0.836516303737469, 0.482962913144690]),
                      np.array([-0.482962913144690, 0.836516303737469, -0.224143868041857, -0.129409522550921])]
    elif band_num == 5:
        kernelInit = [np.array(
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
    else:
        raise Exception('other band not implement')
    # print(f'{kernelInit[2]:.30f}')

    # print(signal.reshape(1,-1))
    model=FMWN(kernelInit=kernelInit,m=[2,3],k=[2,4],level=2,kernelsConstraint='kernelInit',mode='DWT',lossCoeff='None')
    #
    # for name,param in model.named_parameters():
    #     print(name,param)
    out, loss1, loss2=model(signal)
    a=2
    # print(out[0].reshape(1,-1))
    # out2=model(signal2)
    # graph=make_dot(out[0],params=dict(model.named_parameters()))
    # graph.view()
    # print(out2[0].shape,)
    # print(out2[1],)
    # print(out2[2].shape, )

