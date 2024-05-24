
from torch.nn import functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple

# 自定义可学习卷积核
# 'Padding_mode=SAME'
#i input
# 正向卷积，不考虑dialation, o=(i+2p-k)/s + 1,考虑dialation, o=[i+2p-d*(k-1)-1]/s+1

def conv2d_same_padding(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    input_cols = input.size(3)
    filter_rows = weight.size(2)
    filter_cols = weight.size(3)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]  # //下取整 对于input_row/stride是上取整
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    out_cols = (input_cols + stride[1] - 1) // stride[1]  #
    padding_cols = max(0, (out_cols - 1) * stride[1] +
                       (filter_cols - 1) * dilation[1] + 1 - input_cols)
    cols_odd = (padding_cols % 2 != 0)

    # 需要补全的总数为n，n 如果是奇数，那么前面增加 (n-1)/2，后面增加（n+1）/2，n 如果是偶数，那么前后都增加n/2
    if rows_odd or cols_odd:
        input = pad(input, [int(cols_odd),0 ,int(rows_odd), 0 ]) #pytorch 经典补齐优先级，左上，跟反卷积需要对应
        # a=int(cols_odd)
        # input = pad(input, [0, int(cols_odd), 0, int(rows_odd)]) #tensorflow补齐优先级，右下
    # else:
    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)
#周期延拓
# def conv2d_same_padding(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
#     # 函数中padding参数可以无视，实际实现的是padding=same的效果
#     input_rows = input.size(2)
#     input_cols = input.size(3)
#     filter_rows = weight.size(2)
#     filter_cols = weight.size(3)
#     effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
#     out_rows = (input_rows + stride[0] - 1) // stride[0]  # 上取整
#     padding_rows = max(0, (out_rows - 1) * stride[0] +
#                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
#     rows_odd = (padding_rows % 2 != 0)
#     out_cols = (input_cols + stride[1] - 1) // stride[1]  # 上取整
#     padding_cols = max(0, (out_cols - 1) * stride[1] +
#                        (filter_cols - 1) * dilation[1] + 1 - input_cols)
#     cols_odd = (padding_cols % 2 != 0)
#
#     # 需要补全的总数为n，n 如果是奇数，那么前面增加 (n-1)/2，后面增加（n+1）/2，n 如果是偶数，那么前后都增加n/2
#     if rows_odd or cols_odd:
#         # input = pad(input, [int(cols_odd),0 ,int(rows_odd), 0 ]) #pytorch 经典补齐优先级，左上
#         input = pad(input, [padding_cols // 2, padding_cols // 2+ int(cols_odd), padding_rows // 2, padding_rows // 2+int(rows_odd)],mode='reflect') #tensorflow补齐优先级，右下
#     else:
#         input = pad(input, [padding_cols // 2, padding_cols // 2, padding_rows // 2, padding_rows // 2])  # tensorflow补齐优先级，右下
#
#     return F.conv2d(input, weight, bias, stride,
#                     padding=(0, 0),
#                     dilation=dilation, groups=groups)


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel, stride,
                 padding, dilation, transposed, output_padding, groups, bias,padding_mode='zeros',device=None, dtype=None):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.kernel_size = self.kernel.size()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        # 可学习卷积核
        if transposed:
            # in_channels, out_channels // groups, *kernel_size
            self.weight = self.kernel.transpose(0, 1)# 如果是转置卷积，则kernel维度交换
        else:
            # out_channels, in_channels // groups, *kernel_size
            self.weight = self.kernel
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(stride, int):
            stride = _pair(stride)
        if isinstance(padding, int):
            padding = _pair(padding)
        if isinstance(dilation, int):
            dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel, stride, padding, dilation,
            False, _pair(0), groups, bias,**factory_kwargs)
        self.kernel_len = kernel.size()[2]
        self.device = device
        # 修改这里的实现函数
    def forward(self, input,realconv=False):
        if realconv:
            inv_idx=torch.arange(self.kernel_len-1, -1, -1,device=self.device).long()
            weight = self.weight.index_select(2,inv_idx)
            # weight = self.weight[inv_idx]
            return conv2d_same_padding(input, weight, self.bias, self.stride,
                                       self.padding, self.dilation, self.groups)
        else:
            return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)

# 反卷积（转置卷积）,不考虑dialation, o=（i-1)*s-2p+k+outp, 考虑dialation, o=(i-1)*s-2p+d*(k-1)+outp+1
class ConvTranspose2d(_ConvNd):
    def __init__(self,in_channels,out_channels,kernel,stride=1,padding=0,dilation = 1,output_padding=0,
                 groups= 1,bias = False,padding_mode='zeros',device=None,dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(stride, int):
            stride = _pair(stride)
        if isinstance(padding, int):
            padding = _pair(padding)
        if isinstance(dilation, int):
            dilation = _pair(dilation)
        if isinstance(output_padding, int):
            output_padding = _pair(output_padding)
        self.padding_mode=padding_mode
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)
        self.device = device
        self.kernel_len = self.kernel.size()[2]

    def _output_padding(self, input, output_size, stride, padding, kernel_size,
                        num_spatial_dims, dilation= None):
        if output_size is None:
            raise Exception('output_size == None')
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})"
                    .format(num_spatial_dims, input.dim(), num_spatial_dims,
                            num_non_spatial_dims + num_spatial_dims, len(output_size)))
            #  out_padding = output_size -[(input_size-1)*stride+dilation*(kernel_size-1)+1] + 2*padding
            padding_res=[]
            out_padding_res=[]
            for d in range(num_spatial_dims):
                for out_padding in range(0,stride[d]):  # out_padding不超过stride-1
                    padding_nums= out_padding -output_size[d]+(input.size(d + num_non_spatial_dims)-1)*stride[d] + dilation[d]*(kernel_size[d+num_non_spatial_dims]-1)+1
                    if padding_nums < 0:
                        raise Exception('padding_nums<0')
                    if (padding_nums %2 != 0):  # padding_nums奇数，只能分配为一奇数一偶数，continue到下一轮out_padding为奇数
                        continue
                    elif (padding_nums % 2 == 0):
                        padding_res.append(int(padding_nums/2))
                        out_padding_res.append(out_padding)
                        break                   # 跳出里层循环，进入下一个维度
            # print(padding_res,out_padding_res)
            padding_ret = tuple(padding_res)
            out_padding_ret = tuple(out_padding_res)
        return padding_ret,out_padding_ret

    def forward(self, input, output_size= None,realconv=False):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
        assert isinstance(self.padding, tuple)
        num_spatial_dims = 2
        padding, output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,
            num_spatial_dims, self.dilation)#返回padding和output_padding

        if realconv:
            inv_idx = torch.arange(self.kernel_len - 1, -1, -1,device=self.device).long()
            weight = self.weight.index_select(2,inv_idx)
            # weight = self.weight[inv_tensor]
            return F.conv_transpose2d(input, weight, self.bias, self.stride, padding,
            output_padding, self.groups,self.dilation)
        else:
            return F.conv_transpose2d(input, self.weight, self.bias, self.stride, padding,
            output_padding, self.groups,self.dilation)
'''
padding： 
dilation * (kernel_size - 1) - padding 
zero-padding will be added to both sides of each dimension in the input. 
Can be a single number or a tuple (padH, padW). Default: 0
由于这里使用的是pytorch内置的补0方式，所以前面卷积也要使用pytorch内置的补0方式，优先左上，否则重构后会出现相位偏移

output_padding:
additional size added to one side of each dimension in the output shape. Can be a single number or a tuple (out_padH, out_padW). Default: 0
'''





if __name__ == '__main__':
    kernel_h=3
    kernel_w=1
    stride_h=2
    stride_w = 3
    input = torch.arange(1, 36, dtype=torch.float).reshape(1, 1, 5, 7)
    print('input.size:', input.size())
    print('input:', input)
    # 定义卷积核
    kernel = Parameter(torch.randn(1, 1, kernel_h, kernel_w, requires_grad=True))
    conv1 = torch.nn.Conv2d(1, 1, (kernel_h, kernel_w), (stride_h, stride_w), padding=0)
    conv2 = Conv2d(1, 1, kernel=kernel, stride=(stride_h, stride_w))
    print('--------Conv2d--------')
    print('conv1:', conv1)
    print('conv2:', conv2)

    input = torch.autograd.Variable(input)
    output1 = conv1(input)
    print('output1.size:', output1.size())
    # print('output1:', output1.detach())
    output2 = conv2(input)
    print('output2.size:', output2.size())
    # print('output2.shape:', output2.detach().numpy().shape)
    # print('output2:', output2.detach())

    print('------ConvTranspose2d------')
    convtrans1 = torch.nn.ConvTranspose2d(1, 1, (kernel_h, kernel_w), (stride_h, stride_h), padding=0, output_padding=(0, 0))
    input1 = convtrans1(output1)
    print('input1.size:', input1.size())
    kernel2 = Parameter(torch.randn(1, 1, kernel_h, kernel_w, requires_grad=True))
    convtrans2=ConvTranspose2d(1,1,kernel=kernel2,stride=(stride_h,stride_h))
    # convtrans2=torch.nn.ConvTranspose2d(1, 1, (4, 1), (2, 2), padding=0, output_padding=(0, 0))
    input2 = convtrans2(output2,[1,1,5,7])
    print('input2.size:', input2.size())
