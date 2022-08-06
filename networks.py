# This is the networks script0
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#注意力block
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=2):
        super(ChannelAttention, self).__init__()
        #全局平均池化其实就是对每一个通道图所有像素值求平均值，然后得到一个新的1 * 1的通道图
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #得到特征值输入1*1卷积层,  使用卷积代替全连接
        self.fc1 = nn.Conv2d(in_channel, in_channel//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel//ratio, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_pool_out = self.avg_pool(x)
        max_pool_out = self.max_pool(x)
        avg_pool_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        max_pool_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = max_pool_out + avg_pool_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1 #填充为卷积核的一半
        #将得到的两个时空特征卷积为1个
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([avg_pool_out,max_pool_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, in_channel, ratio=2, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.CA = ChannelAttention(in_channel, ratio=ratio)
        self.SA = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        out = x * self.CA(x)
        out = out * self.SA(out)
        return out

class PowerLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    标准偏差层:计算给定'dim'的数据的std
    '''

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        #二维平均池化
        """
        torch.nn.AvgPool2d( kernel_size , stride=None , padding=0 , ceil_mode=False , count_include_pad=True , divisor_override=None )
        desc：在多个平面组成的输入信号上应用2D，池化后图像，长宽均为原来的一半
        param：
        kernelsize：池化核的大小
        stride：窗口的移动步幅，默认与kernel_size大小一致
        """
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


class LGG(nn.Module):
    ##param：（输入的channel，输出的channel，卷积核大小，池化长度,池化步长）
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            #param：（输入的channel，输出的channel，卷积核大小，步长）
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            #param：
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate*pool))
        )
    """
    num_classes:
    input_size:(1,34,512)
    sampling_rate:
    num_T:
    out_graph:
    dropout_rate:
    pool:
    pool_step_rate:
    idx_graph:
    """
    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        # input_size: EEG frequency x channel x datapoint
        super(LGG, self).__init__()
        self.idx = idx_graph
        #卷积窗口大小
        self.window = [0.5, 0.25, 0.125]
        #
        self.pool = pool
        #此次要输入的信道数量，因为不同的图不一样
        self.channel = input_size[1]
        #脑袋区域分为多少个，根据索引来分辨
        self.brain_area = len(self.idx)

        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        #通过设置卷积内核为(1,lenght)和步长为1，我们可以使用conv2d实现1d操作
        # param：（输入的channel，输出的channel，卷积核大小，池化长度,池化步长）
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_t_ = nn.BatchNorm2d(num_T)

        #添加注意力模块
        self.cbam_eeg = CBAMBlock(32)
        self.cbam_eog = CBAMBlock(2)


        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)))
        # diag(W) to assign a weight to each local areas diag(W)为每个局部区域分配权重
        size = self.get_size_temporal(input_size)

        self.avg_pool = nn.AvgPool2d((1, 2))

        """
        torch.nn.parameter.Parameter(data=None, requires_grad=True)
        desc:将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。
        即在定义网络时这个tensor就是一个可以训练的参数了。
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        param:
        data (Tensor) – parameter tensor
        requires_grad (bool, optional) – if the parameter requires gradient.
        See Locally disabling gradient computation for more details. Default: True 
        """
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        #Xavier初始化参数
        nn.init.xavier_uniform_(self.local_filter_weight)

        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)

        # aggregate function
        self.aggregate = Aggregator(self.idx)

        # trainable adj weight for global network
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        #小bn为1dBN，大bn为2d
        self.bn_ = nn.BatchNorm1d(self.brain_area)
        # learn the global network of networks

        self.GCN = GraphConvolution(size[-1], out_graph)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes))

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        con_out = self.BN_t(out)

        #切分数据
        eeg_out = con_out[:, :, 0:32, :]
        eog_out = con_out[:, :, 32:, :]
        #调整维度
        eeg_out = eeg_out.permute(0, 2, 1, 3)
        eog_out = eog_out.permute(0, 2, 1, 3)
        eeg_out = self.cbam_eeg(eeg_out)
        eog_out = self.cbam_eog(eog_out)
        out = torch.cat([eeg_out, eog_out],dim=1)
        out = self.avg_pool(out)

        # out = self.OneXOneConv(out)
        # out = self.BN_t_(out)
        #将维度换位
        #out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        adj = self.get_adj(out)
        out = self.bn(out)
        #图卷积，一行代码即实现
        out = self.GCN(out, adj)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    def local_filter_fun(self, x, w):
        #unsqueeze是升维函数，而后面是重复，这个操作相当于（1.1）-》（x.size,1,1）
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        #F为function函数
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        adj = self.self_similarity(x)   # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s


class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area每个区域内通道数量的列表
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data  x:批量处理x通道x数据
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)


if __name__ == '__main__':
    testx = torch.rand(64, 1, 34, 7680);
    model = LGG(
            num_classes=2,
            input_size=(1, 34, 7680),
            sampling_rate=int(128*1),
            num_T=64,
            out_graph=32,#hidden
            dropout_rate=0.5,
            pool=16,
            pool_step_rate=0.25,
            idx_graph=[2, 2, 2, 2, 4, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2]);
    testy = model(testx);
    print(model)
    # graph = make_dot(testy)
    # graph.render("model", view=False)

    # print(testx.shape)
    # print(testy.shape)