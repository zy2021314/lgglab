import torch
import math
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
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


if __name__ == '__main__':
    testx = torch.rand(64, 2, 64, 347)
    model = CBAMBlock(2)
    testy = model(testx)
    print(model)
    print(testx.shape)
    print(testy.shape)