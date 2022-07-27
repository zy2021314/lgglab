# tested 8/9/2020
import os.path as osp
import scipy.io as sio
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset

"""
该文件的目的是为了将数据集类型建好
"""

class eegDataset(Dataset):
    # x_tensor: (sample影片, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        #断点语句 如果没加载进来则结束程序
        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
