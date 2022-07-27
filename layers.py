import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
class GraphConvolution(Module):
    """
    simple GCN layer
    专门实现GCN层的文件
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        #一种权值初始化方法
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)

        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        #似乎是重整参数的函数
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        #简单的乘且使用relu激活函数
        output = torch.matmul(x, self.weight)-self.bias
        output = F.relu(torch.matmul(adj, output))
        return output

    
