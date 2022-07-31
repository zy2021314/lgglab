import os
import time
import h5py
import numpy as np
import pprint
import random
from networks import *
from eeg_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from vitnetwork import ViT
def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def get_model(args):
    if args.model == 'LGG':
        #保存图的索引格式，{2。2.2----}
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = (args.input_shape[0], channels, args.input_shape[2])
        model = LGG(
            num_classes=args.num_class,
            input_size=input_size,
            sampling_rate=int(args.sampling_rate*args.scale_coefficient),
            num_T=args.T, out_graph=args.hidden,
            dropout_rate=args.dropout,
            pool=args.pool,
            pool_step_rate=args.pool_step_rate,
            idx_graph=idx_local_graph)
    elif  args.model=='VIT':
        #imagesize = args.image_size
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        model = ViT(
            num_classes=args.num_class,
            image_size=(channels, args.input_shape[2]),  # image size is a tuple of (height, width)
            patch_size=(channels, 16),  # patch size is a tuple of (height, width)
            dim=1024,
            depth=1,  # tansformer深度
            heads=16,
            mlp_dim=2048,
            channels=1,
            dropout=0.1,
            emb_dropout=0.1
        )


    return model


def get_dataloader(data, label, batch_size):
    # load the data
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.带标签平滑的NLL损失
    由于某些对象的类别高度不平衡，采用了0.1平滑率的标签平滑
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.标签平滑模块的构造函数。
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)# x: (batch size * class数量)，即log(p(k))
        #unsqueeze为升维的作用，squeeze降维
        # 相当于取出logprobs中的真实标签的那个位置的logit的负值
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))# target: (batch size) 数字标签
        nll_loss = nll_loss.squeeze(1) # (batch size * 1)再squeeze成batch size，即log(p(k))δk,y，δk,y表示除了k=y时该值为1，其余为0
        smooth_loss = -logprobs.mean(dim=-1)# 在class维度取均值，就是对每个样本x的所有类的logprobs取了平均值。
        # smooth_loss = -log(p(k))u(k) = -log(p(k))∗ 1/k
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss# (batch size)
        # loss = (1−ϵ)log(p(k))δk,y + ϵlog(p(k))u(k)
        return loss.mean()# −∑ k=1~K [(1−ϵ)log(p(k))δk,y+ϵlog(p(k))u(k)]






