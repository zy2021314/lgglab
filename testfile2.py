import torch
from torchvision.models import AlexNet
from torchviz import make_dot

x = torch.rand(8, 3, 256, 512)
model = AlexNet()
y = model(x)
g = make_dot(y)
g.view()
#g.render('espnet_model', view=False) # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开