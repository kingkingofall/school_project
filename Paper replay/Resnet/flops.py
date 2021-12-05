import torch
from fvcore.nn import FlopCountAnalysis
from model import  *
from torchvision.models import resnet18

# resnet18
resnet = nn.Sequential()
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), 
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(3, stride=2, padding=1))

l1 = nn.Sequential(*block(64, 64, 2, True))
l2 = nn.Sequential(*block(64, 128, 2))
l3 = nn.Sequential(*block(128, 256, 2))
l4 = nn.Sequential(*block(256, 512, 2))

b2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                    nn.Linear(512, 1000))

resnet.add_module('input layer', b1)
resnet.add_module('block one', l1)
resnet.add_module('block two', l2)
resnet.add_module('block three', l3)
resnet.add_module('block four', l4)
resnet.add_module('ouput layer', b2)

net = resnet18(pretrained=False)


# 计算flops
print("official,flops:{}".format(FlopCountAnalysis(net,\
                                 (torch.rand(1, 3, 224, 224),)).total()))
print("mine,flops:{}".format(FlopCountAnalysis(resnet, \
                                (torch.rand(1, 3, 224, 224),)).total()))

# 计算参数
t = sum(i.numel() for i in net.parameters())
g = sum(i.numel() for i in net.parameters() if i.requires_grad)
t1 = sum(i.numel() for i in resnet.parameters())
g1 = sum(i.numel() for i in resnet.parameters() if i.requires_grad)
print("official,total parameters:{},grad parameters:{}".format(t, g))
print("mine,total parameters:{},grad parameters:{}".format(t1, g1))
