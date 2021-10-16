import torch
import torch.nn as nn
from torchvision.models import resnet, resnet18

class Residual(nn.Module):
    def __init__(self, inputs, outputs, useconv3=False, strides=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, strides, padding=1)
        self.bn1 = nn.BatchNorm2d(outputs)

        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(outputs)       
        
        if  useconv3:
            self.conv3 = nn.Conv2d(inputs, outputs, 1, 2, padding=1)
        else:
            self.conv3 = None

        self.relu = nn.ReLU(inplace=True)

    def forword(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)

def block(inputs, outputs, nums, first=False):
    net = []

    for i in range(nums):
        if not first and i == 0:
            net.append(Residual(inputs, outputs, useconv3=True, strides=2))
        else:
            net.append(Residual(inputs, outputs))        

    return net


if __name__=='__main__':
    resnet = nn.Sequential()
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), 
                        nn.BatchNorm2d(64), nn.ReLU(),
                        nn.MaxPool2d(3, stride=2, padding=1))
    l1 = nn.Sequential(*block(64, 64, 2, True))
    l2 = nn.Sequential(*block(64, 128, 2))
    l3 = nn.Sequential(*block(128, 256, 2))
    l4 = nn.Sequential(*block(256, 512, 2))

    b2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                        nn.Linear(512, 1))

    resnet.add_module('input layer', b1)
    resnet.add_module('layer one', l1)
    resnet.add_module('layer two', l2)
    resnet.add_module('layer three', l3)
    resnet.add_module('layer four', l4)
    resnet.add_module('ouput layer', b2)
    
    print(resnet)
