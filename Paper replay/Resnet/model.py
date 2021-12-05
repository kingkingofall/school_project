import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, inputs, outputs, useconv3=False, strides=1):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(inputs, outputs, 3, strides, padding=1)
        self.bn1 = nn.BatchNorm2d(outputs)

        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(outputs)       
        
        if  useconv3:
            self.conv3 = nn.Conv2d(inputs, outputs, 1, 2, padding=0)
            self.bn3 = nn.BatchNorm2d(outputs)
        else:
            self.conv3 = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        Y = self.relu(self.bn1(self.conv1(x)))
        
        Y = self.bn2(self.conv2(Y))
        
        if self.conv3:
            x = self.bn3(self.conv3(x))

        Y += x
        return self.relu(Y)

def block(inputs, outputs, nums, first=False):
    net = []

    for i in range(nums):

        if not first and i == 0:
            net.append(Residual(inputs, outputs, useconv3=True, strides=2))
        else:
            net.append(Residual(outputs, outputs))        

    return net

