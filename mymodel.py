import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import e

# 激活函数，实现了两种，其他的大家可以自行实现
def simgold(x):
    return 1/(1 + e ** x)

def relu(x):
    if x <= 0:
        return 0 
    else:
        return x

# 线性层简单实现，公式 y = w*x + b
class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size, is_train = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size))
        
        # 判断是否需要偏置，不需要就把它取消
        if is_train:
            self.bias = nn.Parameter(torch.randn(1,))
        else:
            self.bias = None
            
    # 前向传播
    def forward(x):
        return (x * self.weight + self.bias)
    
     
    
def comput_in(x, k):
    return sum(x.dot(k) for (i, j) in zip(x, k))

def comput_in_out(x, k):
    return sum(comput_in(x, j) for j in k)

class Conv2DLayer(nn.Module):
    def __init__(self, kernel_size, input_channel, output_channel,
                 padding, stride,):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(*kernel_size, input_channel))
        self.bais = nn.Parameter(torch.randn(input_size * output_size))
        
    def forward(self, x):
        return comput_in_out(x, self.weight)

class AttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size,
                 input_size, output_size):
        super().__init__()
        
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        
        self.query_weight = nn.Parameter(torch.randn())
        self.key_weight = nn.Parameter(torch.randn())
        self.value_weight = nn.Parameter(torch.randn())
        
    def forward(self, x):
    
        
        return 
    
    
    
if __name__ == '__main__':
    print(torch.cuda.device_count())
    