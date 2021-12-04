import torch
from fvcore.nn import FlopCountAnalysis
from model import  *

# 整体统计
net = VIT_Model(head=16, input_size=256, embedding_size=256, output_size=256,\
                 hidden_size=256 * 4 ,dropout=0, layer_num=1, num_classes=2)

# 相等
net1 = Attention(head=2, embedding_size=256, hidden_size=256)

# 相等
net2 = PatchEmbedding()

# 相等
net3 = VIT_FFC(input_size=256, output_size=256 * 4 , hidden_size=256, dropout=0)

# 相等
net4 = VIT_Block(input_size=256, output_size=256, hidden_size=256*4, head=2, embedding_size=256, dropout=0)

# 计算flops
print(FlopCountAnalysis(net, (torch.rand(1, 3, 224, 224),)).total())
