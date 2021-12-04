import torch
import torch.nn as nn
import torch.nn.functional as F

image_size = 224

def to_2tuple(x):
    return tuple([x] * 2)

# embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, image_size = image_size, embedding_size = 256, patch_size = 8, input_channels = 3):
        super(PatchEmbedding, self).__init__()

        # 计算patch的大小、个数
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.patch_num = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        # 利用卷积作为embedding层
        self.embedding = nn.Conv2d(input_channels, embedding_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.size()

        x = self.embedding(x).flatten(start_dim=2).permute(0, 2, 1)

        return x


# attention
class Attention(nn.Module):
    def __init__(self, head, embedding_size, hidden_size, qk_scale = False):
        super(Attention, self).__init__()

        self.embedding_size = embedding_size
        # 定义qkv的权重矩阵
        self.W_q = nn.Linear(embedding_size, hidden_size)
        self.W_k = nn.Linear(embedding_size, hidden_size)
        self.W_v = nn.Linear(embedding_size, hidden_size)
        self.head = head  
        self.head_size = embedding_size // head
        # 缩放因子
        self.scale = qk_scale or (self.head_size) ** -0.5

        self.dropout1 = nn.Dropout(0.)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.)

    def forward(self, x):

        B, N, C = x.shape

        # 将qkv展开
        Q = self.W_q(x).view(B, self.head, N, self.head_size)
        K = self.W_k(x).view(B, self.head, N, self.head_size)
        V = self.W_v(x).view(B, self.head, N, self.head_size)
        
        # 计算注意力分数
        att = Q @ K.transpose(-2, -1) * self.scale

        # # 对分数进行softmax
        att = F.softmax(att, dim=-1)

        att = self.dropout1(att)

        # 将分数和v相乘
        out = (att @ V).reshape(B, N, C)


        # # 将多头注意力的输出进行拼接
        out = self.fc(out)
        out = self.dropout2(out)

        return out

# FFC
class VIT_FFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout):
        super(VIT_FFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

# vit block
class VIT_Block(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, head, embedding_size, dropout):
        super(VIT_Block, self).__init__()
        output_size = output_size or embedding_size

        self.norm1 = nn.LayerNorm(embedding_size, eps=1e-05)
        self.attention = Attention(head, embedding_size, embedding_size, 0.)

        self.norm2 = nn.LayerNorm(embedding_size, eps=1e-05)
        self.ffc = VIT_FFC(input_size, output_size, hidden_size, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffc(self.norm2(x))
        return x



# 这是官方的初始化方式
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# vit model
class VIT_Model(nn.Module):
    def __init__(self,input_size, output_size, hidden_size,\
         head, embedding_size, dropout, layer_num, num_classes):
        super(VIT_Model, self).__init__()
        self.layer_num = layer_num

        self.embedding = PatchEmbedding(embedding_size=embedding_size)
        self.patch_size = self.embedding.patch_num

        self.pos = nn.Parameter(torch.randn(1, self.patch_size + 1, embedding_size))
        self.cls = nn.Parameter(torch.randn(1, 1, embedding_size))

        self.blocks = nn.Sequential(*[VIT_Block(input_size, output_size, hidden_size, head, embedding_size, dropout) for i in range(layer_num)])
        
        self.norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.fc = nn.Linear(input_size, num_classes)

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.apply(_init_vit_weights)


    def forward(self, x):
        B = x.size(0)
        
        x =  self.embedding(x)

        cls = self.cls.expand((B, -1, -1))

        x = torch.cat([x, cls], dim=1)
        x = x + self.pos

        x = self.blocks(x)
        x = self.norm(x)
        x = self.fc(x[:,0])

        return x
