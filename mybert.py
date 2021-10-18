import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.modules.sparse import Embedding

class FFN_layer(nn.Module):
    def __init__(self, FFN_input, FFN_hidden, FFN_output):
        super().__init__()
        self.dense1 = nn.Linear(FFN_input, FFN_hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(FFN_hidden, FFN_output)
    def forword(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        

class ADD_Norm(nn.Module):
    def __init__(self, dropout, layer_size):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(layer_size)
    def forword(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class Encoder(nn.Module):
    def __init__(self, vocab_size, multihead,
                FFN_input, FFN_hidden, FFN_output,
                dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(vocab_size, multihead)
        self.nor1 = ADD_Norm(dropout, FFN_output)
        self.FFN = FFN_layer(FFN_input,FFN_hidden, FFN_output)
        self.nor2 = ADD_Norm(dropout, FFN_output)

        

    def forword(self, X, lens):
        Y = self.nor1(X, self.attention(X,lens))
        return self.nor2(Y, self.FFN(Y))


class bert(nn.Module):
    def __init__(self, vocab_size, num_hidden, multihead, hiddens,
                FFN_input, FFN_hidden, FFN_output, dropout, max_len=1000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hidden)
        self.segment_embedding = nn.Embedding(2, num_hidden)

        self.blk = nn.Sequential()
        for i in range(hiddens):
            self.blk.add_module('block{i}',
            Encoder(vocab_size, num_hidden, FFN_input=FFN_input, FFN_hidden=FFN_hidden, FFN_output=FFN_output, dropout=dropout))
            
        self.post_embedding = nn.Parameter(torch.randn(1, max_len, num_hidden))

    def forword(self, tokens, segments, X, lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.post_embedding(X)[:,:X.shape[1]]
        for blk in self.blk:
            X = blk(X, lens)
        return X

if __name__=='__main__':
    x = bert(10000, 100, 12, 2, 1024, 1024, 1024, 0.5)
    print(x)

