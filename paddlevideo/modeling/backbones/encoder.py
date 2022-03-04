from dataclasses import replace
from tkinter.tix import Tree
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import matplotlib.pyplot as plt
import math
from paddle import ParamAttr


## 6. MultiHeadAttention  k=1/math.sqrt(fan) bias~U(-k, k)
def init_params(name=None, in_channels=1, kernel_size=1):
    fan_in = in_channels * kernel_size * 1
    k = 1. / math.sqrt(fan_in)
    param_attr = ParamAttr(name=name,
                           initializer=paddle.nn.initializer.Uniform(low=-k,
                                                                     high=k))
    return param_attr


class MultiHeadAttention(nn.Layer):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        # read config
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

  
        # nn.Linear weight initialized by constant(0) defalut, should be kaiming_normal, bias is constant(0) default
        # nn.convND weight should init by kaiming_normal(i.e. default mananer), 
        # and bias should init by k=1/math.sqrt(fan)~U(-k, k), respectively
        self.W_Q = nn.Linear(d_model, d_k * n_heads,
                             weight_attr=nn.initializer.KaimingNormal(),
                             bias_attr=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads,
                             weight_attr=nn.initializer.KaimingNormal(),
                             bias_attr=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads,
                             weight_attr=nn.initializer.KaimingNormal(),
                             bias_attr=False)

        # optional
        self.linear = nn.Linear(n_heads * d_v, d_model,
                                weight_attr=nn.initializer.KaimingNormal(),
                                bias_attr=None)  # for output projection
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        # for FFN  
        self.conv1 = nn.Conv1D(in_channels=d_model, out_channels=d_ff, kernel_size=1,
                               weight_attr=init_params(in_channels=d_model, kernel_size=1),
                               bias_attr=init_params(in_channels=d_model, kernel_size=1)
                               )
        self.FFN_Relu = nn.ReLU()
        self.FFN_dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1D(in_channels=d_ff, out_channels=d_model, kernel_size=1,
                               weight_attr=init_params(in_channels=d_model, kernel_size=1),
                               bias_attr=init_params(in_channels=d_ff, kernel_size=1)
                               )

        self.dropout2 = nn.Dropout(p=dropout)
        self.layer_norm_ffn = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):  # x: [batch_size x len_q x d_model]

        residual, batch_size = x, paddle.shape(x)[0]
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        Q = paddle.transpose(paddle.reshape(self.W_Q(x),shape=[batch_size, -1, self.n_heads, self.d_k]),perm=[0,2,1,3])
        K = paddle.transpose(paddle.reshape(self.W_K(x),shape=[batch_size, -1, self.n_heads, self.d_k]),perm=[0,2,1,3])
        V = paddle.transpose(paddle.reshape(self.W_V(x),shape=[batch_size, -1, self.n_heads, self.d_k]),perm=[0,2,1,3])
        
        ## context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        scores = paddle.matmul(Q, paddle.transpose(K,perm=[0,1,3,2])) / np.sqrt(self.d_k)
        softmax = nn.Softmax(axis=-1)
        attn = softmax(scores)

        context = paddle.matmul(attn, V)
        context =paddle.reshape(paddle.transpose(context, perm=[0,2,1,3]),shape=[batch_size, -1, self.n_heads * self.d_v])
        # context: [batch_size x len_q x n_heads * d_v]
     
        output = self.linear(context)  # optional
        output = self.layer_norm(self.dropout1(output) + residual)  # self.dropout1 is optional
        # output: [batch_size x len_q x d_model]

        residual = output  # output : [batch_size, len_q, d_model]
        
        output = self.FFN_Relu(self.conv1(paddle.transpose(output,perm=[0,2,1])))
        output = self.FFN_dropout(output) # optional
        output = paddle.transpose(self.conv2(output),perm=[0,2,1])

        output = self.layer_norm_ffn(output + residual)  # self.dropout2 is optional
        # output : [batch_size, len_q, d_model]
        # should transpose(1,2)
        return output


## 3. PositionalEncoding 代码实现
class PositionalEncoding(nn.Layer):
    def __init__(self, d_model, seq_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)
        # self.register_buffer('pe', self.pe)  

    def forward(self, x):
        """
        x: [batch_size, d_model, seq_len]
        """
        
        x = paddle.transpose(x, perm=[0,2,1])
        
        pe = paddle.zeros(shape=[self.seq_len, self.d_model])
        #position (max_len, 1)
        position = paddle.unsqueeze(paddle.arange(start=0,end=self.seq_len, dtype='float32'),axis=1)
        
        div_term = paddle.exp(paddle.arange(start=0,end=self.d_model,step=2,dtype='float32') * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = paddle.sin(paddle.multiply(position, div_term))
        pe[:, 1::2] = paddle.cos(paddle.multiply(position, div_term))
        #pe:[max_len*d_model]
        
        pe = paddle.unsqueeze(pe,axis=0)
        x = x + pe  # [batch_size, seq_len, d_model]

        return x  # self.dropout(x)


class Encoder(nn.Layer):  # d_k=d_v 
    def __init__(self, d_model, seq_len, n_layers, d_k, d_v, d_ff, n_heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model, seq_len, dropout)

        # d_k, d_v, d_model, d_ff, n_heads=4
        self.layers = nn.LayerList([MultiHeadAttention(d_k, d_v, d_model, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        '''
        x: [batch_size, d_model, seq_len]
        '''
        x = self.pos_emb(x)

        for layer in self.layers:
            x = layer(x)

        x = paddle.transpose(x,perm=[0,2,1])
        return x
