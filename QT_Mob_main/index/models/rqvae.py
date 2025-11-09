import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000): # d_model是你希望编码到的维度
        super().__init__()
        # 简单实现，对x, y分别编码然后拼接
        self.d_model_half = d_model // 2
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model_half, 2) * (-math.log(10000.0) / self.d_model_half))
        
        pe = torch.zeros(max_len, self.d_model_half)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x): # x的形状是 (batch_size, 2)
        
        # 对x进行归一化，x的形状是 (batch_size, 2)，要求每一个值都在0-1范围内
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-8)
        x = torch.clamp(x, 0.0, 1.0)
        # 假设坐标已经被归一化到 [0, max_len-1]
        x_scaled = (x * (self.pe.size(0) - 1)).long()
        
        pe_x = self.pe[x_scaled[:, 0]]
        pe_y = self.pe[x_scaled[:, 1]]
        
        return torch.cat([pe_x, pe_y], dim=1)

class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        
        ##for loc
        #[2048,1024,512,256,128,64]
        # self.encode_layer_dims =self.layers + [self.e_dim]
       
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        
        

    def forward(self, x, use_sk=True):
        
        ##for loc
        # x_pe = self.pos_encoder(x)
       
        
        
        x = self.encoder(x)
        # 量化后的表示 x_q、量化损失 rq_loss 和索引 indices
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        
         ##for loc
        # 输入 xs 的编码表示，并通过残差向量量化器得到索引 indices
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices


    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon