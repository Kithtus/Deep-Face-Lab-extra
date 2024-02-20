###############################
##############################
#Attention generator
import numpy as np
import torch
import torch.nn as nn
from DeepFakeArchi_torch import *
from torch.utils.data import DataLoader
import gc
import math
from torch.utils.data import Dataset
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ae_dims = 64
e_dims = 32
d_dims = 32
d_mask_dims = 8
input_ch = 3
opts = "ud"
use_fp16=False #half precision: only on GPU
resolution = 512 #512
n_embeddings = 5096
n_heads =4
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model, 1) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, 1)
        pe = torch.sin(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        #x = x + self.pe[:x.size(0)]
        temp = self.pe.unsqueeze(0)
        x = x+ temp[:,:x.shape[1],:]
        return self.dropout(x)

class Encoder_att(nn.Module):

    def __init__(self, in_ch, e_ch, opts, use_fp16=False, **kwargs):
        super().__init__(**kwargs)
        self.in_ch = in_ch
        self.e_ch = e_ch
        self.opts = opts
        conv_dtype = torch.float16 if use_fp16 else torch.float32
        self.pos_enco = PositionalEncoding(256)
        
        if 't' in self.opts:
            self.down1 = Downscale(self.in_ch, self.e_ch, kernel_size=5, conv_dtype=conv_dtype)
            self.res1 = ResidualBlock(self.e_ch, conv_dtype=conv_dtype)
            self.down2 = Downscale(self.e_ch, self.e_ch*2, kernel_size=5, conv_dtype=conv_dtype)
            self.down3 = Downscale(self.e_ch*2, self.e_ch*4, kernel_size=5, conv_dtype=conv_dtype)
            self.down4 = Downscale(self.e_ch*4, self.e_ch*8, kernel_size=5, conv_dtype=conv_dtype)
            self.down5 = Downscale(self.e_ch*8, self.e_ch*8, kernel_size=5, conv_dtype=conv_dtype)
            self.res5 = ResidualBlock(self.e_ch*8, conv_dtype=conv_dtype)
        else:
            self.down1 = DownscaleBlock(self.in_ch, self.e_ch, n_downscales=4 if 't' not in self.opts else 5, kernel_size=5, conv_dtype=conv_dtype)

        

    def forward(self, x):

        if 't' in self.opts:
            x = self.down1(x)
            x = self.res1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            x = self.res5(x)
        else:
            x = self.down1(x)
        x = torch.flatten(x, 2) #flaten all dimension excep batch
        x = torch.permute(x,(0,2,1))

        x = self.pos_enco(x)
        if 'u' in self.opts:
            x = pixel_norm(x,dim = -1)
        return x

    def get_out_res(self, res):
        n_downscales= 4 if 't' not in self.opts else 5
        for i in range(n_downscales):
            res = (res - 3)//2
        return res

    def get_out_ch(self):
        return self.e_ch * 8




class SelfAttention(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, opts, resolution, use_fp16=False, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.use_fp16 = use_fp16
        self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
        self.opts = opts
        self.lowest_dense_res = resolution // (32 if 'd' in opts else 16)
        dtype = torch.float16 if use_fp16 else torch.float32
        self.n_embeddings = n_embeddings
        self.query = nn.Linear( in_ch,  self.ae_out_ch , dtype=dtype)
        self.key = nn.Linear( in_ch,  self.ae_out_ch , dtype=dtype)
        self.value = nn.Linear( in_ch,  self.ae_out_ch , dtype=dtype)
        self.attention = torch.nn.MultiheadAttention(self.ae_out_ch, num_heads = n_heads, batch_first = True)
        self.dense = nn.Linear(self.ae_out_ch * 841 +n_embeddings ,self.lowest_dense_res * self.lowest_dense_res * ae_out_ch)
        self.label_emb = nn.Embedding(2, n_embeddings)
        if 't' not in self.opts:
            self.upscale1 = Upscale(ae_out_ch, ae_out_ch)
        
    def forward(self,encoded,label ):
        batch_size,lenght, emb = encoded.size()
        label_emb = self.label_emb(label)
        #label_emb = label_emb.unsqueeze(1).expand(batch_size,lenght, self.n_embeddings)
        #encoded = torch.concat([encoded, label_emb], dim = -1)
        query = self.query(encoded)
        key = self.key(encoded)
        value = self.value(encoded)
        out = self.attention(query, key, value)[0]
        out = torch.flatten(out,1)
        out = torch.concat([out,label_emb], dim = -1)
        out = self.dense(out)
        out = out.reshape(-1,self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res)

        if 't' not in self.opts:
            out = self.upscale1(out)
        return out
    
    def get_out_res(self):
        return self.lowest_dense_res * 2 if 't' not in self.opts else self.lowest_dense_res

    def get_out_ch(self):
        return self.ae_out_ch


class generator_att(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder_att(in_ch=input_ch, e_ch=e_dims,opts=opts, use_fp16=use_fp16)
        encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2
        self.inter = SelfAttention(in_ch=self.encoder.get_out_ch(), ae_ch=ae_dims, ae_out_ch=ae_dims,opts=opts, resolution=resolution)
        inter_out_ch = self.inter.get_out_ch()
        self.decoder = Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims,opts=opts)
    
    def forward(self,img,label):
        encoded = self.encoder(img)
        latent = self.inter(encoded,label)
        decoded = self.decoder(latent)[0]
        return decoded