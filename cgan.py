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
n_embeddings = 1024

class Inter(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, opts, resolution, use_fp16=False, **kwargs):
        super().__init__(**kwargs)
        self.lowest_dense_res = resolution // (32 if 'd' in opts else 16)
        self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
        self.opts = opts
        in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch
        
        dtype = torch.float16 if use_fp16 else torch.float32

        self.dense1 = nn.Linear( in_ch+n_embeddings, ae_ch , dtype=dtype)
        self.dense2 = nn.Linear( ae_ch, self.lowest_dense_res * self.lowest_dense_res * ae_out_ch,dtype=dtype )
        if 't' not in self.opts:
            self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

        self.label_emb = nn.Embedding(2, n_embeddings)

    def forward(self, inp,label):
        x = torch.cat((inp, self.label_emb(label)),-1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.reshape((-1,self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res))
        
        if 't' not in self.opts:
            x = self.upscale1(x)

        return x

    def get_out_res(self):
        return self.lowest_dense_res * 2 if 't' not in self.opts else self.lowest_dense_res

    def get_out_ch(self):
        return self.ae_out_ch


class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_ch=input_ch, e_ch=e_dims,opts=opts, use_fp16=use_fp16)
        encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2
        self.inter = Inter (in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims,opts=opts, resolution=resolution)
        inter_out_ch = self.inter.get_out_ch()

        self.decoder = Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims,opts=opts)
    
    def forward(self,img,label):
        encoded = self.encoder(img)
        latent = self.inter(encoded,label)
        decoded = self.decoder(latent)[0]
        
        return decoded

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding="same")
        self.max = nn.MaxPool2d(kernel_size=2)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3,padding="same")
        #self.linear1 = nn.Linear(32*64*64+2,128)
        self.linear1 = nn.Linear(32*128*128+2,128)
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64,1)
        self.label_embedding = nn.Embedding(2, 2)
    
    def forward(self,x,labels):
        emb = self.label_embedding(labels)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.max(x)
        
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.max(x)
        x = x.flatten(start_dim = 1)
        
        x = torch.cat([x,emb],-1)
        x = self.lrelu(self.linear1(x))
        x = self.lrelu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x





