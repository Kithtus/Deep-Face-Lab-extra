import numpy as np
import torch
import torch.nn as nn
from DeepFakeArchi_torch import *
from torch.utils.data import DataLoader
import gc
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.io import read_image
import os


FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

ae_dims = 128
e_dims = 64
d_dims = 64
d_mask_dims = 16
input_ch = 3
opts = "ud"
use_fp16=False #half precision: only on GPU
resolution = 512 #512


### COMPONENTS

class Encoder(nn.Module):

    def __init__(self, in_ch, e_ch, opts, use_fp16=False, **kwargs):
        self.in_ch = in_ch
        self.e_ch = e_ch
        self.opts = opts
        conv_dtype = torch.float16 if use_fp16 else torch.float32
            
        super().__init__(**kwargs)
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
        x = torch.flatten(x, 1) #flaten all dimension excep batch
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
    

class Decoder(nn.Module):

    def __init__(self, in_ch, d_ch, d_mask_ch, opts, use_fp16=False,):
        "create Decoder as torch Module using previously defined building blocks"
        super().__init__()
        self.opts = opts
        conv_dtype = torch.float16 if use_fp16 else torch.float32

        if 't' not in self.opts:
            self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
            self.upscale1 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
            self.upscale2 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
            self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
            self.res1 = ResidualBlock(d_ch*4, kernel_size=3)
            self.res2 = ResidualBlock(d_ch*2, kernel_size=3)

            self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
            self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
            self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)

            self.out_conv  = nn.Conv2d( d_ch*2, 3, kernel_size=1, padding='same', dtype=conv_dtype)

            if 'd' in self.opts:
                self.out_conv1 = nn.Conv2d( d_ch*2, 3, kernel_size=3, padding='same', dtype=conv_dtype)
                self.out_conv2 = nn.Conv2d( d_ch*2, 3, kernel_size=3, padding='same', dtype=conv_dtype)
                self.out_conv3 = nn.Conv2d( d_ch*2, 3, kernel_size=3, padding='same', dtype=conv_dtype)
                self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                self.out_convm = nn.Conv2d( d_mask_ch*1, 1, kernel_size=1, padding='same', dtype=conv_dtype)
            else:
                self.out_convm = nn.Conv2d( d_mask_ch*2, 1, kernel_size=1, padding='same', dtype=conv_dtype)
        else:
            self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
            self.upscale1 = Upscale(d_ch*8, d_ch*8, kernel_size=3)
            self.upscale2 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
            self.upscale3 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
            self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
            self.res1 = ResidualBlock(d_ch*8, kernel_size=3)
            self.res2 = ResidualBlock(d_ch*4, kernel_size=3)
            self.res3 = ResidualBlock(d_ch*2, kernel_size=3)

            self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
            self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*8, kernel_size=3)
            self.upscalem2 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
            self.upscalem3 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
            self.out_conv  = nn.Conv2d( d_ch*2, 3, kernel_size=1, padding='same', dtype=conv_dtype)

            if 'd' in self.opts:
                self.out_conv1 = nn.Conv2d( d_ch*2, 3, kernel_size=3, padding='same', dtype=conv_dtype)
                self.out_conv2 = nn.Conv2d( d_ch*2, 3, kernel_size=3, padding='same', dtype=conv_dtype)
                self.out_conv3 = nn.Conv2d( d_ch*2, 3, kernel_size=3, padding='same', dtype=conv_dtype)
                self.upscalem4 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                self.out_convm = nn.Conv2d( d_mask_ch*1, 1, kernel_size=1, padding='same', dtype=conv_dtype)
            else:
                self.out_convm = nn.Conv2d( d_mask_ch*2, 1, kernel_size=1, padding='same', dtype=conv_dtype)

    
        
    def forward(self, z):
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)

        if 't' in self.opts:
            x = self.upscale3(x)
            x = self.res3(x)

        if 'd' in self.opts:
            x = torch.sigmoid( depth_to_space(torch.cat( (self.out_conv(x),
                                                                self.out_conv1(x),
                                                                self.out_conv2(x),
                                                                self.out_conv3(x)), 1), 2) )
        else:
            x = torch.sigmoid(self.out_conv(x))


        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)

        if 't' in self.opts:
            m = self.upscalem3(m)
            if 'd' in self.opts:
                m = self.upscalem4(m)
        else:
            if 'd' in self.opts:
                m = self.upscalem3(m)

        m = torch.sigmoid(self.out_convm(m))

        # if use_fp16:
        #     x = tf.cast(x, tf.float32)
        #     m = tf.cast(m, tf.float32)

        return x, m


lowest_dense_res = resolution // (32 if 'd' in opts else 16)

class Inter(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, opts, use_fp16=False, **kwargs):
        super().__init__(**kwargs)
        self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
        self.opts = opts
        in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch
        
        dtype = torch.float16 if use_fp16 else torch.float32

        self.dense1 = nn.Linear( in_ch+2, ae_ch , dtype=dtype)
        self.dense2 = nn.Linear( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch,dtype=dtype )
        if 't' not in self.opts:
            self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

        self.label_emb = nn.Embedding(2, 2)


    def forward(self, inp,label):
        x = torch.cat((inp,self.label_emb(label)),-1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.reshape((-1,self.ae_out_ch, lowest_dense_res, lowest_dense_res))
        if 't' not in self.opts:
            x = self.upscale1(x)

        return x

    def get_out_res(self):
        return lowest_dense_res * 2 if 't' not in self.opts else lowest_dense_res

    def get_out_ch(self):
        return self.ae_out_ch


class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_ch=input_ch, e_ch=e_dims,opts=opts, use_fp16=use_fp16)
        encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2

        self.inter = Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims,opts=opts)
        inter_out_ch = self.inter.get_out_ch()

        self.decoder = Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims,opts=opts)
    
    def forward(self,img,label):
        encoded = self.encoder(img)
        latent = self.inter(encoded,label)
        decoded = self.decoder(latent)

        return decoded

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding="same")
        self.max = nn.MaxPool2d(kernel_size=2)
        self.lrelu = nn.LeakReLU(negative_slope = 0.1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3,padding="same")
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

        x = x.flatten()
        x = torch.cat([x,emb],-1)
        x = self.lrelu(self.linear1(x))
        x = self.lrelu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
### Loader
class ImageSet_thomas(Dataset):
    """The simplest image loader possible"""
    def __init__(self, main_dir, use_fp16=False):
        """Just take the folder path as input"""
        self.main_dir = main_dir
        self.img_dst = os.listdir(main_dir+"/aligned_dst")
        self.img_src = os.listdir(main_dir+"/aligned_src")
        self.use_fp16 = use_fp16

    def __len__(self):
        return len(self.img_dst) + len(self.img_src)

    def __getitem__(self, idx):
        if idx < len(self.img_dst):
            img_loc = os.path.join(self.main_dir+"/aligned_dst", self.img_dst[idx])
            label = torch.tensor(0)
        else:
            img_loc = os.path.join(self.main_dir+"/aligned_src", self.img_dst[idx-len(self.img_dst)])
            label = torch.tensor(1)
        image = read_image(img_loc)
        return image.float() , label


### training
ds = ImageSet_thomas(main_dir="workspace/aligned_all/")
device = "cuda"
batch_size = 4
trainloader = DataLoader(ds,batch_size=batch_size)
loss = nn.MSELoss()
n_epochs = 200
G = generator()
D = discriminator()

G = G.to(device)
D = D.to(device)
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(trainloader):
        batch_size = imgs.shape[0]

        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        #generator training
        optimizer_G.zero_grad()

        gen_imgs = generator(imgs,labels)
        validity = discriminator(gen_imgs, labels)

        g_loss = loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        #discriminator training
        optimizer_D.zero_grad()

        validity_real = discriminator(real_imgs, labels)
        d_real_loss = loss(validity_real, valid)

        validity_fake = discriminator(gen_imgs.detach(), labels)
        d_fake_loss = loss(validity_fake, fake)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(trainloader), d_loss.item(), g_loss.item())
        )

                

