import numpy as np
import torch
import torch.nn as nn
from DeepFakeArchi_torch import *
from torch.utils.data import DataLoader
from cgan import  generator, discriminator
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.io import read_image
import os
from torchvision.transforms import GaussianBlur
from utils import DSSIM_Loss
from utils import ImageSet_thomas
batch_size = 64
resolution = 512
blur_size = resolution//20
k_size = blur_size + 1 - blur_size%2
blur = GaussianBlur(kernel_size=(k_size,k_size),sigma=blur_size)




#ds = ImageSet_thomas(main_dir="workspace")
ds = ImageSet_thomas(main_dir="img_512")
device = "cuda"
trainloader = DataLoader(ds,batch_size=batch_size)
loss_func = nn.MSELoss()
n_epochs = 3000
G = generator()
D = discriminator()
lr = 1e-5
betas = (0.5,0.999)
G = G.to(device)
D = D.to(device)
optimizer_G = torch.optim.Adam(G.parameters(), lr = lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
scheduler = ReduceLROnPlateau(optimizer_G, patience=20, eps=1e-8, factor=.5)
criterion = DSSIM_Loss(data_range=1, size_average=True, channel=3)
for epoch in range(n_epochs):
    li_gen = []
    li_disc = []
    print(epoch)
    for i, (imgs, labels,mask) in enumerate(trainloader):
        batch_size = imgs.shape[0]
        valid = torch.ones((batch_size,1), requires_grad=False, device=device)
        fake = torch.zeros((batch_size,1), requires_grad=False, device=device)

        real_imgs = imgs.float().to(device)
        labels = labels.long().to(device)
        mask = blur(mask).to(device)
        #generator training
        optimizer_G.zero_grad()
        imgs = imgs.clone().float().to(device)
        gen_imgs = G(imgs,labels)
        gen_imgs = imgs * (1-mask ) + gen_imgs * mask
        validity = D(gen_imgs, labels)
        g_loss = 0.05*loss_func(validity, valid) + criterion(imgs,gen_imgs)

        #g_loss = criterion(imgs,gen_imgs)
        g_loss.backward()
        li_gen.append(g_loss)
        optimizer_G.step()

        #discriminator training
        optimizer_D.zero_grad()

        validity_real = D(real_imgs, labels)
        d_real_loss = loss_func(validity_real, valid)

        validity_fake = D(gen_imgs.detach(), labels)
        d_fake_loss = loss_func(validity_fake, fake)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        li_disc.append(d_loss)
    #scheduler.step(torch.mean(torch.stack(li_gen)))
    if epoch % 50 == 0:
        torch.save(G.state_dict(), f"cgan_{epoch}.pth")
    print(torch.mean(torch.stack(li_gen)))
    print(torch.mean(torch.stack(li_disc)))
    print()

             
