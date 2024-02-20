import torch.nn as nn
import torch


class Downscale(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=5, *kwargs):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size,
                               stride=2, padding="same")
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        super().__init__(*kwargs)
    
        
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.activation(x)
        return x

    def get_out_ch(self):
        return self.out_ch

class DownscaleBlock(nn.Module):

    def __init__(self, in_ch, ch, n_downscales, kernel_size):
        self.downs = []
        last_ch = in_ch

        for i in range(n_downscales):
            cur_ch = ch*( min(2**i, 8)  )
            self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size))
            last_ch = self.downs[-1].get_out_ch()
    
    def forward(self, inp):
        x = inp
        for down in self.downs:
            x = down(x)
        return x


def depth_to_space(x,size):
    x = torch.permute(x,(0,2,3,1))
    b,h,w,c = x.shape
    oh, ow = h * size, w * size
    oc = c // (size * size)
    x = x.reshape((-1,h,w,size,size,oc,))
    x = torch.permute(x,(0, 1, 3, 2, 4, 5))
    x = x.reshape((-1, oc, oh, ow))
    return x

class Upscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, *kwargs):
        self.conv1 = nn.Conv1d(in_ch, out_ch*4, kernel_size,
                               padding="same")
        self.activation = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.activation(x)
        x = depth_to_space(x,2)
        return x
