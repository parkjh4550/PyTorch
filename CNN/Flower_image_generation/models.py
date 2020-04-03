import torch
from torch import nn, optim

class GNet(nn.Module):
    def __init__(self, nz=100, ngf=32):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.gnet = nn.Sequential(

            #Transposed Convolution image size
            # => out_size = {(in_size - 1) * stride } - 2*padding + kernal_size + output_padding

            # input : nz => in_size = 1
            # kernel => kernel_size=4 , stride = 1, padding=0
            nn.ConvTranspose2d(self.nz, self.ngf*8, 4, 1,0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.gnet(x)
        return out

class DNet(nn.Module):
    def __init__(self, ndf=32):
        super().__init__()
        self.ndf = ndf
        self.dnet = nn.Sequential(
            nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*8, 1, 4, 2, 0, bias=False)
        )

    def forward(self, x):
        # out : (batch, 1, 1, 1)
        out = self.dnet(x)
        # (batch, 1, 1, 1) -> (batch,
        return out.squeeze()

