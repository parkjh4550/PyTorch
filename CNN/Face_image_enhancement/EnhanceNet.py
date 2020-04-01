import torch
from torch import nn, optim

class EnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Conv layer
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            # DeConv Layer
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,4,stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        )

    def forward(self, img):
        out_img = self.net(img)
        return out_img
