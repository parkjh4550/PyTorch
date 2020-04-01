# dataset link : http://vis-www.cs.umass.edu/lfw/ -> click "All images aligned with deep funneling" link

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from PairImageFolder import DownsizedPairImageFolder
from EnhanceNet import EnhanceNet

from utils import train_net, save_result_img, path_check
import os

if __name__ == '__main__':

    #dataset
    train_data = DownsizedPairImageFolder('./dataset/train', transform=transforms.ToTensor())
    test_data = DownsizedPairImageFolder('./dataset/test', transform=transforms.ToTensor())

    #dataloader
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    # network
    net = EnhanceNet()

    # training
    train_net(net, train_loader, test_loader, device='cuda:0')

    # save result
    dst = './result'
    f_name = 'cnn_upscale.jpg'

    path_check(dst)
    save_result_img(net, test_data, os.path.join(dst,f_name))