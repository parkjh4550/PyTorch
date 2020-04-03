# dataset link : http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import GNet, DNet
from utils import train_net

if __name__ == '__main__':

    img_data = ImageFolder('./dataset', transform=transforms.Compose([transforms.Resize(80),
                                                                      transforms.CenterCrop(64),
                                                                      transforms.ToTensor()]))

    batch_size = 64
    img_loader = DataLoader(img_data, batch_size=batch_size, shuffle=True)

    g = GNet().to('cuda:0')
    d = DNet().to('cuda:0')

    train_net(g, d, img_loader, batch_size=batch_size, device='cuda:0')
