import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tqdm

from torchvision.datasets import FashionMNIST
from torchvision import transforms

from model import ConvNet
from utils import train_net, eval_net

if __name__ == '__main__':
    # load train, test dataset
    fashion_mnist_train = FashionMNIST("./dataset",
                                       train=True, download=True,
                                       transform=transforms.ToTensor())
    fashion_mnist_test = FashionMNIST("./dataset",
                                      train=False, download=True,
                                      transform=transforms.ToTensor())
    # set dataloader
    batch_size = 128

    train_loader = DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=False)

    net = ConvNet()
    train_net(net, train_loader, test_loader, n_iter=20, device="cuda:0")