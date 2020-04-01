# dataset download link : https://github.com/lucidfrontier45/PyTorch-Book/raw/master/data/taco_andburrito.tar.gz

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from utils import train_net

from torchvision import models # pre-trained model

if __name__ == '__main__':
    # Dataset
    # RandomCrop(size)
    # CenterCrop(size)
    train_imgs = ImageFolder('./dataset/train/',
                             transform=transforms.Compose([
                                 transforms.RandomCrop(224),
                                 transforms.ToTensor()
                             ]))
    test_imgs = ImageFolder('./dataset/test/',
                            transform=transforms.Compose([
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                            ]))

    # DataLoader
    train_loader = DataLoader(train_imgs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_imgs, batch_size=32, shuffle=False)

    # print classes
    print(train_imgs)
    print("classes : ", train_imgs.classes)
    print("class to idx : ", train_imgs.class_to_idx)

    # pre-trained model
    net = models.resnet18(pretrained=True)

    # turn of auto_grad
    for p in net.parameters():
        p.requires_grad = False

    # change the FC layer
    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim, 2)

    # if we only want the feature, we can put a IdentityLayer.
    # This is just one of many other methods.
    """
    class IdentityLayer(nn.Module):
        def forward(self, x):
            return x
    net.fc = IdentityLayer()
    """

    # start training
    train_net(net, train_loader, test_loader, n_iter=20, device='cuda:0')