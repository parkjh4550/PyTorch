import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision.utils import save_image

import numpy as np
import math
import tqdm
import os

def psnr(mse, max_v =1.0):
    # evaluation method for a signal restore
    return 10 * math.log10(max_v**2 / mse)

def eval_net(net, data_loader, device='cpu'):
    net.eval()
    ys = []
    ypreds = []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            y_pred = net(x)

        ys.append(y)
        ypreds.append(y_pred)

    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)

    score = nn.functional.mse_loss(ypreds, ys).item()
    return score

def train_net(net, train_loader, test_loader, optimizer=optim.Adam, loss_fn=nn.MSELoss(),n_iter=10, device='cpu'):
    train_losses = []
    val_acc = []
    optim = optimizer(net.parameters())

    net.to(device)
    for epoch in range(n_iter):
        net.train()
        n = 0       #num of used samples
        score = 0   #evaluation score
        running_loss = 0.0  #sum of losses in this epoch
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            #xx = torch.from_numpy(np.asarray(xx))
            #yy = torch.from_numpy(np.asarray(yy))

            xx, yy = xx.to(device), yy.to(device)
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            n += len(xx)

        train_losses.append(running_loss/len(train_loader))

        val_acc.append(eval_net(net, test_loader, device))

        # print the result
        print("epoch : {} \n\t train_losses : {} \n\t val_acc: {}".format(
            epoch, psnr(train_losses[-1]), psnr(val_acc[-1])), flush=True)


def save_result_img(net, test_dataset, dst):

    # Randomly select 4 images from dataloader
    random_test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    print('type of random_test_loader : ', type(random_test_loader))
    it = iter(random_test_loader)
    print('type of test_loader iterator: ', type(it))
    x, y = next(it)

    # Bilinear method
    bl_recon = torch.nn.functional.upsample(x, 128, mode='bilinear', align_corners=True)
    # CNN
    yp = net(x.to('cuda:0'))

    #save iamges
    # [target, bilinear, prediction]
    # nrow = # of samples
    save_image(torch.cat([y, bl_recon, yp], 0), dst , nrow=4)

def path_check(p):
    if not os.path.isdir(p):
        os.mkdir(p)
        print('directory is made : ', p)
    print('directory already exist')