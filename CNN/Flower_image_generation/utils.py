import torch
from torch import nn, optim
from torchvision.utils import save_image

from statistics import mean
import tqdm

def train_dcgan(g, d, data_loader, batch_size= 64, lr=0.0002, betas=(0.5, 0.999), optimizer=optim.Adam, device='cpu'):

    optim_g, optim_d = optimizer(g.parameters(), lr, betas), optimizer(d.parameters(), lr, betas)

    # target values
    ones = torch.ones(batch_size).to(device)
    zeros = torch.zeros(batch_size).to(device)

    loss_f = nn.BCEWithLogitsLoss()
    log_loss_g, log_loss_d = [], []

    for real_img, _ in tqdm.tqdm(data_loader):
        batch_len = len(real_img)
        real_img = real_img.to(device)

        ##### TRAIN Generator
        # generate a fake image from the random variable.
        z = torch.randn(batch_len, g.nz, 1, 1).to(device)
        fake_img = g(z)

        # save only fake image
        fake_img_tensor = fake_img.detach()

        out = d(fake_img)

        loss_g = loss_f(out, ones[:batch_len])
        log_loss_g.append(loss_g.item())

        # calc gradient
        g.zero_grad(), d.zero_grad()
        loss_g.backward()
        optim_g.step()

        ##### TRAIN Discriminator
        # real image loss
        real_out = d(real_img)
        loss_d_real = loss_f(real_out, ones[:batch_len])

        # fake image loss
        fake_out = d(fake_img_tensor)
        loss_d_fake = loss_f(fake_out, zeros[:batch_len])

        # calc total loss
        loss_d = loss_d_fake + loss_d_real
        log_loss_d.append(loss_d.item())

        # calc gradients and update
        d.zero_grad(), g.zero_grad()
        loss_d.backward()
        optim_d.step()

    return mean(log_loss_g), mean(log_loss_d)

def train_net(g, d, data_loader, n_iter=10, batch_size=64, device='cpu'):
    for epoch in range(n_iter):
        train_dcgan(g, d, data_loader, batch_size, device=device)

        # save every 10 epochs
        if epoch % 10 == 0:
            # save parameters
            torch.save(
                g.state_dict(),
                './model/g_{:03d}.prm'.format(epoch),
                pickle_protocol=4
            )

            torch.save(
                d.state_dict(),
                './model/d_{:03d}.prm'.format(epoch),
                pickle_protocol=4
            )
            # check the trained model
            fixed_z = torch.randn(batch_size, g.nz, 1, 1).to(device)
            generated_img = g(fixed_z)
            save_image(generated_img, './model/{:03d}.jpg'.format(epoch))

