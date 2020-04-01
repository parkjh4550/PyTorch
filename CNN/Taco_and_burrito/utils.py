import torch
from torch import nn, optim
import tqdm


def eval_net(net, data_loader, device='cpu'):
    net.eval()

    ys, ypreds = [], []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            _, y_pred = net(x).max(1)

        ys.append(y)
        ypreds.append(y_pred)

    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)

    # calc accuracy
    acc = (ys==ypreds).float().sum() / len(ys)
    return acc.item()

def train_net(net, train_loader, test_loader, only_fc=True,
              optim=optim.Adam,
              loss_fn=nn.CrossEntropyLoss(),
              n_iter=10, device='cpu'):
    net.to(device)

    train_losses = []
    train_acc, val_acc = [], []

    if only_fc:
        # only optimize the fc layer
        optimizer = optim(net.fc.parameters())
    else:
        optimizer = optim(net.parameters())

    for epoch in range(n_iter):
        net.train()
        n = 0
        n_acc = 0
        running_loss = 0.0
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader),
                                     total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)

            out = net(xx)

            loss = loss_fn(out, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n += len(xx)
            _, y_pred = out.max(1)

            n_acc += (y_pred == yy).float().sum().item()
        train_losses.append(running_loss/i)
        train_acc.append(n_acc / n)

        # test the model
        val_acc.append(eval_net(net, test_loader, device))

        # print the result
        print("epoch : {} \n\t train_losses : {} \n\t train_acc : {} \n\t val_acc: {}".format(
            epoch, train_losses[-1], train_acc[-1], val_acc[-1]), flush=True)
