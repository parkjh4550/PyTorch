import torch
from torch import nn, optim
import tqdm


# eval step
def eval_net(net, data_loader, device="cpu"):
    net.to(device)
    net.eval()
    ys, ypreds = [], []

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        # turn off the auto grad calculation
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)

    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)

    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

# training step
def train_net(net, train_loader, test_loader, optim_cls=optim.Adam,
              loss_fn=nn.CrossEntropyLoss(), n_iter=10, device="cpu"):
    net.to(device)
    train_losses= []
    train_acc, val_acc = [], []

    optim = optim_cls(net.parameters())

    for epoch in range(n_iter):
        running_loss = 0.0

        net.train() # set training mode
        n, n_acc = 0, 0

        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)

            out = net(xx)

            loss = loss_fn(out, yy)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            n += len(xx)

            _, y_pred = out.max(1)
            n_acc += (yy==y_pred).float().sum().item()

        train_losses.append(running_loss/i)
        # train data accuracy
        train_acc.append(n_acc/n)
        # eval accuracy
        val_acc.append(eval_net(net, test_loader, device))

        # print the result
        print("epoch : {} \n\t train_losses : {} \n\t train_acc : {} \n\t val_acc: {}".format(
            epoch,train_losses[-1], train_acc[-1], val_acc[-1]), flush=True)
