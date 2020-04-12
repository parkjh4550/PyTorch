import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from statistics import mean
import tqdm

from SequenceTaggingNet import SequenceTaggingNet
from IMDBDataset import IMDBDataset

def eval_net(net, data_loader, device="cpu"):
    net.eval()
    ys, ypreds = [], []

    for x, y, l in data_loader:
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)

        with torch.no_grad():
            y_pred = net(x, l=l)
            y_pred = (y_pred >0).long().unsqueeze(0)

            ys.append(y)
            ypreds.append(y_pred)

    # list( 2 dim ) -> torch (1 dim array)
    ys = torch.cat(ys)      # ys = torch.cat(ys, dim=0)
    ypreds = torch.cat(ypreds)      # ypreds = torch.cat(ypreds, dim=0)
    acc = (ys == ypreds).float().sum() /len(ys)

    return acc.item()

if __name__ == '__main__':
    # Dataset load
    train_data = IMDBDataset("./aclImdb/")
    test_data = IMDBDataset("./aclImdb/", train=False)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    # network
    vocab_size = train_data.vocab_size
    net = SequenceTaggingNet(vocab_size+1, num_layers=2)
    net.to("cuda:0")

    opt = optim.Adam(net.parameters())
    loss_f = nn.BCEWithLogitsLoss() # Binary Cross Entropy with logits

    for epoch in range(10):
        losses = []
        net.train()
        #for itr, (x, y, l) in enumerate(train_loader):
        for x, y, l in tqdm.tqdm(train_loader):
            x = x.to("cuda:0")
            y = y.to("cuda:0")
            l = l.to("cuda:0")


            y_pred = net(x, l=l)
            y_pred = y_pred.unsqueeze(0)
            #print(x.shape, y.shape, l.shape, y_pred.shape)
            loss = loss_f(y_pred, y.float())

            net.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        train_acc = eval_net(net, train_loader, "cuda:0")
        test_acc = eval_net(net, test_loader, "cuda:0")
        #print(epoch)
        print("=======epoch {}\ntrain_acc : {}\ntest_acc : {}".format(epoch, train_acc, test_acc))