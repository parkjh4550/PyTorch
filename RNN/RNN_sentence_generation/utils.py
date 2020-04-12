import string
import torch
from torch import nn, optim

from statistics import mean
import tqdm

def build_vocab():
    # build all printable ASCII characters
    all_chars = string.printable
    vocab_size = len(all_chars)
    vocab_dict = dict((c,i) for (i, c) in enumerate(all_chars))

    return all_chars, vocab_size, vocab_dict

def str2ints(s, vocab_dict):
    # string -> int list
    return [vocab_dict[c] for c in s]


def ints2str(x, vocab_array):
    # int list -> string
    return "".join([vocab_array[i] for i in x])

def generate_seq(net, text_dataset, start_phrase='The King said ',
                 length=200, temperature=0.8, device='cpu'):

    net.to(device)
    net.eval()
    result = [] # save output

    # string -> tensor
    start_tensor = torch.tensor(
        str2ints(start_phrase, text_dataset.vocab_dict),
        dtype=torch.int64
    ).to(device)

    # attach a batch size dim
    x0 = start_tensor.unsqueeze(0)

    # model prediction
    o, h = net(x0)
    print('output shape : ', o.shape)

    # output -> probability
    print('o[:,-1] shape: ', o[:,-1].shape)
    print('o[:,-1].view(-1) shape : ', o[:,-1].view(-1).shape)
    out_dist = o[:,-1].view(-1).exp()

    top_i = torch.multinomial(out_dist, 1)[0]

    for i in range(length):
        inp = torch.tensor([[top_i]], dtype=torch.int64)
        inp = inp.to(device)

        o, h = net(inp, h)
        out_dist = o.view(-1).exp()
        top_i = torch.multinomial(out_dist, 1)[0]

        result.append(top_i)

    return start_phrase + ints2str(result, text_dataset.char_arr)


def train_net(net, data_loader, dataset, n_iter=10, optimizer=optim.Adam, loss_f=nn.CrossEntropyLoss(), device='cpu'):
    #net.to(device)
    #net.cuda()
    optim = optimizer(net.parameters())

    for epoch in range(n_iter):
        #net = net.to(device)
        net.train()
        losses = []
        for data in tqdm.tqdm(data_loader):
            x = data[:, :-1]
            y = data[:, 1:]

            x, y = x.to(device), y.to(device)

            y_pred, _ = net(x)

            loss = loss_f(y_pred.view(-1, dataset.vocab_size), y.view(-1))
            net.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())

        print(epoch, mean(losses))
        #print(generate_seq(net, dataset, device))
    with torch.no_grad():
        print(generate_seq(net, dataset, device))