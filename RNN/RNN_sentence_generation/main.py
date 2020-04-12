# dataset : https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ShakespeareDataset import ShakespeareDataset
from model import SequenceGenerationNet
from utils import train_net

if __name__ == '__main__':
    batch_size = 32

    text_dataset = ShakespeareDataset('./dataset/tinyshakespeare/input.txt', chunk_size=200)
    data_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)

    net = SequenceGenerationNet(num_embeddings=text_dataset.vocab_size,
                                embedding_dim=20,
                                hidden_size=50,
                                num_layers=2,
                                dropout=0.1)
    net = net.to('cuda:0')

    train_net(net, data_loader, text_dataset,
              n_iter=2, device='cuda:0')

