# dataset link : http://www.manythings.org/anki/
# spa-eng dataset
from torch.utils.data import DataLoader
from torch import nn, optim

from model import Encoder, Decoder
from TranslationPairDataset import TranslationPairDataset
from utils import train_model, load_model
import os
if __name__ == '__main__':
    #parameters
    batch_size = 64
    max_len = 10
    data_path = './dataset/spa-eng/spa.txt'

    #load dataset
    ds = TranslationPairDataset(data_path, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    #model define
    enc = Encoder(len(ds.src_word_list), 100, 100, 2)
    dec = Decoder(len(ds.trg_word_list), 100, 100, 2)
    optimizer = optim.Adam
    loss_f = nn.CrossEntropyLoss()

    # checkpoint load
    check_dir = './checkpoint'
    if os.listdir(check_dir):
        enc, dec = load_model(enc, dec, check_dir)

    #training
    train_model(enc, dec, ds, loader, optimizer=optimizer, loss_f=loss_f, n_epoch=30, device='cuda:0')