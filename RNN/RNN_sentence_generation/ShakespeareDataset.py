import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import tqdm

from utils import build_vocab, str2ints

class ShakespeareDataset(Dataset):
    def __init__(self, path, chunk_size=200):
        # build vocab
        self.char_arr, self.vocab_size, self.vocab_dict = build_vocab()

        # read file and (str -> ints)
        data = str2ints(open(path).read().strip(), self.vocab_dict)\

        # ints -> tensor
        data = torch.tensor(data, dtype=torch.int64).split(chunk_size)

        # check the last item
        if len(data[-1]) < chunk_size:
            data = data[:-1]

        self.data = data
        self.n_chunks = len(self.data)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        return self.data[idx]