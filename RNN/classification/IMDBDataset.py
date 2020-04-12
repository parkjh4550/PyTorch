import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pathlib
import glob
from utils import text2ids, list2tensor

class IMDBDataset(Dataset):
    def __init__(self, dir_path, train=True, max_len=100, padding=True):
        self.max_len = max_len
        self.padding = padding

        path = pathlib.Path(dir_path)
        vocab_path = path.joinpath("imdb.vocab")

        # load vacabulary
        self.vocab_array = vocab_path.open().read().strip().splitlines()

        # build vacab dict
        self.vocab_dict = dict((w, i+1) for (i, w) in enumerate(self.vocab_array))

        # data path
        if train:
            target_path = path.joinpath("train")
        else:
            target_path = path.joinpath("test")

        # data load
        pos_files = sorted(glob.glob(str(target_path.joinpath("pos/*.txt"))))
        neg_files = sorted(glob.glob(str(target_path.joinpath("neg/*.txt"))))

        # labeling
        # pos : 1
        # neg : 0
        self.labeled_files = list(zip([0]*len(neg_files), neg_files)) + list(zip([1]*len(pos_files), pos_files))

    @property
    def vocab_size(self):
        return len(self.vocab_array)

    def __len__(self):
        return len(self.labeled_files)

    def __getitem__(self, idx):
        #print(idx)
        label, f = self.labeled_files[idx]

        # change to small letters
        data = open(f, encoding='UTF8').read().lower()
        #print('1 :', data)
        # text -> id list
        data = text2ids(data, self.vocab_dict)
        #print('2 :', data)
        # id list -> tensor
        data, n_tokens = list2tensor(data, self.max_len, self.padding)
        #print('3 :', data)
        #print(data.shape, label, n_tokens)
        #print(label)
        #print(n_tokens)
        return data, label, n_tokens