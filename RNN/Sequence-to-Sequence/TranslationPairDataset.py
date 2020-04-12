import torch
from torch.utils.data import Dataset
import itertools
from utils import normalize, parse_line, build_vocab, words2tensor


class TranslationPairDataset(Dataset):
    def __init__(self, path, max_len=15):
        self.max_len = max_len
        # filter long sentences
        def filter_pair(p):
            return not(len(p[0])>self.max_len or len(p[1])>self.max_len)

        # open a file, parse and filter it
        with open(path, encoding='UTF8') as fp:
            pairs = map(parse_line, fp)
            pairs = filter(filter_pair, pairs)  # only true items will be returned
            pairs = list(pairs)

        # split into src and trg
        src = [p[0] for p in pairs]
        trg = [p[1] for p in pairs]

        """
        # nested list -> list
        i_src = itertools.chain.from_iterable(src)
        i_trg = itertools.chain.from_iterable(trg)

        # build vocab
        self.src_word_list, self.src_word_dict = build_vocab(i_src)     
        self.trg_word_list, self.trg_word_dict = build_vocab(i_trg)
        """
        self.src_word_list, self.src_word_dict = build_vocab(itertools.chain.from_iterable(src))
        self.trg_word_list, self.trg_word_dict = build_vocab(itertools.chain.from_iterable(trg))

        #word -> tensor
        self.src_data = [words2tensor(words, self.src_word_dict, max_len) for words in src]
        self.trg_data = [words2tensor(words, self.trg_word_dict, max_len) for words in trg]
        if len(self.src_data) == len(self.trg_data):
            print('Construct src, trg data : ',len(self.src_data))

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src, lsrc = self.src_data[idx]
        trg, ltrg = self.trg_data[idx]
        return src, lsrc, trg, ltrg
