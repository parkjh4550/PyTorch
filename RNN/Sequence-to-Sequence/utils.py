import re
import collections
import itertools
import torch
from torch import nn, optim

import tqdm
from statistics import mean
# normalize
# parse_line
# build_vocab
# words2tensor

remove_marks_regex = re.compile("[\,\(\)\[\]\*:;¡¿]|<.*?>")
shift_marks_regex = re.compile("([?!\.])")

UNK = 0
SOS = 1
EOS = 2

def normalize(text):
    # 1. change all characters to lowe cases
    text = text.lower()

    # 2. remove all special tokens
    text = remove_marks_regex.sub("", text)

    # 3. insert spaces
    text = shift_marks_regex.sub("", text)

    return text

def parse_line(line):
    # sentences -> word tokens

    # 1. normalize a sentence
    line = normalize(line)

    # 2. sentence -> token
    # split with tabs
    src, trg = line.split("\t")[0],  line.split("\t")[1]

    # remove spaces of the both ends
    # split with inside spaces
    src_tokens = src.strip().split()
    trg_tokens = trg.strip().split()

    return src_tokens, trg_tokens

def build_vocab(tokens):
    # text -> words -> indexes

    print("=======build vocab======")
    # counts all words in the sentence
    counts = collections.Counter(tokens)
    print('counted results type : ', type(counts.items()))
    print('counted results  : ',counts.items())
    # sort all words according to the counts
    sorted_counts = sorted(counts.items(), key=lambda c: c[1], reverse=True)
    print('sorted results 0~5 words : ', sorted_counts[:5])

    #build dictionary
    # word : index
    word_list = ['<UNK>', "<SOS>", "<EOS>"] + [ x[0] for x in sorted_counts]
    word_dict = dict((w,i) for i, w in enumerate(word_list))

    print("=============end")
    return word_list, word_dict

def words2tensor(words, word_dict, max_len, padding=0):
    # word -> pytorch tensor
    # attach EOS token
    words = words + ['<EOS>']

    # token -> indexes
    words = [word_dict.get(w, 0) for w in words]
    seq_len = len(words)

    # if shorter than max_len, do padding
    if seq_len<max_len+1:
        words = words +[padding] * (max_len + 1 -seq_len)

    # return tensor
    return torch.tensor(words, dtype=torch.int64), torch.tensor(seq_len, dtype=torch.int64)


def translate(input_str, enc, dec, src_dict, trg_word_list, max_len=15, device='cpu'):
    # words-> tokens
    words = normalize(input_str).split()

    # tokens -> tnesor, sequence_length
    input_tensor, seq_len = words2tensor(words, src_dict, max_len=max_len)
    input_tensor = input_tensor.unsqueeze(0)
    seq_len = [seq_len]

    # prepare special tokens
    sos_inputs = torch.tensor(SOS, dtype=torch.int64)

    input_tensor = input_tensor.to(device)
    sos_inputs = sos_inputs.to(device)

    # encoder output
    ctx_vec = enc(input_tensor, l=seq_len)

    # set the decoder's initial state
    z = sos_inputs
    h = ctx_vec
    results = []
    for i in range(max_len):
        o, h = dec(z.view(1,1), h)

        # max : [ value, indices ]
        # we only need index
        wi = o.detach().view(-1).max(0)[1]

        if wi.item() == EOS:
            break
        results.append(wi.item())
        # the predicted output will be given to next step
        z = wi

    return " ".join(trg_word_list[i] for i in results)

def to2D(x):
    shapes = x.shape
    return x.reshape(shapes[0] * shapes[1], -1)

def train_model(enc, dec, dataset, dataloader, optimizer=optim.Adam, loss_f =nn.CrossEntropyLoss(), n_epoch=10,  device='cpu'):
    print('device type : ', device)
    enc.to(device), dec.to(device)
    opt_enc = optimizer(enc.parameters(), 0.002)
    opt_dec = optimizer(dec.parameters(), 0.002)

    for epoch in range(n_epoch):
        enc.train(), dec.train()
        losses = []
        for x, lx, y, ly in tqdm.tqdm(dataloader):
            # sort inputs for pack_padded_input
            lx, sort_idx = lx.sort(descending=True)
            x, y, ly = x[sort_idx], y[sort_idx], ly[sort_idx]

            x, y= x.to(device), y.to(device)
            lx, ly = lx.to(device), ly.to(device)

            # encoder output
            ctx_vec = enc(x, l=lx)

            ly, sort_idx = ly.sort(descending=True)
            y = y[sort_idx]

            h0 = (ctx_vec[0][:, sort_idx, :], ctx_vec[1][:, sort_idx, :] )
            z = y[:, :-1].detach()

            z[z==-100] = 0

            # decoder output
            o, _ = dec(z, h0, l=ly-1)
            loss = loss_f(to2D(o[:]), to2D(y[:,1:max(ly)]).squeeze())

            # backpropagation
            enc.zero_grad(), dec.zero_grad()
            loss.backward()
            opt_enc.step(), opt_dec.step()

            losses.append(loss.item())

        # show losses and translation result
        print('========epoch {} : {}'.format(epoch, mean(losses)))

        enc.eval(), dec.eval()
        print('trainslation result ======\n', translate('I am a student.', enc, dec, src_dict=dataset.src_word_dict, trg_word_list=dataset.trg_word_list, max_len=dataset.max_len, device=device))

        # save model
        if (epoch+1)%10==0:
            # if we save models which are on the gpu,
            # it automatically load them on gpu when we use torch.load
            # if the computer doesn't have a gpu, it gets an error.
            # So we move the model's parameters to CPU and save them
            enc.to('cpu'), dec.to('cpu')
            enc_param, dec_param = enc.state_dict(), dec.state_dict()


            torch.save(enc_param, './checkpoint/enc_epoch_{}.prm'.format(epoch+1), pickle_protocol=4)
            torch.save(dec_param, './checkpoint/dec_epoch_{}.prm'.format(epoch+1), pickle_protocol=4)

            enc.to(device), dec.to(device)