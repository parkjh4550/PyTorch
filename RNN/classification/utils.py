import re
import torch

remove_marks_regex = re.compile("[,\.\(\)\[\]\*:;]|<.*?>")
shift_marks_regex = re.compile("([?!])")

def text2ids(text, vocab_dict):
    # remove special tokens
    text = remove_marks_regex.sub("", text)
    # put " " instead of ! or ?
    text =shift_marks_regex.sub(r' \1', text)
    #print('1-1 : ', text)
    #tokenize
    tokens = text.split()
    return [vocab_dict.get(token, 0) for token in tokens]

def list2tensor(token_idxes, max_len=100, padding=True):
    if len(token_idxes) > max_len:
        token_idxes = token_idxes[:max_len]
        n_tokens = len(token_idxes)

    else:
        n_tokens = len(token_idxes)
        if padding:
            token_idxes = token_idxes + [0] *(max_len - n_tokens)
    return torch.tensor(token_idxes, dtype=torch.int64), n_tokens