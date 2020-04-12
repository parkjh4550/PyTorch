import torch
from torch import nn, optim

class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dims=50, hidden_size =50,
                 num_layers=1, dropout=0.2):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dims)
        self.lstm = nn.LSTM(embedding_dims, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, h0=None, l=None):
        x = self.emb(x)
        if l is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True)
        _, h = self.lstm(x, h0)
        return h


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dims=50, hidden_size=50,
                 num_layers=1, dropout=0.2):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dims)
        self.lstm = nn.LSTM(embedding_dims, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.linear = nn.Linear(hidden_size, num_embeddings)

    def forward(self, x, h, l=None):
        x = self.emb(x)
        if l is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True)

        x, h = self.lstm(x, h)
        if l is not None:
            x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0)[0]
        x = self.linear(x)

        return x, h