import torch
from torch import nn, optim

class SequenceTaggingNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, hidden_size=50, num_layers=1, drop_out=0.2):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h0 =None, l=None):
        # (batch_size, step_size) -> (batch_size, step_size, embedding_dim)
        x = self.emb(x)

        # (batch_size, step_size, embedding_dim) -> (batch_size, step_size, hidden_dim)
        x, h = self.lstm(x, h0)

        # (batch_size, step_size, hidden_dim) -> (batch_size, 1)
        if l is not None:
            # if original lenght is given, use it.
            x = x[list(range(len(x))), l-1, :]
        else:
            x = x[:,-1,:]

        # text feature -> linear layer
        x = self.linear(x)

        # (batch_size, 1) -> (batch_size, )
        x = x.squeeze()
        return x