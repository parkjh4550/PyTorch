import torch
from torch import nn, optim

class SequenceGenerationNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, hidden_size=50, num_layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        self.linear = nn.Linear(hidden_size, num_embeddings)

    def forward(self, x, h0=None):
        x = self.emb(x)
        x, h = self.lstm(x, h0)
        x = self.linear(x)
        return x, h