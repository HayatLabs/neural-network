import torch
import torch.nn as nn


class QuerySenseBrain(nn.Module):
    """Small recurrent encoder-decoder used by the V2 training script."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, trg):
        _, hidden = self.encoder(self.embedding(src))
        decoded, _ = self.decoder(self.embedding(trg), hidden)
        return self.output(decoded)