import torch
from torch import nn as nn
import numpy

class w2vtorch(nn.module):
    def __init__(self, num_embedding, embedding_dim) -> None:
        super().__init__()
        self.embed_v = nn.Embedding(num_embedding, embedding_dim)
        self.embed_u = nn.Embedding(num_embedding, embedding_dim)

    def skip_gram(self, center, contexts_and_negatives, embed_v, embed_u):
        v = self.embed_v(center) # shape of (n, 1, d)
        u = self.embed_u(contexts_and_negatives) # shape of (n, m, d)
        pred = torch.bmm(v, u.permute(0, 2, 1)) # bmm((n, 1, d), (n, d, m)) => shape of (n, 1, m)
        return pred

    