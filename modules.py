import math

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, function):
        super(Residual, self).__init__()
        self.function = function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x) + x


def Downsample(dim: int) -> nn.Module:
    return nn.Conv2d(dim, dim, 4, 2, 1)


def Upsample(dim: int) -> nn.Module:
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


class PositionalEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
