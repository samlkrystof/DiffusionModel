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

class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super(Block, self).__init__()
        self.convolution = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: tuple = None) -> torch.Tensor:
        x = self.convolution(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim=None, groups: int = 8):
        super(ResnetBlock, self).__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = torch.unsqueeze(torch.unsqueeze(time_emb, 2), 3) + h

        h = self.block2(h)
        return h + self.res_conv(x)

class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super(ConvNextBlock, self).__init__()
        self.mlp(
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.first_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if self.mlp is not None and time_emb is not None:
            condition = self.mlp(time_emb)
            h = h + torch.unsqueeze(torch.unsqueeze(condition, 2), 3)

        h = self.net(h)
        return h + self.res_conv(x)


