import math
from functools import partial
from typing import List, Tuple

import torch
from torch import nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, function):
        super(Residual, self).__init__()
        self.function = function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x) + x


def Downsample(dim: int, dim_out: int = None) -> nn.Module:
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out or dim, 3, padding=1)
    )


def Upsample(dim: int, dim_out: int = None) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, dim_out or dim, 3, padding=1)
    )


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

class WeightStandardizedConv2d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", "var", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * torch.rsqrt(var + eps)

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super(Block, self).__init__()
        self.convolution = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
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

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super(Attention, self).__init__()
        hidden_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b (h, c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        dots = torch.einsum('b h d i,b h d j->b h i j', q, k)
        dots = dots - dots.amax(dim=-1, keepdim=True).detach()
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h d j-> b h i d', attn, v)
        out = rearrange(out, 'b h (x,y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super(LinearAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim: int, function):
        super(PreNorm, self).__init__()
        self.function = function
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.function(x)


class UNet(nn.Module):
    def __init__(self, dim: int, init_dim: int = None, out_dim: int = None, dim_mults: Tuple[int] = (1, 2, 4, 8),
                 channels: int = 3, self_condition: bool = False, resnet_block_groups: int = 4):
        super(UNet, self).__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = init_dim or dim
        self.input_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            PositionalEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        num_resolutions = len(in_out)

        for i, (in_dim, out_dim) in enumerate(in_out):
            is_last = i >= (num_resolutions - 1)

            self.down.append(nn.ModuleList([
                block_class(in_dim, in_dim, time_emb_dim=time_dim),
                block_class(in_dim, in_dim, time_emb_dim=time_dim),
                Residual(PreNorm(in_dim, LinearAttention(in_dim))),
                Downsample(in_dim, out_dim)
                if not is_last
                else nn.Conv2d(in_dim, out_dim, 3, padding=1)
            ]))

        middle_dim = dims[-1]
        self.mid_first_block = block_class(middle_dim, middle_dim, time_emb_dim=time_dim)
        self.mid_attention = Residual(PreNorm(middle_dim, Attention(middle_dim)))
        self.mid_second_block = block_class(middle_dim, middle_dim, time_emb_dim=time_dim)

        for i, (in_dim, out_dim) in enumerate(reversed(in_out)):
            is_last = i == (len(in_out) - 1)

            self.up.append(nn.ModuleList([
                block_class(in_dim + out_dim, out_dim, time_emb_dim=time_dim),
                block_class(in_dim + out_dim, out_dim, time_emb_dim=time_dim),
                Residual(PreNorm(out_dim, LinearAttention(out_dim))),
                Upsample(out_dim, in_dim)
                if not is_last
                else nn.Conv2d(out_dim, in_dim, 3, padding=1)
            ]))

        self.out_dim = out_dim or channels
        self.end_res_block = block_class(dim * 2, dim, time_emb_dim=time_dim)
        self.end_conv = nn.Conv2d(dim, self.out_dim, 1)


    def forward(self, x: torch.Tensor, time: torch.Tensor, x_self_cond: torch.Tensor = None) -> torch.Tensor:
        if self.self_condition:
            x_self_cond = x_self_cond or torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.input_conv(x)
        copy = x.clone()

        time_transformed = self.time_mlp(time)

        residuals = []

        for block1, block2, attention, downsample in self.down:
            x = block1(x, time_transformed)
            residuals.append(x)

            x = block2(x, time_transformed)
            x = attention(x)
            residuals.append(x)

            x = downsample(x)

        x = self.mid_first_block(x, time_transformed)
        x = self.mid_attention(x)
        x = self.mid_second_block(x, time_transformed)

        for block1, block2, attention, upsample in self.up:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = block1(x, time_transformed)

            x = torch.cat((x, residuals.pop()), dim = 1)
            x = block2(x, time_transformed)
            x = attention(x)
            x = upsample(x)

        x = torch.cat((x, copy), dim=1)

        x = self.end_res_block(x, time_transformed)
        return self.end_conv(x)
