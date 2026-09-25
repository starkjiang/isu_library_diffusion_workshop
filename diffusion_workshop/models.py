"""Reference U-Net: the network students assemble across Labs 1-4.

Shapes in comments use B=batch, C=channels, H/W=height/width.
"""
import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class GELUConvBlock(nn.Module):
    """Conv 3x3 -> GroupNorm -> GELU. Keeps H and W unchanged."""

    def __init__(self, in_ch, out_ch, n_groups):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(n_groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class RearrangePoolBlock(nn.Module):
    """Halve H and W without throwing pixels away.

    Each 2x2 patch is moved into the channel axis (C -> 4C), then a conv learns
    how to mix those four values back down to C channels.
    """

    def __init__(self, in_ch, n_groups):
        super().__init__()
        self.rearrange = Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        self.conv = GELUConvBlock(4 * in_ch, in_ch, n_groups)

    def forward(self, x):
        return self.conv(self.rearrange(x))


class ResidualConvBlock(nn.Module):
    """Two conv blocks with a skip: out = conv1(x) + conv2(conv1(x))."""

    def __init__(self, in_ch, out_ch, n_groups):
        super().__init__()
        self.conv1 = GELUConvBlock(in_ch, out_ch, n_groups)
        self.conv2 = GELUConvBlock(out_ch, out_ch, n_groups)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1 + self.conv2(x1)


class DownBlock(nn.Module):
    """(B, in_ch, H, W) -> (B, out_ch, H/2, W/2)"""

    def __init__(self, in_ch, out_ch, n_groups):
        super().__init__()
        self.model = nn.Sequential(
            GELUConvBlock(in_ch, out_ch, n_groups),
            GELUConvBlock(out_ch, out_ch, n_groups),
            RearrangePoolBlock(out_ch, n_groups),
        )

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    """Concatenate the skip connection, then (B, 2*in_ch, H, W) -> (B, out_ch, 2H, 2W)."""

    def __init__(self, in_ch, out_ch, n_groups):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(2 * in_ch, out_ch, kernel_size=2, stride=2),
            GELUConvBlock(out_ch, out_ch, n_groups),
            GELUConvBlock(out_ch, out_ch, n_groups),
            GELUConvBlock(out_ch, out_ch, n_groups),
        )

    def forward(self, x, skip):
        return self.model(torch.cat((x, skip), dim=1))


class SinusoidalPositionEmbedBlock(nn.Module):
    """Timestep t (B,) -> (B, dim) vector of sines and cosines at many frequencies."""

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "embedding size must be even"
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=time.device) / max(half - 1, 1))
        args = time.float()[:, None] * freqs[None, :]
        return torch.cat((args.sin(), args.cos()), dim=-1)


class EmbedBlock(nn.Module):
    """Small MLP that turns a vector (B, input_dim) into a feature map (B, emb_dim, 1, 1)."""

    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Unflatten(1, (emb_dim, 1, 1)),
        )

    def forward(self, x):
        return self.model(x.view(-1, self.input_dim))


class UNet(nn.Module):
    """Noise-prediction U-Net with timestep and (optional) context conditioning.

    forward(x, t, c=None, c_mask=None)
        x      (B, img_ch, img_size, img_size)  noisy image x_t
        t      (B,)  integer timesteps
        c      (B, c_embed_dim) context vector (one-hot label, CLIP embedding, ...)
        c_mask (B, 1) of 0/1. 0 drops the context for that sample (classifier-free guidance)
    """

    def __init__(self, T, img_ch=1, img_size=16, down_chs=(64, 64, 128),
                 t_embed_dim=8, c_embed_dim=10, n_groups_small=8, n_groups_big=32):
        super().__init__()
        assert img_size % 4 == 0, "img_size must be divisible by 4 (two down-samplings)"
        self.T, self.c_embed_dim = T, c_embed_dim
        c0, c1, c2 = down_chs
        latent = img_size // 4

        # Encoder
        self.down0 = ResidualConvBlock(img_ch, c0, n_groups_small)
        self.down1 = DownBlock(c0, c1, n_groups_big)
        self.down2 = DownBlock(c1, c2, n_groups_big)
        self.to_vec = nn.Sequential(nn.Flatten(), nn.GELU())

        # Bottleneck
        self.dense_emb = nn.Sequential(
            nn.Linear(c2 * latent**2, c1), nn.GELU(),
            nn.Linear(c1, c1), nn.GELU(),
            nn.Linear(c1, c2 * latent**2), nn.GELU(),
        )

        # Conditioning
        self.sinusoidal_time = SinusoidalPositionEmbedBlock(t_embed_dim)
        self.t_emb1 = EmbedBlock(t_embed_dim, c2)
        self.t_emb2 = EmbedBlock(t_embed_dim, c1)
        self.c_emb1 = EmbedBlock(c_embed_dim, c2)
        self.c_emb2 = EmbedBlock(c_embed_dim, c1)

        # Decoder
        self.up0 = nn.Sequential(nn.Unflatten(1, (c2, latent, latent)), GELUConvBlock(c2, c2, n_groups_big))
        self.up1 = UpBlock(c2, c1, n_groups_big)
        self.up2 = UpBlock(c1, c0, n_groups_big)
        self.out = nn.Sequential(
            nn.Conv2d(2 * c0, c0, 3, 1, 1),
            nn.GroupNorm(n_groups_small, c0),
            nn.GELU(),
            nn.Conv2d(c0, img_ch, 3, 1, 1),
        )

    def forward(self, x, t, c=None, c_mask=None):
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        latent = self.dense_emb(self.to_vec(down2))

        t = self.sinusoidal_time(t)
        t_emb1, t_emb2 = self.t_emb1(t), self.t_emb2(t)

        if c is None:                       # unconditional use: behave as if context was dropped
            c = torch.zeros(x.shape[0], self.c_embed_dim, device=x.device)
        elif c_mask is not None:
            c = c * c_mask
        c_emb1, c_emb2 = self.c_emb1(c), self.c_emb2(c)

        up0 = self.up0(latent)
        up1 = self.up1(c_emb1 * up0 + t_emb1, down2)   # context scales, time shifts
        up2 = self.up2(c_emb2 * up1 + t_emb2, down1)
        return self.out(torch.cat((up2, down0), dim=1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
