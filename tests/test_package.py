"""Fast unit tests for the reference implementation:  pytest -q"""
import os

os.environ["WORKSHOP_SMOKE"] = "1"

import torch

from diffusion_workshop.ddpm import DDPM, get_context_mask
from diffusion_workshop.models import RearrangePoolBlock, SinusoidalPositionEmbedBlock, UNet


def test_forward_closed_form_matches_step_by_step_statistics():
    d = DDPM(T=200)
    x0 = torch.ones(4000, 1, 2, 2)
    x = x0.clone()
    for t in range(200):
        x = d.a[t].sqrt() * x + d.B[t].sqrt() * torch.randn_like(x)
    jump, _ = d.q(x0, torch.full((4000,), 199))
    assert abs(x.mean() - jump.mean()) < 0.02 and abs(x.std() - jump.std()) < 0.02


def test_reverse_step_inverts_forward_step_given_true_noise():
    d = DDPM(T=50)
    x0 = torch.randn(3, 1, 8, 8)
    x1, eps = d.q(x0, torch.zeros(3, dtype=torch.long))
    assert torch.allclose(d.reverse_q(x1, 0, eps), x0, atol=1e-5)


def test_unet_shapes_and_conditioning():
    m = UNet(T=50, img_ch=3, img_size=32, down_chs=(32, 32, 64), c_embed_dim=12).eval()
    x, t, c = torch.randn(2, 3, 32, 32), torch.tensor([0, 49]), torch.randn(2, 12)
    keep, drop = torch.ones(2, 1), torch.zeros(2, 1)
    assert m(x, t, c, keep).shape == x.shape
    assert not torch.allclose(m(x, t, c, keep), m(x, t, c, drop))      # context matters
    assert torch.allclose(m(x, t, c, drop), m(x, t), atol=1e-6)        # dropped == unconditional


def test_rearrange_pool_and_time_embedding():
    assert RearrangePoolBlock(8, 4)(torch.randn(2, 8, 16, 16)).shape == (2, 8, 8, 8)
    e = SinusoidalPositionEmbedBlock(16)(torch.arange(10))
    assert e.shape == (10, 16) and len({tuple(r.tolist()) for r in e.round(decimals=4)}) == 10


def test_context_mask_rate_and_guided_sampler():
    mask = get_context_mask(torch.zeros(20000, 10), 0.25)
    assert mask.shape == (20000, 1) and 0.73 < mask.mean() < 0.77
    d = DDPM(T=5)
    m = UNet(T=5, img_size=8, down_chs=(32, 32, 32))
    out = d.sample_w(m, torch.eye(10)[:3], (1, 8, 8), w=1.0)
    assert out.shape == (3, 1, 8, 8) and torch.isfinite(out).all()
