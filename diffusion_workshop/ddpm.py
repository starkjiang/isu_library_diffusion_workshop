"""Reference DDPM: noise schedule, forward process q, loss, reverse step, samplers.

Notation follows Ho et al. 2020 ("Denoising Diffusion Probabilistic Models"):
    beta_t        variance of the noise added at step t
    alpha_t       1 - beta_t
    alpha_bar_t   alpha_1 * alpha_2 * ... * alpha_t
"""
import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, T=300, beta_start=1e-4, beta_end=0.03, device="cpu"):
        self.T, self.device = T, device
        self.B = torch.linspace(beta_start, beta_end, T, device=device)
        self.a = 1.0 - self.B
        self.a_bar = torch.cumprod(self.a, dim=0)
        self.sqrt_a_bar = self.a_bar.sqrt()                     # weight on the clean image
        self.sqrt_one_minus_a_bar = (1 - self.a_bar).sqrt()     # weight on the noise
        self.sqrt_a_inv = (1 / self.a).sqrt()
        self.pred_noise_coeff = (1 - self.a) / (1 - self.a_bar).sqrt()

    # ---- forward process ------------------------------------------------- #
    def q(self, x_0, t, noise=None):
        """Jump straight from x_0 to x_t. t is a (B,) tensor of integer timesteps."""
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.sqrt_a_bar[t, None, None, None] * x_0 + self.sqrt_one_minus_a_bar[t, None, None, None] * noise
        return x_t, noise

    def get_loss(self, model, x_0, t, *model_args):
        """MSE between the true noise and the noise the model predicts from x_t."""
        x_t, noise = self.q(x_0, t)
        return F.mse_loss(model(x_t, t, *model_args), noise)

    # ---- reverse process ------------------------------------------------- #
    @torch.no_grad()
    def reverse_q(self, x_t, t, e_t):
        """One denoising step x_t -> x_{t-1}, given the predicted noise e_t. t is an int."""
        u_t = self.sqrt_a_inv[t] * (x_t - self.pred_noise_coeff[t] * e_t)
        if t == 0:
            return u_t                                          # last step: no fresh noise
        return u_t + self.B[t].sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model, shape, keep_every=None):
        """Unconditional sampling. Returns x_0 (and a list of frames if keep_every is set)."""
        model.eval()
        x_t = torch.randn(shape, device=self.device)
        frames = []
        for t in range(self.T - 1, -1, -1):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            x_t = self.reverse_q(x_t, t, model(x_t, t_batch))
            if keep_every and (t % keep_every == 0):
                frames.append(x_t.cpu())
        model.train()
        return (x_t, frames) if keep_every else x_t

    @torch.no_grad()
    def sample_w(self, model, c, img_shape, w=1.0, keep_every=None):
        """Classifier-free guided sampling.

        c          (N, c_dim) one context vector per image to generate
        img_shape  (C, H, W)
        w          guidance weight: e = (1 + w) * e_conditional - w * e_unconditional
        """
        model.eval()
        n = c.shape[0]
        x_t = torch.randn(n, *img_shape, device=self.device)
        c2 = c.repeat(2, 1)                                     # first half keeps context,
        mask = torch.ones(2 * n, 1, device=self.device)         # second half has it dropped
        mask[n:] = 0.0
        frames = []
        for t in range(self.T - 1, -1, -1):
            t_batch = torch.full((2 * n,), t, device=self.device, dtype=torch.long)
            e = model(x_t.repeat(2, 1, 1, 1), t_batch, c2, mask)
            e_keep, e_drop = e[:n], e[n:]
            e_t = (1 + w) * e_keep - w * e_drop
            x_t = self.reverse_q(x_t, t, e_t)
            if keep_every and (t % keep_every == 0):
                frames.append(x_t.cpu())
        model.train()
        return (x_t, frames) if keep_every else x_t


def get_context_mask(c, drop_prob):
    """Bernoulli mask of shape (B, 1): 1 = keep this sample's context, 0 = drop it.

    Each sample is dropped independently with probability drop_prob. The U-Net
    multiplies the context by this mask, so dropped samples see an all-zero context.
    """
    keep_prob = torch.full((c.shape[0], 1), 1.0 - drop_prob, device=c.device)
    return torch.bernoulli(keep_prob)
