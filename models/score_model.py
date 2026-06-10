"""Noise Conditional Score Model (NCSM) + reconstruction-through-diffusion.

Train s_theta(x, sigma) ~ grad_x log p_sigma(x) by denoising score matching
over a geometric noise ladder. At test time, noise an event to sigma_max and
run annealed Langevin dynamics back down the ladder; anomaly score is the
reconstruction error. Signal events lie off the learned background manifold,
so the background-trained score cannot return them to their origin.

Plain MLP (no diffusion library): the content is in the objective and sampler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def noise_schedule(sigma_min=0.01, sigma_max=3.0, n_levels=10):
    """Geometric noise levels sigma_l, l = 0..L-1 (ascending)."""
    l = torch.arange(n_levels, dtype=torch.float32)
    return sigma_min * (sigma_max / sigma_min) ** (l / (n_levels - 1))


class ScoreNet(nn.Module):
    """s_theta(x, sigma): input concat(x, log sigma) -> score, same dim as x."""

    def __init__(self, dim=8, hidden=(128, 256, 128),
                 sigma_min=0.01, sigma_max=3.0, n_levels=10):
        super().__init__()
        self.dim = dim
        self.register_buffer("sigmas", noise_schedule(sigma_min, sigma_max, n_levels))

        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, h3), nn.SiLU(),
            nn.Linear(h3, dim),
        )

    def forward(self, x, sigma):
        """x: (B, dim); sigma: (B,) or (B,1). Returns score (B, dim)."""
        log_sigma = torch.log(sigma).reshape(-1, 1)
        return self.net(torch.cat([x, log_sigma], dim=1))

    def training_loss(self, x):
        """Weighted denoising score matching loss, lambda(sigma) = sigma^2."""
        b = x.shape[0]
        idx = torch.randint(0, self.sigmas.shape[0], (b,), device=x.device)
        sigma = self.sigmas[idx].reshape(-1, 1)
        eps = torch.randn_like(x)
        x_tilde = x + sigma * eps
        target = -eps / sigma                      # = (x - x_tilde) / sigma^2
        pred = self.forward(x_tilde, sigma.reshape(-1))
        lam = sigma ** 2                            # Song & Ermon 2019 weighting
        return torch.mean(lam * (pred - target) ** 2)

    @torch.no_grad()
    def reconstruct(self, x, n_steps=100, step_scale=0.1):
        """Anneal sigma from sigma_max to sigma_min and run Langevin dynamics.

        Kept for the Stage-4 convergence check (a noised background event should
        settle as steps increase). It is NOT used for the anomaly score: starting
        at sigma_max=3 (>> the unit-variance data) discards x and samples a
        generic background point, so ||x - rec||^2 is dominated by noise (AUC~0.5).
        """
        sigma_max = float(self.sigmas[-1])
        sigma_min = float(self.sigmas[0])
        cur = x + sigma_max * torch.randn_like(x)
        ratio = (sigma_min / sigma_max) ** (1.0 / max(n_steps - 1, 1))
        sigma_t = sigma_max
        for _ in range(n_steps):
            s = torch.full((x.shape[0],), sigma_t, device=x.device)
            score = self.forward(cur, s)
            step = step_scale * sigma_t
            cur = cur + 0.5 * step ** 2 * score + step * torch.randn_like(x)
            sigma_t *= ratio
        return cur

    @torch.no_grad()
    def anomaly_score(self, x, seeds=2, seed=0):
        """Reconstruction-through-denoising error, averaged over the noise ladder.

        For each level we draw x_tilde = x + sigma*eps and form the model's
        one-step (Tweedie) reconstruction x_hat = x_tilde + sigma^2 * s(x_tilde,
        sigma) = the score net's estimate of E[x | x_tilde]. ||x - x_hat||^2 is
        small when x lies on the learned background manifold (the score points
        back to x) and large for off-manifold signal (the score points toward the
        background, away from x). Averaging over the full ladder and a few noise
        draws gives a low-variance score; a fixed seed keeps it reproducible so
        the validation metric is stable across epochs.

        This is the same reconstruction idea as the Langevin sampler but made
        stable -- see reconstruct() for why the literal sigma_max recipe fails.
        """
        g = torch.Generator(device=x.device).manual_seed(seed)
        acc = torch.zeros(x.shape[0], device=x.device)
        for sig in self.sigmas.tolist():
            s_vec = torch.full((x.shape[0],), sig, device=x.device)
            for _ in range(seeds):
                eps = torch.randn(x.shape, generator=g, device=x.device)
                x_tilde = x + sig * eps
                x_hat = x_tilde + sig ** 2 * self.forward(x_tilde, s_vec)
                acc += torch.sum((x - x_hat) ** 2, dim=1)
        return acc / (self.sigmas.shape[0] * seeds)


def build(config):
    m = config.get("model", {})
    return ScoreNet(
        dim=m.get("input_dim", 8),
        hidden=tuple(m.get("hidden", [128, 256, 128])),
        sigma_min=m.get("sigma_min", 0.01),
        sigma_max=m.get("sigma_max", 3.0),
        n_levels=m.get("n_levels", 10),
    )
