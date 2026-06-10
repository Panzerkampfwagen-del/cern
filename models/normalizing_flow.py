"""Masked Autoregressive Flow (MAF) with from-scratch MADE masking.

Each MADE layer outputs (mu, alpha) per dimension under an autoregressive
constraint built with degree masks (Germain et al. 2015). The affine
transform z_i = (x_i - mu_i) * exp(-alpha_i) has log|det| = -sum_i alpha_i.
Stacking layers with permutations lets every dim condition on every other.

Density is exact (no variational bound), and we implement the masking
ourselves rather than using nflows / normflows, per the constraints.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """Linear layer whose weights are gated by a fixed binary mask."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.copy_(mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Outputs (mu, alpha) for an autoregressive affine transform.

    Network: Linear(D,H) -> GELU -> Linear(H,H) -> GELU -> Linear(H, 2D),
    with MADE connectivity masks so output dim i depends only on inputs j < i.
    """

    def __init__(self, dim=8, hidden=(128, 128), alpha_scale=3.0):
        super().__init__()
        self.dim = dim
        self.alpha_scale = alpha_scale

        self.l1 = MaskedLinear(dim, hidden[0])
        self.l2 = MaskedLinear(hidden[0], hidden[1])
        self.l3 = MaskedLinear(hidden[1], 2 * dim)

        # Degrees: inputs 1..D; hidden units cycle through 1..D-1; outputs
        # repeat the input degrees for the mu block and the alpha block.
        deg_in = torch.arange(1, dim + 1)
        deg_h1 = torch.arange(hidden[0]) % (dim - 1) + 1
        deg_h2 = torch.arange(hidden[1]) % (dim - 1) + 1
        deg_out = torch.cat([deg_in, deg_in])

        # in->hidden / hidden->hidden use >=, hidden->output uses strict >.
        self.l1.set_mask((deg_h1[:, None] >= deg_in[None, :]).float())
        self.l2.set_mask((deg_h2[:, None] >= deg_h1[None, :]).float())
        self.l3.set_mask((deg_out[:, None] > deg_h2[None, :]).float())

    def forward(self, x):
        h = F.gelu(self.l1(x))
        h = F.gelu(self.l2(h))
        out = self.l3(h)
        mu, alpha = out[:, :self.dim], out[:, self.dim:]
        alpha = self.alpha_scale * torch.tanh(alpha)  # numerical stability
        return mu, alpha


class MAF(nn.Module):
    def __init__(self, dim=8, n_layers=8, hidden=(128, 128), seed=0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [MADE(dim, hidden) for _ in range(n_layers)])
        g = torch.Generator().manual_seed(seed)
        perms = []
        for _ in range(n_layers):
            perms.append(torch.randperm(dim, generator=g))
        self.register_buffer("perms", torch.stack(perms))

    def log_prob(self, x):
        """Exact log-density log p(x), shape (batch,)."""
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device)
        for layer, perm in zip(self.layers, self.perms):
            z = z[:, perm]
            mu, alpha = layer(z)
            z = (z - mu) * torch.exp(-alpha)
            log_det = log_det - alpha.sum(dim=1)
        base = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim * math.log(2 * math.pi)
        return base + log_det

    def training_loss(self, x):
        """Negative log-likelihood."""
        return -self.log_prob(x).mean()

    @torch.no_grad()
    def anomaly_score(self, x):
        """Anomaly score = -log p(x); higher = more anomalous."""
        return -self.log_prob(x)


def build(config):
    m = config.get("model", {})
    return MAF(
        dim=m.get("input_dim", 8),
        n_layers=m.get("n_layers", 8),
        hidden=tuple(m.get("hidden", [128, 128])),
        seed=m.get("seed", 0),
    )
