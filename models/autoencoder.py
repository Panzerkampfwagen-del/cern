"""Autoencoder baseline. Anomaly score = reconstruction error.

The spec lists encoder hidden sizes [64, 32, 16] and separately a latent
dim of 4 ("bottleneck, force compression"). We honour both: the listed
sizes are the hidden stack and 4 is the true bottleneck, giving
8 -> 64 -> 32 -> 16 -> 4 -> 16 -> 32 -> 64 -> 8. latent_dim is configurable.
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim=8, hidden=(64, 32, 16), latent_dim=4):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        enc += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_loss(self, x):
        """Mean MSE reconstruction loss over batch and features."""
        x_hat = self.forward(x)
        return torch.mean((x - x_hat) ** 2)

    @torch.no_grad()
    def anomaly_score(self, x):
        """Per-event reconstruction error ||x - x_hat||^2 (sum over features)."""
        x_hat = self.forward(x)
        return torch.sum((x - x_hat) ** 2, dim=1)


def build(config):
    m = config.get("model", {})
    return Autoencoder(
        input_dim=m.get("input_dim", 8),
        hidden=tuple(m.get("hidden", [64, 32, 16])),
        latent_dim=m.get("latent_dim", 4),
    )
