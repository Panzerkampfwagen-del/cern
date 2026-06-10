"""Forward-pass shapes and finite losses for all three models."""

import torch

from models.autoencoder import Autoencoder
from models.normalizing_flow import MAF
from models.score_model import ScoreNet


def _x(b=32, d=8):
    return torch.randn(b, d)


def test_autoencoder():
    m = Autoencoder()
    x = _x()
    assert m(x).shape == x.shape
    assert torch.isfinite(m.training_loss(x))
    assert m.anomaly_score(x).shape == (32,)


def test_flow():
    m = MAF()
    x = _x()
    assert m.log_prob(x).shape == (32,)
    loss = m.training_loss(x)
    assert torch.isfinite(loss)
    assert m.anomaly_score(x).shape == (32,)


def test_score_model():
    m = ScoreNet()
    x = _x()
    sigma = torch.full((32,), 0.5)
    assert m(x, sigma).shape == x.shape
    assert torch.isfinite(m.training_loss(x))
    rec = m.reconstruct(x, n_steps=10)
    assert rec.shape == x.shape
    assert m.anomaly_score(x, seeds=1).shape == (32,)
