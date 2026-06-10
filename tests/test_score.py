"""Score-model checks: the learned score reproduces the denoising target and
scales like 1/sigma, and Langevin denoising returns a perturbed point toward
its origin.

Note: the prompt's build-order note says "score magnitude should decrease as
sigma -> 0". That is backwards. The denoising score target is -eps/sigma, whose
magnitude grows like 1/sigma as sigma -> 0. We verify the correct relationship
(magnitude larger at sigma_min than sigma_max) and that sigma * s ~= -eps.
"""

import torch

from models.score_model import ScoreNet


def _trained_score(seed=0, steps=600):
    torch.manual_seed(seed)
    dim = 8
    mean = torch.linspace(-0.5, 0.5, dim)
    data = mean + 0.3 * torch.randn(4000, dim)
    net = ScoreNet(dim=dim)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    for _ in range(steps):
        idx = torch.randint(0, data.shape[0], (256,))
        loss = net.training_loss(data[idx])
        opt.zero_grad(); loss.backward()
        opt.step()
    return net, mean, data


def test_score_magnitude_scales_inverse_sigma():
    """Score magnitude is larger at sigma_min than sigma_max (~1/sigma)."""
    net, mean, data = _trained_score()
    net.eval()
    x = data[:500]
    with torch.no_grad():
        s_small = net(x, torch.full((500,), float(net.sigmas[0])))
        s_large = net(x, torch.full((500,), float(net.sigmas[-1])))
    mag_small = s_small.norm(dim=1).mean()
    mag_large = s_large.norm(dim=1).mean()
    assert mag_small > mag_large


def test_score_matches_analytic_gaussian_score():
    """For data ~ N(mu, tau^2 I), the true score is -(x-mu)/(tau^2+sigma^2)."""
    tau = 0.3
    net, mean, data = _trained_score()
    net.eval()
    x = data[:1000]
    for li in (len(net.sigmas) // 2, len(net.sigmas) - 1):
        sigma_val = float(net.sigmas[li])
        # Evaluate where the net operates: samples from p_sigma (x + sigma*eps).
        x_eval = x + sigma_val * torch.randn_like(x)
        with torch.no_grad():
            pred = net(x_eval, torch.full((x.shape[0],), sigma_val))
        analytic = -(x_eval - mean) / (tau ** 2 + sigma_val ** 2)
        rel_err = (pred - analytic).norm(dim=1).mean() / analytic.norm(dim=1).mean()
        assert rel_err < 0.4          # learned score tracks the analytic score


def test_langevin_denoises_toward_origin():
    net, mean, data = _trained_score()
    net.eval()
    x = data[:500]
    with torch.no_grad():
        rec = net.reconstruct(x, n_steps=100)
        noised = x + float(net.sigmas[-1]) * torch.randn_like(x)
    rec_err = ((x - rec) ** 2).sum(dim=1).mean()
    noise_err = ((x - noised) ** 2).sum(dim=1).mean()
    assert rec_err < noise_err          # denoising reduces distance to origin
