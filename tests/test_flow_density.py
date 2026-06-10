"""MADE autoregressive masking + flow density on a toy Gaussian."""

import torch

from models.normalizing_flow import MADE, MAF


def test_made_is_autoregressive():
    """mu_i and alpha_i must not depend on input dims j >= i."""
    dim = 8
    made = MADE(dim=dim, hidden=(64, 64))
    made.eval()
    x = torch.randn(1, dim, requires_grad=True)

    def mu_of(x_):
        mu, _ = made(x_)
        return mu[0]

    jac = torch.autograd.functional.jacobian(mu_of, x).reshape(dim, dim)
    for i in range(dim):
        for j in range(i, dim):           # output i must ignore inputs j >= i
            assert jac[i, j].abs() < 1e-7


def test_flow_density_higher_for_in_distribution():
    torch.manual_seed(0)
    dim = 8
    mean = torch.linspace(-1, 1, dim)
    data = mean + 0.3 * torch.randn(4000, dim)

    flow = MAF(dim=dim, n_layers=4, hidden=(64, 64))
    opt = torch.optim.AdamW(flow.parameters(), lr=1e-3)
    for _ in range(400):
        idx = torch.randint(0, data.shape[0], (256,))
        loss = flow.training_loss(data[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()

    flow.eval()
    in_dist = mean + 0.3 * torch.randn(1000, dim)
    out_dist = mean + 5.0 + 0.3 * torch.randn(1000, dim)
    with torch.no_grad():
        lp_in = flow.log_prob(in_dist).mean()
        lp_out = flow.log_prob(out_dist).mean()
    assert lp_in > lp_out + 5.0          # clearly higher density in-distribution
