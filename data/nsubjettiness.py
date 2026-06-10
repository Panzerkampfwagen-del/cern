"""N-subjettiness tau_N implemented from scratch.

tau_N = (1/d0) * sum_k pT_k * min_a dR(k, axis_a)
with d0 = R0 * sum_k pT_k and dR = sqrt(deta^2 + dphi^2).

Subjet axes are found by pT-weighted k-means (Lloyd) on the (eta, phi)
positions of the jet constituents. tau21 = tau2 / tau1 measures 2-prong
vs 1-prong substructure: low for boson (W/Z -> qq) jets, high for QCD.
"""

import numpy as np


def delta_phi(phi1, phi2):
    """Signed phi difference wrapped to [-pi, pi]."""
    d = phi1 - phi2
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    """Angular distance in the eta-phi plane with phi wrapping."""
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta * deta + dphi * dphi)


def _circular_mean(phi, weights):
    """pT-weighted circular mean of phi angles."""
    s = np.sum(weights * np.sin(phi))
    c = np.sum(weights * np.cos(phi))
    return np.arctan2(s, c)


def kmeans_axes(pt, eta, phi, n_axes, max_iter=20, tol=1e-4):
    """pT-weighted k-means on (eta, phi). Returns axes of shape (n_axes, 2).

    Initialised deterministically with the n_axes highest-pT constituents so
    the result is reproducible without an RNG seed.
    """
    n = pt.shape[0]
    if n <= n_axes:
        # Fewer constituents than requested axes: pad with the existing points.
        ax_eta = np.empty(n_axes)
        ax_phi = np.empty(n_axes)
        ax_eta[:n] = eta
        ax_phi[:n] = phi
        ax_eta[n:] = eta[-1] if n > 0 else 0.0
        ax_phi[n:] = phi[-1] if n > 0 else 0.0
        return np.stack([ax_eta, ax_phi], axis=1)

    seed_idx = np.argsort(pt)[::-1][:n_axes]
    ax_eta = eta[seed_idx].copy()
    ax_phi = phi[seed_idx].copy()

    for _ in range(max_iter):
        # Assign each constituent to its nearest axis.
        deta = eta[:, None] - ax_eta[None, :]
        dphi = delta_phi(phi[:, None], ax_phi[None, :])
        dr2 = deta * deta + dphi * dphi
        assign = np.argmin(dr2, axis=1)

        new_eta = ax_eta.copy()
        new_phi = ax_phi.copy()
        for a in range(n_axes):
            mask = assign == a
            w = pt[mask]
            if w.sum() <= 0.0:
                continue
            new_eta[a] = np.sum(w * eta[mask]) / w.sum()
            new_phi[a] = _circular_mean(phi[mask], w)

        shift = np.max(np.abs(new_eta - ax_eta)) + np.max(np.abs(delta_phi(new_phi, ax_phi)))
        ax_eta, ax_phi = new_eta, new_phi
        if shift < tol:
            break

    return np.stack([ax_eta, ax_phi], axis=1)


def tau_n(pt, eta, phi, n, R0=1.0):
    """N-subjettiness tau_N for one jet's constituents (1-D arrays)."""
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    d0 = R0 * np.sum(pt)
    if d0 <= 0.0:
        return 0.0

    axes = kmeans_axes(pt, eta, phi, n)
    deta = eta[:, None] - axes[None, :, 0]
    dphi = delta_phi(phi[:, None], axes[None, :, 1])
    dr = np.sqrt(deta * deta + dphi * dphi)
    min_dr = np.min(dr, axis=1)
    return float(np.sum(pt * min_dr) / d0)


def tau21(pt, eta, phi, R0=1.0, eps=1e-8):
    """tau2 / tau1 for one jet. Returns 0 if tau1 is ~0 (degenerate jet)."""
    t1 = tau_n(pt, eta, phi, 1, R0)
    if t1 < eps:
        return 0.0
    t2 = tau_n(pt, eta, phi, 2, R0)
    return t2 / t1
