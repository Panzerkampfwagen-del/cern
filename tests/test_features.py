"""tau21 sanity on synthetic jets + feature-extraction smoke test."""

import numpy as np

from data.jet_features import N_FEATURES, cluster_chunk, _features_for_chunk
from data.nsubjettiness import delta_phi, delta_r, tau21


def test_delta_phi_wraps():
    assert abs(delta_phi(3.0, -3.0)) < 0.5      # near 2pi apart -> small
    assert abs(delta_phi(0.1, -0.1) - 0.2) < 1e-9
    assert delta_r(0.0, 0.0, 0.0, 0.0) == 0.0


def test_tau21_two_prong_lower_than_one_prong():
    rng = np.random.default_rng(0)
    n = 150
    # One-prong: single tight blob.
    eta1 = 0.05 * rng.standard_normal(n)
    phi1 = 0.05 * rng.standard_normal(n)
    pt1 = rng.uniform(1, 10, n)
    t21_one = tau21(pt1, eta1, phi1)

    # Two-prong: two well-separated subjets.
    half = n // 2
    eta2 = np.concatenate([0.3 + 0.05 * rng.standard_normal(half),
                           -0.3 + 0.05 * rng.standard_normal(n - half)])
    phi2 = np.concatenate([0.3 + 0.05 * rng.standard_normal(half),
                           -0.3 + 0.05 * rng.standard_normal(n - half)])
    pt2 = rng.uniform(1, 10, n)
    t21_two = tau21(pt2, eta2, phi2)

    assert t21_two < t21_one
    assert t21_two < 0.5          # genuine 2-prong is small
    assert 0.0 <= t21_one <= 1.5


def test_feature_extraction_shapes():
    rng = np.random.default_rng(1)
    N = 200
    pt = np.zeros((1, N)); eta = np.zeros((1, N)); phi = np.zeros((1, N))
    pt[0, :60] = rng.uniform(1, 20, 60)
    eta[0, :60] = 0.5 + 0.1 * rng.standard_normal(60)
    phi[0, :60] = 1.0 + 0.1 * rng.standard_normal(60)
    pt[0, 60:120] = rng.uniform(1, 20, 60)
    eta[0, 60:120] = -0.8 + 0.1 * rng.standard_normal(60)
    phi[0, 60:120] = -2.0 + 0.1 * rng.standard_normal(60)

    jets, consts, nev = cluster_chunk(pt, eta, phi)
    feats = _features_for_chunk(jets, consts, nev)
    assert feats.shape == (1, N_FEATURES)
    assert np.isfinite(feats).all()
    assert feats[0, 6] > 0.0      # m_jj positive
    assert feats[0, 2] >= feats[0, 3]   # pT_j1 >= pT_j2 (sorted)
