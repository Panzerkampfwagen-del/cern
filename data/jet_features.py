"""Turn raw particle 4-vectors into the 8 high-level dijet features.

Pipeline per event:
  1. anti-kT (R=1.0) clustering with fastjet (clustering only)
  2. keep the two leading-pT jets
  3. per jet: mass, pT, eta, phi, and tau21 (from scratch, nsubjettiness.py)
  4. m_jj = invariant mass of the dijet system, delta_eta_jj = |eta1 - eta2|

Output feature order (FEATURE_NAMES):
  m_j1, m_j2, pT_j1, pT_j2, tau21_j1, tau21_j2, m_jj, delta_eta_jj
"""

import numpy as np

from .nsubjettiness import tau21

FEATURE_NAMES = [
    "m_j1", "m_j2", "pT_j1", "pT_j2",
    "tau21_j1", "tau21_j2", "m_jj", "delta_eta_jj",
]
N_FEATURES = len(FEATURE_NAMES)
JET_R = 1.0
PT_MIN = 20.0  # minimum jet pT (GeV) to be considered a leading jet


def _import_fastjet():
    import awkward as ak
    import fastjet
    import vector
    vector.register_awkward()
    return ak, fastjet, vector


def particles_to_p4(pt, eta, phi):
    """Massless 4-momentum components from (pT, eta, phi)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = pt * np.cosh(eta)
    return px, py, pz, E


def cluster_chunk(pt2d, eta2d, phi2d):
    """anti-kT cluster a chunk of events.

    pt2d/eta2d/phi2d: (n_events, n_particles) padded arrays (pT==0 = padding).
    Returns an awkward array of jets (Momentum4D, sorted by pT desc) and an
    awkward array of their constituents, both indexed [event][jet].
    """
    ak, fastjet, _ = _import_fastjet()

    mask = pt2d > 0.0
    n_events = pt2d.shape[0]
    counts = mask.sum(axis=1)

    flat_pt = pt2d[mask]
    flat_eta = eta2d[mask]
    flat_phi = phi2d[mask]
    px, py, pz, E = particles_to_p4(flat_pt, flat_eta, flat_phi)

    flat = ak.zip(
        {"px": px, "py": py, "pz": pz, "E": E},
        with_name="Momentum4D",
    )
    parts = ak.unflatten(flat, ak.Array(counts.astype(np.int64)))

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, JET_R)
    cs = fastjet.ClusterSequence(parts, jetdef)
    jets = cs.inclusive_jets(min_pt=PT_MIN)
    consts = cs.constituents(min_pt=PT_MIN)

    # Sort jets (and matching constituents) by descending pT within each event.
    order = ak.argsort(jets.pt, axis=1, ascending=False)
    return jets[order], consts[order], n_events


def event_features(jet_pt, jet_eta, jet_phi, jet_mass, c_pt, c_eta, c_phi):
    """Compute the 8 features for one event given its two leading jets.

    Scalars for the two jets plus per-jet constituent (pt, eta, phi) lists.
    Returns a length-8 numpy array, or None if fewer than two jets.
    """
    if len(jet_pt) < 2:
        return None

    t21_1 = tau21(c_pt[0], c_eta[0], c_phi[0])
    t21_2 = tau21(c_pt[1], c_eta[1], c_phi[1])

    # Dijet invariant mass from the two leading jet 4-vectors.
    px1, py1, pz1, E1 = particles_to_p4(jet_pt[0], jet_eta[0], jet_phi[0])
    px2, py2, pz2, E2 = particles_to_p4(jet_pt[1], jet_eta[1], jet_phi[1])
    # Jets are massive, so add the mass back into energy.
    E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + jet_mass[0] ** 2)
    E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + jet_mass[1] ** 2)
    Es, pxs, pys, pzs = E1 + E2, px1 + px2, py1 + py2, pz1 + pz2
    m_jj = np.sqrt(max(Es**2 - pxs**2 - pys**2 - pzs**2, 0.0))

    return np.array([
        jet_mass[0], jet_mass[1],
        jet_pt[0], jet_pt[1],
        t21_1, t21_2,
        m_jj, abs(jet_eta[0] - jet_eta[1]),
    ], dtype=np.float64)


def _features_for_chunk(jets, consts, n_events):
    """Vectorised-clustered chunk -> (n_events, 8) features (NaN row if <2 jets)."""
    import awkward as ak

    jpt = ak.to_list(jets.pt)
    jeta = ak.to_list(jets.eta)
    jphi = ak.to_list(jets.phi)
    jmass = ak.to_list(jets.mass)
    cpt = ak.to_list(consts.pt)
    ceta = ak.to_list(consts.eta)
    cphi = ak.to_list(consts.phi)

    out = np.full((n_events, N_FEATURES), np.nan, dtype=np.float64)
    for i in range(n_events):
        f = event_features(
            np.asarray(jpt[i]), np.asarray(jeta[i]), np.asarray(jphi[i]),
            np.asarray(jmass[i]),
            [np.asarray(c) for c in cpt[i]],
            [np.asarray(c) for c in ceta[i]],
            [np.asarray(c) for c in cphi[i]],
        )
        if f is not None:
            out[i] = f
    return out


def extract_features(read_chunks):
    """Drive feature extraction over an iterable of (pt2d, eta2d, phi2d, labels).

    read_chunks yields padded particle arrays + labels per chunk. Returns
    (features (M, 8), labels (M,)) keeping only events with two valid jets.
    """
    feats, labs = [], []
    n_seen = 0
    for pt2d, eta2d, phi2d, labels in read_chunks:
        jets, consts, n_events = cluster_chunk(pt2d, eta2d, phi2d)
        chunk_feats = _features_for_chunk(jets, consts, n_events)
        valid = ~np.isnan(chunk_feats).any(axis=1)
        feats.append(chunk_feats[valid])
        labs.append(np.asarray(labels)[valid])
        n_seen += n_events
        print(f"[jet_features] processed {n_seen} events, "
              f"kept {sum(len(f) for f in feats)}", flush=True)
    return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)
