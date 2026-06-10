"""Background vs signal distributions of the 8 high-level features.

The tau21 panels are the physics sanity check: QCD background jets peak high
(0.5-0.9), W/Z signal jets peak low (0.3-0.5).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_features
from data.jet_features import FEATURE_NAMES


def plot_features(features, labels, out="results/feature_distributions.png"):
    bg = labels == 0
    sig = labels == 1
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for j, ax in enumerate(axes.flat):
        lo, hi = np.percentile(features[:, j], [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(features[bg, j], bins=bins, density=True, histtype="step",
                color="C0", label="background", linewidth=1.5)
        ax.hist(features[sig, j], bins=bins, density=True, histtype="step",
                color="C3", label="signal", linewidth=1.5)
        ax.set_title(FEATURE_NAMES[j])
        ax.set_yticks([])
        if j == 0:
            ax.legend()
    fig.suptitle("LHCO R&D: background (QCD) vs signal (W'->WZ)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[feature_plots] saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.npz")
    ap.add_argument("--out", default="results/feature_distributions.png")
    args = ap.parse_args()
    features, labels = load_features(args.features)
    plot_features(features, labels, args.out)

    # Print tau21 peak locations as a numeric physics check.
    for j, name in [(4, "tau21_j1"), (5, "tau21_j2")]:
        bg_peak = np.median(features[labels == 0, j])
        sig_peak = np.median(features[labels == 1, j])
        print(f"[feature_plots] {name}: background median {bg_peak:.3f}, "
              f"signal median {sig_peak:.3f}")


if __name__ == "__main__":
    main()
