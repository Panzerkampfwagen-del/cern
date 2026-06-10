"""Anomaly score histograms, background vs signal, per model.

Clear separation between the two distributions means a good detector.
Reads results/<name>_eval.npz produced by evaluate.py.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRETTY = {"autoencoder": "Autoencoder", "flow": "Normalizing Flow",
          "score": "Score Model"}


def plot_one(ax, eval_path, title):
    d = np.load(eval_path)
    scores, labels = d["scores"], d["labels"]
    lo, hi = np.percentile(scores, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax.hist(scores[labels == 0], bins=bins, density=True, histtype="step",
            color="C0", label="background", linewidth=1.5)
    ax.hist(scores[labels == 1], bins=bins, density=True, histtype="step",
            color="C3", label="signal", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("anomaly score")
    ax.set_yticks([])
    ax.legend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["autoencoder", "flow", "score"])
    ap.add_argument("--eval-dir", default="results")
    ap.add_argument("--out", default="results/score_distributions.png")
    args = ap.parse_args()

    present = [(n, os.path.join(args.eval_dir, f"{n}_eval.npz")) for n in args.names
               if os.path.exists(os.path.join(args.eval_dir, f"{n}_eval.npz"))]
    if not present:
        print("[score_distributions] no eval files found; run evaluate.py first")
        return
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 4.5),
                             squeeze=False)
    for ax, (name, path) in zip(axes[0], present):
        plot_one(ax, path, PRETTY.get(name, name))
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"[score_distributions] saved {args.out}")


if __name__ == "__main__":
    main()
