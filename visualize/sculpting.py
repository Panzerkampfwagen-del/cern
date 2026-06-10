"""Dijet-mass sculpting check for all three models.

For each model, overlay the background m_jj distribution before and after an
anomaly cut. A good detector leaves the background m_jj shape unchanged (low
JS divergence): sculpting would carve a fake bump that mimics a resonance.
This is the most important validation for a real physics result.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import sculpting_js

PRETTY = {"autoencoder": "Autoencoder", "flow": "Normalizing Flow",
          "score": "Score Model"}


def plot_one(ax, eval_path, title, keep_fraction=0.01):
    d = np.load(eval_path)
    scores, labels, mjj = d["scores"], d["labels"], d["mjj"]
    bg = labels == 0
    js, thr = sculpting_js(scores, mjj, labels, keep_fraction=keep_fraction)
    surviving = bg & (scores >= thr)
    edges = np.histogram_bin_edges(mjj[bg], bins=40)
    ax.hist(mjj[bg], bins=edges, density=True, histtype="step", color="C0",
            label="background (all)", linewidth=1.5)
    ax.hist(mjj[surviving], bins=edges, density=True, histtype="step",
            color="C3", label=f"surviving top {keep_fraction:.0%}", linewidth=1.5)
    ax.set_title(f"{title}\nJS div = {js:.4f}")
    ax.set_xlabel("m_jj [GeV]")
    ax.set_yticks([])
    ax.legend()
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["autoencoder", "flow", "score"])
    ap.add_argument("--keep-fraction", type=float, default=0.01)
    ap.add_argument("--eval-dir", default="results")
    ap.add_argument("--out", default="results/sculpting.png")
    args = ap.parse_args()

    present = [(n, os.path.join(args.eval_dir, f"{n}_eval.npz")) for n in args.names
               if os.path.exists(os.path.join(args.eval_dir, f"{n}_eval.npz"))]
    if not present:
        print("[sculpting] no eval files found; run evaluate.py first")
        return

    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 4.5),
                             squeeze=False)
    for ax, (name, path) in zip(axes[0], present):
        js = plot_one(ax, path, PRETTY.get(name, name), args.keep_fraction)
        flag = "N/A" if np.isnan(js) else ("OK" if js < 0.02 else "SCULPTING")
        print(f"[sculpting] {name}: JS div = {js:.4f}  [{flag}]")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"[sculpting] saved {args.out}")


if __name__ == "__main__":
    main()
