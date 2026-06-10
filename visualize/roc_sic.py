"""ROC and SIC curves overlaid for all available models."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRETTY = {"autoencoder": "Autoencoder", "flow": "Normalizing Flow",
          "score": "Score Model"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["autoencoder", "flow", "score"])
    ap.add_argument("--eval-dir", default="results")
    ap.add_argument("--out", default="results/roc_sic.png")
    args = ap.parse_args()

    present = [(n, os.path.join(args.eval_dir, f"{n}_eval.npz")) for n in args.names
               if os.path.exists(os.path.join(args.eval_dir, f"{n}_eval.npz"))]
    if not present:
        print("[roc_sic] no eval files found; run evaluate.py first")
        return

    fig, (ax_roc, ax_sic) = plt.subplots(1, 2, figsize=(13, 5))
    for i, (name, path) in enumerate(present):
        d = np.load(path)
        fpr, tpr, sic = d["fpr"], d["tpr"], d["sic"]
        ax_roc.plot(fpr, tpr, color=f"C{i}", label=PRETTY.get(name, name))
        mask = fpr > 1e-4
        ax_sic.plot(fpr[mask], sic[mask], color=f"C{i}",
                    label=PRETTY.get(name, name))

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set_xlabel("background efficiency (FPR)")
    ax_roc.set_ylabel("signal efficiency (TPR)")
    ax_roc.set_title("ROC")
    ax_roc.legend()

    ax_sic.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax_sic.set_xscale("log")
    ax_sic.set_xlabel("background efficiency (FPR)")
    ax_sic.set_ylabel("SIC = TPR / sqrt(FPR)")
    ax_sic.set_title("Significance Improvement")
    ax_sic.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"[roc_sic] saved {args.out}")


if __name__ == "__main__":
    main()
