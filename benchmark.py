"""Head-to-head benchmark table + permutation feature importance.

Importance = drop in AUC when one input feature is shuffled across the test
set (breaking its correlation with the rest). The most discriminating feature
shows the largest drop. Physics expectation: tau21 should rank top for W'->WZ.
"""

import argparse
import json

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from data.jet_features import FEATURE_NAMES
from evaluate import load_trained
from train import compute_scores

DEFAULT_CONFIGS = [
    "configs/autoencoder.yaml",
    "configs/flow.yaml",
    "configs/score_model.yaml",
]
PRETTY = {"autoencoder": "Autoencoder", "flow": "Normalizing Flow",
          "score": "Score Model"}


def permutation_importance(config, device="cpu", seed=0, topk=3):
    """AUC drop per feature when that feature column is shuffled."""
    model, _norm, _splits, (test_x, labels, _mjj) = load_trained(config, device)
    base_auc = roc_auc_score(labels, compute_scores(model, test_x, device=device))
    rng = np.random.default_rng(seed)
    drops = []
    for j in range(test_x.shape[1]):
        xp = test_x.copy()
        xp[:, j] = xp[rng.permutation(len(xp)), j]
        auc_j = roc_auc_score(labels, compute_scores(model, xp, device=device))
        drops.append(base_auc - auc_j)
    drops = np.array(drops)
    order = np.argsort(drops)[::-1][:topk]
    return [(FEATURE_NAMES[j], float(drops[j])) for j in order]


def load_metrics(config):
    with open(config["out"] + "_metrics.json") as f:
        metrics = json.load(f)
    with open(config["out"] + "_history.json") as f:
        metrics["train_time_sec"] = json.load(f)["train_time_sec"]
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eps-s", type=float, default=0.01)
    args = ap.parse_args()

    rows, importances = [], {}
    for cfg_path in args.configs:
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        rows.append(load_metrics(config))
        importances[config["name"]] = permutation_importance(config, args.device)

    print(f"\n[cern] LHC Olympics R&D benchmark  signal: W'->WZ  "
          f"eps_s={args.eps_s}")
    print(f"  {'Model':<18} | {'AUC':<5} | {'Max SIC':<7} | "
          f"{'mJJ sculpting (JS div)':<22} | Train time")
    print("  " + "-" * 18 + "|" + "-" * 7 + "|" + "-" * 9 + "|"
          + "-" * 24 + "|" + "-" * 12)
    for m in rows:
        name = PRETTY.get(m["name"], m["name"])
        print(f"  {name:<18} | {m['auc']:.2f}  |  {m['max_sic']:.2f}   |"
              f"       {m['sculpting_js']:.3f}            |   "
              f"{m['train_time_sec'] / 60:.1f} min")

    print("\n  Permutation feature importance (top-3 by AUC drop):")
    for name, feats in importances.items():
        pretty = PRETTY.get(name, name)
        s = ", ".join(f"{fn} ({d:+.3f})" for fn, d in feats)
        print(f"    {pretty:<18}: {s}")
    print("  Physics expectation: tau21_j1 / tau21_j2 should rank highest "
          "for W'->WZ.\n")


if __name__ == "__main__":
    main()
