"""Evaluate a trained model on the labelled test set.

Metrics:
  AUC          - ranking quality of the anomaly score
  SIC curve    - significance improvement = TPR / sqrt(FPR); report max
  sculpting    - JS divergence between background m_jj before/after an
                 anomaly cut (a good detector leaves background m_jj flat)

Labels are used here only -- never to compute the anomaly score.

  python evaluate.py --config configs/autoencoder.yaml
"""

import argparse
import importlib
import json

import numpy as np
import torch
import yaml
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score, roc_curve

from data.dataset import Normalizer, load_features, make_splits
from data.jet_features import FEATURE_NAMES
from train import MODEL_MODULES, compute_scores

MJJ_IDX = FEATURE_NAMES.index("m_jj")


def sic_curve(labels, scores, fpr_floor=1e-4):
    """Return (fpr, tpr, sic) with sic = tpr / sqrt(fpr); max SIC ignores
    the low-statistics region fpr < fpr_floor."""
    # drop_intermediate=False: sklearn's default prunes ROC points collinear in
    # (FPR, TPR), but SIC = TPR/sqrt(FPR) is a nonlinear reweighting, so a pruned
    # interior point can be the true SIC maximizer -- keep all thresholds.
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    sic = np.zeros_like(fpr)
    valid = fpr > 0
    sic[valid] = tpr[valid] / np.sqrt(fpr[valid])
    max_sic = float(sic[fpr >= fpr_floor].max()) if np.any(fpr >= fpr_floor) else 0.0
    return fpr, tpr, sic, max_sic


def sculpting_js(scores, mjj, labels, keep_fraction=0.01, bins=40):
    """JS divergence between background m_jj before and after an anomaly cut.

    Cut keeps the most anomalous keep_fraction of events (high score). The
    diagnostic uses true background only, so a low value means the cut did
    not sculpt the background m_jj shape into a fake bump.
    """
    bg = labels == 0
    mjj_bg = mjj[bg]
    thr = np.quantile(scores, 1.0 - keep_fraction)
    surviving = bg & (scores >= thr)
    edges = np.histogram_bin_edges(mjj_bg, bins=bins)
    p_counts, _ = np.histogram(mjj_bg, bins=edges)
    q_counts, _ = np.histogram(mjj[surviving], bins=edges)
    if surviving.sum() < 2 or q_counts.sum() == 0:
        return float("nan"), thr      # too few survivors to compare shapes
    p = p_counts / p_counts.sum() + 1e-12
    q = q_counts / q_counts.sum() + 1e-12
    # jensenshannon returns the distance (sqrt of divergence); square it.
    return float(jensenshannon(p, q, base=2) ** 2), thr


def load_trained(config, device="cpu"):
    """Rebuild splits, load normaliser + checkpoint. Returns model, test data."""
    features, labels = load_features(config["data"]["features"])
    splits = make_splits(features, labels, config["data"]["splits"],
                         seed=config["data"].get("seed", 0))
    normalizer = Normalizer.load(config["out"] + "_norm.npz")

    mod = importlib.import_module(MODEL_MODULES[config["name"]])
    model = mod.build(config).to(device)
    model.load_state_dict(torch.load(config["out"] + ".pt", map_location=device,
                                     weights_only=True))
    model.eval()

    test_x = normalizer.transform(splits["test"][0]).astype(np.float32)
    test_labels = splits["test"][1]
    test_mjj = splits["test"][0][:, MJJ_IDX]
    return model, normalizer, splits, (test_x, test_labels, test_mjj)


def evaluate(config, device="cpu"):
    model, normalizer, splits, (test_x, labels, mjj) = load_trained(config, device)
    scores = compute_scores(model, test_x, device=device)

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, sic, max_sic = sic_curve(labels, scores)
    js, thr = sculpting_js(scores, mjj, labels)

    np.savez(config["out"] + "_eval.npz",
             scores=scores, labels=labels, mjj=mjj,
             fpr=fpr, tpr=tpr, sic=sic)
    metrics = {"name": config["name"], "auc": auc, "max_sic": max_sic,
               "sculpting_js": js, "cut_threshold": float(thr)}
    with open(config["out"] + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[eval:{config['name']}] AUC={auc:.4f}  maxSIC={max_sic:.3f}  "
          f"sculpting_JS={js:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    evaluate(config, device=args.device)


if __name__ == "__main__":
    main()
