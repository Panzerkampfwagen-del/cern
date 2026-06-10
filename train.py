"""Unified training loop for the three anomaly models.

All models expose training_loss(x) and anomaly_score(x), so one Trainer
drives them. Models train on background only; the validation separation
metric (mean signal score - mean background score) uses validation labels
for early stopping and logging only -- never as model input.

  python train.py --config configs/autoencoder.yaml
"""

import argparse
import importlib
import json
import os
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from data.dataset import Normalizer, get_dataloaders, load_features, make_splits

MODEL_MODULES = {
    "autoencoder": "models.autoencoder",
    "flow": "models.normalizing_flow",
    "score": "models.score_model",
}


def build_model(config):
    mod = importlib.import_module(MODEL_MODULES[config["name"]])
    return mod.build(config)


def compute_scores(model, x, batch_size=8192, device="cpu"):
    """Anomaly score for every row of x (numpy out). Batched for memory."""
    model.eval()
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device)
            out.append(model.anomaly_score(xb).cpu())
    return torch.cat(out).numpy()


def separation(scores, labels):
    """Mean score on signal minus mean score on background."""
    return float(scores[labels == 1].mean() - scores[labels == 0].mean())


class Trainer:
    def __init__(self, device="cpu"):
        self.device = device

    def fit(self, model, train_loader, val_data, config):
        tr = config["train"]
        model = model.to(self.device)
        val_x, val_labels = val_data

        opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                                weight_decay=tr.get("weight_decay", 0.0))
        sched = None
        if tr.get("cosine", False):
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=tr["epochs"])
        grad_clip = tr.get("grad_clip", None)
        log_every = tr.get("log_every", 10)
        patience = tr.get("patience", 20)
        metric = tr.get("early_stop_metric", "separation")

        history = {"train_loss": [], "val_sep": [], "val_auc": [], "val_metric": []}
        best = -np.inf
        best_state = None
        wait = 0
        ckpt = config["out"] + ".pt"

        for epoch in range(1, tr["epochs"] + 1):
            model.train()
            running, nb = 0.0, 0
            for xb in train_loader:
                xb = xb.to(self.device)
                loss = model.training_loss(xb)
                opt.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                running += loss.item()
                nb += 1
            if sched:
                sched.step()
            train_loss = running / max(nb, 1)

            scores = compute_scores(model, val_x, device=self.device)
            sep = separation(scores, val_labels)
            auc = float(roc_auc_score(val_labels, scores))
            # Checkpoint-selection objective. "auc" tracks the reported metric and
            # is far more stable than mean-separation (whose tail sensitivity
            # misranks the flow and the diffusion score); "separation" is the spec
            # default; "loss" falls back to training loss.
            sel = {"auc": auc, "separation": sep, "loss": -train_loss}[metric]
            history["train_loss"].append(train_loss)
            history["val_sep"].append(sep)
            history["val_auc"].append(auc)
            history["val_metric"].append(sel)

            if epoch % log_every == 0 or epoch == 1:
                print(f"[train:{config['name']}] epoch {epoch:4d}  "
                      f"loss {train_loss:.4f}  val_sep {sep:+.4f}  "
                      f"val_auc {auc:.4f}", flush=True)

            if sel > best:
                best = sel
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                torch.save(best_state, ckpt)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"[train:{config['name']}] early stop at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(os.path.dirname(config["out"]), exist_ok=True)

    features, labels = load_features(config["data"]["features"])
    splits = make_splits(features, labels, config["data"]["splits"],
                         seed=config["data"].get("seed", 0))

    normalizer = Normalizer(log_features=config["data"].get("log_features"))
    normalizer.fit(splits["train"][0])
    normalizer.save(config["out"] + "_norm.npz")

    train_loader, _, _ = get_dataloaders(
        splits, normalizer, batch_size=config["train"]["batch"])
    val_x = normalizer.transform(splits["val"][0]).astype(np.float32)
    val_labels = splits["val"][1]
    sub = config["train"].get("val_subsample")
    if sub:  # val is pre-shuffled, so a head slice is a random subsample
        val_x, val_labels = val_x[:sub], val_labels[:sub]

    torch.manual_seed(config["data"].get("seed", 0))
    model = build_model(config)
    print(f"[train:{config['name']}] params: "
          f"{sum(p.numel() for p in model.parameters())}  device={args.device}")

    trainer = Trainer(device=args.device)
    t0 = time.time()
    history = trainer.fit(model, train_loader, (val_x, val_labels), config)
    train_time = time.time() - t0

    with open(config["out"] + "_history.json", "w") as f:
        json.dump({"history": history, "train_time_sec": train_time}, f, indent=2)
    print(f"[train:{config['name']}] done in {train_time / 60:.1f} min  "
          f"-> {config['out']}.pt")


if __name__ == "__main__":
    main()
