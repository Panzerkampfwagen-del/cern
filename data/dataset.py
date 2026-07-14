"""Dataset assembly: splits, normalisation, PyTorch wrappers.

Anomaly models train on background ONLY. Validation and test are mixed
(background + signal); their labels are used only for the validation
separation metric and for AUC/SIC at evaluation time, never as model input.

Note on signal fraction: the spec gives explicit split counts (val 80k bg +
20k signal, test 200k bg + 10k signal) and separately names eps_s = 0.01 as
the physics scenario. These disagree numerically. AUC and SIC(eps_b)=TPR/sqrt(eps_b)
are both prevalence-independent, so the reported metrics are unaffected; we use
the explicit counts and treat eps_s = 0.01 as the nominal scenario label.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_features(path):
    """Load extracted features. Accepts a .npz with 'features' and 'labels'."""
    d = np.load(path)
    return d["features"].astype(np.float64), d["labels"].astype(np.int64)


def make_splits(features, labels, sizes, seed=0):
    """Partition into train (bg only) / val / test (mixed).

    sizes: dict with val_bg, val_sig, test_bg, test_sig. Train takes all
    remaining background. Returns dict of (features, labels) per split.
    """
    rng = np.random.default_rng(seed)
    bg_idx = np.where(labels == 0)[0]
    sig_idx = np.where(labels == 1)[0]
    rng.shuffle(bg_idx)
    rng.shuffle(sig_idx)

    need_bg = sizes["val_bg"] + sizes["test_bg"]
    need_sig = sizes["val_sig"] + sizes["test_sig"]
    if len(bg_idx) < need_bg + 1 or len(sig_idx) < need_sig:
        raise ValueError(
            f"not enough events: bg {len(bg_idx)} (need >{need_bg}), "
            f"sig {len(sig_idx)} (need {need_sig})")

    vb = bg_idx[: sizes["val_bg"]]
    tb = bg_idx[sizes["val_bg"]: need_bg]
    train_bg = bg_idx[need_bg:]
    vs = sig_idx[: sizes["val_sig"]]
    ts = sig_idx[sizes["val_sig"]: need_sig]

    def pack(idx):
        return features[idx], labels[idx]

    val_idx = np.concatenate([vb, vs]); rng.shuffle(val_idx)
    test_idx = np.concatenate([tb, ts]); rng.shuffle(test_idx)

    splits = {
        "train": pack(train_bg),
        "val": pack(val_idx),
        "test": pack(test_idx),
    }
    assert splits["train"][1].sum() == 0, "signal leaked into training set"
    return splits


class Normalizer:
    """Standardise features to zero mean / unit variance from training stats.

    Optionally log1p-transforms a subset of features first (heavy-tailed pT /
    mass columns) before standardising. Disabled by default to match the spec.
    """

    def __init__(self, log_features=None):
        self.log_features = list(log_features) if log_features else []
        self.mean = None
        self.std = None

    def _pre(self, x):
        x = x.copy()
        for j in self.log_features:
            x[:, j] = np.log1p(np.clip(x[:, j], 0.0, None))
        return x

    def fit(self, x):
        xp = self._pre(x)
        self.mean = xp.mean(axis=0)
        self.std = xp.std(axis=0) + 1e-8
        return self

    def transform(self, x):
        return (self._pre(x) - self.mean) / self.std

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std,
                 log_features=np.array(self.log_features, dtype=np.int64))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls(log_features=d["log_features"].tolist())
        obj.mean = d["mean"]
        obj.std = d["std"]
        return obj


class FeatureDataset(Dataset):
    """Wraps a normalised feature matrix as float32 tensors."""

    def __init__(self, x):
        self.x = torch.as_tensor(x, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]


def get_dataloaders(splits, normalizer, batch_size, num_workers=0):
    """Build train/val/test loaders. Loaders yield normalised feature tensors."""
    train_x = normalizer.transform(splits["train"][0])
    val_x = normalizer.transform(splits["val"][0])
    test_x = normalizer.transform(splits["test"][0])

    train_loader = DataLoader(FeatureDataset(train_x), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(FeatureDataset(val_x), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(FeatureDataset(test_x), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
