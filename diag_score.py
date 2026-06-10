"""Diagnostic: scan anomaly-score variants for the trained score net.

For each noise level (and combinations) report BOTH test AUC and m_jj sculpting
JS, so we can pick a score that discriminates without keying on m_jj.
"""
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from evaluate import load_trained, sculpting_js

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def batched(fn, x, bs=8192):
    out = []
    for i in range(0, len(x), bs):
        out.append(fn(torch.as_tensor(x[i:i + bs], dtype=torch.float32, device=DEV)))
    return torch.cat(out).cpu().numpy()


def tweedie(model, levels, seeds=4, seed=0):
    @torch.no_grad()
    def f(x):
        g = torch.Generator(device=x.device).manual_seed(seed)
        acc = torch.zeros(x.shape[0], device=x.device)
        for sig in levels:
            sv = torch.full((x.shape[0],), sig, device=x.device)
            for _ in range(seeds):
                xt = x + sig * torch.randn(x.shape, generator=g, device=x.device)
                xhat = xt + sig ** 2 * model.forward(xt, sv)
                acc += ((x - xhat) ** 2).sum(1)
        return acc / (len(levels) * seeds)
    return f


def report(name, scores, labels, mjj):
    auc = roc_auc_score(labels, scores)
    js, _ = sculpting_js(scores, mjj, labels)
    print(f"  {name:26s} AUC={auc:.4f}  sculpt_JS={js:.4f}")
    return auc, js


def main():
    with open("configs/score_model.yaml") as f:
        config = yaml.safe_load(f)
    model, _, _, (test_x, labels, mjj) = load_trained(config, device=DEV)
    sig = [float(s) for s in model.sigmas.tolist()]
    print("per-single-sigma Tweedie:")
    for s in sig:
        report(f"sigma={s:.3f}", batched(tweedie(model, [s]), test_x), labels, mjj)
    print("combinations:")
    report("all sigma", batched(tweedie(model, sig), test_x), labels, mjj)
    report("low 4", batched(tweedie(model, sig[:4]), test_x), labels, mjj)
    report("top 3", batched(tweedie(model, sig[-3:]), test_x), labels, mjj)
    report("mid 0.13-0.85", batched(tweedie(model, sig[4:8]), test_x), labels, mjj)


if __name__ == "__main__":
    main()
