# cern

Unsupervised anomaly detection for LHC collision events, comparing three
detection paradigms on the **LHC Olympics 2020 R&D** benchmark.

The autoencoder, not the score model, is the best physics-motivated detector:
its bottleneck forces it to learn jet substructure (τ₂₁ ranks top-2, exactly as
`W'→WZ` physics predicts), and it barely sculpts the mass spectrum (JS=0.030).
The score model achieves the highest raw AUC but discriminates via m_jj and
sculpts catastrophically (JS=0.551) — it would produce a false discovery.

**Demonstrates:** unsupervised anomaly detection, normalizing flows (MADE from
scratch), NCSM score-based models, permutation feature importance, mass sculpting
analysis. LHC Olympics 2020 R&D benchmark.

The three paradigms:

- **Reconstruction-based** — an autoencoder, anomaly score = reconstruction error
- **Density-based** — a masked autoregressive flow (MAF), anomaly score = −log p(x)
- **Score-based** — a noise-conditional diffusion model (NCSM), anomaly score =
  reconstruction-through-denoising error

The models train on Standard-Model background (QCD dijets) only and are scored
on their ability to flag a hypothetical `W'→WZ→qqqq` signal — without ever
seeing a label during training. The contribution is the **comparative analysis**:
for this signal, which paradigm detects new physics best, and *why*.

Reference: LHC Olympics 2020 challenge, Kasieczka et al. 2021.

---

## Results

Full benchmark on the labelled test set (200k background + 10k signal),
real LHCO R&D data, 1.1M events processed end to end:

| Model            | AUC      | Max SIC | m_jj sculpting (JS) | Top features (perm. importance) |
|------------------|----------|---------|---------------------|---------------------------------|
| Autoencoder      | 0.727    | 1.22    | **0.030**           | m_jj, **τ21_j2, τ21_j1**         |
| Normalizing Flow | 0.636    | 1.16    | 0.051               | Δη_jj, m_jj, pT_j2               |
| Score (diffusion)| **0.744**| 1.25    | 0.551               | m_jj, Δη_jj                      |

### The comparative finding

The three paradigms succeed (or fail) by keying on **different physics**:

- **Autoencoder** — the 4-dim bottleneck forces it to learn background
  *structure*, so the signal surfaces as anomalous **substructure**: τ₂₁ ranks
  top-2, exactly as `W'→WZ` physics predicts (the bosons give 2-prong jets).
  Strong AUC and by far the **least mass sculpting** — the best *physics* detector.
- **Score / diffusion** — highest raw AUC, but it discriminates via **global
  kinematics (m_jj)** and therefore sculpts the background mass spectrum
  catastrophically (JS 0.55). This is intrinsic to score matching, not a tuning
  miss: a useful score is only learned where the noised background has support,
  which near the off-manifold signal happens only at large σ — a coarse,
  m_jj-dominated regime. High AUC, but it would manufacture a fake bump.
- **Flow** — underperforms both, and its validation AUC *peaks at epoch 1* then
  degrades as the negative log-likelihood improves: a clean demonstration that
  **likelihood ≠ anomaly score** (the better it models the background m_jj tail,
  the more "normal" the resonance looks).

Plots: [feature_distributions](results/feature_distributions.png),
[roc_sic](results/roc_sic.png),
[score_distributions](results/score_distributions.png),
[sculpting](results/sculpting.png).

---

## Setup

Create the dedicated `cern_anom` conda environment (Python 3.11, CUDA torch):

```bash
conda env create -f environment.yml   # creates cern_anom
# or, to install manually into an existing env:
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

```bash
conda activate cern_anom
PY=python
```

Key dependencies: `torch==2.5.1+cu121`, `numpy`, `pyyaml`, `tables` (PyTables),
`fastjet`, `scipy`, `scikit-learn`. `matplotlib` is optional (arrays are always
saved as `.npz`; PNGs render only if it is present).

---

## Data

The R&D dataset is `events_anomalydetection_v2.h5` (~2.9 GB) from
[Zenodo record 6466204](https://zenodo.org/record/6466204): 1M background +
100k signal events (9.09% signal), each an array of up to 700 particle
4-vectors. The 8 high-level dijet features are computed from the raw particles
in this repo (feature engineering is part of the exercise).

### Downloading (gotchas handled)

Zenodo throttles single connections to ~60 KB/s. Use the bundled parallel
downloader (16 ranged connections, ~10× faster, resumable):

```bash
$PY data/parallel_download.py --connections 16
# then ALWAYS verify:
md5sum data/raw/events_anomalydetection_v2.h5
# expected: 629789d55813be3860781b084ae7f1de
```

The file is an old pandas-0.15.2 fixed frame, **blosc-compressed**. Modern
`pandas.read_hdf` *and* `h5py` both fail on it (byte-string metadata / missing
blosc plugin). It is read directly via **PyTables** in
[data/download.py](data/download.py): `/df/block0_values` is `(1.1M, 2101)` —
columns 0–2099 are 700 × (pT, η, φ), column 2100 is the truth label.

### Feature engineering

- [data/jet_features.py](data/jet_features.py) — anti-kT clustering (`fastjet`),
  two leading jets, then (m, pT, η, φ), m_jj, Δη_jj per event.
- [data/nsubjettiness.py](data/nsubjettiness.py) — **N-subjettiness τ_N
  implemented from scratch** (k-means subjet axes), τ21 = τ2/τ1. Verified on real
  data: QCD background peaks high (median 0.56), W/Z signal peaks low (0.29).
- [data/dataset.py](data/dataset.py) — splits, per-feature normalization
  (statistics from background-only training data), optional log-transform of the
  heavy-tailed mass/pT columns for the flow/score models. Training set contains
  **zero** signal.

---

## Running

All commands use the project env (see [Setup](#setup)), run from the repo root:

```bash
conda activate cern_anom
PY=python
```

**Quickstart — the models are already trained; just see the results** (instant,
uses the saved checkpoints in `results/`):

```bash
$PY benchmark.py        # comparison table + permutation feature importance
$PY -m pytest -q        # 11 tests
```

**Re-make the plots** (written to `results/`):

```bash
$PY visualize/feature_plots.py --features results/features.npz --out results/feature_distributions.png
$PY visualize/roc_sic.py
$PY visualize/score_distributions.py
$PY visualize/sculpting.py
```

**Re-train / re-evaluate one model** (~3–20 min each on GPU):

```bash
$PY train.py    --config configs/autoencoder.yaml    # or flow.yaml / score_model.yaml
$PY evaluate.py --config configs/autoencoder.yaml
```

`run_fix.sh` retrains just the flow + score models.

**Everything end-to-end** (assumes the dataset is downloaded & md5-verified):

```bash
bash run_pipeline.sh
```

This runs: inspect → feature extraction (1.1M events) → physics sanity plot →
train + evaluate all three → benchmark table → all plots. Logs to
`results/run_pipeline.log`.

**From a clean machine (no data yet)** — fetch and verify the dataset first:

```bash
$PY data/parallel_download.py --connections 16     # ~2.9 GB, resumable
md5sum data/raw/events_anomalydetection_v2.h5      # must be 629789d55813be3860781b084ae7f1de
bash run_pipeline.sh
```

The two data-building stages can also be run on their own:

```bash
$PY -c "from data.download import inspect; inspect()"            # structure + label balance
$PY extract.py --workers 2 --chunk-size 3000 --out results/features.npz
```

> Memory note: feature extraction OOMs above 2 fastjet workers on a ~8 GB box.
> `run_pipeline.sh` is pinned to `--workers 2 --chunk-size 3000`; chunks are
> checkpointed under `results/feat_chunks/` and the run is resumable. Keep those
> flags if you call `extract.py` by hand.

---

## Models

All three expose `training_loss(x)` and `anomaly_score(x)`, so one
[Trainer](train.py) drives them. Hyperparameters live in [configs/](configs/),
not in the model code.

- [models/autoencoder.py](models/autoencoder.py) — `8→64→32→16→(latent 4)→…→8`
  MLP, MSE reconstruction loss, score = ‖x − x̂‖².
- [models/normalizing_flow.py](models/normalizing_flow.py) — 8 stacked **MADE**
  layers with hand-built autoregressive masks (no `nflows`/`normflows`), random
  permutations between layers; exact log-density, score = −log p(x).
- [models/score_model.py](models/score_model.py) — NCSM score network (plain MLP)
  trained by weighted denoising score matching over a geometric σ ladder. Anomaly
  score = **Tweedie one-step denoising error** averaged over the ladder.

  > Note: the literal "noise to σ_max=3, run Langevin back down, measure
  > ‖x − x_rec‖²" recipe gives AUC ≈ 0.50 — starting at 3σ on unit-variance data
  > discards x and samples a *generic* background point, so the error is pure
  > noise. The Tweedie one-step reconstruction is the same idea made stable. The
  > Langevin sampler is kept in `reconstruct()` for the convergence check.

Checkpoint selection for the flow and score models uses **validation AUC** (the
reported metric), which is far more stable than the mean-separation score whose
tail-sensitivity misranks both.

---

## Evaluation

[evaluate.py](evaluate.py) computes, using labels *only* here (never for scoring):

- **AUC** of the anomaly score.
- **SIC curve** — significance improvement TPR/√FPR; report max.
- **Mass sculpting** — Jensen-Shannon divergence between the background m_jj
  distribution before and after an anomaly cut. A good detector leaves it flat;
  sculpting it into a fake bump is a false discovery.

[benchmark.py](benchmark.py) builds the comparison table and **permutation
feature importance** (AUC drop when each feature is shuffled).

---

## Layout

```
data/        download (PyTables reader) + parallel downloader, jet features,
             from-scratch N-subjettiness, dataset/splits/normalization
models/      autoencoder, MAF (hand-masked MADE), NCSM score net
train.py     unified trainer (AUC / separation / loss early-stopping)
evaluate.py  ROC/AUC, SIC, sculpting JS
benchmark.py comparison table + permutation importance
extract.py   parallel feature extraction over the full dataset (checkpointed)
visualize/   feature distributions, ROC/SIC, score histograms, sculpting
configs/     per-model YAML hyperparameters
tests/       pytest suite (11 tests, fully offline on synthetic data)
run_pipeline.sh  end-to-end driver;  run_fix.sh  retrain flow+score
diag_score.py    per-σ AUC-vs-sculpting diagnostic for the score model
results/     checkpoints, metrics, plots, cached features (gitignored)
```

---

## Tests

```bash
$PY -m pytest -q     # 11 passed
```

Covers τ21 on synthetic jets, model forward-pass shapes and finite losses, that
the flow assigns higher log-density to in-distribution data, and that the score
network behaves correctly under denoising — all offline, no dataset required.

---

## Limitations and next steps

- **The flow misses its 0.68 target** (0.636), and **none of the three meet the
  strict JS < 0.02 sculpting bar.** m_jj is an input feature, so without explicit
  mass-decorrelation the density and score methods key on it and sculpt (the AE
  comes closest at 0.030). A plain unconditional MAF is genuinely limited here.
- **The fix is the same thread:** a **mass-decorrelated, m_jj-conditional density**
  (ANODE / CWoLa-style) should clear the flow's target, and together with explicit
  decorrelation (DisCo / planing) is the route to JS < 0.02 for the density and
  score methods.
