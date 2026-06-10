#!/usr/bin/env bash
# Post-download pipeline: inspect -> extract -> train/eval x3 -> benchmark -> plots.
# Run only after the dataset is fully downloaded. `inspect` parses HDF5 metadata
# (stored at the end of the file), so it fails loudly if the file is incomplete.
set -u
cd "$(dirname "$0")"
PY=/home/aryan/anaconda3/envs/qiskit_clean/bin/python
LOG=results/run_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "[pipeline] $(date) starting"

echo "[pipeline] === inspect (also an integrity check) ==="
$PY -c "from data.download import inspect; inspect()" || { echo "[pipeline] FILE INCOMPLETE/CORRUPT"; exit 1; }

echo "[pipeline] === feature extraction (1.1M events) ==="
# 2 workers / small chunks: this box has ~2.6 GB free and swap is full, so a
# larger pool OOM-kills workers. Chunks are checkpointed, so this is resumable.
$PY extract.py --chunk-size 3000 --workers 2 --out results/features.npz || exit 1

echo "[pipeline] === physics sanity: feature distributions ==="
$PY visualize/feature_plots.py --features results/features.npz \
    --out results/feature_distributions.png || exit 1

for cfg in configs/autoencoder.yaml configs/flow.yaml configs/score_model.yaml; do
  echo "[pipeline] === train $cfg ==="
  $PY train.py --config "$cfg" || exit 1
  echo "[pipeline] === evaluate $cfg ==="
  $PY evaluate.py --config "$cfg" || exit 1
done

echo "[pipeline] === benchmark ==="
$PY benchmark.py || exit 1

echo "[pipeline] === plots ==="
$PY visualize/score_distributions.py || exit 1
$PY visualize/roc_sic.py || exit 1
$PY visualize/sculpting.py || exit 1

echo "[pipeline] $(date) DONE"
