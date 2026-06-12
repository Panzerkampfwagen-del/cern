#!/usr/bin/env bash
# Retrain flow + score with AUC-based checkpoint selection and the fixed
# diffusion anomaly score, then re-evaluate, re-benchmark, re-plot. The
# autoencoder (AUC 0.73) is left as-is; benchmark/plots reuse its results.
set -u
cd "$(dirname "$0")"
PY=/home/aryan/anaconda3/envs/cern_anom/bin/python
LOG=results/run_fix.log
exec > >(tee -a "$LOG") 2>&1

echo "[fix] $(date) start"
for cfg in configs/flow.yaml configs/score_model.yaml; do
  echo "[fix] === train $cfg ==="
  $PY train.py --config "$cfg" || exit 1
  echo "[fix] === eval $cfg ==="
  $PY evaluate.py --config "$cfg" || exit 1
done

echo "[fix] === benchmark ==="
$PY benchmark.py || exit 1
echo "[fix] === plots ==="
$PY visualize/score_distributions.py || exit 1
$PY visualize/roc_sic.py || exit 1
$PY visualize/sculpting.py || exit 1
echo "[fix] $(date) DONE"
