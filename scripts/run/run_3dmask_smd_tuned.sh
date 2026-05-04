#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_3d_mask.py"
GPU=3

mkdir -p logs runs/3dmask_smd_tuned

# SMD 28 entities, lr=1e-4, batch=512
# 1 seed first (28 entities), fast enough on GPU 3

COMMON_ARGS=(
  --input_dir /home/mschae/oraclead_transfer/processed/SMD
  --dataset OTHER
  --epochs 80
  --batch 512
  --lr 1e-4
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4 --enc_layers 2 --dec_layers 2
  --grad_clip 1.0
  --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  --use_median_vus_window
  --diagnose_components
  --seeds 0
  --out_dir runs/3dmask_smd_tuned
)

echo "[$(date)] Starting 3dmask SMD 28 entities (seed 0)"

CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u "$RUNNER" \
  "${COMMON_ARGS[@]}" \
  > "logs/3dmask_smd_tuned.log" 2>&1

echo "[$(date)] 3dmask SMD tuned done."
