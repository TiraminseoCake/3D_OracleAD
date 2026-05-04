#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_3d_mask.py"
GPU=2

mkdir -p logs runs/3dmask_psm_tuned

COMMON_ARGS=(
  --input_dir /home/mschae/oraclead_transfer/processed/PSM
  --entities PSM
  --dataset PSM
  --epochs 80
  --batch 256
  --lr 5e-4
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4 --enc_layers 2 --dec_layers 2
  --grad_clip 1.0
  --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  --use_median_vus_window
  --save_per_seed
  --diagnose_components
)

for seed in 0 1 2 3 4; do
  echo "[$(date)] Starting 3dmask PSM seed $seed"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u "$RUNNER" \
    "${COMMON_ARGS[@]}" \
    --seeds "$seed" \
    --out_dir "runs/3dmask_psm_tuned/seed${seed}" \
    > "logs/3dmask_psm_tuned_seed${seed}.log" 2>&1
  echo "[$(date)] Done seed $seed"
done
echo "[$(date)] 3dmask PSM tuned all done."
