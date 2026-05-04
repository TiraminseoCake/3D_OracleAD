#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_3d_mask.py"

mkdir -p logs

# ============================================================
# Ablation 1: w/o SLS (lam_dev=0) — no structural deviation loss
# ============================================================
ABLATION="ablation_swat_3dmask_no_sls"
mkdir -p "runs/$ABLATION"

NO_SLS_ARGS=(
  --input_dir /home/mschae/oraclead_transfer/processed/SWaT
  --entities swat
  --dataset SWaT
  --epochs 80
  --batch 128
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4 --enc_layers 2 --dec_layers 2
  --grad_clip 1.0
  --lam_recon 0.1
  --lam_dev 0.0
  --lam_sparse 0.05
  --use_median_vus_window
  --save_per_seed
  --diagnose_components
)

run_no_sls() {
  local gpu="$1"; local seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -u "$RUNNER" \
    "${NO_SLS_ARGS[@]}" \
    --seeds "$seed" \
    --out_dir "runs/$ABLATION/seed${seed}" \
    > "logs/${ABLATION}_seed${seed}.log" 2>&1
}

echo "[$(date)] Starting ablation: w/o SLS (lam_dev=0)"

(
  run_no_sls 0 0
  run_no_sls 0 3
) &

(
  run_no_sls 1 1
  run_no_sls 1 4
) &

(
  run_no_sls 2 2
) &

wait
echo "[$(date)] w/o SLS done."

# ============================================================
# Ablation 2: w/o recon (lam_recon=0) — no reconstruction loss
# ============================================================
ABLATION2="ablation_swat_3dmask_no_recon"
mkdir -p "runs/$ABLATION2"

NO_RECON_ARGS=(
  --input_dir /home/mschae/oraclead_transfer/processed/SWaT
  --entities swat
  --dataset SWaT
  --epochs 80
  --batch 128
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4 --enc_layers 2 --dec_layers 2
  --grad_clip 1.0
  --lam_recon 0.0
  --lam_dev 3.0
  --lam_sparse 0.05
  --use_median_vus_window
  --save_per_seed
  --diagnose_components
)

run_no_recon() {
  local gpu="$1"; local seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -u "$RUNNER" \
    "${NO_RECON_ARGS[@]}" \
    --seeds "$seed" \
    --out_dir "runs/$ABLATION2/seed${seed}" \
    > "logs/${ABLATION2}_seed${seed}.log" 2>&1
}

echo "[$(date)] Starting ablation: w/o recon (lam_recon=0)"

(
  run_no_recon 0 0
  run_no_recon 0 3
) &

(
  run_no_recon 1 1
  run_no_recon 1 4
) &

(
  run_no_recon 2 2
) &

wait
echo "[$(date)] w/o recon done."

# ============================================================
# Full model baseline (for fair comparison with same grad_clip)
# ============================================================
ABLATION3="ablation_swat_3dmask_full"
mkdir -p "runs/$ABLATION3"

FULL_ARGS=(
  --input_dir /home/mschae/oraclead_transfer/processed/SWaT
  --entities swat
  --dataset SWaT
  --epochs 80
  --batch 128
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4 --enc_layers 2 --dec_layers 2
  --grad_clip 1.0
  --lam_recon 0.1
  --lam_dev 3.0
  --lam_sparse 0.05
  --use_median_vus_window
  --save_per_seed
  --diagnose_components
)

run_full() {
  local gpu="$1"; local seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -u "$RUNNER" \
    "${FULL_ARGS[@]}" \
    --seeds "$seed" \
    --out_dir "runs/$ABLATION3/seed${seed}" \
    > "logs/${ABLATION3}_seed${seed}.log" 2>&1
}

echo "[$(date)] Starting full model baseline"

(
  run_full 0 0
  run_full 0 3
) &

(
  run_full 1 1
  run_full 1 4
) &

(
  run_full 2 2
) &

wait
echo "[$(date)] Full model done."

echo "[$(date)] All 3 ablation conditions complete."
