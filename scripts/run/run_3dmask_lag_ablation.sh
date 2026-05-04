#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_3d_mask.py"
mkdir -p logs

COMMON=(
  --input_dir /home/mschae/oraclead_transfer/processed/SWaT
  --entities swat
  --dataset SWaT
  --epochs 80
  --batch 128
  --L 10
  --lag_win 5
  --d 64 --heads 4 --enc_layers 2 --dec_layers 2
  --grad_clip 1.0
  --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  --use_median_vus_window
  --save_per_seed
  --diagnose_components
)

run_seed() {
  local gpu="$1"; local seed="$2"; local tau="$3"; local tag="$4"
  mkdir -p "runs/${tag}/seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -u "$RUNNER" \
    "${COMMON[@]}" \
    --tau_max "$tau" \
    --seeds "$seed" \
    --out_dir "runs/${tag}/seed${seed}" \
    > "logs/${tag}_seed${seed}.log" 2>&1
}

# ============================================================
# tau=1: GPU 0 (seed 0,3 순차) + GPU 1 (seed 1,4) + GPU 2 (seed 2)
# ============================================================
TAG1="ablation_3dmask_tau1"
echo "[$(date)] Starting tau_max=1"

(
  run_seed 0 0 1 "$TAG1"
  run_seed 0 3 1 "$TAG1"
) &

(
  run_seed 1 1 1 "$TAG1"
  run_seed 1 4 1 "$TAG1"
) &

(
  run_seed 2 2 1 "$TAG1"
) &

wait
echo "[$(date)] tau_max=1 done."

# ============================================================
# tau=3: same GPU layout
# ============================================================
TAG3="ablation_3dmask_tau3"
echo "[$(date)] Starting tau_max=3"

(
  run_seed 0 0 3 "$TAG3"
  run_seed 0 3 3 "$TAG3"
) &

(
  run_seed 1 1 3 "$TAG3"
  run_seed 1 4 3 "$TAG3"
) &

(
  run_seed 2 2 3 "$TAG3"
) &

wait
echo "[$(date)] tau_max=3 done."

echo "[$(date)] All lag ablation complete."
