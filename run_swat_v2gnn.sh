#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_causal_v2_gnn.py"

mkdir -p logs runs/swat_v2gnn

COMMON_ARGS=(
  --input_dir /home/mschae/oraclead_transfer/processed/SWaT
  --entities swat
  --dataset SWaT
  --epochs 80
  --batch 128
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4
  --gnn_layers 2
  --grad_clip 1.0
  --use_median_vus_window
  --save_per_seed
  --diagnose_components
  --no_cte
)

run_seed() {
  local gpu="$1"
  local seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -u "$RUNNER" \
    "${COMMON_ARGS[@]}" \
    --seeds "$seed" \
    --out_dir "runs/swat_v2gnn/seed${seed}" \
    > "logs/swat_v2gnn_seed${seed}.log" 2>&1
}

(
  run_seed 0 0
  run_seed 0 3
) &

(
  run_seed 1 1
  run_seed 1 4
) &

(
  run_seed 2 2
) &

wait
echo "All seeds done."
