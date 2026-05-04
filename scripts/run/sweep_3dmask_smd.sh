#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_3d_mask.py"
GPU=3
OUTBASE="runs/3dmask_smd_sweep"
mkdir -p logs "$OUTBASE"

# 대표 entity 3개: 고(machine-1-1), 중(machine-2-4), 저(machine-1-2)
ENTITIES="machine-1-1,machine-2-4,machine-1-2"

COMMON=(
  --input_dir /home/mschae/oraclead_transfer/processed/SMD
  --entities "$ENTITIES"
  --dataset OTHER
  --epochs 30
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4 --enc_layers 2 --dec_layers 2
  --grad_clip 1.0
  --use_median_vus_window
  --diagnose_components
  --seeds 0
)

run_config() {
  local tag="$1"; shift
  echo "[$(date)] SMD 3dmask $tag"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u "$RUNNER" \
    "${COMMON[@]}" "$@" \
    --out_dir "$OUTBASE/$tag" \
    > "logs/3dmask_smd_sweep_${tag}.log" 2>&1
  echo "[$(date)] Done $tag"
  grep 'seed 0.*A-PR' "logs/3dmask_smd_sweep_${tag}.log" 2>/dev/null
  echo "---"
}

run_config "lr5e4_b512" --lr 5e-4 --batch 512 --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
run_config "lr1e3_b512" --lr 1e-3 --batch 512 --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
run_config "lr1e4_b512" --lr 1e-4 --batch 512 --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
run_config "lr5e4_b256" --lr 5e-4 --batch 256 --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
run_config "lr5e4_b512_lowdev"  --lr 5e-4 --batch 512 --lam_recon 0.1 --lam_dev 1.0 --lam_sparse 0.05
run_config "lr5e4_b512_highdev" --lr 5e-4 --batch 512 --lam_recon 0.1 --lam_dev 5.0 --lam_sparse 0.05

echo ""
echo "=== SMD 3dmask SWEEP SUMMARY ==="
for tag in lr5e4_b512 lr1e3_b512 lr1e4_b512 lr5e4_b256 lr5e4_b512_lowdev lr5e4_b512_highdev; do
  echo "=== $tag ==="
  grep 'seed 0.*A-PR' "logs/3dmask_smd_sweep_${tag}.log" 2>/dev/null || echo "(not done)"
done
