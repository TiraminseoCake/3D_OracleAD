#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_3d_mask.py"
mkdir -p logs

# ============================================================
# PSM sweep on GPU 2
# ============================================================
sweep_psm() {
  local GPU=2
  local OUTBASE="runs/3dmask_psm_sweep"
  mkdir -p "$OUTBASE"

  COMMON=(
    --input_dir /home/mschae/oraclead_transfer/processed/PSM
    --entities PSM
    --dataset PSM
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
    echo "[$(date)] PSM $tag"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u "$RUNNER" \
      "${COMMON[@]}" "$@" \
      --out_dir "$OUTBASE/$tag" \
      > "logs/3dmask_psm_sweep_${tag}.log" 2>&1
    grep 'seed 0.*A-PR' "logs/3dmask_psm_sweep_${tag}.log" 2>/dev/null
  }

  run_config "lr5e5_b1024" --lr 5e-5 --batch 1024 --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  run_config "lr5e4_b1024" --lr 5e-4 --batch 1024 --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  run_config "lr1e3_b1024" --lr 1e-3 --batch 1024 --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  run_config "lr5e4_b256"  --lr 5e-4 --batch 256  --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  run_config "lr5e4_b1024_lowdev" --lr 5e-4 --batch 1024 --lam_recon 0.1 --lam_dev 1.0 --lam_sparse 0.05
  run_config "lr5e4_b1024_highdev" --lr 5e-4 --batch 1024 --lam_recon 0.1 --lam_dev 5.0 --lam_sparse 0.05

  echo ""
  echo "=== PSM 3dmask SWEEP SUMMARY ==="
  for tag in lr5e5_b1024 lr5e4_b1024 lr1e3_b1024 lr5e4_b256 lr5e4_b1024_lowdev lr5e4_b1024_highdev; do
    printf "%-30s " "$tag"
    grep 'seed 0.*A-PR' "logs/3dmask_psm_sweep_${tag}.log" 2>/dev/null || echo "(not done)"
  done
}

# ============================================================
# MSL sweep on GPU 3
# ============================================================
sweep_msl() {
  local GPU=3
  local OUTBASE="runs/3dmask_msl_sweep"
  mkdir -p "$OUTBASE"

  COMMON=(
    --input_dir /home/mschae/oraclead_transfer/processed/MSL
    --entities msl
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
    echo "[$(date)] MSL $tag"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -u "$RUNNER" \
      "${COMMON[@]}" "$@" \
      --out_dir "$OUTBASE/$tag" \
      > "logs/3dmask_msl_sweep_${tag}.log" 2>&1
    grep 'seed 0.*A-PR' "logs/3dmask_msl_sweep_${tag}.log" 2>/dev/null
  }

  run_config "lr5e4_b512"  --lr 5e-4 --batch 512  --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  run_config "lr1e3_b512"  --lr 1e-3 --batch 512  --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  run_config "lr1e4_b512"  --lr 1e-4 --batch 512  --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  run_config "lr5e4_b256"  --lr 5e-4 --batch 256  --lam_recon 0.1 --lam_dev 3.0 --lam_sparse 0.05
  run_config "lr5e4_b512_lowdev"  --lr 5e-4 --batch 512 --lam_recon 0.1 --lam_dev 1.0 --lam_sparse 0.05
  run_config "lr5e4_b512_highdev" --lr 5e-4 --batch 512 --lam_recon 0.1 --lam_dev 5.0 --lam_sparse 0.05

  echo ""
  echo "=== MSL 3dmask SWEEP SUMMARY ==="
  for tag in lr5e4_b512 lr1e3_b512 lr1e4_b512 lr5e4_b256 lr5e4_b512_lowdev lr5e4_b512_highdev; do
    printf "%-30s " "$tag"
    grep 'seed 0.*A-PR' "logs/3dmask_msl_sweep_${tag}.log" 2>/dev/null || echo "(not done)"
  done
}

# 병렬 실행
sweep_psm &
sweep_msl &
wait

echo "[$(date)] All 3dmask sweeps done."
