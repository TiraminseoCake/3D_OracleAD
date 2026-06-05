#!/usr/bin/env bash
set -euo pipefail

PYBIN="${PYBIN:-python}"
RUNNER="src/runners/oraclead_npz_runner_causal_v2_pcmci.py"
mkdir -p logs runs/psm_sweep_final

COMMON=(
  --input_dir /home/mschae/oraclead_transfer/processed/PSM
  --entities PSM
  --dataset PSM
  --epochs 30
  --L 10 --tau_max 5 --lag_win 5
  --d 64 --heads 4
  --grad_clip 1.0
  --prior pcmci --pcmci_alpha 0.05 --pcmci_subsample 10000
  --use_median_vus_window
  --diagnose_components
  --no_calibrate_scores
  --seeds 0
)

run_config() {
  local gpu="$1"; local tag="$2"; shift 2
  local outdir="runs/psm_sweep_final/$tag"
  mkdir -p "$outdir"
  echo "[$(date)] GPU$gpu $tag"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -u "$RUNNER" \
    "${COMMON[@]}" "$@" \
    --out_dir "$outdir" \
    > "logs/psm_sweep_${tag}.log" 2>&1
  grep 'seed 0.*A-PR' "logs/psm_sweep_${tag}.log" 2>/dev/null
}

BATCHES=(256 512 1024 2048)
LRS=("1e-4" "5e-4" "1e-3" "5e-3")

# GPU 2: 절반
(
  for b in 256 512; do
    for lr in "${LRS[@]}"; do
      tag="b${b}_lr${lr}"
      run_config 2 "$tag" --batch $b --lr $lr
    done
  done
) &

# GPU 3: 나머지 절반
(
  for b in 1024 2048; do
    for lr in "${LRS[@]}"; do
      tag="b${b}_lr${lr}"
      run_config 3 "$tag" --batch $b --lr $lr
    done
  done
) &

wait

echo ""
echo "============================================"
echo "PSM FULL SWEEP SUMMARY"
echo "============================================"
printf "%-25s %10s %10s %10s\n" "Config" "A-PR" "F1" "V-PR"
echo "-------------------------------------------------------------"
for b in "${BATCHES[@]}"; do
  for lr in "${LRS[@]}"; do
    tag="b${b}_lr${lr}"
    result=$(grep 'seed 0.*A-PR' "logs/psm_sweep_${tag}.log" 2>/dev/null | tail -1)
    if [ -n "$result" ]; then
      apr=$(echo "$result" | grep -oP 'A-PR=\K[0-9.]+')
      f1=$(echo "$result" | grep -oP 'F1=\K[0-9.]+')
      vpr=$(echo "$result" | grep -oP 'VUS-PR=\K[0-9.]+')
      printf "%-25s %10s %10s %10s\n" "$tag" "$apr" "$f1" "$vpr"
    else
      printf "%-25s %10s\n" "$tag" "(not done)"
    fi
  done
done
