#!/usr/bin/env bash
set -euo pipefail

RUNNER="oraclead_npz_runner_2d_paper.py"
SMD_DIR="$HOME/datasets/processed/SMD"

# config 선택: base | residual | residual_unique
CFG="${1:-residual}"

OUT_BASE="$HOME/datasets/raw/runs/smd_${CFG}"
OUT0="${OUT_BASE}_gpu0"
OUT1="${OUT_BASE}_gpu1"
mkdir -p "$OUT0" "$OUT1"

# 엔티티 목록 생성
mapfile -t ENTS < <(ls -1 "$SMD_DIR"/*.npz | xargs -n1 basename | sed 's/\.npz$//')

ENTS0=()
ENTS1=()
for i in "${!ENTS[@]}"; do
  if (( i % 2 == 0 )); then ENTS0+=("${ENTS[$i]}"); else ENTS1+=("${ENTS[$i]}"); fi
done

E0=$(IFS=,; echo "${ENTS0[*]}")
E1=$(IFS=,; echo "${ENTS1[*]}")

# 공통 옵션
COMMON=(--input_dir "$SMD_DIR" --dataset SMD)

# config별 옵션
EXTRA=()
case "$CFG" in
  base)
    EXTRA=()
    ;;
  residual)
    EXTRA=(--mhsa_residual)
    ;;
  residual_unique)
    EXTRA=(--mhsa_residual --thr_mode unique --rf1_thr_mode unique)
    ;;
  *)
    echo "Unknown CFG: $CFG (use base|residual|residual_unique)"
    exit 1
    ;;
esac

echo "[INFO] CFG=$CFG"
echo "[INFO] GPU0 entities: ${#ENTS0[@]} | GPU1 entities: ${#ENTS1[@]}"
echo "[INFO] OUT0=$OUT0"
echo "[INFO] OUT1=$OUT1"

# 두 GPU 병렬 실행
CUDA_VISIBLE_DEVICES=0 python3 -u "$RUNNER" "${COMMON[@]}" "${EXTRA[@]}" \
  --entities "$E0" --out_dir "$OUT0" > "${OUT0}/log.txt" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 -u "$RUNNER" "${COMMON[@]}" "${EXTRA[@]}" \
  --entities "$E1" --out_dir "$OUT1" > "${OUT1}/log.txt" 2>&1 &

wait
echo "[DONE] both GPUs finished"

# summary 병합(단순 concat)
python3 - <<PY
import os, pandas as pd
out0="${OUT0}"; out1="${OUT1}"
s0=os.path.join(out0,"summary.csv")
s1=os.path.join(out1,"summary.csv")
dfs=[]
for s in [s0,s1]:
    if os.path.exists(s):
        dfs.append(pd.read_csv(s))
if not dfs:
    raise SystemExit("No summary.csv found")
df=pd.concat(dfs, axis=0, ignore_index=True)
out=os.path.join("${OUT_BASE}_merged_summary.csv")
df.to_csv(out, index=False)
print("Saved merged summary:", out)
print("Rows:", len(df))
PY

