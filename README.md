# OracleAD reproduction & variants (2D/3D SLS)

## What this repo contains
- 2D OracleAD runner (paper-aligned settings)
- 3D SLS variants (research experiments)
- Dataset preprocessing scripts (PSM/SMD)
- TSAD metrics utilities (F1, Range-F1, AUC, VUS)

## Install
pip install -r requirements.txt

## Data
See data/README.md

## Run (example: SMD)
python3 src/runners/oraclead_npz_runner_2d_paper.py \
  --input_dir <processed/SMD> \
  --dataset SMD \
  --~~~_paper.py 가 논문 형식따른 코드

## Multi-GPU (SMD split)
bash scripts/run/run_smd_all_split2gpu.sh residual
