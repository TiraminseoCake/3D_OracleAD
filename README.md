# OracleAD-3D: Multi-Lag Structural Deviation for Time Series Anomaly Detection

## Overview

OracleAD-3D is a multivariate time series anomaly detection model that detects anomalies by combining **prediction error (P)** with **structural deviation (D)** from a learned normal reference (SLS).

### Key Ideas

1. **Multi-Lag Encoding**: Each variable is independently encoded via LSTM at multiple time lags (τ=1~5), capturing temporal dependencies at different delays.
2. **Self-consistent Lag Structure (SLS)**: During training, the model accumulates a reference of normal pairwise embedding distances across variables and lags.
3. **Multiplicative Anomaly Scoring**: `A = P × D` — anomalies require both prediction failure and structural disruption, reducing false positives.

### Architecture

```
Input X [B, L, N]
│
├── τ=1: Local Window → Per-Variable LSTM × N → MHSA → C*_τ=1
├── τ=2: Local Window → Per-Variable LSTM × N → MHSA → C*_τ=2
├── ...
└── τ=5: Local Window → Per-Variable LSTM × N → MHSA → C*_τ=5
                                                          │
                                                    stack → C_all [B, 5, N, d]
                                                          │
                                              ┌───────────┴───────────┐
                                              │                       │
                                        Reconstruction           Prediction
                                        (LSTM Decoder)     (Weighted Aggregation)
                                              │                       │
                                         recon [B,L-1,N]         pred [B,N]

Scoring:
  P = |x_true - pred|          → Prediction Error
  D = ||dist(C_all) - SLS||    → Structural Deviation
  A = P × D                    → Anomaly Score
```

## Project Structure

```
.
├── model/
│   └── oraclead_npz_runner_3d_mask.py    # Main model + training + evaluation
├── metrics/
│   └── paper_eval/                        # Evaluation metrics (AUC-PR, F1, VUS, etc.)
├── scripts/
│   ├── run_swat_3dmask_ablation.sh        # SWaT ablation study
│   ├── run_3dmask_psm_tuned.sh            # PSM with tuned hyperparameters
│   ├── run_3dmask_smd_tuned.sh            # SMD with tuned hyperparameters
│   ├── run_3dmask_lag_ablation.sh         # Multi-lag ablation (τ=1,3,5)
│   ├── sweep_3dmask_psm_msl.sh           # PSM/MSL hyperparameter sweep
│   └── sweep_3dmask_smd.sh               # SMD hyperparameter sweep
├── figures/                               # Visualization results
│   ├── fig_attack_zoomin.png              # Attack response timeline
│   ├── fig_pred_weights.png               # Learned variable dependencies
│   ├── fig_pxd_vs_ppd.png                # P×D vs P+D comparison
│   ├── fig_dependency_network.png         # Top-20 dependency network
│   ├── fig_ablation_comprehensive.png     # Full ablation study
│   ├── fig_sls_necessity.png              # Why SLS matters
│   ├── fig_lag_aware_sls.png              # Per-lag structure analysis
│   ├── sls_heatmap_swat.png              # SLS reference heatmap
│   └── sls_normal_vs_anomaly.png         # Normal vs anomaly comparison
├── results/                               # Experiment results (CSV)
├── requirements.txt
└── README.md
```

## Results

### Main Results (5 seed average, best hyperparameters)

| Dataset | F1 | R-F1 | Aff-F | A-ROC | A-PR | V-ROC | V-PR |
|---------|-----|------|-------|-------|------|-------|------|
| **SWaT** | 79.69 | 29.27 | 78.96 | 87.89 | 78.27 | 83.20 | 67.57 |
| **PSM** | 56.58 | 50.90 | 77.25 | 75.12 | 53.41 | 69.27 | 46.89 |
| **SMD** | 48.59 | 31.35 | 78.02 | 82.86 | 44.26 | 83.86 | 38.78 |

### Ablation Study (SWaT, 5 seed average)

**Score Combination:**

| Condition | F1 | A-PR | V-PR |
|-----------|-----|------|------|
| P only | 80.90 | 80.30 | 76.99 |
| D only | 52.51 | 46.40 | 36.99 |
| P + D | 78.26 | 74.64 | 56.46 |
| **P × D** | **79.69** | **78.27** | **67.57** |

**Multi-Lag:**

| τ_max | F1 | A-PR | V-PR |
|-------|-----|------|------|
| 1 | 78.86 | 76.34 | 65.16 |
| 3 | 79.67 | 77.94 | 67.51 |
| **5** | **79.69** | **78.27** | **67.57** |

## Quick Start

### Data Format

Input `.npz` files should contain:
- `train`: Training data `[T_train, N]` (normal data only)
- `test`: Test data `[T_test, N]` (normal + anomaly)
- `label`: Binary labels `[T_test]` (0=normal, 1=anomaly)

### Training & Evaluation

```bash
python model/oraclead_npz_runner_3d_mask.py \
  --input_dir /path/to/data \
  --entities entity_name \
  --dataset SWaT \
  --epochs 80 \
  --batch 128 \
  --L 10 \
  --tau_max 5 \
  --lag_win 5 \
  --d 64 \
  --heads 4 \
  --enc_layers 2 \
  --dec_layers 2 \
  --grad_clip 1.0 \
  --lam_recon 0.1 \
  --lam_dev 3.0 \
  --lam_sparse 0.05 \
  --use_median_vus_window \
  --save_per_seed \
  --diagnose_components \
  --out_dir runs/output
```

### Optimal Hyperparameters per Dataset

| Dataset | lr | batch | lam_dev |
|---------|-----|-------|---------|
| SWaT | 5e-4 | 128 | 3.0 |
| PSM | 5e-4 | 256 | 3.0 |
| SMD | 5e-4 | 256 | 3.0 |

## Requirements

```
torch>=1.9
numpy
scikit-learn
pandas
matplotlib (optional, for visualization)
tensorboard or tensorboardX (optional, for logging)
```
