# -*- coding: utf-8 -*-
"""
oraclead_npz_runner_causal_v2_gnn.py

Causal improvements over oraclead_npz_runner_causalgraph.py:

  [FIX 1] JSD (Jensen-Shannon Divergence) replaces symmetric KL.
          Bounded in [0, log2], true metric, well-behaved when supports differ.

  [FIX 2] Acyclicity loss REMOVED.
          For lag>=1 time series, temporal ordering already guarantees DAG.
          The NOTEARS loss on tau=1 was nearly always 0 due to softmax-normalized
          pred_weights, adding cost without benefit.

  [FIX 3] Two-stage Conditional TE prior.
          Stage 1: unconditional TE → rough directed graph.
          Stage 2: identify confounders as common parents (z->src AND z->tgt)
          via the rough graph. Mediators (src->m->tgt) are excluded because
          TE(src->z) is high for mediators, not for confounders.
          Also uses reduced bins for z (num_bins//2) to mitigate B^4 sparsity.

  [FIX 4] Learnable causal MHSA mask.
          Replaces frozen TE gate mask with nn.Parameter initialized from TE gate.
          Model can refine attention structure as it learns better causal edges.
          Warmup ramp prevents mask from dominating early training.

  [FIX 5] Interaction safety:
          - CTE prior errors don't permanently corrupt attention (mask is learnable)
          - Warmup ensures model learns basic representations before mask activates
          - has_* flags stored as register_buffer (survives save/load)
          - Intervention sampling uses np.random.Generator (reproducible)
"""
import argparse
import glob
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# TensorBoard writer (fallback to tensorboardX)
# ============================================================
SummaryWriter = None
_TB_BACKEND = None
try:
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
    SummaryWriter = _TorchSummaryWriter
    _TB_BACKEND = "torch.utils.tensorboard"
except Exception:
    try:
        from tensorboardX import SummaryWriter as _XSummaryWriter
        SummaryWriter = _XSummaryWriter
        _TB_BACKEND = "tensorboardX"
    except Exception:
        SummaryWriter = None
        _TB_BACKEND = None

plt = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from metrics.paper_eval.metrics_api import get_metrics as paper_get_metrics


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# General utils
# ============================================================
def standardize_train_test(train: np.ndarray, test: np.ndarray):
    train = train.astype(np.float32)
    test = test.astype(np.float32)

    train = np.where(np.isfinite(train), train, np.nan)
    test = np.where(np.isfinite(test), test, np.nan)

    col_mean = np.nanmean(train, axis=0, keepdims=True).astype(np.float32)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)

    train = np.where(np.isnan(train), col_mean, train).astype(np.float32)
    test = np.where(np.isnan(test), col_mean, test).astype(np.float32)

    mu = train.mean(axis=0, keepdims=True).astype(np.float32)
    var = ((train - mu) ** 2).mean(axis=0, keepdims=True).astype(np.float32)
    sd = np.sqrt(var).astype(np.float32)
    sd = np.where(sd == 0.0, 1.0, sd).astype(np.float32)

    train_z = (train - mu) / sd
    test_z = (test - mu) / sd
    return train_z.astype(np.float32), test_z.astype(np.float32), mu, sd


def reduce_label(y, T):
    y = np.asarray(y)
    if y.ndim == 2:
        y = (y.sum(axis=1) > 0).astype(np.int32)
    else:
        y = y.astype(np.int32)
    if len(y) != T:
        raise ValueError(f"label length mismatch: {len(y)} != {T}")
    return y


def anomaly_segments(y01: np.ndarray):
    y01 = np.asarray(y01).astype(np.int32)
    segs = []
    in_seg = False
    s = 0
    for i, v in enumerate(y01):
        if v == 1 and not in_seg:
            s = i
            in_seg = True
        elif v == 0 and in_seg:
            segs.append((s, i - 1))
            in_seg = False
    if in_seg:
        segs.append((s, len(y01) - 1))
    return segs


def get_median_anomaly_length(y01: np.ndarray):
    segs = anomaly_segments(y01)
    if len(segs) == 0:
        return 100
    lens = [e - s + 1 for s, e in segs]
    med = int(np.median(lens))
    return max(med, 1)


def pct(x):
    return (float(x) * 100.0) if np.isfinite(x) else float("nan")


def safe_mean_std(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def robust_loc_scale(arr, eps: float = 1e-6):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.std(arr))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return med, scale


def robust_zscore(arr, center, scale, clip_min=0.0):
    z = (np.asarray(arr, dtype=np.float64) - float(center)) / max(float(scale), 1e-6)
    if clip_min is not None:
        z = np.maximum(z, float(clip_min))
    return z.astype(np.float32)


def make_pseudo_env_ids(num_windows: int, num_envs: int):
    num_envs = max(int(num_envs), 1)
    if num_envs == 1 or num_windows <= 1:
        return np.zeros((num_windows,), dtype=np.int64)
    idx = np.arange(num_windows, dtype=np.int64)
    env = np.floor(idx * num_envs / max(num_windows, 1)).astype(np.int64)
    env = np.clip(env, 0, num_envs - 1)
    return env


def normalize_vector_torch(x: torch.Tensor, eps: float = 1e-12):
    x = x.clamp_min(0.0)
    s = x.sum()
    if torch.isfinite(s) and float(s.detach().cpu()) > eps:
        return x / (s + eps)
    return torch.full_like(x, 1.0 / float(max(x.numel(), 1)))


def make_self_causal_fallback_torch(tau_max: int, N: int, device, dtype):
    out = torch.zeros(tau_max, N, N, device=device, dtype=dtype)
    diag = torch.arange(N, device=device)
    out[0, diag, diag] = 1.0
    return out


def normalize_causal_tensor_torch(x: torch.Tensor, eps: float = 1e-12):
    """
    Normalize nonnegative causal tensor over (tau, source) for each target.
    Supports [tau, src, tgt] and [B, tau, src, tgt].
    """
    if x.dim() == 3:
        tau_max, N, _ = x.shape
        flat = x.clamp_min(0.0).reshape(tau_max * N, N)
        colsum = flat.sum(dim=0, keepdim=True)

        fallback = torch.zeros_like(flat)
        diag = torch.arange(N, device=flat.device)
        fallback[diag, diag] = 1.0

        flat = torch.where(colsum > eps, flat / colsum.clamp_min(eps), fallback)
        return flat.view(tau_max, N, N)

    if x.dim() == 4:
        B, tau_max, N, _ = x.shape
        flat = x.clamp_min(0.0).reshape(B, tau_max * N, N)
        colsum = flat.sum(dim=1, keepdim=True)

        fallback = torch.zeros_like(flat)
        diag = torch.arange(N, device=flat.device)
        fallback[:, diag, diag] = 1.0

        flat = torch.where(colsum > eps, flat / colsum.clamp_min(eps), fallback)
        return flat.view(B, tau_max, N, N)

    raise ValueError(f"normalize_causal_tensor_torch expects 3D or 4D tensor, got {x.dim()}D.")


def fit_score_calibrator(train_scores: dict):
    out = {}
    for key in ["P_raw", "C_raw", "G_raw"]:
        center, scale = robust_loc_scale(train_scores[key])
        out[key] = {"center": center, "scale": scale}
    return out


def apply_score_calibrator(raw_scores: dict, calibrator: dict, clip_min: float = 0.0,
                           alpha: float = 1.0, beta: float = 1.0):
    Pn = robust_zscore(raw_scores["P_raw"], calibrator["P_raw"]["center"], calibrator["P_raw"]["scale"], clip_min)
    Cn = robust_zscore(raw_scores["C_raw"], calibrator["C_raw"]["center"], calibrator["C_raw"]["scale"], clip_min)
    Gn = robust_zscore(raw_scores["G_raw"], calibrator["G_raw"]["center"], calibrator["G_raw"]["scale"], clip_min)
    S = (float(alpha) * Cn + float(beta) * Gn).astype(np.float32)
    A = (Pn * S).astype(np.float32)
    return {"P": Pn, "C": Cn, "G": Gn, "S": S, "A": A}


def score_components_to_timeline(comp_dict, Tt, start):
    out = {}
    for k, v in comp_dict.items():
        arr = np.full((Tt,), np.nan, dtype=np.float32)
        arr[start:] = np.asarray(v, dtype=np.float32)
        out[k + "_t"] = arr
    return out


def prediction_train_loss(x_true, pred, loss_type: str = "l1"):
    diff = x_true - pred
    if loss_type == "l2root":
        return diff.pow(2).sum(dim=-1).sqrt().mean()
    return diff.abs().mean()


def reconstruction_train_loss(x_true, recon, loss_type: str = "l1"):
    diff = x_true - recon
    if loss_type == "l2root":
        return diff.pow(2).sum(dim=(1, 2)).sqrt().mean()
    return diff.abs().mean()


# ============================================================
# TE causal prior (original, unconditional)
# ============================================================
def _fit_equal_width_bins(series_TN: np.ndarray, num_bins: int):
    T, N = series_TN.shape
    edges = []
    for i in range(N):
        col = series_TN[:, i].astype(np.float64)
        col = col[np.isfinite(col)]
        if col.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(col.min()), float(col.max())
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
        edges.append(np.linspace(lo, hi, int(num_bins) + 1)[1:-1])
    return edges


def _digitize_with_edges(series_TN: np.ndarray, edges):
    T, N = series_TN.shape
    out = np.zeros((T, N), dtype=np.int64)
    for i, e in enumerate(edges):
        out[:, i] = np.digitize(series_TN[:, i], e, right=False)
    return out


def _transfer_entropy_discrete_1lag(x_disc: np.ndarray,
                                    y_disc: np.ndarray,
                                    tau: int,
                                    num_bins: int,
                                    eps: float = 1e-12) -> float:
    tau = int(tau)
    B = int(num_bins)
    T = len(x_disc)

    t0 = max(tau, 1)
    if T - t0 <= 1:
        return 0.0

    y_t = y_disc[t0:]
    y_prev = y_disc[t0 - 1:T - 1]
    x_prev = x_disc[t0 - tau:T - tau]

    M = int(y_t.shape[0])
    if M <= 1:
        return 0.0

    xyz_code = (y_t * B + y_prev) * B + x_prev
    yz_code = y_prev * B + x_prev
    yy_code = y_t * B + y_prev
    y_code = y_prev

    c_xyz = np.bincount(xyz_code, minlength=B * B * B).astype(np.float64)
    c_yz = np.bincount(yz_code, minlength=B * B).astype(np.float64)
    c_yy = np.bincount(yy_code, minlength=B * B).astype(np.float64)
    c_y = np.bincount(y_code, minlength=B).astype(np.float64)

    nz = np.flatnonzero(c_xyz > 0)
    if nz.size == 0:
        return 0.0

    yt = nz // (B * B)
    rem = nz % (B * B)
    yp = rem // B
    xp = rem % B

    num = c_xyz[nz] * c_y[yp]
    den = c_yz[yp * B + xp] * c_yy[yt * B + yp]

    te_nat = np.sum((c_xyz[nz] / float(M)) * np.log((num + eps) / (den + eps)))
    te_bits = te_nat / np.log(2.0)
    return float(max(te_bits, 0.0))


# ============================================================
# [FIX 3] Two-stage Conditional TE prior
#
# Problem with v1: correlation-based confounder selection confuses
# mediators (X->M->Y) with confounders (Z->X, Z->Y).
# Conditioning on a mediator blocks the real causal path.
#
# Solution: 2-stage approach
#   Stage 1: Compute unconditional TE for all pairs → rough graph
#   Stage 2: For each (src, tgt), identify candidate confounders
#            as variables z with high TE *to both* src and tgt
#            (i.e. z is a common parent, not a descendant of src).
#            A mediator has high TE *from* src, so it's excluded.
#   Also: reduce bins for CTE (B_cond < B) to mitigate B^4 sparsity.
# ============================================================
def _conditional_te_discrete_1lag(x_disc: np.ndarray,
                                   y_disc: np.ndarray,
                                   z_disc: np.ndarray,
                                   tau: int,
                                   num_bins_xy: int,
                                   num_bins_z: int,
                                   eps: float = 1e-12) -> float:
    """
    CTE_{x->y|z}(tau).
    Uses separate bin counts for (x,y) vs z to reduce sparsity:
      x,y use num_bins_xy bins; z uses num_bins_z bins (typically smaller).
    Total bin combinations: Bxy^2 * Bz * Bxy = Bxy^3 * Bz (not Bxy^4).
    """
    tau = int(tau)
    Bxy = int(num_bins_xy)
    Bz = int(num_bins_z)
    T = len(x_disc)
    t0 = max(tau, 1)
    if T - t0 <= 1:
        return 0.0

    y_t    = y_disc[t0:]
    y_prev = y_disc[t0 - 1:T - 1]
    x_prev = x_disc[t0 - tau:T - tau]
    # re-bin z to fewer bins to reduce sparsity
    z_prev = np.clip(z_disc[t0 - 1:T - 1] * Bz // max(Bxy, 1), 0, Bz - 1).astype(np.int64)

    M = int(y_t.shape[0])
    if M <= 1:
        return 0.0

    # joint codes with mixed bin sizes
    cond_code    = y_prev * Bz + z_prev                                  # (y_{t-1}, z_{t-1})
    full_code    = ((y_t * Bxy + y_prev) * Bz + z_prev) * Bxy + x_prev  # (y_t, y_{t-1}, z_{t-1}, x_{t-tau})
    cond_x_code  = cond_code * Bxy + x_prev                             # (y_{t-1}, z_{t-1}, x_{t-tau})
    yt_cond_code = (y_t * Bxy + y_prev) * Bz + z_prev                   # (y_t, y_{t-1}, z_{t-1})

    S_full    = Bxy * Bxy * Bz * Bxy
    S_cond    = Bxy * Bz
    S_cond_x  = Bxy * Bz * Bxy
    S_yt_cond = Bxy * Bxy * Bz

    c_full    = np.bincount(full_code,    minlength=S_full).astype(np.float64)
    c_cond    = np.bincount(cond_code,    minlength=S_cond).astype(np.float64)
    c_cond_x  = np.bincount(cond_x_code,  minlength=S_cond_x).astype(np.float64)
    c_yt_cond = np.bincount(yt_cond_code, minlength=S_yt_cond).astype(np.float64)

    nz = np.flatnonzero(c_full > 0)
    if nz.size == 0:
        return 0.0

    # decode indices
    rem = nz.copy()
    yt_idx = rem // (Bxy * Bz * Bxy); rem = rem % (Bxy * Bz * Bxy)
    yp_idx = rem // (Bz * Bxy);       rem = rem % (Bz * Bxy)
    zp_idx = rem // Bxy
    xp_idx = rem % Bxy

    cond_idx    = yp_idx * Bz + zp_idx
    cond_x_idx  = cond_idx * Bxy + xp_idx
    yt_cond_idx = (yt_idx * Bxy + yp_idx) * Bz + zp_idx

    num = c_full[nz] * c_cond[cond_idx]
    den = c_cond_x[cond_x_idx] * c_yt_cond[yt_cond_idx]

    cte_nat = np.sum((c_full[nz] / float(M)) * np.log((num + eps) / (den + eps)))
    cte_bits = cte_nat / np.log(2.0)
    return float(max(cte_bits, 0.0))


def _find_confounders_from_rough_graph(rough_te: np.ndarray, src: int, tgt: int,
                                        confounder_thresh: float = 0.1) -> list:
    """
    Identify candidate confounders for (src->tgt) using a rough TE graph.

    A confounder z has:  z->src (high TE) AND z->tgt (high TE)
    A mediator m has:    src->m (high TE) AND m->tgt (high TE)

    We select z where TE(z->src) and TE(z->tgt) are both above threshold,
    but TE(src->z) is NOT high (excludes mediators/descendants).

    rough_te: [N, N] where rough_te[i, j] = max over tau of TE(i->j)
    Returns list of z indices (can be empty).
    """
    N = rough_te.shape[0]
    if N < 3:
        return []

    candidates = []
    for z in range(N):
        if z == src or z == tgt:
            continue
        # z -> src and z -> tgt must be strong (common parent pattern)
        te_z_to_src = rough_te[z, src]
        te_z_to_tgt = rough_te[z, tgt]
        # src -> z must be weak (exclude mediators/descendants)
        te_src_to_z = rough_te[src, z]

        if (te_z_to_src > confounder_thresh and
            te_z_to_tgt > confounder_thresh and
            te_src_to_z < te_z_to_src * 0.5):  # z causes src, not the reverse
            score = te_z_to_src + te_z_to_tgt  # confounding strength
            candidates.append((z, score))

    # return top-1 confounder (strongest common parent)
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [candidates[0][0]]


def build_cte_causal_prior(train_TN: np.ndarray,
                           tau_max: int,
                           num_bins: int = 8,
                           num_chunks: int = 32,
                           chunk_len: int = 256,
                           threshold: float = 0.0,
                           self_mass: float = 0.25,
                           seed: int = 0):
    """
    [FIX 3] Two-stage Conditional TE prior.

    Stage 1: Compute unconditional TE to get rough directed graph.
    Stage 2: For each (src, tgt) pair, use the rough graph to identify
             true confounders (common parents) vs mediators, then
             compute CTE conditioning only on confounders.

    Key improvements over v1:
    - Confounder selection uses causal structure, not correlation
    - Reduced z-bins (num_bins//2) to mitigate B^4 sparsity
    - Falls back to unconditional TE when no confounder is found

    Returns:
        te_weight: [tau_max, N, N]
        te_gate:   [tau_max, N, N]
    """
    train_TN = np.asarray(train_TN, dtype=np.float32)
    T, N = train_TN.shape
    tau_max = int(tau_max)
    num_bins_z = max(num_bins // 2, 2)  # fewer bins for conditioning var

    if T < tau_max + 3:
        raise ValueError(f"train length too short for tau_max={tau_max}: T={T}")

    edges = _fit_equal_width_bins(train_TN, num_bins=num_bins)
    disc_all = _digitize_with_edges(train_TN, edges)

    min_len = max(tau_max + 3, 16)
    if T <= max(int(chunk_len), min_len):
        starts = [0]
        actual_len = T
    else:
        rng = np.random.default_rng(seed)
        actual_len = max(min(int(chunk_len), T), min_len)
        max_start = T - actual_len
        starts = rng.integers(0, max_start + 1, size=max(int(num_chunks), 1)).tolist()

    # ---- Stage 1: unconditional TE -> rough graph ----
    te_stage1 = np.zeros((tau_max, N, N), dtype=np.float64)
    used = 0
    for s in starts:
        seg = disc_all[s:s + actual_len]
        if seg.shape[0] < min_len:
            continue
        used += 1
        for tau in range(1, tau_max + 1):
            for src in range(N):
                x = seg[:, src]
                for tgt in range(N):
                    if src == tgt:
                        continue
                    y = seg[:, tgt]
                    te_stage1[tau - 1, src, tgt] += _transfer_entropy_discrete_1lag(
                        x, y, tau=tau, num_bins=num_bins
                    )
    if used == 0:
        raise RuntimeError("No valid chunks for CTE stage-1.")
    te_stage1 /= float(used)

    # rough graph: max TE over lags for each (src, tgt)
    rough_te_max = te_stage1.max(axis=0)  # [N, N]
    # adaptive threshold: median of nonzero values
    nz_vals = rough_te_max[rough_te_max > 0]
    conf_thresh = float(np.median(nz_vals)) if nz_vals.size > 0 else 0.1

    print(f"  [CTE] stage-1 done: rough graph density="
          f"{(rough_te_max > conf_thresh).sum()}/{N*N}, "
          f"confounder_thresh={conf_thresh:.4f}", flush=True)

    # ---- Stage 2: CTE conditioning on identified confounders ----
    # pre-compute confounder map
    confounder_map = {}
    for src in range(N):
        for tgt in range(N):
            if src == tgt:
                continue
            confounder_map[(src, tgt)] = _find_confounders_from_rough_graph(
                rough_te_max, src, tgt, confounder_thresh=conf_thresh
            )

    n_conditioned = sum(1 for v in confounder_map.values() if len(v) > 0)
    print(f"  [CTE] confounder map: {n_conditioned}/{len(confounder_map)} pairs have confounders",
          flush=True)

    te_acc = np.zeros((tau_max, N, N), dtype=np.float64)
    used2 = 0
    for s in starts:
        seg = disc_all[s:s + actual_len]
        if seg.shape[0] < min_len:
            continue
        used2 += 1
        for tau in range(1, tau_max + 1):
            for src in range(N):
                x = seg[:, src]
                for tgt in range(N):
                    if src == tgt:
                        continue
                    y = seg[:, tgt]
                    confounders = confounder_map[(src, tgt)]

                    if confounders:
                        z_idx = confounders[0]
                        z = seg[:, z_idx]
                        val = _conditional_te_discrete_1lag(
                            x, y, z, tau=tau,
                            num_bins_xy=num_bins,
                            num_bins_z=num_bins_z,
                        )
                    else:
                        # no confounder identified → use unconditional TE
                        val = _transfer_entropy_discrete_1lag(
                            x, y, tau=tau, num_bins=num_bins
                        )
                    te_acc[tau - 1, src, tgt] += val

    if used2 == 0:
        raise RuntimeError("No valid chunks for CTE stage-2.")

    te_raw = te_acc / float(used2)
    te_raw[te_raw < float(threshold)] = 0.0

    diag = np.arange(N)
    te_raw[0, diag, diag] = np.maximum(te_raw[0, diag, diag], float(self_mass))

    flat = te_raw.reshape(tau_max * N, N)
    colsum = flat.sum(axis=0, keepdims=True)

    fallback = np.zeros_like(flat)
    fallback[diag, diag] = 1.0

    flat = np.where(colsum > 1e-12, flat / np.clip(colsum, 1e-12, None), fallback)
    te_weight = flat.reshape(tau_max, N, N).astype(np.float32)

    nz = te_raw[te_raw > 0]
    scale = float(np.quantile(nz, 0.75)) if nz.size > 0 else 1.0
    te_gate = np.clip(te_raw / max(scale, 1e-12), 0.0, 1.0).astype(np.float32)
    te_gate[0, diag, diag] = 1.0

    return te_weight, te_gate


# Keep original build_te_causal_prior for --no_cte fallback
def build_te_causal_prior(train_TN: np.ndarray,
                          tau_max: int,
                          num_bins: int = 8,
                          num_chunks: int = 32,
                          chunk_len: int = 256,
                          threshold: float = 0.0,
                          self_mass: float = 0.25,
                          seed: int = 0):
    train_TN = np.asarray(train_TN, dtype=np.float32)
    T, N = train_TN.shape
    tau_max = int(tau_max)

    if T < tau_max + 3:
        raise ValueError(f"train length too short for tau_max={tau_max}: T={T}")

    edges = _fit_equal_width_bins(train_TN, num_bins=num_bins)
    disc_all = _digitize_with_edges(train_TN, edges)

    te_acc = np.zeros((tau_max, N, N), dtype=np.float64)

    min_len = max(tau_max + 3, 16)
    if T <= max(int(chunk_len), min_len):
        starts = [0]
        actual_len = T
    else:
        rng = np.random.default_rng(seed)
        actual_len = max(min(int(chunk_len), T), min_len)
        max_start = T - actual_len
        starts = rng.integers(0, max_start + 1, size=max(int(num_chunks), 1)).tolist()

    used = 0
    for s in starts:
        seg = disc_all[s:s + actual_len]
        if seg.shape[0] < min_len:
            continue
        used += 1
        for tau in range(1, tau_max + 1):
            for src in range(N):
                x = seg[:, src]
                for tgt in range(N):
                    if src == tgt:
                        continue
                    y = seg[:, tgt]
                    te_acc[tau - 1, src, tgt] += _transfer_entropy_discrete_1lag(
                        x, y, tau=tau, num_bins=num_bins
                    )

    if used == 0:
        raise RuntimeError("No valid chunks were available for TE prior estimation.")

    te_raw = te_acc / float(used)
    te_raw[te_raw < float(threshold)] = 0.0

    diag = np.arange(N)
    te_raw[0, diag, diag] = np.maximum(te_raw[0, diag, diag], float(self_mass))

    flat = te_raw.reshape(tau_max * N, N)
    colsum = flat.sum(axis=0, keepdims=True)
    fallback = np.zeros_like(flat)
    fallback[diag, diag] = 1.0
    flat = np.where(colsum > 1e-12, flat / np.clip(colsum, 1e-12, None), fallback)
    te_weight = flat.reshape(tau_max, N, N).astype(np.float32)

    nz = te_raw[te_raw > 0]
    scale = float(np.quantile(nz, 0.75)) if nz.size > 0 else 1.0
    te_gate = np.clip(te_raw / max(scale, 1e-12), 0.0, 1.0).astype(np.float32)
    te_gate[0, diag, diag] = 1.0

    return te_weight, te_gate


# ============================================================
# Dataset
# ============================================================
class SlidingWindowDataset(Dataset):
    def __init__(self, series_TN: np.ndarray, L: int, env_ids=None, return_env: bool = False):
        self.x = series_TN.astype(np.float32)
        self.L = int(L)
        self.T, self.N = self.x.shape
        self.return_env = bool(return_env)
        if self.T < self.L:
            raise ValueError(f"T={self.T} < L={self.L}")
        self.W = self.T - self.L + 1
        if env_ids is None:
            self.env_ids = np.zeros((self.W,), dtype=np.int64)
        else:
            env_ids = np.asarray(env_ids, dtype=np.int64)
            if len(env_ids) != self.W:
                raise ValueError(f"env_ids length mismatch: {len(env_ids)} != {self.W}")
            self.env_ids = env_ids

    def __len__(self):
        return self.W

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx:idx + self.L])
        if self.return_env:
            return x, int(self.env_ids[idx])
        return x


# ============================================================
# Model blocks
# ============================================================
class TemporalAttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1, bias=True)

    def forward(self, H):
        a = torch.softmax(self.score(H).squeeze(-1), dim=1)
        return (H * a.unsqueeze(-1)).sum(dim=1)


class PerVarEncoder(nn.Module):
    def __init__(self, d: int, num_layers: int, dropout: float):
        super().__init__()
        do = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(1, d, batch_first=True, num_layers=num_layers, dropout=do)
        self.pool = TemporalAttnPool(d)

    def forward(self, x):
        H, _ = self.lstm(x)
        return self.pool(H)


class PerVarReconDecoder(nn.Module):
    def __init__(self, d: int, L: int, num_layers: int, dropout: float):
        super().__init__()
        self.out_len = L - 1
        self.d = d
        self.num_layers = num_layers
        do = dropout if num_layers > 1 else 0.0

        self.init_h = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, num_layers * d),
        )
        self.init_c = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, num_layers * d),
        )

        self.lstm = nn.LSTM(1, d, batch_first=True, num_layers=num_layers, dropout=do)
        self.out = nn.Linear(d, 1)

    def forward(self, c):
        B, d = c.shape
        z = torch.zeros(B, self.out_len, 1, device=c.device, dtype=c.dtype)
        h0 = torch.tanh(self.init_h(c)).view(self.num_layers, B, d).contiguous()
        c0 = torch.tanh(self.init_c(c)).view(self.num_layers, B, d).contiguous()
        Y, _ = self.lstm(z, (h0, c0))
        O = self.out(Y).squeeze(-1)
        return O


# [V2+GNN] Causal GNN Layer — message passing along causal edges
class CausalGNNLayer(nn.Module):
    def __init__(self, d, num_layers=2, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'msg_fn': nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)),
                'upd_fn': nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)),
                'norm': nn.LayerNorm(d),
            }))
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, adj, mask_var=None):
        # H: [B, N, d], adj: [N, N]
        # mask_var: block outgoing messages from this variable during intervention
        if mask_var is not None:
            adj = adj.clone()
            adj[int(mask_var), :] = 0.0
        for layer in self.layers:
            msg = layer['msg_fn'](H)
            agg = torch.einsum('bsd,st->btd', msg, adj)
            H = H + self.dropout(layer['upd_fn'](agg))
            H = layer['norm'](H)
        return H


def pairwise_sq_l2(C):
    A2 = (C * C).sum(dim=2)
    G = torch.bmm(C, C.transpose(1, 2))
    D = A2.unsqueeze(2) + A2.unsqueeze(1) - 2.0 * G
    return torch.clamp(D, min=0.0)


def pairwise_l2(C):
    return (pairwise_sq_l2(C) + 1e-12).sqrt()


def logit_from_prob(p: float):
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


# ============================================================
# Main model
# ============================================================
class OracleAD3DCausalEffect(nn.Module):
    def __init__(self, N: int, L: int, tau_max: int, d: int, heads: int,
                 enc_layers: int, dec_layers: int, dropout: float,
                 mhsa_residual: bool = False,
                 lag_fusion: str = "mean",
                 lag_win: int = 5,
                 pred_temp: float = 1.0,
                 self_loop_bias: float = 1.0,
                 lag_source_topk: int = 0,
                 dynamic_graph: bool = True,
                 graph_hidden: int = 16,
                 gate_init: float = 0.15,
                 te_prior_blend: float = 0.35,
                 # [FIX 4] learnable causal attention mask
                 causal_attn_mask_scale: float = 0.5,
                 causal_mask_warmup_epochs: int = 5,
                 # [V2+GNN] GNN layers
                 gnn_layers: int = 2):
        super().__init__()
        self.N = N
        self.L = L
        self.tau_max = tau_max
        self.d = d
        self.lag_fusion = lag_fusion
        self.lag_win = int(lag_win)
        self.pred_temp = float(pred_temp)
        self.self_loop_bias = float(self_loop_bias)
        self.lag_source_topk = int(lag_source_topk)
        self.dynamic_graph = bool(dynamic_graph)
        self.graph_hidden = int(graph_hidden)
        self.te_prior_blend = float(te_prior_blend)
        self.causal_attn_mask_scale = float(causal_attn_mask_scale)
        self.causal_mask_warmup_epochs = int(causal_mask_warmup_epochs)
        self._current_epoch = 0  # set by training loop

        if tau_max >= L:
            raise ValueError(f"tau_max={tau_max} must be < L={L}")
        if self.lag_win <= 0:
            raise ValueError(f"lag_win must be >= 1, got {self.lag_win}")

        self.encoders = nn.ModuleList([PerVarEncoder(d, enc_layers, dropout) for _ in range(N)])
        self.gnn = CausalGNNLayer(d, gnn_layers, dropout)  # [V2+GNN]
        self.mhsa = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.recon_decoders = nn.ModuleList([PerVarReconDecoder(d, L, dec_layers, dropout) for _ in range(N)])
        self.mhsa_residual = mhsa_residual

        self.pred_logits = nn.Parameter(torch.zeros(tau_max, N, N))
        with torch.no_grad():
            diag_idx = torch.arange(N)
            self.pred_logits[:, diag_idx, diag_idx] += self.self_loop_bias

        self.edge_log_alpha = nn.Parameter(torch.full((tau_max, N, N), logit_from_prob(gate_init)))

        if self.dynamic_graph:
            self.dynamic_q = nn.Linear(d, self.graph_hidden, bias=False)
            self.dynamic_k = nn.Linear(d, self.graph_hidden, bias=False)
            self.dynamic_scale = 1.0 / math.sqrt(max(self.graph_hidden, 1))

        self.edge_value_head = nn.Linear(d, N, bias=True)
        self.pred_bias = nn.Parameter(torch.zeros(N))

        # persistent references (not saved in state_dict flags — see register_buffer for tensors)
        self.register_buffer("cls_ref", torch.zeros(tau_max, N, N), persistent=True)
        self.register_buffer("w_ref", torch.zeros(tau_max, N, N), persistent=True)
        self.register_buffer("_has_cls_ref", torch.zeros(1, dtype=torch.bool), persistent=True)
        self.register_buffer("_has_w_ref", torch.zeros(1, dtype=torch.bool), persistent=True)

        self.register_buffer("te_prior_weight", torch.zeros(tau_max, N, N), persistent=True)
        self.register_buffer("te_prior_gate", torch.zeros(tau_max, N, N), persistent=True)
        self.register_buffer("_has_te_prior", torch.zeros(1, dtype=torch.bool), persistent=True)

        # [FIX 4] Learnable causal attention mask logits: [tau_max, N, N]
        # Initialized to zeros (uniform attention). set_te_prior() initializes
        # from TE gate. Then the model learns to refine the mask during training.
        self.causal_mask_logits = nn.Parameter(torch.zeros(tau_max, N, N))

    # Properties so callers can use model.has_cls_ref as before
    @property
    def has_cls_ref(self):
        return bool(self._has_cls_ref.item())

    @has_cls_ref.setter
    def has_cls_ref(self, v):
        self._has_cls_ref.fill_(int(bool(v)))

    @property
    def has_w_ref(self):
        return bool(self._has_w_ref.item())

    @has_w_ref.setter
    def has_w_ref(self, v):
        self._has_w_ref.fill_(int(bool(v)))

    @property
    def has_te_prior(self):
        return bool(self._has_te_prior.item())

    @has_te_prior.setter
    def has_te_prior(self, v):
        self._has_te_prior.fill_(int(bool(v)))

    def reset_refs(self):
        self.w_ref.zero_()
        self.has_w_ref = False

        if self.has_te_prior:
            self.cls_ref.copy_(self.te_prior_weight)
            self.has_cls_ref = True
        else:
            self.cls_ref.zero_()
            self.has_cls_ref = False

    def edge_gate(self):
        return torch.sigmoid(self.edge_log_alpha)

    def gate_sparsity(self):
        return self.edge_gate().mean()

    def lag_monotonic_penalty(self):
        if self.tau_max <= 1:
            return torch.tensor(0.0, device=self.edge_log_alpha.device)
        gate = self.edge_gate()
        return F.relu(gate[1:] - gate[:-1]).mean()

    @torch.no_grad()
    def set_te_prior(self, te_weight: torch.Tensor, te_gate: torch.Tensor = None, init_scale: float = 0.25):
        te_weight = te_weight.to(device=self.pred_logits.device, dtype=self.pred_logits.dtype).clamp_min(0.0)

        if te_gate is None:
            te_gate = (te_weight > 0).to(dtype=self.pred_logits.dtype)
        else:
            te_gate = te_gate.to(device=self.pred_logits.device, dtype=self.pred_logits.dtype).clamp(0.0, 1.0)

        te_weight = normalize_causal_tensor_torch(te_weight)
        diag = torch.arange(self.N, device=te_weight.device)
        te_gate[0, diag, diag] = 1.0

        self.te_prior_weight.copy_(te_weight)
        self.te_prior_gate.copy_(te_gate)
        self.has_te_prior = True

        self.cls_ref.copy_(te_weight)
        self.has_cls_ref = True

        if init_scale > 0.0:
            prior_score = torch.log(te_weight.clamp_min(1e-8))
            g = te_gate.clamp(1e-4, 1.0 - 1e-4)
            prior_alpha = torch.log(g / (1.0 - g))
            self.pred_logits.add_(float(init_scale) * prior_score)
            self.edge_log_alpha.add_(0.5 * float(init_scale) * prior_alpha)

        # [FIX 4] Initialize learnable causal mask from TE gate logits
        g = te_gate.clamp(1e-4, 1.0 - 1e-4)
        self.causal_mask_logits.data.copy_(torch.log(g / (1.0 - g)) * 0.5)

    def _effective_gate(self):
        gate = self.edge_gate()
        if not self.has_te_prior:
            return gate
        return gate * (0.05 + 0.95 * self.te_prior_gate)

    def _compute_dynamic_delta(self, C_all):
        if not self.dynamic_graph:
            return None
        q = self.dynamic_q(C_all)
        k = self.dynamic_k(C_all)
        delta = torch.einsum("btsh,btih->btsi", q, k) * self.dynamic_scale
        return delta

    def _normalize_weight_tensor(self, score, gate):
        tau_max, N, _ = self.pred_logits.shape
        temp = max(self.pred_temp, 1e-6)

        if score.dim() == 3:
            flat_s = (score / temp).reshape(tau_max * N, N)
            flat_g = gate.reshape(tau_max * N, N)
            flat_s = flat_s - flat_s.max(dim=0, keepdim=True).values
            unnorm = torch.exp(flat_s) * flat_g
            if self.lag_source_topk > 0:
                k = min(self.lag_source_topk, tau_max * N)
                _, idx = torch.topk(unnorm, k=k, dim=0)
                mask = torch.zeros_like(unnorm)
                mask.scatter_(0, idx, 1.0)
                unnorm = unnorm * mask
            weights = unnorm / (unnorm.sum(dim=0, keepdim=True) + 1e-12)
            return weights.view(tau_max, N, N)

        if score.dim() == 4:
            B = score.shape[0]
            flat_s = (score / temp).reshape(B, tau_max * N, N)
            flat_g = gate.reshape(1, tau_max * N, N)
            flat_s = flat_s - flat_s.max(dim=1, keepdim=True).values
            unnorm = torch.exp(flat_s) * flat_g
            if self.lag_source_topk > 0:
                k = min(self.lag_source_topk, tau_max * N)
                _, idx = torch.topk(unnorm, k=k, dim=1)
                mask = torch.zeros_like(unnorm)
                mask.scatter_(1, idx, 1.0)
                unnorm = unnorm * mask
            weights = unnorm / (unnorm.sum(dim=1, keepdim=True) + 1e-12)
            return weights.view(B, tau_max, N, N)

        raise ValueError(f"score dim must be 3 or 4, got {score.dim()}")

    def get_pred_weights(self, local_delta=None):
        gate = self._effective_gate()

        if self.has_te_prior and self.te_prior_blend > 0.0:
            prior_bias = self.te_prior_blend * torch.log(self.te_prior_weight.clamp_min(1e-8))
        else:
            prior_bias = 0.0

        if local_delta is None:
            score = self.pred_logits + prior_bias
        else:
            if torch.is_tensor(prior_bias):
                score = self.pred_logits.unsqueeze(0) + prior_bias.unsqueeze(0) + local_delta
            else:
                score = self.pred_logits.unsqueeze(0) + local_delta
        return self._normalize_weight_tensor(score, gate)

    def pred_weight_entropy(self, weights=None):
        if weights is None:
            weights = self.get_pred_weights()
        tau_max, N, _ = self.pred_logits.shape
        denom = max(math.log(max(tau_max * N, 2)), 1e-6)
        if weights.dim() == 3:
            flat = weights.reshape(tau_max * N, N)
            return -(flat * torch.log(flat + 1e-12)).sum(dim=0).mean() / denom
        if weights.dim() == 4:
            flat = weights.reshape(weights.shape[0], tau_max * N, N)
            return -(flat * torch.log(flat + 1e-12)).sum(dim=1).mean() / denom
        raise ValueError(f"weights dim must be 3 or 4, got {weights.dim()}")

    # --------------------------------------------------------
    # [FIX 1] JSD (Jensen-Shannon Divergence)
    # Unlike symmetric KL which is unbounded, JSD is a true metric
    # (bounded in [0, log2]), well-behaved when p and q have
    # different supports. JSD = 0.5*KL(p||m) + 0.5*KL(q||m), m=(p+q)/2
    # --------------------------------------------------------
    def causal_prior_losses(self, pred_weights=None):
        zero = torch.tensor(0.0, device=self.pred_logits.device)
        if not self.has_te_prior:
            return zero, zero

        if pred_weights is None:
            pred_weights = self.get_pred_weights()

        if pred_weights.dim() == 4:
            w_mean = pred_weights.mean(dim=0)
        else:
            w_mean = pred_weights

        p = w_mean.reshape(self.tau_max * self.N, self.N).clamp_min(1e-8)
        q = self.te_prior_weight.reshape(self.tau_max * self.N, self.N).clamp_min(1e-8)

        # JSD: bounded, symmetric, true metric
        m = 0.5 * (p + q)
        loss_te_w = 0.5 * (F.kl_div(m.log(), p, reduction="batchmean")
                         + F.kl_div(m.log(), q, reduction="batchmean"))

        loss_te_g = F.binary_cross_entropy(
            self.edge_gate().clamp(1e-4, 1.0 - 1e-4),
            self.te_prior_gate.clamp(1e-4, 1.0 - 1e-4),
        )
        return loss_te_w, loss_te_g

    # --------------------------------------------------------
    # [FIX 4] Learnable causal attention mask
    #
    # v1 problems fixed:
    #   1. Frozen prior → now learnable (nn.Parameter, initialized from TE gate)
    #   2. Scale too aggressive → warmup ramp (0→scale over warmup_epochs)
    #   3. Bad prior contaminates everything → learnable logits can diverge
    #      from prior as model learns better structure
    #
    # mask[tgt, src] = warmup_scale * sigmoid(causal_mask_logits[tau, src, tgt])
    # sigmoid output in (0,1) → converted to log-domain additive bias.
    # --------------------------------------------------------
    def _causal_attn_mask(self, tau: int) -> torch.Tensor | None:
        if self.causal_attn_mask_scale <= 0.0:
            return None

        # warmup: ramp from 0 to full scale over warmup epochs
        if self.causal_mask_warmup_epochs > 0 and self._current_epoch > 0:
            ramp = min(float(self._current_epoch) / float(self.causal_mask_warmup_epochs), 1.0)
        else:
            ramp = 0.0  # epoch 0 = no mask yet (let model learn basic representations first)

        if ramp <= 0.0:
            return None

        # learnable soft gate for this lag
        soft_gate = torch.sigmoid(self.causal_mask_logits[tau - 1])  # [src, tgt] in (0,1)
        # log-domain bias: mask[tgt, src] for PyTorch MHA convention
        log_bias = torch.log(soft_gate.clamp_min(1e-6)).T  # [tgt, src]
        return (self.causal_attn_mask_scale * ramp) * log_bias

    def forward(self, X, mask_tau=None, mask_var=None, mask_fill_value=0.0):
        B, L, N = X.shape
        lag_embeds = []

        for tau in range(1, self.tau_max + 1):
            end = L - tau
            start = max(0, end - self.lag_win)

            c_list = []
            for i in range(N):
                x_i = X[:, start:end, i].unsqueeze(-1)
                if (mask_tau is not None) and (mask_var is not None):
                    if (tau == int(mask_tau)) and (i == int(mask_var)):
                        x_i = torch.full_like(x_i, float(mask_fill_value))
                ci = self.encoders[i](x_i)
                c_list.append(ci)

            C_tau = torch.stack(c_list, dim=1)  # [B, src, d]

            # [V2+GNN] GNN message passing along causal graph
            adj_tau = self._effective_gate()[tau - 1]  # [N, N]
            gnn_mask = int(mask_var) if (mask_tau is not None and mask_var is not None and tau == int(mask_tau)) else None
            C_tau = self.gnn(C_tau, adj_tau, mask_var=gnn_mask)

            # [FIX 4] apply causal mask to MHSA
            causal_mask = self._causal_attn_mask(tau)
            A_tau, _ = self.mhsa(C_tau, C_tau, C_tau,
                                  attn_mask=causal_mask,
                                  need_weights=False)
            C_star_tau = (C_tau + A_tau) if self.mhsa_residual else A_tau
            lag_embeds.append(C_star_tau)

        C_all = torch.stack(lag_embeds, dim=1)  # [B, tau, src, d]

        if self.lag_fusion == "max":
            C_recon = C_all.max(dim=1).values
        else:
            C_recon = C_all.mean(dim=1)

        recon_list = []
        for i in range(N):
            recon_list.append(self.recon_decoders[i](C_recon[:, i, :]))
        recon = torch.stack(recon_list, dim=-1)

        local_delta = self._compute_dynamic_delta(C_all)
        pred_weights = self.get_pred_weights(local_delta=local_delta)

        edge_value = self.edge_value_head(C_all)
        edge_effect = edge_value * pred_weights
        edge_strength = pred_weights * edge_value.abs()
        pred = edge_effect.sum(dim=(1, 2)) + self.pred_bias

        return recon, pred, C_all, pred_weights, edge_value, edge_effect, edge_strength, self.edge_gate(), local_delta


# ============================================================
# TensorBoard helpers
# ============================================================
def tb_log_score_histograms(writer, prefix, step, labels, score_t_dict, start_idx):
    if writer is None:
        return
    valid = np.isfinite(score_t_dict["A_t"][start_idx:])
    yv = labels[start_idx:][valid]
    for key in ["P_t", "C_t", "G_t", "S_t", "A_t"]:
        sv = score_t_dict[key][start_idx:][valid]
        if len(sv) > 0:
            writer.add_histogram(f"{prefix}/scores/{key}_all", sv, step)
        if (yv == 1).sum() > 0:
            writer.add_histogram(f"{prefix}/scores/{key}_anom", sv[yv == 1], step)
        if (yv == 0).sum() > 0:
            writer.add_histogram(f"{prefix}/scores/{key}_norm", sv[yv == 0], step)


def tb_log_score_curves(writer, prefix, step, labels, score_t_dict, max_points=2000):
    if writer is None or plt is None:
        return
    T = len(labels)
    idx = np.arange(T)
    sel = np.linspace(0, T - 1, max_points).astype(int) if T > max_points else idx

    fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(sel, labels[sel], linewidth=1.0)
    axes[0].set_ylabel("label")
    for ax, key in zip(axes[1:], ["P_t", "C_t", "G_t", "S_t", "A_t"]):
        ax.plot(sel, score_t_dict[key][sel], linewidth=1.0)
        ax.set_ylabel(key[:-2])
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    try:
        writer.add_figure(f"{prefix}/figures/score_curves", fig, global_step=step)
    except Exception as e:
        print(f"[warn] tb figure logging failed for {prefix} step {step}: {e}", flush=True)
    finally:
        plt.close(fig)


# ============================================================
# Scoring
# ============================================================
def prediction_score(err: torch.Tensor, args):
    if args.p_agg == "mean":
        return err.mean(dim=1)
    if args.p_agg == "max":
        return err.max(dim=1).values
    k = min(int(args.p_topk), err.shape[1])
    return err.topk(k, dim=1).values.mean(dim=1)


def matrix_deviation_per_tau(diff: torch.Tensor, agg: str = "fro", topk: int = 3):
    if agg == "fro":
        return torch.sqrt(diff.pow(2).mean(dim=(2, 3)) + 1e-12)
    row_dev = diff.abs().mean(dim=3)
    if agg == "maxrow":
        return row_dev.max(dim=2).values
    k = min(int(topk), row_dev.shape[2])
    return row_dev.topk(k, dim=2).values.mean(dim=2)


def lag_aggregate(per_tau: torch.Tensor, mode: str = "mean"):
    if mode == "max":
        return per_tau.max(dim=1).values
    return per_tau.mean(dim=1)


@torch.no_grad()
def score_windows_raw(model, series_TN, device, batch, args):
    model.eval()
    ds = SlidingWindowDataset(series_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    W = len(ds)
    P_w = np.zeros((W,), dtype=np.float32)
    C_w = np.zeros((W,), dtype=np.float32)
    G_w = np.zeros((W,), dtype=np.float32)

    cls_ref = model.cls_ref.detach()
    w_ref = model.w_ref.detach()
    offset = 0

    for X in loader:
        X = X.to(device)
        recon, pred, C_all, pred_weights, edge_value, edge_effect, edge_strength, gate, local_delta = model(X)

        x_true_next = X[:, -1, :]
        err = (x_true_next - pred).abs()
        P = prediction_score(err, args)

        if model.has_cls_ref:
            cls_cur = normalize_causal_tensor_torch(edge_strength)
            cdiff = cls_cur - cls_ref.unsqueeze(0)
            C_per_tau = matrix_deviation_per_tau(cdiff, agg=args.c_agg, topk=args.c_topk)
            Cscore = lag_aggregate(C_per_tau, mode=args.causal_lag_agg)
        else:
            Cscore = torch.zeros_like(P)

        if pred_weights.dim() == 3:
            pred_weights_b = pred_weights.unsqueeze(0).expand(X.shape[0], -1, -1, -1)
        else:
            pred_weights_b = pred_weights

        if model.has_w_ref:
            gdiff = pred_weights_b - w_ref.unsqueeze(0)
            G_per_tau = matrix_deviation_per_tau(gdiff, agg=args.g_agg, topk=args.g_topk)
            Gscore = lag_aggregate(G_per_tau, mode=args.graph_lag_agg)
        else:
            Gscore = torch.zeros_like(P)

        bsz = X.shape[0]
        P_w[offset:offset + bsz] = P.detach().cpu().numpy().astype(np.float32)
        C_w[offset:offset + bsz] = Cscore.detach().cpu().numpy().astype(np.float32)
        G_w[offset:offset + bsz] = Gscore.detach().cpu().numpy().astype(np.float32)
        offset += bsz

    return {"P_raw": P_w, "C_raw": C_w, "G_raw": G_w}


def score_windows(model, series_TN, device, batch, args, calibrator=None):
    raw = score_windows_raw(model, series_TN, device, batch, args)
    if calibrator is not None:
        cal = apply_score_calibrator(
            raw,
            calibrator,
            clip_min=args.calib_clip_min,
            alpha=args.score_alpha,
            beta=args.score_beta,
        )
    else:
        cal = {
            "P": raw["P_raw"].astype(np.float32),
            "C": raw["C_raw"].astype(np.float32),
            "G": raw["G_raw"].astype(np.float32),
        }
        cal["S"] = (args.score_alpha * cal["C"] + args.score_beta * cal["G"]).astype(np.float32)
        cal["A"] = (cal["P"] * cal["S"]).astype(np.float32)

    out = {}
    out.update(raw)
    out.update(cal)
    return out


# ============================================================
# Intervention utilities
# ============================================================
def intervene_local_window(X: torch.Tensor, tau: int, src: int, lag_win: int,
                           mode: str = "permute", fill_value: float = 0.0):
    B, L, N = X.shape
    tau = int(tau)
    src = int(src)
    end = L - tau
    start = max(0, end - int(lag_win))
    if start >= end:
        return X.clone()

    Xp = X.clone()
    if mode == "permute":
        if B <= 1:
            return Xp
        rng_idx = torch.randperm(B, device=X.device)
        Xp[:, start:end, src] = X[rng_idx, start:end, src]
    elif mode == "fill":
        Xp[:, start:end, src] = float(fill_value)
    else:
        raise ValueError(f"Unknown intervention mode: {mode}")
    return Xp


def sample_intervention_pairs(tau_max: int, N: int, num_pairs: int, rng: np.random.Generator):
    pairs = []
    for _ in range(max(int(num_pairs), 0)):
        tau = int(rng.integers(1, tau_max + 1))
        src = int(rng.integers(0, N))
        pairs.append((tau, src))
    return pairs


def permutation_alignment_and_epoch_cls(model, X, x_true_next, base_abs_err_ref, edge_strength, args, rng):
    zero = torch.tensor(0.0, device=X.device)
    cls_sum = torch.zeros(model.tau_max, model.N, model.N, device=X.device)
    cls_cnt = torch.zeros(model.tau_max, model.N, 1, device=X.device)

    if args.perm_pairs_per_batch <= 0 or X.shape[0] <= 1:
        return zero, cls_sum, cls_cnt

    pairs = sample_intervention_pairs(model.tau_max, model.N, args.perm_pairs_per_batch, rng)
    losses = []

    edge_strength_mean = edge_strength.mean(dim=0)

    for tau, src in pairs:
        if args.perm_mode == "fill":
            _, pred_perm, _, _, _, _, _, _, _ = model(
                X, mask_tau=tau, mask_var=src, mask_fill_value=args.mask_fill_value,
            )
        else:
            Xp = intervene_local_window(
                X, tau=tau, src=src, lag_win=model.lag_win,
                mode=args.perm_mode, fill_value=args.mask_fill_value,
            )
            # Pass mask info so GNN blocks outgoing messages from intervened variable
            _, pred_perm, _, _, _, _, _, _, _ = model(Xp, mask_tau=tau, mask_var=src)

        delta_pos = torch.clamp((x_true_next - pred_perm).abs() - base_abs_err_ref, min=0.0).mean(dim=0)

        cls_sum[tau - 1, src, :] += delta_pos.detach()
        cls_cnt[tau - 1, src, 0] += 1.0

        delta_sum = float(delta_pos.sum().detach().cpu())
        if not np.isfinite(delta_sum) or delta_sum <= 1e-12:
            continue

        cur = edge_strength_mean[tau - 1, src, :]
        losses.append(F.mse_loss(normalize_vector_torch(cur), normalize_vector_torch(delta_pos)))

    if len(losses) == 0:
        return zero, cls_sum, cls_cnt
    return torch.stack(losses).mean(), cls_sum, cls_cnt


def invariance_loss_from_tensor(tensor4d, env_ids):
    if tensor4d.dim() != 4:
        return torch.tensor(0.0, device=tensor4d.device)
    env_ids = env_ids.to(tensor4d.device)
    uniq = torch.unique(env_ids)
    if len(uniq) <= 1:
        return torch.tensor(0.0, device=tensor4d.device)
    env_means = []
    for e in uniq:
        m = (env_ids == e)
        if m.any():
            env_means.append(tensor4d[m].mean(dim=0))
    if len(env_means) <= 1:
        return torch.tensor(0.0, device=tensor4d.device)
    E = torch.stack(env_means, dim=0)
    return ((E - E.mean(dim=0, keepdim=True)) ** 2).mean()


def graph_stability_loss(pred_weights, w_ref):
    if pred_weights.dim() == 3:
        diff = pred_weights - w_ref
    else:
        diff = pred_weights - w_ref.unsqueeze(0)
    return diff.pow(2).mean()


def causal_structure_loss(edge_strength, cls_ref):
    if edge_strength.dim() == 4:
        cur = edge_strength.mean(dim=0)
    else:
        cur = edge_strength

    p = normalize_causal_tensor_torch(cur).reshape(cur.shape[0] * cur.shape[1], cur.shape[2]).clamp_min(1e-8)
    q = normalize_causal_tensor_torch(cls_ref).reshape(cls_ref.shape[0] * cls_ref.shape[1], cls_ref.shape[2]).clamp_min(1e-8)
    return F.kl_div(p.log(), q, reduction="batchmean")


# ============================================================
# Train
# ============================================================
def train_one_seed(model, train_TN, device,
                   epochs, batch, lr, weight_decay,
                   lam_task=1.0,
                   lam_causal=1.0,
                   lam_graphreg=0.05,
                   lam_robust=0.10,
                   cls_ema=0.9,
                   wref_ema=0.9,
                   start_cls_epoch=5,
                   start_wref_epoch=3,
                   grad_clip=0.0,
                   train_loss_type="l1",
                   recon_loss_type="l1",
                   writer=None, writer_prefix="",
                   args=None):
    model.reset_refs()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    W = train_TN.shape[0] - model.L + 1
    env_ids = make_pseudo_env_ids(W, args.num_envs)
    ds = SlidingWindowDataset(train_TN, model.L, env_ids=env_ids, return_env=True)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False, num_workers=0)

    Tlag = model.tau_max
    N = model.N

    W_RECON   = 0.05
    W_TE_GATE = 0.50
    W_GRAPH   = 0.50
    W_LAGMONO = 0.50
    W_INV     = 0.50

    # [FIX 3] use seeded rng for reproducible intervention sampling
    perm_rng = np.random.default_rng(args.seeds[0] if hasattr(args, "seeds") else 0)

    for ep in range(1, epochs + 1):
        model.train()
        model._current_epoch = ep  # [FIX 4] inform mask warmup

        w_sum = torch.zeros(Tlag, N, N, device=device)
        w_cnt = 0

        cls_sum = torch.zeros(Tlag, N, N, device=device)
        cls_cnt = torch.zeros(Tlag, N, 1, device=device)

        stats = {
            "task": 0.0, "causal": 0.0, "graphreg": 0.0, "robust": 0.0, "total": 0.0,
            "pred": 0.0, "recon": 0.0, "tew": 0.0, "teg": 0.0,
            "cstruct": 0.0, "graph": 0.0, "gate": 0.0, "lagmono": 0.0,
            "perm": 0.0, "inv": 0.0,
        }
        steps = 0
        last_use_cstruct = False
        last_use_graph_loss = False

        for X, env in loader:
            X = X.to(device)
            env = torch.as_tensor(env, device=device, dtype=torch.long)

            recon, pred, C_all, pred_weights, edge_value, edge_effect, edge_strength, gate, local_delta = model(X)
            xL = X[:, -1, :]
            xpast = X[:, :model.L - 1, :]

            loss_pred = prediction_train_loss(xL, pred, loss_type=train_loss_type)
            loss_recon = reconstruction_train_loss(xpast, recon, loss_type=recon_loss_type)

            if pred_weights.dim() == 4:
                w_epoch_mean = pred_weights.mean(dim=0).detach()
            else:
                w_epoch_mean = pred_weights.detach()
            w_sum += w_epoch_mean
            w_cnt += 1

            use_cstruct_loss = (ep >= start_cls_epoch) and model.has_cls_ref
            use_graph_loss = (ep >= start_wref_epoch) and model.has_w_ref
            last_use_cstruct = use_cstruct_loss
            last_use_graph_loss = use_graph_loss

            loss_cstruct = causal_structure_loss(edge_strength, model.cls_ref) if use_cstruct_loss \
                else torch.tensor(0.0, device=device)

            loss_graph = graph_stability_loss(pred_weights, model.w_ref) if use_graph_loss \
                else torch.tensor(0.0, device=device)

            loss_gate = model.gate_sparsity()
            loss_lagmono = model.lag_monotonic_penalty()
            loss_te_w, loss_te_g = model.causal_prior_losses(pred_weights)

            base_abs_err_ref = (xL - pred).abs().detach()
            loss_perm, batch_cls_sum, batch_cls_cnt = permutation_alignment_and_epoch_cls(
                model, X, xL, base_abs_err_ref, edge_strength, args, perm_rng
            )
            cls_sum += batch_cls_sum
            cls_cnt += batch_cls_cnt

            loss_inv = invariance_loss_from_tensor(edge_strength, env)

            group_task = loss_pred + W_RECON * loss_recon

            group_causal = loss_te_w + W_TE_GATE * loss_te_g
            if use_cstruct_loss:
                group_causal = group_causal + loss_cstruct

            group_graphreg = loss_gate + W_LAGMONO * loss_lagmono
            if use_graph_loss:
                group_graphreg = group_graphreg + W_GRAPH * loss_graph

            group_robust = loss_perm + W_INV * loss_inv

            loss = (
                lam_task    * group_task
                + lam_causal  * group_causal
                + lam_graphreg * group_graphreg
                + lam_robust  * group_robust
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            stats["task"]     += float(group_task.detach().cpu())
            stats["causal"]   += float(group_causal.detach().cpu())
            stats["graphreg"] += float(group_graphreg.detach().cpu())
            stats["robust"]   += float(group_robust.detach().cpu())
            stats["total"]    += float(loss.detach().cpu())
            stats["pred"]     += float(loss_pred.detach().cpu())
            stats["recon"]    += float(loss_recon.detach().cpu())
            stats["tew"]      += float(loss_te_w.detach().cpu())
            stats["teg"]      += float(loss_te_g.detach().cpu())
            stats["cstruct"]  += float(loss_cstruct.detach().cpu())
            stats["graph"]    += float(loss_graph.detach().cpu())
            stats["gate"]     += float(loss_gate.detach().cpu())
            stats["lagmono"]  += float(loss_lagmono.detach().cpu())
            stats["perm"]     += float(loss_perm.detach().cpu())
            stats["inv"]      += float(loss_inv.detach().cpu())
            steps += 1

        with torch.no_grad():
            epoch_w = w_sum / max(w_cnt, 1)
            if not model.has_w_ref:
                model.w_ref.copy_(epoch_w)
                model.has_w_ref = True
            else:
                if wref_ema <= 0.0:
                    model.w_ref.copy_(epoch_w)
                else:
                    beta = float(wref_ema)
                    model.w_ref.mul_(beta).add_(epoch_w * (1.0 - beta))

            if model.has_cls_ref:
                fallback_cls = model.cls_ref
            elif model.has_te_prior:
                fallback_cls = model.te_prior_weight
            else:
                fallback_cls = make_self_causal_fallback_torch(Tlag, N, device=device, dtype=cls_sum.dtype)

            epoch_cls_raw = torch.where(
                cls_cnt > 0,
                cls_sum / cls_cnt.clamp_min(1.0),
                fallback_cls
            )
            epoch_cls = normalize_causal_tensor_torch(epoch_cls_raw)

            if not model.has_cls_ref:
                model.cls_ref.copy_(epoch_cls)
                model.has_cls_ref = True
            else:
                if cls_ema <= 0.0:
                    model.cls_ref.copy_(epoch_cls)
                else:
                    beta = float(cls_ema)
                    model.cls_ref.mul_(beta).add_(epoch_cls * (1.0 - beta))
                    model.cls_ref.copy_(normalize_causal_tensor_torch(model.cls_ref))

        avg = {k: v / max(steps, 1) for k, v in stats.items()}
        print(
            f"  [ep {ep:02d}] "
            f"task={avg['task']:.6f} causal={avg['causal']:.6f} "
            f"graphreg={avg['graphreg']:.6f} robust={avg['robust']:.6f} "
            f"total={avg['total']:.6f} | "
            f"pred={avg['pred']:.6f} recon={avg['recon']:.6f} "
            f"tew={avg['tew']:.6f} teg={avg['teg']:.6f} "
            f"cstruct={avg['cstruct']:.6f} graph={avg['graph']:.6f} "
            f"gate={avg['gate']:.6f} lagmono={avg['lagmono']:.6f} "
            f"perm={avg['perm']:.6f} inv={avg['inv']:.6f} "
            f"has_cls={model.has_cls_ref} use_cstruct={last_use_cstruct} "
            f"has_wref={model.has_w_ref} use_graph={last_use_graph_loss}",
            flush=True,
        )

        if writer is not None:
            writer.add_scalar(f"{writer_prefix}/train/group_task",     avg["task"],     ep)
            writer.add_scalar(f"{writer_prefix}/train/group_causal",   avg["causal"],   ep)
            writer.add_scalar(f"{writer_prefix}/train/group_graphreg", avg["graphreg"], ep)
            writer.add_scalar(f"{writer_prefix}/train/group_robust",   avg["robust"],   ep)
            writer.add_scalar(f"{writer_prefix}/train/total_loss",     avg["total"],    ep)

            writer.add_scalar(f"{writer_prefix}/train/pred_loss",         avg["pred"],    ep)
            writer.add_scalar(f"{writer_prefix}/train/recon_loss",        avg["recon"],   ep)
            writer.add_scalar(f"{writer_prefix}/train/te_weight_loss",    avg["tew"],     ep)
            writer.add_scalar(f"{writer_prefix}/train/te_gate_loss",      avg["teg"],     ep)
            writer.add_scalar(f"{writer_prefix}/train/causal_struct_loss",avg["cstruct"], ep)
            writer.add_scalar(f"{writer_prefix}/train/graph_loss",        avg["graph"],   ep)
            writer.add_scalar(f"{writer_prefix}/train/gate_loss",         avg["gate"],    ep)
            writer.add_scalar(f"{writer_prefix}/train/lagmono_loss",      avg["lagmono"], ep)
            writer.add_scalar(f"{writer_prefix}/train/perm_loss",         avg["perm"],    ep)
            writer.add_scalar(f"{writer_prefix}/train/inv_loss",          avg["inv"],     ep)

            with torch.no_grad():
                gate_np = model.edge_gate().detach().cpu().numpy()
                pw_global = model.get_pred_weights().detach().cpu().numpy()
                writer.add_scalar(f"{writer_prefix}/train/gate_mean",           float(gate_np.mean()),  ep)
                writer.add_scalar(f"{writer_prefix}/train/gate_max",            float(gate_np.max()),   ep)
                writer.add_scalar(f"{writer_prefix}/train/pred_weight_mean",    float(pw_global.mean()), ep)
                writer.add_scalar(f"{writer_prefix}/train/pred_weight_max",     float(pw_global.max()),  ep)
                writer.add_scalar(f"{writer_prefix}/train/pred_weight_entropy",
                                  float(model.pred_weight_entropy().detach().cpu()), ep)
                writer.add_scalar(f"{writer_prefix}/train/w_ref_mean",   float(model.w_ref.mean().detach().cpu()),   ep)
                writer.add_scalar(f"{writer_prefix}/train/cls_ref_mean", float(model.cls_ref.mean().detach().cpu()), ep)
                writer.add_scalar(f"{writer_prefix}/train/cls_ref_std",  float(model.cls_ref.std().detach().cpu()),  ep)

            if ep % 10 == 0:
                writer.add_histogram(f"{writer_prefix}/train/gate_hist",        gate_np,  ep)
                writer.add_histogram(f"{writer_prefix}/train/pred_weight_hist", pw_global, ep)
                writer.add_histogram(f"{writer_prefix}/train/w_ref_hist",   model.w_ref.detach().cpu().numpy(),   ep)
                writer.add_histogram(f"{writer_prefix}/train/cls_ref_hist", model.cls_ref.detach().cpu().numpy(), ep)


# ============================================================
# Local intervention contribution analysis
# ============================================================
@torch.no_grad()
def compute_intervention_contribution_3d(model, test_TN, device, batch, args):
    model.eval()
    ds = SlidingWindowDataset(test_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    tau_max = model.tau_max
    N = model.N

    raw_sum = np.zeros((tau_max, N, N), dtype=np.float64)
    pos_sum = np.zeros((tau_max, N, N), dtype=np.float64)
    n_windows = 0

    for X in loader:
        X = X.to(device)
        x_true_next = X[:, -1, :]
        _, pred_base, _, _, _, _, _, _, _ = model(X)
        base_err = (x_true_next - pred_base).abs()
        B = X.shape[0]

        for tau in range(1, tau_max + 1):
            for src in range(N):
                if args.intervention_mode == "fill":
                    _, pred_int, _, _, _, _, _, _, _ = model(
                        X, mask_tau=tau, mask_var=src, mask_fill_value=args.mask_fill_value,
                    )
                else:
                    Xp = intervene_local_window(
                        X, tau=tau, src=src, lag_win=model.lag_win,
                        mode=args.intervention_mode, fill_value=args.mask_fill_value
                    )
                    _, pred_int, _, _, _, _, _, _, _ = model(Xp)

                int_err = (x_true_next - pred_int).abs()
                delta = int_err - base_err
                raw_sum[tau - 1, src, :] += delta.sum(dim=0).detach().cpu().numpy().astype(np.float64)
                pos_sum[tau - 1, src, :] += torch.clamp(delta, min=0.0).sum(dim=0).detach().cpu().numpy().astype(np.float64)
        n_windows += B

    if n_windows == 0:
        raise RuntimeError("No test windows available for intervention contribution analysis.")

    G_raw_tau = raw_sum / float(n_windows)
    G_pos_tau = pos_sum / float(n_windows)
    return {
        "G_raw_tau":           G_raw_tau.astype(np.float32),
        "G_pos_tau":           G_pos_tau.astype(np.float32),
        "G_raw_lag_mean":      G_raw_tau.mean(axis=0).astype(np.float32),
        "G_pos_lag_mean":      G_pos_tau.mean(axis=0).astype(np.float32),
        "G_raw_lag_max":       G_raw_tau.max(axis=0).astype(np.float32),
        "G_pos_lag_max":       G_pos_tau.max(axis=0).astype(np.float32),
        "source_strength_tau": G_pos_tau.sum(axis=2).astype(np.float32),
        "target_received_tau": G_pos_tau.sum(axis=1).astype(np.float32),
    }


def topk_edges_from_matrix(M: np.ndarray, topk: int):
    M = np.asarray(M)
    N1, N2 = M.shape
    flat = M.reshape(-1)
    order = np.argsort(flat)[::-1]
    out = []
    for idx in order:
        val = flat[idx]
        if not np.isfinite(val):
            continue
        src = idx // N2
        tgt = idx % N2
        out.append((src, tgt, float(val)))
        if len(out) >= topk:
            break
    return out


def topk_edges_from_tensor(T: np.ndarray, topk: int):
    T = np.asarray(T)
    tau_max, N1, N2 = T.shape
    flat = T.reshape(-1)
    order = np.argsort(flat)[::-1]
    out = []
    for idx in order:
        val = flat[idx]
        if not np.isfinite(val):
            continue
        tau = idx // (N1 * N2)
        rem = idx % (N1 * N2)
        src = rem // N2
        tgt = rem % N2
        out.append((tau + 1, src, tgt, float(val)))
        if len(out) >= topk:
            break
    return out


def print_intervention_contrib_summary(name: str, G_pos_tau: np.ndarray, topk: int = 10):
    G_pos_lag_mean = G_pos_tau.mean(axis=0)
    print(f"\n[{name}] intervention contribution top-{topk} edges (lag-mean, positive delta)", flush=True)
    for rank, (src, tgt, val) in enumerate(topk_edges_from_matrix(G_pos_lag_mean, topk), start=1):
        print(f"  {rank:02d}. src={src:02d} -> tgt={tgt:02d} : {val:.6f}", flush=True)
    print(f"[{name}] intervention contribution top-{topk} lag-specific edges", flush=True)
    for rank, (tau, src, tgt, val) in enumerate(topk_edges_from_tensor(G_pos_tau, topk), start=1):
        print(f"  {rank:02d}. tau={tau:02d} src={src:02d} -> tgt={tgt:02d} : {val:.6f}", flush=True)


def save_intervention_contrib_csv(csv_path: str, G_raw_tau: np.ndarray, G_pos_tau: np.ndarray):
    import pandas as pd
    tau_max, N, _ = G_raw_tau.shape
    rows = []
    for tau in range(tau_max):
        for src in range(N):
            for tgt in range(N):
                rows.append({
                    "tau": tau + 1, "source": src, "target": tgt,
                    "raw_delta":      float(G_raw_tau[tau, src, tgt]),
                    "positive_delta": float(G_pos_tau[tau, src, tgt]),
                })
    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False)


# ============================================================
# Paper eval helper
# ============================================================
def paper_eval_one(score_series_1d, y01, start_idx, args):
    score = score_series_1d[start_idx:].astype(np.float64)
    labels = y01[start_idx:].astype(np.int32)

    m = (~np.isnan(score)) & np.isfinite(score)
    score = score[m]
    labels = labels[m]

    if score.size == 0:
        return {k: float("nan") for k in [
            "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
            "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F",
        ]}

    sliding_window = get_median_anomaly_length(labels) if args.use_median_vus_window else args.paper_slidingWindow

    return paper_get_metrics(
        score=score, labels=labels, slidingWindow=sliding_window,
        pred=None, version=args.paper_vus_version, thre=args.paper_vus_thre,
    )


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="OracleAD causal v2 runner")
    ap.add_argument("--input_dir",  type=str, required=True)
    ap.add_argument("--entities",   type=str, default="")
    ap.add_argument("--dataset",    type=str, default="PSM", choices=["PSM", "SMD", "SWaT", "OTHER"])

    # model / training
    ap.add_argument("--L",          type=int,   default=10)
    ap.add_argument("--tau_max",    type=int,   default=5)
    ap.add_argument("--lag_win",    type=int,   default=5)
    ap.add_argument("--batch",      type=int,   default=1024)
    ap.add_argument("--epochs",     type=int,   default=80)
    ap.add_argument("--d",          type=int,   default=64)
    ap.add_argument("--heads",      type=int,   default=4)
    ap.add_argument("--dropout",    type=float, default=0.0)
    ap.add_argument("--enc_layers", type=int,   default=2)
    ap.add_argument("--dec_layers", type=int,   default=2)

    # grouped losses
    ap.add_argument("--lam_task",     type=float, default=1.0)
    ap.add_argument("--lam_causal",   type=float, default=1.0)
    ap.add_argument("--lam_graphreg", type=float, default=0.05)
    ap.add_argument("--lam_robust",   type=float, default=0.10)
    ap.add_argument("--cls_ema",           type=float, default=0.9)
    ap.add_argument("--wref_ema",          type=float, default=0.9)
    ap.add_argument("--start_cls_epoch",   type=int,   default=5)
    ap.add_argument("--start_wref_epoch",  type=int,   default=3)
    ap.add_argument("--grad_clip",         type=float, default=0.0)

    ap.add_argument("--lr",           type=float, default=0.0,
                    help="0 => paper defaults (PSM=5e-5, others=5e-4)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seeds",        type=str,   default="0,1,2,3,4")
    ap.add_argument("--train_loss_type", type=str, default="l1", choices=["l1", "l2root"])
    ap.add_argument("--recon_loss_type", type=str, default="l1", choices=["l1", "l2root"])

    # graph / causal structure
    ap.add_argument("--pred_temp",       type=float, default=1.0)
    ap.add_argument("--self_loop_bias",  type=float, default=1.0)
    ap.add_argument("--lag_source_topk", type=int,   default=0)
    ap.add_argument("--graph_hidden",    type=int,   default=16)
    ap.add_argument("--gate_init",       type=float, default=0.15)
    ap.add_argument("--num_envs",        type=int,   default=4)
    ap.add_argument("--perm_pairs_per_batch", type=int, default=2)
    ap.add_argument("--perm_mode",       type=str,   default="permute", choices=["permute", "fill"])
    ap.add_argument("--dynamic_graph",    dest="dynamic_graph", action="store_true")
    ap.add_argument("--no_dynamic_graph", dest="dynamic_graph", action="store_false")
    ap.set_defaults(dynamic_graph=True)

    # [FIX 4] learnable causal MHSA mask
    ap.add_argument("--causal_attn_mask_scale", type=float, default=0.5,
                    help="Scale for learnable causal attention mask. 0 to disable.")
    ap.add_argument("--causal_mask_warmup", type=int, default=5,
                    help="Epochs before causal mask reaches full strength (0=no warmup).")

    # [V2+GNN]
    ap.add_argument("--gnn_layers", type=int, default=2,
                    help="Number of GNN message passing layers.")

    # [FIX 3] conditional TE prior
    ap.add_argument("--use_cte",    dest="use_cte", action="store_true",
                    help="Use 2-stage Conditional TE prior (graph-based confounder detection).")
    ap.add_argument("--no_cte",     dest="use_cte", action="store_false",
                    help="Use original unconditional TE prior.")
    ap.set_defaults(use_cte=True)

    # TE prior shared args
    ap.add_argument("--te_bins",       type=int,   default=8)
    ap.add_argument("--te_num_chunks", type=int,   default=32)
    ap.add_argument("--te_chunk_len",  type=int,   default=256)
    ap.add_argument("--te_threshold",  type=float, default=0.0)
    ap.add_argument("--te_self_mass",  type=float, default=0.25)
    ap.add_argument("--te_prior_blend",type=float, default=0.35)
    ap.add_argument("--te_init_scale", type=float, default=0.25)
    ap.add_argument("--te_seed",       type=int,   default=0)

    # scoring
    ap.add_argument("--p_agg",       type=str,   default="mean", choices=["mean", "max", "topk"])
    ap.add_argument("--p_topk",      type=int,   default=3)
    ap.add_argument("--c_agg",       type=str,   default="fro",  choices=["fro", "maxrow", "topkrow"])
    ap.add_argument("--c_topk",      type=int,   default=3)
    ap.add_argument("--g_agg",       type=str,   default="fro",  choices=["fro", "maxrow", "topkrow"])
    ap.add_argument("--g_topk",      type=int,   default=3)
    ap.add_argument("--causal_lag_agg", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--graph_lag_agg",  type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--lag_fusion",  type=str,   default="mean", choices=["mean", "max"])
    ap.add_argument("--score_alpha", type=float, default=1.0)
    ap.add_argument("--score_beta",  type=float, default=1.0)
    ap.add_argument("--calib_clip_min", type=float, default=0.0)
    ap.add_argument("--calibrate_scores",    dest="calibrate_scores", action="store_true")
    ap.add_argument("--no_calibrate_scores", dest="calibrate_scores", action="store_false")
    ap.set_defaults(calibrate_scores=True)

    # paper eval params
    ap.add_argument("--paper_slidingWindow", type=int,   default=100)
    ap.add_argument("--paper_vus_version",   type=str,   default="opt", choices=["opt", "opt_mem"])
    ap.add_argument("--paper_vus_thre",      type=int,   default=250)
    ap.add_argument("--use_median_vus_window", action="store_true")

    # misc
    ap.add_argument("--mhsa_residual",      action="store_true")
    ap.add_argument("--diagnose_components",action="store_true")

    # tensorboard
    ap.add_argument("--use_tensorboard", action="store_true")
    ap.add_argument("--tb_root",      type=str, default="runs/tensorboard/oraclead_causal_v2")
    ap.add_argument("--tb_histograms",action="store_true")
    ap.add_argument("--tb_figures",   action="store_true")

    # intervention contribution analysis
    ap.add_argument("--mask_contrib",       action="store_true")
    ap.add_argument("--intervention_mode",  type=str,   default="permute", choices=["permute", "fill"])
    ap.add_argument("--mask_fill_value",    type=float, default=0.0)
    ap.add_argument("--mask_batch",         type=int,   default=0)
    ap.add_argument("--mask_topk",          type=int,   default=10)
    ap.add_argument("--mask_save_csv",      action="store_true")

    ap.add_argument("--out_dir",       type=str, default="runs/oraclead_causal_v2")
    ap.add_argument("--save_per_seed", action="store_true")
    args = ap.parse_args()

    if args.use_tensorboard and SummaryWriter is None:
        raise ImportError(
            "TensorBoard writer is unavailable. Install one of:\n"
            "  pip install tensorboard\n"
            "  pip install tensorboardX"
        )
    if args.tb_figures and plt is None:
        print("[warn] matplotlib not available, tb_figures will be ignored.", flush=True)
        args.tb_figures = False

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    if args.use_tensorboard:
        print(f"tensorboard backend: {_TB_BACKEND}", flush=True)

    lr = float(args.lr) if (args.lr and args.lr > 0) else (5e-5 if args.dataset == "PSM" else 5e-4)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    # expose seeds on args so train_one_seed can pick up seed for perm_rng
    args.seeds = seeds
    mask_batch = args.mask_batch if args.mask_batch > 0 else args.batch

    if args.entities:
        wanted = [e.strip() for e in args.entities.split(",") if e.strip()]
        files = [os.path.join(args.input_dir, f"{e}.npz") for e in wanted]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))

    rows = []
    for f in files:
        if not os.path.exists(f):
            print(f"[skip] file not found: {f}", flush=True)
            continue

        name = os.path.splitext(os.path.basename(f))[0]
        data = np.load(f)
        train = data["train"].astype(np.float32)
        test  = data["test"].astype(np.float32)
        y = reduce_label(data["label"], test.shape[0])

        if train.ndim == 1:
            train = train[:, None]
        if test.ndim == 1:
            test = test[:, None]
        if train.shape[1] != test.shape[1]:
            print("[skip]", name, "N mismatch", flush=True)
            continue

        train_z, test_z, mu, sd = standardize_train_test(train, test)
        N = train_z.shape[1]
        if train_z.shape[0] < args.L + 1 or test_z.shape[0] < args.L + 1:
            print("[skip]", name, "too short", flush=True)
            continue

        # [FIX 3] choose prior builder
        prior_builder = build_cte_causal_prior if args.use_cte else build_te_causal_prior
        prior_label   = "CTE" if args.use_cte else "TE"
        print(f"\n=== {name} (Ttr={train_z.shape[0]}, Tte={test_z.shape[0]}, N={N}) "
              f"| prior={prior_label} ===", flush=True)

        te_weight_np, te_gate_np = prior_builder(
            train_z,
            tau_max=args.tau_max,
            num_bins=args.te_bins,
            num_chunks=args.te_num_chunks,
            chunk_len=args.te_chunk_len,
            threshold=args.te_threshold,
            self_mass=args.te_self_mass,
            seed=args.te_seed,
        )

        print(
            f"lr={lr} L={args.L} tau_max={args.tau_max} lag_win={args.lag_win} "
            f"batch={args.batch} enc/dec={args.enc_layers}/{args.dec_layers} "
            f"lam_task={args.lam_task} lam_causal={args.lam_causal} "
            f"lam_graphreg={args.lam_graphreg} lam_robust={args.lam_robust} "
            f"causal_attn_mask_scale={args.causal_attn_mask_scale} "
            f"cls_ema={args.cls_ema} wref_ema={args.wref_ema} "
            f"prior={prior_label} te_bins={args.te_bins} "          # [FIX 3]
            f"te_prior_blend={args.te_prior_blend} te_init_scale={args.te_init_scale}",
            flush=True,
        )

        metrics = []
        for seed in seeds:
            print(f"\n[seed {seed}] training ...", flush=True)
            set_seed(seed)
            args.seeds = [seed]  # expose current seed for perm_rng

            writer = None
            if args.use_tensorboard:
                log_dir = os.path.join(args.tb_root, name, f"seed{seed}")
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                if hasattr(writer, "add_text"):
                    writer.add_text("config/backend", str(_TB_BACKEND), 0)
                    writer.add_text("config/entity",  name, 0)
                    writer.add_text("config/prior",   prior_label, 0)

            model = OracleAD3DCausalEffect(
                N=N,
                L=args.L,
                tau_max=args.tau_max,
                d=args.d,
                heads=args.heads,
                enc_layers=args.enc_layers,
                dec_layers=args.dec_layers,
                dropout=args.dropout,
                mhsa_residual=args.mhsa_residual,
                lag_fusion=args.lag_fusion,
                lag_win=args.lag_win,
                pred_temp=args.pred_temp,
                self_loop_bias=args.self_loop_bias,
                lag_source_topk=args.lag_source_topk,
                dynamic_graph=args.dynamic_graph,
                graph_hidden=args.graph_hidden,
                gate_init=args.gate_init,
                te_prior_blend=args.te_prior_blend,
                causal_attn_mask_scale=args.causal_attn_mask_scale,
                causal_mask_warmup_epochs=args.causal_mask_warmup,
                gnn_layers=args.gnn_layers,
            ).to(device)

            model.set_te_prior(
                torch.from_numpy(te_weight_np),
                torch.from_numpy(te_gate_np),
                init_scale=args.te_init_scale,
            )

            train_one_seed(
                model,
                train_z,
                device,
                epochs=args.epochs,
                batch=args.batch,
                lr=lr,
                weight_decay=args.weight_decay,
                lam_task=args.lam_task,
                lam_causal=args.lam_causal,
                lam_graphreg=args.lam_graphreg,
                lam_robust=args.lam_robust,
                cls_ema=args.cls_ema,
                wref_ema=args.wref_ema,
                start_cls_epoch=args.start_cls_epoch,
                start_wref_epoch=args.start_wref_epoch,
                grad_clip=args.grad_clip,
                train_loss_type=args.train_loss_type,
                recon_loss_type=args.recon_loss_type,
                writer=writer,
                writer_prefix=name,
                args=args,
            )

            calibrator = None
            if args.calibrate_scores:
                print(f"[seed {seed}] fitting robust score calibrator on train windows ...", flush=True)
                train_scores = score_windows(model, train_z, device, batch=args.batch, args=args, calibrator=None)
                calibrator = fit_score_calibrator(train_scores)

            test_scores = score_windows(model, test_z, device, batch=args.batch, args=args, calibrator=calibrator)

            Tt = test_z.shape[0]
            start = args.L - 1
            score_t_dict = score_components_to_timeline(
                {k: test_scores[k] for k in ["P", "C", "G", "S", "A", "P_raw", "C_raw", "G_raw"]},
                Tt=Tt, start=start,
            )
            P_t = score_t_dict["P_t"]
            C_t = score_t_dict["C_t"]
            G_t = score_t_dict["G_t"]
            S_t = score_t_dict["S_t"]
            A_t = score_t_dict["A_t"]

            mtr_P = paper_eval_one(P_t, y, start, args)
            mtr_C = paper_eval_one(C_t, y, start, args)
            mtr_G = paper_eval_one(G_t, y, start, args)
            mtr_S = paper_eval_one(S_t, y, start, args)
            mtr_A = paper_eval_one(A_t, y, start, args)

            if args.diagnose_components:
                print(
                    f"[seed {seed}] paper_eval components\n"
                    f"  P-only : A-PR={pct(mtr_P['AUC-PR']):.2f}  VUS-PR={pct(mtr_P['VUS-PR']):.2f}  F1={pct(mtr_P['Standard-F1']):.2f}\n"
                    f"  C-only : A-PR={pct(mtr_C['AUC-PR']):.2f}  VUS-PR={pct(mtr_C['VUS-PR']):.2f}  F1={pct(mtr_C['Standard-F1']):.2f}\n"
                    f"  G-only : A-PR={pct(mtr_G['AUC-PR']):.2f}  VUS-PR={pct(mtr_G['VUS-PR']):.2f}  F1={pct(mtr_G['Standard-F1']):.2f}\n"
                    f"  S=C+G  : A-PR={pct(mtr_S['AUC-PR']):.2f}  VUS-PR={pct(mtr_S['VUS-PR']):.2f}  F1={pct(mtr_S['Standard-F1']):.2f}\n"
                    f"  A=P*S  : A-PR={pct(mtr_A['AUC-PR']):.2f}  VUS-PR={pct(mtr_A['VUS-PR']):.2f}  F1={pct(mtr_A['Standard-F1']):.2f}",
                    flush=True,
                )

            A_PR  = float(mtr_A["AUC-PR"])
            A_ROC = float(mtr_A["AUC-ROC"])
            VUS_PR  = float(mtr_A["VUS-PR"])
            VUS_ROC = float(mtr_A["VUS-ROC"])
            F1    = float(mtr_A["Standard-F1"])
            PA_F1 = float(mtr_A["PA-F1"])
            EV_F1 = float(mtr_A["Event-based-F1"])
            R_F1  = float(mtr_A["R-based-F1"])
            Aff_F1= float(mtr_A["Affiliation-F"])

            metrics.append((A_PR, A_ROC, F1, PA_F1, EV_F1, R_F1, Aff_F1, VUS_ROC, VUS_PR))

            mask_out = None
            if args.mask_contrib:
                print(f"[seed {seed}] running local intervention contribution analysis ...", flush=True)
                mask_out = compute_intervention_contribution_3d(model, test_z, device, batch=mask_batch, args=args)
                print_intervention_contrib_summary(name, mask_out["G_pos_tau"], topk=args.mask_topk)

            print(
                f"[seed {seed}] "
                f"A-PR={pct(A_PR):.2f}  A-ROC={pct(A_ROC):.2f}  "
                f"F1={pct(F1):.2f}  PA-F1={pct(PA_F1):.2f}  EV-F1={pct(EV_F1):.2f}  "
                f"R-F1={pct(R_F1):.2f}  Aff-F={pct(Aff_F1):.2f}  "
                f"VUS-ROC={pct(VUS_ROC):.2f}  VUS-PR={pct(VUS_PR):.2f}",
                flush=True,
            )

            if writer is not None:
                writer.add_scalar(f"{name}/eval/AUC_PR",    A_PR,   seed)
                writer.add_scalar(f"{name}/eval/AUC_ROC",   A_ROC,  seed)
                writer.add_scalar(f"{name}/eval/F1",        F1,     seed)
                writer.add_scalar(f"{name}/eval/PA_F1",     PA_F1,  seed)
                writer.add_scalar(f"{name}/eval/Event_F1",  EV_F1,  seed)
                writer.add_scalar(f"{name}/eval/R_F1",      R_F1,   seed)
                writer.add_scalar(f"{name}/eval/Aff_F",     Aff_F1, seed)
                writer.add_scalar(f"{name}/eval/VUS_ROC",   VUS_ROC,seed)
                writer.add_scalar(f"{name}/eval/VUS_PR",    VUS_PR, seed)
                writer.add_scalar(f"{name}/eval/P_mean", float(np.nanmean(P_t)), seed)
                writer.add_scalar(f"{name}/eval/C_mean", float(np.nanmean(C_t)), seed)
                writer.add_scalar(f"{name}/eval/G_mean", float(np.nanmean(G_t)), seed)
                writer.add_scalar(f"{name}/eval/S_mean", float(np.nanmean(S_t)), seed)
                with torch.no_grad():
                    gate_np = model.edge_gate().detach().cpu().numpy()
                    writer.add_scalar(f"{name}/eval/gate_mean",           float(gate_np.mean()), seed)
                    writer.add_scalar(f"{name}/eval/gate_max",            float(gate_np.max()),  seed)
                    writer.add_scalar(f"{name}/eval/pred_weight_entropy",
                                      float(model.pred_weight_entropy().detach().cpu()), seed)
                    writer.add_scalar(f"{name}/eval/w_ref_mean",   float(model.w_ref.mean().detach().cpu()),   seed)
                    writer.add_scalar(f"{name}/eval/cls_ref_mean", float(model.cls_ref.mean().detach().cpu()), seed)
                if args.tb_histograms:
                    tb_log_score_histograms(writer, name, seed, y, score_t_dict, start)
                if args.tb_figures:
                    tb_log_score_curves(writer, name, seed, y, score_t_dict)
                if mask_out is not None:
                    writer.add_scalar(f"{name}/mask/G_pos_mean",           float(mask_out["G_pos_tau"].mean()),          seed)
                    writer.add_scalar(f"{name}/mask/G_pos_max",            float(mask_out["G_pos_tau"].max()),           seed)
                    writer.add_scalar(f"{name}/mask/G_raw_mean",           float(mask_out["G_raw_tau"].mean()),          seed)
                    writer.add_scalar(f"{name}/mask/source_strength_mean", float(mask_out["source_strength_tau"].mean()),seed)
                    writer.add_scalar(f"{name}/mask/target_received_mean", float(mask_out["target_received_tau"].mean()),seed)
                writer.flush()
                writer.close()

            if args.save_per_seed:
                save_kwargs = {
                    "A_t": A_t, "S_t": S_t, "P_t": P_t, "C_t": C_t, "G_t": G_t,
                    "P_raw_t": score_t_dict["P_raw_t"],
                    "C_raw_t": score_t_dict["C_raw_t"],
                    "G_raw_t": score_t_dict["G_raw_t"],
                    "y": y,
                    "cls_ref":             model.cls_ref.detach().cpu().numpy().astype(np.float32),
                    "w_ref":               model.w_ref.detach().cpu().numpy().astype(np.float32),
                    "gate":                model.edge_gate().detach().cpu().numpy().astype(np.float32),
                    "pred_weights_global": model.get_pred_weights().detach().cpu().numpy().astype(np.float32),
                    "te_prior_weight":     model.te_prior_weight.detach().cpu().numpy().astype(np.float32),
                    "te_prior_gate":       model.te_prior_gate.detach().cpu().numpy().astype(np.float32),
                    "mu": mu, "sd": sd,
                }
                if calibrator is not None:
                    for comp in ["P", "C", "G"]:
                        save_kwargs[f"{comp}_center"] = np.array([calibrator[f"{comp}_raw"]["center"]], dtype=np.float32)
                        save_kwargs[f"{comp}_scale"]  = np.array([calibrator[f"{comp}_raw"]["scale"]],  dtype=np.float32)
                if mask_out is not None:
                    save_kwargs.update({
                        "G_raw_tau":           mask_out["G_raw_tau"],
                        "G_pos_tau":           mask_out["G_pos_tau"],
                        "G_raw_lag_mean":      mask_out["G_raw_lag_mean"],
                        "G_pos_lag_mean":      mask_out["G_pos_lag_mean"],
                        "G_raw_lag_max":       mask_out["G_raw_lag_max"],
                        "G_pos_lag_max":       mask_out["G_pos_lag_max"],
                        "source_strength_tau": mask_out["source_strength_tau"],
                        "target_received_tau": mask_out["target_received_tau"],
                    })
                np.savez(os.path.join(args.out_dir, f"{name}_seed{seed}.npz"), **save_kwargs)

                if args.mask_save_csv and mask_out is not None:
                    save_intervention_contrib_csv(
                        os.path.join(args.out_dir, f"{name}_seed{seed}_intervention_contrib.csv"),
                        mask_out["G_raw_tau"], mask_out["G_pos_tau"],
                    )

        A_PR_m,   A_PR_s   = safe_mean_std([m[0] for m in metrics])
        A_ROC_m,  A_ROC_s  = safe_mean_std([m[1] for m in metrics])
        F1_m,     F1_s     = safe_mean_std([m[2] for m in metrics])
        PA_m,     PA_s     = safe_mean_std([m[3] for m in metrics])
        EV_m,     EV_s     = safe_mean_std([m[4] for m in metrics])
        R_F1_m,   R_F1_s   = safe_mean_std([m[5] for m in metrics])
        Aff_m,    Aff_s    = safe_mean_std([m[6] for m in metrics])
        VUS_ROC_m,VUS_ROC_s= safe_mean_std([m[7] for m in metrics])
        VUS_PR_m, VUS_PR_s = safe_mean_std([m[8] for m in metrics])

        print(f"\n[{name}] mean±std over {len(seeds)} seeds:", flush=True)
        print(f"  A-PR          {pct(A_PR_m):.2f} ± {pct(A_PR_s):.2f}", flush=True)
        print(f"  A-ROC         {pct(A_ROC_m):.2f} ± {pct(A_ROC_s):.2f}", flush=True)
        print(f"  Standard-F1   {pct(F1_m):.2f} ± {pct(F1_s):.2f}", flush=True)
        print(f"  PA-F1         {pct(PA_m):.2f} ± {pct(PA_s):.2f}", flush=True)
        print(f"  Event-F1      {pct(EV_m):.2f} ± {pct(EV_s):.2f}", flush=True)
        print(f"  R-based-F1    {pct(R_F1_m):.2f} ± {pct(R_F1_s):.2f}", flush=True)
        print(f"  Affiliation-F {pct(Aff_m):.2f} ± {pct(Aff_s):.2f}", flush=True)
        print(f"  VUS-ROC       {pct(VUS_ROC_m):.2f} ± {pct(VUS_ROC_s):.2f}", flush=True)
        print(f"  VUS-PR        {pct(VUS_PR_m):.2f} ± {pct(VUS_PR_s):.2f}", flush=True)

        rows.append((name, A_PR_m, A_ROC_m, F1_m, PA_m, EV_m, R_F1_m, Aff_m, VUS_ROC_m, VUS_PR_m))

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=[
            "entity", "AUC_PR", "AUC_ROC",
            "F1", "PA_F1", "Event_F1", "R_F1", "Aff_F",
            "VUS_ROC", "VUS_PR",
        ])
        df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
        print("\nSaved summary:", os.path.join(args.out_dir, "summary.csv"), flush=True)


if __name__ == "__main__":
    main()
