import argparse, os, glob, sys, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

# optional figure logging
plt = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ============================================================
# Make sure `src/` is on PYTHONPATH
# ============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/runners
_SRC_DIR  = os.path.abspath(os.path.join(_THIS_DIR, ".."))      # .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from metrics.paper_eval.metrics_api import get_metrics as paper_get_metrics


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Utils
# ============================================================
def standardize_train_test(train: np.ndarray, test: np.ndarray):
    train = train.astype(np.float32)
    test  = test.astype(np.float32)

    train = np.where(np.isfinite(train), train, np.nan)
    test  = np.where(np.isfinite(test),  test,  np.nan)

    col_mean = np.nanmean(train, axis=0, keepdims=True).astype(np.float32)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)

    train = np.where(np.isnan(train), col_mean, train).astype(np.float32)
    test  = np.where(np.isnan(test),  col_mean, test ).astype(np.float32)

    mu = train.mean(axis=0, keepdims=True).astype(np.float32)
    var = ((train - mu) ** 2).mean(axis=0, keepdims=True).astype(np.float32)
    sd = np.sqrt(var).astype(np.float32)
    sd = np.where(sd == 0.0, 1.0, sd).astype(np.float32)

    train_z = (train - mu) / sd
    test_z  = (test  - mu) / sd
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


# ============================================================
# Dataset
# ============================================================
class SlidingWindowDataset(Dataset):
    def __init__(self, series_TN: np.ndarray, L: int):
        self.x = series_TN.astype(np.float32)
        self.L = int(L)
        self.T, self.N = self.x.shape
        if self.T < self.L:
            raise ValueError(f"T={self.T} < L={self.L}")

    def __len__(self):
        return self.T - self.L + 1

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx:idx+self.L])  # (L, N)


# ============================================================
# 3D OracleAD model (lag-local + lag/source weighted prediction)
# ============================================================
class TemporalAttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1, bias=True)

    def forward(self, H):  # (B, T, d)
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
        O = self.out(Y).squeeze(-1)  # (B, L-1)
        return O


class PerVarPredictor(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)  # (B,)


def pairwise_sq_l2(C):
    # C: (B, N, d)
    A2 = (C * C).sum(dim=2)
    G = torch.bmm(C, C.transpose(1, 2))
    D = A2.unsqueeze(2) + A2.unsqueeze(1) - 2.0 * G
    return torch.clamp(D, min=0.0)


def pairwise_l2(C):
    return (pairwise_sq_l2(C) + 1e-12).sqrt()


def pairwise_lag_distance(C_all, use_squared_l2=False):
    # C_all: (B, tau, N, d) -> (B, tau, N, N)
    B, T, N, d = C_all.shape
    flat = C_all.reshape(B * T, N, d)
    if use_squared_l2:
        D = pairwise_sq_l2(flat)
    else:
        D = pairwise_l2(flat)
    return D.reshape(B, T, N, N)


class OracleAD3D(nn.Module):
    def __init__(self, N: int, L: int, tau_max: int, d: int, heads: int,
                 enc_layers: int, dec_layers: int, dropout: float,
                 mhsa_residual: bool = False, lag_fusion: str = "mean",
                 lag_win: int = 5,
                 pred_temp: float = 1.0,
                 self_loop_bias: float = 1.0,
                 lag_source_topk: int = 0):
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

        if tau_max >= L:
            raise ValueError(f"tau_max={tau_max} must be < L={L}")
        if self.lag_win <= 0:
            raise ValueError(f"lag_win must be >= 1, got {self.lag_win}")

        self.encoders = nn.ModuleList([PerVarEncoder(d, enc_layers, dropout) for _ in range(N)])
        self.mhsa = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)

        # reconstruction branch
        self.recon_decoders = nn.ModuleList([PerVarReconDecoder(d, L, dec_layers, dropout) for _ in range(N)])

        # prediction branch
        self.pred_heads = nn.ModuleList([PerVarPredictor(d) for _ in range(N)])

        self.mhsa_residual = mhsa_residual

        # learnable lag/source -> target weights
        # shape: (tau, src, tgt)
        self.pred_logits = nn.Parameter(torch.zeros(tau_max, N, N))
        with torch.no_grad():
            for tau in range(tau_max):
                self.pred_logits[tau].fill_(0.0)
                diag_idx = torch.arange(N)
                self.pred_logits[tau, diag_idx, diag_idx] += self.self_loop_bias

        # SLS: (tau_max, N, N)
        self.register_buffer("sls", torch.zeros(tau_max, N, N), persistent=True)
        self.has_sls = False

    def reset_sls(self):
        self.sls.zero_()
        self.has_sls = False

    def _get_pred_weights(self):
        """
        Returns:
          weights: (tau, src, tgt)
        Softmax over flattened (tau, src) per target.
        """
        tau_max, N, _ = self.pred_logits.shape
        flat = (self.pred_logits / max(self.pred_temp, 1e-6)).reshape(tau_max * N, N)  # (tau*src, tgt)
        weights_flat = torch.softmax(flat, dim=0)

        if self.lag_source_topk > 0:
            k = min(self.lag_source_topk, tau_max * N)
            vals, idx = torch.topk(weights_flat, k=k, dim=0)
            mask = torch.zeros_like(weights_flat)
            mask.scatter_(0, idx, 1.0)
            weights_flat = weights_flat * mask
            weights_flat = weights_flat / (weights_flat.sum(dim=0, keepdim=True) + 1e-12)

        weights = weights_flat.view(tau_max, N, N)
        return weights

    def pred_weight_entropy(self):
        """
        Normalized entropy of lag/source weights across (tau, src) for each target.
        Lower => sparser.
        """
        weights = self._get_pred_weights()
        tau_max, N, _ = weights.shape
        flat = weights.reshape(tau_max * N, N)  # (tau*src, tgt)
        ent = -(flat * torch.log(flat + 1e-12)).sum(dim=0).mean()
        if tau_max * N > 1:
            ent = ent / math.log(tau_max * N)
        return ent

    def forward(self, X, mask_tau=None, mask_var=None, mask_fill_value=0.0):
        """
        X: (B, L, N)

        mask_tau: int or None
            1..tau_max. 해당 lag branch 입력(local lag window)에서만 masking 수행.
        mask_var: int or None
            0..N-1. 해당 source variable만 masking 수행.
        mask_fill_value: float
            표준화 공간에서 주입할 값 (기본 0.0 = train mean).
        """
        B, L, N = X.shape
        lag_embeds = []

        for tau in range(1, self.tau_max + 1):
            # local lag window centered near lag tau
            # target time = L-1
            # exact lag point is around index (L-1-tau)
            # use window [end-lag_win, end), where end = L-tau
            end = L - tau
            start = max(0, end - self.lag_win)

            c_list = []
            for i in range(N):
                x_i = X[:, start:end, i].unsqueeze(-1)  # (B, local_len, 1)

                # inference-time local lag window masking for contribution analysis
                if (mask_tau is not None) and (mask_var is not None):
                    if (tau == int(mask_tau)) and (i == int(mask_var)):
                        x_i = torch.full_like(x_i, float(mask_fill_value))

                ci = self.encoders[i](x_i)
                c_list.append(ci)

            C_tau = torch.stack(c_list, dim=1)  # (B, N, d)

            A_tau, _ = self.mhsa(C_tau, C_tau, C_tau, need_weights=False)
            C_star_tau = (C_tau + A_tau) if self.mhsa_residual else A_tau
            lag_embeds.append(C_star_tau)

        C_all = torch.stack(lag_embeds, dim=1)  # (B, tau_max, N, d)

        # ----------------------------------------------------
        # reconstruction branch: keep stable fused representation
        # ----------------------------------------------------
        if self.lag_fusion == "max":
            C_recon = C_all.max(dim=1).values  # (B, N, d)
        else:
            C_recon = C_all.mean(dim=1)        # (B, N, d)

        recon_list = []
        for i in range(N):
            r = self.recon_decoders[i](C_recon[:, i, :])
            recon_list.append(r)
        recon = torch.stack(recon_list, dim=-1)  # (B, L-1, N)

        # ----------------------------------------------------
        # prediction branch: lag/source -> target weighted aggregation
        # ----------------------------------------------------
        pred_weights = self._get_pred_weights()  # (tau, src, tgt)

        # context for each target i:
        # z_i = sum_tau,sum_src w[tau,src,i] * C_all[:,tau,src,:]
        pred_ctx = torch.einsum("btsd,tsi->bid", C_all, pred_weights)  # (B, tgt, d)

        pred_list = []
        for i in range(N):
            p = self.pred_heads[i](pred_ctx[:, i, :])  # (B,)
            pred_list.append(p)
        pred = torch.stack(pred_list, dim=-1)  # (B, N)

        return recon, pred, C_all, pred_weights


# ============================================================
# TensorBoard helpers
# ============================================================
def tb_log_score_histograms(writer, prefix, step, labels, P_t, D_t, A_t, start_idx):
    if writer is None:
        return

    valid = np.isfinite(A_t[start_idx:])
    yv = labels[start_idx:][valid]
    pv = P_t[start_idx:][valid]
    dv = D_t[start_idx:][valid]
    av = A_t[start_idx:][valid]

    if len(pv) > 0:
        writer.add_histogram(f"{prefix}/scores/P_all", pv, step)
        writer.add_histogram(f"{prefix}/scores/D_all", dv, step)
        writer.add_histogram(f"{prefix}/scores/A_all", av, step)

    if (yv == 1).sum() > 0:
        writer.add_histogram(f"{prefix}/scores/P_anom", pv[yv == 1], step)
        writer.add_histogram(f"{prefix}/scores/D_anom", dv[yv == 1], step)
        writer.add_histogram(f"{prefix}/scores/A_anom", av[yv == 1], step)

    if (yv == 0).sum() > 0:
        writer.add_histogram(f"{prefix}/scores/P_norm", pv[yv == 0], step)
        writer.add_histogram(f"{prefix}/scores/D_norm", dv[yv == 0], step)
        writer.add_histogram(f"{prefix}/scores/A_norm", av[yv == 0], step)


def tb_log_score_curves(writer, prefix, step, labels, P_t, D_t, A_t, max_points=2000):
    if writer is None or plt is None:
        return

    T = len(labels)
    idx = np.arange(T)
    if T > max_points:
        sel = np.linspace(0, T - 1, max_points).astype(int)
    else:
        sel = idx

    y = labels[sel]
    p = P_t[sel]
    d = D_t[sel]
    a = A_t[sel]

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(sel, y, linewidth=1.0)
    axes[0].set_ylabel("label")

    axes[1].plot(sel, p, linewidth=1.0)
    axes[1].set_ylabel("P_t")

    axes[2].plot(sel, d, linewidth=1.0)
    axes[2].set_ylabel("D_t")

    axes[3].plot(sel, a, linewidth=1.0)
    axes[3].set_ylabel("A_t")
    axes[3].set_xlabel("time")

    fig.tight_layout()

    try:
        writer.add_figure(f"{prefix}/figures/score_curves", fig, global_step=step)
    except Exception as e:
        print(f"[warn] tb figure logging failed for {prefix} step {step}: {e}", flush=True)
    finally:
        plt.close(fig)


# ============================================================
# Train
# ============================================================
def train_one_seed(model, train_TN, device,
                   epochs, batch, lr, weight_decay,
                   lam_recon, lam_dev, lam_sparse, sls_ema,
                   start_sls_epoch=5,
                   use_squared_l2=False,
                   grad_clip=0.0,
                   writer=None, writer_prefix=""):
    model.reset_sls()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = SlidingWindowDataset(train_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)

    Tlag = model.tau_max
    N = model.N

    for ep in range(1, epochs + 1):
        model.train()

        sls_sum = torch.zeros(Tlag, N, N, device=device)
        sls_cnt = 0

        lp, lrn, ld, ls = 0.0, 0.0, 0.0, 0.0
        steps = 0
        last_use_dev_loss = False

        for X in loader:
            X = X.to(device)
            recon, pred, C_all, pred_weights = model(X)

            xL = X[:, -1, :]              # (B, N)
            xpast = X[:, :model.L-1, :]   # (B, L-1, N)

            loss_pred = ((xL - pred).pow(2).sum(dim=-1)).sqrt().mean()
            loss_recon = ((xpast - recon).pow(2).sum(dim=(1, 2))).sqrt().mean()

            D = pairwise_lag_distance(C_all, use_squared_l2=use_squared_l2)  # (B, tau, N, N)
            sls_sum += D.mean(dim=0).detach()
            sls_cnt += 1

            use_dev_loss = (ep >= start_sls_epoch) and model.has_sls
            last_use_dev_loss = use_dev_loss

            loss_sparse = model.pred_weight_entropy()

            if use_dev_loss:
                diff = D - model.sls.unsqueeze(0)
                Ndim = diff.size(2)
                loss_dev = (diff.pow(2).sum(dim=(1, 2, 3)) / float(Tlag * Ndim * Ndim)).mean()
                loss = loss_pred + lam_recon * loss_recon + lam_dev * loss_dev + lam_sparse * loss_sparse
            else:
                loss_dev = torch.tensor(0.0, device=device)
                loss = loss_pred + lam_recon * loss_recon + lam_sparse * loss_sparse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            lp += float(loss_pred.detach().cpu())
            lrn += float(loss_recon.detach().cpu())
            ld += float(loss_dev.detach().cpu())
            ls += float(loss_sparse.detach().cpu())
            steps += 1

        with torch.no_grad():
            epoch_sls = sls_sum / max(sls_cnt, 1)  # (tau, N, N)
            if not model.has_sls:
                model.sls.copy_(epoch_sls)
                model.has_sls = True
            else:
                if sls_ema <= 0.0:
                    model.sls.copy_(epoch_sls)
                else:
                    beta = float(sls_ema)
                    model.sls.mul_(beta).add_(epoch_sls * (1.0 - beta))

        pred_avg = lp / max(steps, 1)
        recon_avg = lrn / max(steps, 1)
        dev_avg = ld / max(steps, 1)
        sparse_avg = ls / max(steps, 1)

        print(
            f"  [ep {ep:02d}] pred={pred_avg:.6f} recon={recon_avg:.6f} "
            f"dev={dev_avg:.3e} sparse={sparse_avg:.6f} "
            f"has_sls={model.has_sls} use_dev={last_use_dev_loss}",
            flush=True
        )

        if writer is not None:
            writer.add_scalar(f"{writer_prefix}/train/pred_loss", pred_avg, ep)
            writer.add_scalar(f"{writer_prefix}/train/recon_loss", recon_avg, ep)
            writer.add_scalar(f"{writer_prefix}/train/dev_loss", dev_avg, ep)
            writer.add_scalar(f"{writer_prefix}/train/sparse_loss", sparse_avg, ep)

            with torch.no_grad():
                pw = model._get_pred_weights().detach().cpu().numpy()
                writer.add_scalar(f"{writer_prefix}/train/pred_weight_mean", float(pw.mean()), ep)
                writer.add_scalar(f"{writer_prefix}/train/pred_weight_max", float(pw.max()), ep)
                writer.add_scalar(f"{writer_prefix}/train/pred_weight_entropy", float(model.pred_weight_entropy().detach().cpu()), ep)

            if model.has_sls:
                writer.add_scalar(f"{writer_prefix}/train/sls_mean", float(model.sls.mean().detach().cpu()), ep)
                writer.add_scalar(f"{writer_prefix}/train/sls_std", float(model.sls.std().detach().cpu()), ep)
                writer.add_scalar(f"{writer_prefix}/train/sls_fro", float(torch.linalg.norm(model.sls.detach()).cpu()), ep)

                if ep % 10 == 0:
                    writer.add_histogram(f"{writer_prefix}/train/sls_hist", model.sls.detach().cpu().numpy(), ep)
                    writer.add_histogram(f"{writer_prefix}/train/pred_weight_hist", pw, ep)


# ============================================================
# Scoring
# ============================================================
def prediction_score(err: torch.Tensor, args):
    # err: (B, N)
    if args.p_agg == "mean":
        return err.mean(dim=1)
    elif args.p_agg == "max":
        return err.max(dim=1).values
    else:
        k = min(int(args.p_topk), err.shape[1])
        return err.topk(k, dim=1).values.mean(dim=1)


def lag_deviation_per_tau(diff: torch.Tensor, args):
    # diff: (B, tau, N, N) -> (B, tau)
    if args.d_agg == "fro":
        return torch.linalg.norm(diff, ord="fro", dim=(2, 3))
    else:
        row_dev = diff.abs().mean(dim=3)  # (B, tau, N)
        if args.d_agg == "maxrow":
            return row_dev.max(dim=2).values
        else:
            k = min(int(args.d_topk), row_dev.shape[2])
            return row_dev.topk(k, dim=2).values.mean(dim=2)


def lag_aggregate(per_tau: torch.Tensor, args):
    # per_tau: (B, tau)
    if args.lag_agg == "max":
        return per_tau.max(dim=1).values
    else:
        return per_tau.mean(dim=1)


@torch.no_grad()
def score_series(model, test_TN, device, batch, args):
    model.eval()
    ds = SlidingWindowDataset(test_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    W = len(ds)
    P_w = np.zeros((W,), dtype=np.float32)
    D_w = np.zeros((W,), dtype=np.float32)

    sls = model.sls.detach()  # (tau, N, N)
    offset = 0
    for X in loader:
        X = X.to(device)
        recon, pred, C_all, pred_weights = model(X)

        x_true_next = X[:, -1, :]
        err = (x_true_next - pred).abs()
        P = prediction_score(err, args)

        Dm = pairwise_lag_distance(C_all, use_squared_l2=args.use_squared_l2)  # (B, tau, N, N)
        diff = Dm - sls.unsqueeze(0)
        per_tau = lag_deviation_per_tau(diff, args)  # (B, tau)
        Dscore = lag_aggregate(per_tau, args)        # (B,)

        bsz = X.shape[0]
        P_w[offset:offset + bsz] = P.detach().cpu().numpy().astype(np.float32)
        D_w[offset:offset + bsz] = Dscore.detach().cpu().numpy().astype(np.float32)
        offset += bsz

    A_w = (P_w * D_w).astype(np.float32)
    return P_w, D_w, A_w


# ============================================================
# Local lag masking contribution analysis
# ============================================================
@torch.no_grad()
def compute_mask_contribution_3d(model, test_TN, device, batch, args):
    """
    lag별 / source별 local lag window masking이 target prediction error를 얼마나 증가시키는지 측정.

    반환:
      G_raw_tau: (tau, src, tgt)
          mean(masked_err - base_err)
      G_pos_tau: (tau, src, tgt)
          mean(relu(masked_err - base_err))
    """
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
        x_true_next = X[:, -1, :]  # (B, N)

        _, pred_base, _, _ = model(X)
        base_err = (x_true_next - pred_base).abs()  # (B, N)
        B = X.shape[0]

        for tau in range(1, tau_max + 1):
            for src in range(N):
                _, pred_mask, _, _ = model(
                    X,
                    mask_tau=tau,
                    mask_var=src,
                    mask_fill_value=args.mask_fill_value
                )
                mask_err = (x_true_next - pred_mask).abs()  # (B, N)
                delta = mask_err - base_err                 # (B, N)

                raw_sum[tau - 1, src, :] += delta.sum(dim=0).detach().cpu().numpy().astype(np.float64)
                pos_sum[tau - 1, src, :] += torch.clamp(delta, min=0.0).sum(dim=0).detach().cpu().numpy().astype(np.float64)

        n_windows += B

    if n_windows == 0:
        raise RuntimeError("No test windows available for mask contribution analysis.")

    G_raw_tau = raw_sum / float(n_windows)
    G_pos_tau = pos_sum / float(n_windows)

    out = {
        "G_raw_tau": G_raw_tau.astype(np.float32),
        "G_pos_tau": G_pos_tau.astype(np.float32),
        "G_raw_lag_mean": G_raw_tau.mean(axis=0).astype(np.float32),
        "G_pos_lag_mean": G_pos_tau.mean(axis=0).astype(np.float32),
        "G_raw_lag_max": G_raw_tau.max(axis=0).astype(np.float32),
        "G_pos_lag_max": G_pos_tau.max(axis=0).astype(np.float32),
        "source_strength_tau": G_pos_tau.sum(axis=2).astype(np.float32),   # (tau, src)
        "target_received_tau": G_pos_tau.sum(axis=1).astype(np.float32),   # (tau, tgt)
    }
    return out


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


def print_mask_contrib_summary(name: str, G_pos_tau: np.ndarray, topk: int = 10):
    G_pos_lag_mean = G_pos_tau.mean(axis=0)

    print(f"\n[{name}] local-lag-mask contribution top-{topk} edges (lag-mean, positive delta)", flush=True)
    for rank, (src, tgt, val) in enumerate(topk_edges_from_matrix(G_pos_lag_mean, topk), start=1):
        print(f"  {rank:02d}. src={src:02d} -> tgt={tgt:02d} : {val:.6f}", flush=True)

    print(f"[{name}] local-lag-mask contribution top-{topk} lag-specific edges", flush=True)
    for rank, (tau, src, tgt, val) in enumerate(topk_edges_from_tensor(G_pos_tau, topk), start=1):
        print(f"  {rank:02d}. tau={tau:02d} src={src:02d} -> tgt={tgt:02d} : {val:.6f}", flush=True)


def save_mask_contrib_csv(csv_path: str, G_raw_tau: np.ndarray, G_pos_tau: np.ndarray):
    import pandas as pd

    tau_max, N, _ = G_raw_tau.shape
    rows = []
    for tau in range(tau_max):
        for src in range(N):
            for tgt in range(N):
                rows.append({
                    "tau": tau + 1,
                    "source": src,
                    "target": tgt,
                    "raw_delta": float(G_raw_tau[tau, src, tgt]),
                    "positive_delta": float(G_pos_tau[tau, src, tgt]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


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
        return {
            "AUC-PR": float("nan"), "AUC-ROC": float("nan"),
            "VUS-PR": float("nan"), "VUS-ROC": float("nan"),
            "Standard-F1": float("nan"), "PA-F1": float("nan"),
            "Event-based-F1": float("nan"), "R-based-F1": float("nan"),
            "Affiliation-F": float("nan"),
        }

    sliding_window = get_median_anomaly_length(labels) if args.use_median_vus_window else args.paper_slidingWindow

    return paper_get_metrics(
        score=score,
        labels=labels,
        slidingWindow=sliding_window,
        pred=None,
        version=args.paper_vus_version,
        thre=args.paper_vus_thre
    )


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--entities", type=str, default="")
    ap.add_argument("--dataset", type=str, default="PSM", choices=["PSM", "SMD", "SWaT", "OTHER"])

    # model/training
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--tau_max", type=int, default=5)
    ap.add_argument("--lag_win", type=int, default=5, help="local lag window length for each lag branch")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=80)

    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--dec_layers", type=int, default=2)

    ap.add_argument("--lam_recon", type=float, default=0.1)
    ap.add_argument("--lam_dev", type=float, default=3.0)
    ap.add_argument("--lam_sparse", type=float, default=0.05,
                    help="entropy sparsity penalty on learned lag/source->target prediction weights")
    ap.add_argument("--sls_ema", type=float, default=0.0)
    ap.add_argument("--start_sls_epoch", type=int, default=5)
    ap.add_argument("--use_squared_l2", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0이면 gradient clipping 안 함")

    ap.add_argument("--lr", type=float, default=0.0, help="0 => paper defaults (PSM=5e-5, others=5e-4)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")

    # lag/source prediction weighting
    ap.add_argument("--pred_temp", type=float, default=1.0, help="temperature for learned lag/source prediction weights")
    ap.add_argument("--self_loop_bias", type=float, default=1.0, help="initial bias added to source==target logits")
    ap.add_argument("--lag_source_topk", type=int, default=0,
                    help="optional hard top-k over flattened (lag,source) per target after softmax; 0 disables")

    # scoring aggregation
    ap.add_argument("--p_agg", type=str, default="mean", choices=["mean", "max", "topk"])
    ap.add_argument("--p_topk", type=int, default=3)
    ap.add_argument("--d_agg", type=str, default="fro", choices=["fro", "maxrow", "topkrow"])
    ap.add_argument("--d_topk", type=int, default=3)
    ap.add_argument("--lag_agg", type=str, default="mean", choices=["mean", "max"],
                    help="lag별 deviation score를 최종 D로 집계하는 방식")
    ap.add_argument("--lag_fusion", type=str, default="mean", choices=["mean", "max"],
                    help="reconstruction branch에서 lag embeddings를 합치는 방식")

    # paper eval params
    ap.add_argument("--paper_slidingWindow", type=int, default=100)
    ap.add_argument("--paper_vus_version", type=str, default="opt", choices=["opt", "opt_mem"])
    ap.add_argument("--paper_vus_thre", type=int, default=250)
    ap.add_argument("--use_median_vus_window", action="store_true")

    # misc
    ap.add_argument("--mhsa_residual", action="store_true")
    ap.add_argument("--diagnose_components", action="store_true",
                    help="Print paper_eval metrics for P-only / D-only / A(=P*D).")

    # tensorboard
    ap.add_argument("--use_tensorboard", action="store_true")
    ap.add_argument("--tb_root", type=str, default="runs/tensorboard/oraclead_3d_mask")
    ap.add_argument("--tb_histograms", action="store_true")
    ap.add_argument("--tb_figures", action="store_true")

    # local lag masking contribution
    ap.add_argument("--mask_contrib", action="store_true",
                    help="Run inference-time lag-specific local-window masking contribution analysis.")
    ap.add_argument("--mask_fill_value", type=float, default=0.0,
                    help="Value injected into standardized local lag window during masking. Default 0.0 (= train mean).")
    ap.add_argument("--mask_batch", type=int, default=0,
                    help="Batch size for masking analysis. 0 => use --batch.")
    ap.add_argument("--mask_topk", type=int, default=10,
                    help="How many top contribution edges to print.")
    ap.add_argument("--mask_save_csv", action="store_true",
                    help="Save flattened (tau, source, target) contribution table as CSV.")

    ap.add_argument("--out_dir", type=str, default="runs/oraclead_3d_mask")
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
    mask_batch = args.mask_batch if args.mask_batch > 0 else args.batch

    if args.entities:
        wanted = [e.strip() for e in args.entities.split(",") if e.strip()]
        files = [os.path.join(args.input_dir, f"{e}.npz") for e in wanted]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))

    rows = []
    for f in files:
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

        print(f"\n=== {name} (Ttr={train_z.shape[0]}, Tte={test_z.shape[0]}, N={N}) ===", flush=True)
        print(
            f"lr={lr} AdamW(wd={args.weight_decay}) "
            f"L={args.L} tau_max={args.tau_max} lag_win={args.lag_win} "
            f"batch={args.batch} enc/dec={args.enc_layers}/{args.dec_layers} "
            f"lam_recon={args.lam_recon} lam_dev={args.lam_dev} lam_sparse={args.lam_sparse} "
            f"pred_temp={args.pred_temp} self_loop_bias={args.self_loop_bias} lag_source_topk={args.lag_source_topk} "
            f"mhsa_residual={args.mhsa_residual} grad_clip={args.grad_clip} "
            f"start_sls_epoch={args.start_sls_epoch} use_squared_l2={args.use_squared_l2} "
            f"p_agg={args.p_agg}(k={args.p_topk}) d_agg={args.d_agg}(k={args.d_topk}) "
            f"lag_agg={args.lag_agg} lag_fusion={args.lag_fusion} "
            f"use_median_vus_window={args.use_median_vus_window} "
            f"paper_eval(slidingWindow={args.paper_slidingWindow}, version={args.paper_vus_version}, thre={args.paper_vus_thre}) "
            f"mask_contrib={args.mask_contrib} mask_fill_value={args.mask_fill_value} mask_batch={mask_batch}",
            flush=True
        )

        metrics = []
        for seed in seeds:
            print(f"\n[seed {seed}] training ...", flush=True)
            set_seed(seed)

            writer = None
            if args.use_tensorboard:
                log_dir = os.path.join(args.tb_root, name, f"seed{seed}")
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                if hasattr(writer, "add_text"):
                    writer.add_text("config/backend", str(_TB_BACKEND), 0)
                    writer.add_text("config/entity", name, 0)

            model = OracleAD3D(
                N=N, L=args.L, tau_max=args.tau_max, d=args.d, heads=args.heads,
                enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                dropout=args.dropout, mhsa_residual=args.mhsa_residual,
                lag_fusion=args.lag_fusion,
                lag_win=args.lag_win,
                pred_temp=args.pred_temp,
                self_loop_bias=args.self_loop_bias,
                lag_source_topk=args.lag_source_topk
            ).to(device)

            train_one_seed(
                model, train_z, device,
                epochs=args.epochs, batch=args.batch, lr=lr, weight_decay=args.weight_decay,
                lam_recon=args.lam_recon, lam_dev=args.lam_dev, lam_sparse=args.lam_sparse,
                sls_ema=args.sls_ema,
                start_sls_epoch=args.start_sls_epoch,
                use_squared_l2=args.use_squared_l2,
                grad_clip=args.grad_clip,
                writer=writer, writer_prefix=name
            )

            P_w, D_w, A_w = score_series(model, test_z, device, batch=args.batch, args=args)

            Tt = test_z.shape[0]
            start = args.L - 1

            P_t = np.full((Tt,), np.nan, dtype=np.float32)
            D_t = np.full((Tt,), np.nan, dtype=np.float32)
            A_t = np.full((Tt,), np.nan, dtype=np.float32)
            P_t[start:] = P_w
            D_t[start:] = D_w
            A_t[start:] = A_w

            mtr_P = paper_eval_one(P_t, y, start, args)
            mtr_D = paper_eval_one(D_t, y, start, args)
            mtr_A = paper_eval_one(A_t, y, start, args)

            if args.diagnose_components:
                print(
                    f"[seed {seed}] paper_eval components\n"
                    f"  P-only: A-PR={pct(mtr_P['AUC-PR']):.2f}  VUS-PR={pct(mtr_P['VUS-PR']):.2f}  "
                    f"F1={pct(mtr_P['Standard-F1']):.2f}  R-F1={pct(mtr_P['R-based-F1']):.2f}\n"
                    f"  D-only: A-PR={pct(mtr_D['AUC-PR']):.2f}  VUS-PR={pct(mtr_D['VUS-PR']):.2f}  "
                    f"F1={pct(mtr_D['Standard-F1']):.2f}  R-F1={pct(mtr_D['R-based-F1']):.2f}\n"
                    f"  A=P*D:  A-PR={pct(mtr_A['AUC-PR']):.2f}  VUS-PR={pct(mtr_A['VUS-PR']):.2f}  "
                    f"F1={pct(mtr_A['Standard-F1']):.2f}  R-F1={pct(mtr_A['R-based-F1']):.2f}",
                    flush=True
                )

            A_PR   = float(mtr_A["AUC-PR"])
            A_ROC  = float(mtr_A["AUC-ROC"])
            VUS_PR = float(mtr_A["VUS-PR"])
            VUS_ROC= float(mtr_A["VUS-ROC"])
            F1     = float(mtr_A["Standard-F1"])
            PA_F1  = float(mtr_A["PA-F1"])
            EV_F1  = float(mtr_A["Event-based-F1"])
            R_F1   = float(mtr_A["R-based-F1"])
            Aff_F1 = float(mtr_A["Affiliation-F"])

            metrics.append((A_PR, A_ROC, F1, PA_F1, EV_F1, R_F1, Aff_F1, VUS_ROC, VUS_PR))

            print(
                f"[seed {seed}]  "
                f"A-PR={pct(A_PR):.2f}  A-ROC={pct(A_ROC):.2f}  "
                f"F1={pct(F1):.2f}  PA-F1={pct(PA_F1):.2f}  EventF1={pct(EV_F1):.2f}  "
                f"R-F1={pct(R_F1):.2f}  Aff-F={pct(Aff_F1):.2f}  "
                f"VUS-ROC={pct(VUS_ROC):.2f}  VUS-PR={pct(VUS_PR):.2f}",
                flush=True
            )

            # -----------------------------
            # local lag masking contribution
            # -----------------------------
            mask_out = None
            if args.mask_contrib:
                print(f"[seed {seed}] computing local-lag-mask contribution ...", flush=True)
                mask_out = compute_mask_contribution_3d(
                    model, test_z, device, batch=mask_batch, args=args
                )

                G_raw_tau = mask_out["G_raw_tau"]
                G_pos_tau = mask_out["G_pos_tau"]

                print_mask_contrib_summary(name, G_pos_tau, topk=args.mask_topk)

                np.savez(
                    os.path.join(args.out_dir, f"{name}_seed{seed}_mask_contrib.npz"),
                    G_raw_tau=mask_out["G_raw_tau"],
                    G_pos_tau=mask_out["G_pos_tau"],
                    G_raw_lag_mean=mask_out["G_raw_lag_mean"],
                    G_pos_lag_mean=mask_out["G_pos_lag_mean"],
                    G_raw_lag_max=mask_out["G_raw_lag_max"],
                    G_pos_lag_max=mask_out["G_pos_lag_max"],
                    source_strength_tau=mask_out["source_strength_tau"],
                    target_received_tau=mask_out["target_received_tau"],
                    pred_weights=model._get_pred_weights().detach().cpu().numpy().astype(np.float32),
                    mu=mu,
                    sd=sd,
                )

                if args.mask_save_csv:
                    save_mask_contrib_csv(
                        os.path.join(args.out_dir, f"{name}_seed{seed}_mask_contrib.csv"),
                        G_raw_tau, G_pos_tau
                    )

            if writer is not None:
                writer.add_scalar(f"{name}/eval/AUC_PR", A_PR, seed)
                writer.add_scalar(f"{name}/eval/AUC_ROC", A_ROC, seed)
                writer.add_scalar(f"{name}/eval/F1", F1, seed)
                writer.add_scalar(f"{name}/eval/PA_F1", PA_F1, seed)
                writer.add_scalar(f"{name}/eval/Event_F1", EV_F1, seed)
                writer.add_scalar(f"{name}/eval/R_F1", R_F1, seed)
                writer.add_scalar(f"{name}/eval/Aff_F", Aff_F1, seed)
                writer.add_scalar(f"{name}/eval/VUS_ROC", VUS_ROC, seed)
                writer.add_scalar(f"{name}/eval/VUS_PR", VUS_PR, seed)

                if args.diagnose_components:
                    writer.add_scalar(f"{name}/diag/P_AUC_PR", float(mtr_P["AUC-PR"]), seed)
                    writer.add_scalar(f"{name}/diag/P_VUS_PR", float(mtr_P["VUS-PR"]), seed)
                    writer.add_scalar(f"{name}/diag/D_AUC_PR", float(mtr_D["AUC-PR"]), seed)
                    writer.add_scalar(f"{name}/diag/D_VUS_PR", float(mtr_D["VUS-PR"]), seed)
                    writer.add_scalar(f"{name}/diag/A_AUC_PR", float(mtr_A["AUC-PR"]), seed)
                    writer.add_scalar(f"{name}/diag/A_VUS_PR", float(mtr_A["VUS-PR"]), seed)

                with torch.no_grad():
                    pw = model._get_pred_weights().detach().cpu().numpy()
                    writer.add_scalar(f"{name}/eval/pred_weight_entropy", float(model.pred_weight_entropy().detach().cpu()), seed)
                    writer.add_scalar(f"{name}/eval/pred_weight_max", float(pw.max()), seed)

                if args.tb_histograms:
                    tb_log_score_histograms(writer, name, seed, y, P_t, D_t, A_t, start)

                if args.tb_figures:
                    tb_log_score_curves(writer, name, seed, y, P_t, D_t, A_t)

                if mask_out is not None:
                    writer.add_scalar(f"{name}/mask/G_pos_mean", float(mask_out["G_pos_tau"].mean()), seed)
                    writer.add_scalar(f"{name}/mask/G_pos_max", float(mask_out["G_pos_tau"].max()), seed)
                    writer.add_scalar(f"{name}/mask/G_raw_mean", float(mask_out["G_raw_tau"].mean()), seed)
                    writer.add_scalar(f"{name}/mask/source_strength_mean", float(mask_out["source_strength_tau"].mean()), seed)
                    writer.add_scalar(f"{name}/mask/target_received_mean", float(mask_out["target_received_tau"].mean()), seed)

                writer.flush()
                writer.close()

            if args.save_per_seed:
                np.savez(
                    os.path.join(args.out_dir, f"{name}_seed{seed}.npz"),
                    A_t=A_t, P_t=P_t, D_t=D_t, y=y,
                    sls=model.sls.detach().cpu().numpy(),
                    pred_weights=model._get_pred_weights().detach().cpu().numpy().astype(np.float32),
                    mu=mu, sd=sd
                )

        A_PR_m, A_PR_s       = safe_mean_std([m[0] for m in metrics])
        A_ROC_m, A_ROC_s     = safe_mean_std([m[1] for m in metrics])
        F1_m, F1_s           = safe_mean_std([m[2] for m in metrics])
        PA_m, PA_s           = safe_mean_std([m[3] for m in metrics])
        EV_m, EV_s           = safe_mean_std([m[4] for m in metrics])
        R_F1_m, R_F1_s       = safe_mean_std([m[5] for m in metrics])
        Aff_m, Aff_s         = safe_mean_std([m[6] for m in metrics])
        VUS_ROC_m, VUS_ROC_s = safe_mean_std([m[7] for m in metrics])
        VUS_PR_m, VUS_PR_s   = safe_mean_std([m[8] for m in metrics])

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
            "entity",
            "AUC_PR", "AUC_ROC",
            "F1", "PA_F1", "Event_F1", "R_F1", "Aff_F",
            "VUS_ROC", "VUS_PR"
        ])
        df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
        print("\nSaved summary:", os.path.join(args.out_dir, "summary.csv"), flush=True)


if __name__ == "__main__":
    main()