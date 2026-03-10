import argparse, os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
# StandardScaler(train-fit)  (Appendix D.1)
# ============================================================
def standardize_train_test(train: np.ndarray, test: np.ndarray):
    train = train.astype(np.float32)
    test  = test.astype(np.float32)

    # StandardScaler 전에 NaN/Inf 처리 필요 (논문에 명시가 없지만 현실적으로 필수)
    train = np.where(np.isfinite(train), train, np.nan)
    test  = np.where(np.isfinite(test),  test,  np.nan)

    col_mean = np.nanmean(train, axis=0, keepdims=True).astype(np.float32)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)

    train = np.where(np.isnan(train), col_mean, train).astype(np.float32)
    test  = np.where(np.isnan(test),  col_mean, test ).astype(np.float32)

    mu = train.mean(axis=0, keepdims=True).astype(np.float32)
    var = ((train - mu) ** 2).mean(axis=0, keepdims=True).astype(np.float32)  # ddof=0 (StandardScaler)
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


# ============================================================
# Sliding window dataset
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
        return torch.from_numpy(self.x[idx:idx+self.L])  # (L,N)


# ============================================================
# Metrics helpers
# ============================================================
def _mask_valid(y, score):
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(score).astype(np.float64)
    m = ~np.isnan(s)
    return y[m], s[m]

def _segments(y01):
    segs=[]
    in_seg=False; s=0
    for i,v in enumerate(y01):
        if v==1 and not in_seg:
            in_seg=True; s=i
        elif v==0 and in_seg:
            in_seg=False; segs.append((s,i-1))
    if in_seg:
        segs.append((s,len(y01)-1))
    return segs

def _overlap_len(a,b):
    s=max(a[0],b[0]); e=min(a[1],b[1])
    return max(0, e-s+1)


# ============================================================
# AUC-PR / AUC-ROC  (Appendix B.4)
# ============================================================
def auc_pr(y_true, score):
    y, s = _mask_valid(y_true, score)
    npos = int((y==1).sum())
    if npos==0: return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tps = np.cumsum(y==1)
    fps = np.cumsum(y==0)
    prec = tps / np.maximum(tps + fps, 1)
    rec  = tps / npos
    prec = np.concatenate([[1.0], prec])
    rec  = np.concatenate([[0.0], rec])
    return float(np.trapz(prec, rec))

def auc_roc(y_true, score):
    y, s = _mask_valid(y_true, score)
    npos = int((y==1).sum())
    nneg = int((y==0).sum())
    if npos==0 or nneg==0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    ranks = np.arange(1, len(y_sorted)+1, dtype=np.float64)
    sum_ranks_pos = ranks[y_sorted==1].sum()
    auc = (sum_ranks_pos - npos*(npos+1)/2.0) / (npos*nneg)
    return float(auc)


# ============================================================
# Point-wise F1 (Appendix B.1) + Optimal threshold search
# - thr_mode: quantile (fast) / unique (exact, possibly slow)
# ============================================================
def f1_point(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    if tp==0: return 0.0
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def best_f1_point(y_true, score, thr_mode="quantile", n_q=1200):
    y, s = _mask_valid(y_true, score)
    if int(y.sum())==0:
        return 0.0, float("nan")

    if thr_mode == "unique":
        thr_list = np.unique(s)[::-1]
    else:
        qs = np.linspace(0.0, 1.0, n_q)
        thr_list = np.unique(np.quantile(s, qs))[::-1]

    best=-1.0; best_thr=float(thr_list[0])
    for thr in thr_list:
        yp = (s >= thr).astype(np.int32)
        val = f1_point(y, yp)
        if val>best:
            best=val; best_thr=float(thr)
    return float(best), float(best_thr)


# ============================================================
# Range-F1 (TSB-AD style)  Appendix B.2
# - bias-adjusted weighted coverage ω
# - CF cardinality penalty
# - ER existence reward (alpha term)
# - precision computed symmetrically with alpha=0 (paper)
# ============================================================
def _bias_weights(length: int, bias: str):
    if length <= 0:
        return np.array([], dtype=np.float64)
    if bias == "flat":
        w = np.ones(length, dtype=np.float64)
    elif bias == "front":
        w = np.linspace(1.0, 0.0, length, dtype=np.float64)
        w = np.maximum(w, 0.0)
    elif bias == "back":
        w = np.linspace(0.0, 1.0, length, dtype=np.float64)
        w = np.maximum(w, 0.0)
    elif bias == "middle":
        mid = (length - 1) / 2.0
        idx = np.arange(length, dtype=np.float64)
        w = 1.0 - np.abs(idx - mid) / max(mid, 1.0)
        w = np.maximum(w, 0.0)
    else:
        w = np.ones(length, dtype=np.float64)

    s = w.sum()
    if s <= 0:
        return np.ones(length, dtype=np.float64) / length
    return w / s

def _weighted_coverage(seg, y_pred01, bias: str):
    s,e = seg
    L = e - s + 1
    w = _bias_weights(L, bias)
    hit = y_pred01[s:e+1].astype(np.float64)
    return float((w * hit).sum())

def _cf(seg, pred_segs):
    s,e = seg
    k = 0
    for ps,pe in pred_segs:
        if _overlap_len((s,e), (ps,pe)) > 0:
            k += 1
    if k == 0:
        return 0.0
    return 1.0 / k

def _existence_reward(gt_segs, pred_segs):
    if len(gt_segs) == 0:
        return 0.0
    hit = 0
    for g in gt_segs:
        ok = False
        for p in pred_segs:
            if _overlap_len(g,p) > 0:
                ok = True
                break
        hit += 1 if ok else 0
    return hit / len(gt_segs)

def range_f1_tsb(y_true01, y_pred01, alpha=0.0, bias="flat"):
    y_true01 = y_true01.astype(np.int32)
    y_pred01 = y_pred01.astype(np.int32)

    gt_segs = _segments(y_true01)
    pr_segs = _segments(y_pred01)
    if len(gt_segs) == 0:
        return 0.0

    # Recall_range
    rec_terms = []
    for g in gt_segs:
        omega = _weighted_coverage(g, y_pred01, bias=bias)
        cf = _cf(g, pr_segs)
        rec_terms.append(omega * cf)
    recall_range = float(np.mean(rec_terms)) + float(alpha) * _existence_reward(gt_segs, pr_segs)

    # Precision_range (swap roles), alpha=0 per paper
    if len(pr_segs) == 0:
        precision_range = 0.0
    else:
        prec_terms = []
        for p in pr_segs:
            omega = _weighted_coverage(p, y_true01, bias=bias)
            cf = _cf(p, gt_segs)
            prec_terms.append(omega * cf)
        precision_range = float(np.mean(prec_terms))

    if precision_range + recall_range == 0.0:
        return 0.0
    return float(2.0 * precision_range * recall_range / (precision_range + recall_range))

def best_range_f1_tsb(y_true, score, thr_mode="quantile", n_q=1200, alpha=0.0, bias="flat"):
    y, s = _mask_valid(y_true, score)
    if int(y.sum()) == 0:
        return 0.0, float("nan")

    if thr_mode == "unique":
        thr_list = np.unique(s)[::-1]
    else:
        qs = np.linspace(0.0, 1.0, n_q)
        thr_list = np.unique(np.quantile(s, qs))[::-1]

    best=-1.0; best_thr=float(thr_list[0])
    for thr in thr_list:
        yp = (s >= thr).astype(np.int32)
        val = range_f1_tsb(y, yp, alpha=alpha, bias=bias)
        if val > best:
            best = val
            best_thr = float(thr)
    return float(best), float(best_thr)


# ============================================================
# Affiliation-F1 (Appendix B.3) - optional if package exists
# ============================================================
def affiliation_f1_optional(y_true, score):
    try:
        from affiliation.metrics import pr_from_events  # type: ignore
    except Exception:
        return float("nan"), float("nan")

    y, s = _mask_valid(y_true, score)
    gt = _segments(y)
    if len(gt)==0:
        return float("nan"), float("nan")

    # threshold search (light)
    qs = np.linspace(0.0, 1.0, 200)
    thr_list = np.unique(np.quantile(s, qs))[::-1]
    best=-1.0; best_thr=float(thr_list[0])

    for thr in thr_list:
        yp = (s >= thr).astype(np.int32)
        pr = _segments(yp)
        P,R = pr_from_events(pr, gt)
        val = 0.0 if (P+R)==0 else float(2*P*R/(P+R))
        if val>best:
            best=val; best_thr=float(thr)
    return float(best), float(best_thr)


# ============================================================
# VUS (Appendix B.5) via tolerance dilation and averaging over omegas
# ============================================================
def dilate_labels(y_true, omega: int):
    y = y_true.copy().astype(np.int32)
    if omega <= 0:
        return y
    segs = _segments(y)
    yd = np.zeros_like(y)
    T = len(y)
    for s,e in segs:
        ss = max(0, s-omega)
        ee = min(T-1, e+omega)
        yd[ss:ee+1] = 1
    return yd

def vus_roc_pr(y_true, score, omegas):
    y, s = _mask_valid(y_true, score)
    if int(y.sum())==0:
        return float("nan"), float("nan")
    aucs_roc=[]
    aucs_pr=[]
    for om in omegas:
        yd = dilate_labels(y, int(om))
        aucs_roc.append(auc_roc(yd, s))
        aucs_pr.append(auc_pr(yd, s))
    return float(np.nanmean(aucs_roc)), float(np.nanmean(aucs_pr))


# ============================================================
# OracleAD 2D model (paper-aligned)
# - per-variable LSTM Encoder/Decoder depth=2
# - attention pooling -> c_i
# - MHSA refine -> c_i*
# - decoder input z=all-zero vector length L
# ============================================================
class TemporalAttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1, bias=True)
    def forward(self, H):  # (B, L-1, d)
        a = torch.softmax(self.score(H).squeeze(-1), dim=1)
        return (H * a.unsqueeze(-1)).sum(dim=1)

class PerVarEncoder(nn.Module):
    def __init__(self, d: int, num_layers: int, dropout: float):
        super().__init__()
        do = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(1, d, batch_first=True, num_layers=num_layers, dropout=do)
        self.pool = TemporalAttnPool(d)
    def forward(self, x):
        H,_ = self.lstm(x)
        return self.pool(H)

class PerVarDecoder(nn.Module):
    def __init__(self, d: int, L: int, num_layers: int, dropout: float):
        super().__init__()
        self.L = L
        self.d = d
        self.num_layers = num_layers
        do = dropout if num_layers > 1 else 0.0
        self.init_h = nn.Linear(d, num_layers * d)
        self.init_c = nn.Linear(d, num_layers * d)
        self.lstm = nn.LSTM(1, d, batch_first=True, num_layers=num_layers, dropout=do)
        self.out = nn.Linear(d, 1)

    def forward(self, c_star):
        B,d = c_star.shape
        z = torch.zeros(B, self.L, 1, device=c_star.device, dtype=c_star.dtype)
        h0 = torch.tanh(self.init_h(c_star)).view(self.num_layers, B, d).contiguous()
        c0 = torch.tanh(self.init_c(c_star)).view(self.num_layers, B, d).contiguous()
        Y,_ = self.lstm(z, (h0, c0))
        O = self.out(Y).squeeze(-1)     # (B,L)
        recon = O[:, :self.L-1]
        pred  = O[:, self.L-1]
        return recon, pred

def pairwise_sq_l2(C):
    A2 = (C*C).sum(dim=2)
    G  = torch.bmm(C, C.transpose(1,2))
    D  = A2.unsqueeze(2) + A2.unsqueeze(1) - 2.0 * G
    return torch.clamp(D, min=0.0)

class OracleAD2D(nn.Module):
    def __init__(self, N: int, L: int, d: int, heads: int,
                 enc_layers: int, dec_layers: int, dropout: float,
                 mhsa_residual: bool = False):
        super().__init__()
        self.N=N; self.L=L; self.d=d
        self.encoders = nn.ModuleList([PerVarEncoder(d, enc_layers, dropout) for _ in range(N)])
        self.mhsa = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.decoders = nn.ModuleList([PerVarDecoder(d, L, dec_layers, dropout) for _ in range(N)])
        self.mhsa_residual = mhsa_residual

        self.register_buffer("sls", torch.zeros(N, N), persistent=True)
        self.has_sls = False

    def reset_sls(self):
        self.sls.zero_()
        self.has_sls = False

    def forward(self, X):  # (B,L,N)
        B,L,N = X.shape
        c_list=[]
        for i in range(N):
            ci = self.encoders[i](X[:, :L-1, i].unsqueeze(-1))
            c_list.append(ci)
        C = torch.stack(c_list, dim=1)  # (B,N,d)

        A,_ = self.mhsa(C, C, C, need_weights=False)
        C_star = (C + A) if self.mhsa_residual else A

        recon_list=[]; pred_list=[]
        for i in range(N):
            r,p = self.decoders[i](C_star[:, i, :])
            recon_list.append(r); pred_list.append(p)
        recon = torch.stack(recon_list, dim=-1)  # (B,L-1,N)
        pred  = torch.stack(pred_list,  dim=-1)  # (B,N)
        return recon, pred, C_star


# ============================================================
# Train / Score
# ============================================================
def train_one_seed(model, train_TN, device,
                   epochs, batch, lr, weight_decay,
                   lam_recon, lam_dev, sls_ema):
    model.reset_sls()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # Appendix D.1

    ds = SlidingWindowDataset(train_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)

    N = model.N
    for ep in range(1, epochs+1):
        model.train()
        sls_sum = torch.zeros(N, N, device=device)
        sls_cnt = 0

        lp=lrn=ld=0.0; steps=0
        for X in loader:
            X = X.to(device)
            recon, pred, C_star = model(X)

            xL = X[:, -1, :]
            xpast = X[:, :model.L-1, :]

            loss_pred  = torch.mean((xL - pred) ** 2)
            loss_recon = torch.mean((xpast - recon) ** 2)

            D = pairwise_sq_l2(C_star)
            sls_sum += D.mean(dim=0).detach()
            sls_cnt += 1

            if model.has_sls:
                diff = D - model.sls.unsqueeze(0)
                loss_dev = torch.mean(diff**2)
                loss = loss_pred + lam_recon*loss_recon + lam_dev*loss_dev
            else:
                loss_dev = torch.tensor(0.0, device=device)
                loss = loss_pred + lam_recon*loss_recon

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            lp += float(loss_pred.detach().cpu())
            lrn += float(loss_recon.detach().cpu())
            ld += float(loss_dev.detach().cpu())
            steps += 1

        with torch.no_grad():
            epoch_sls = sls_sum / max(sls_cnt, 1)
            if not model.has_sls:
                model.sls.copy_(epoch_sls)
                model.has_sls = True
            else:
                if sls_ema <= 0.0:
                    model.sls.copy_(epoch_sls)
                else:
                    beta = float(sls_ema)
                    model.sls.mul_(beta).add_(epoch_sls * (1.0 - beta))

        print(f"  [ep {ep:02d}] pred={lp/steps:.6f} recon={lrn/steps:.6f} dev={ld/steps:.3e} has_sls={model.has_sls}", flush=True)


@torch.no_grad()
def score_series(model, test_TN, device, batch):
    model.eval()
    ds = SlidingWindowDataset(test_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    W = len(ds)
    P_w = np.zeros((W,), dtype=np.float32)
    D_w = np.zeros((W,), dtype=np.float32)

    sls = model.sls.detach()
    offset = 0
    for X in loader:
        X = X.to(device)
        recon, pred, C_star = model(X)

        xL = X[:, -1, :]
        P = (xL - pred).abs().mean(dim=1)  # Eq.(15)

        Dm = pairwise_sq_l2(C_star)
        diff = Dm - sls.unsqueeze(0)
        fro = torch.linalg.norm(diff, ord="fro", dim=(1,2))  # Eq.(16)

        bsz = X.shape[0]
        P_w[offset:offset+bsz] = P.detach().cpu().numpy().astype(np.float32)
        D_w[offset:offset+bsz] = fro.detach().cpu().numpy().astype(np.float32)
        offset += bsz

    A_w = (P_w * D_w).astype(np.float32)  # Eq.(17)
    return P_w, D_w, A_w


def evaluate_metrics(y, A_t, vus_omegas, rf1_alpha=0.0, rf1_bias="flat",
                     thr_mode="quantile", rf1_thr_mode="quantile"):
    A_PR  = auc_pr(y, A_t)
    A_ROC = auc_roc(y, A_t)

    F1, _   = best_f1_point(y, A_t, thr_mode=thr_mode, n_q=1200)
    R_F1, _ = best_range_f1_tsb(y, A_t, thr_mode=rf1_thr_mode, n_q=1200,
                                alpha=rf1_alpha, bias=rf1_bias)

    Aff_F1,_ = affiliation_f1_optional(y, A_t)
    VUS_ROC, VUS_PR = vus_roc_pr(y, A_t, vus_omegas)

    return A_PR, A_ROC, F1, R_F1, Aff_F1, VUS_ROC, VUS_PR


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--entities", type=str, default="")
    ap.add_argument("--dataset", type=str, default="PSM", choices=["PSM","SMD","SWaT","OTHER"])

    # paper defaults (Appendix D.2)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=80)

    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--dec_layers", type=int, default=2)

    ap.add_argument("--lam_recon", type=float, default=0.1)
    ap.add_argument("--lam_dev", type=float, default=3.0)
    ap.add_argument("--sls_ema", type=float, default=0.0)

    # paper lr (Appendix D.1)
    ap.add_argument("--lr", type=float, default=0.0, help="0 => paper defaults (PSM=5e-5, others=5e-4)")
    ap.add_argument("--weight_decay", type=float, default=0.01)  # AdamW default in PyTorch

    # seeds (Appendix C suggests 5 seeds)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")

    # VUS omegas (Appendix B.5): not specified exactly, default 0..10
    ap.add_argument("--vus_omegas", type=str, default="0,1,2,3,4,5,6,7,8,9,10")

    # Range-F1 params (Appendix B.2)
    ap.add_argument("--rf1_alpha", type=float, default=0.0)
    ap.add_argument("--rf1_bias", type=str, default="flat", choices=["flat","front","back","middle"])

    # threshold search modes
    ap.add_argument("--thr_mode", type=str, default="quantile", choices=["quantile","unique"],
                    help="F1 optimal threshold search")
    ap.add_argument("--rf1_thr_mode", type=str, default="quantile", choices=["quantile","unique"],
                    help="Range-F1 optimal threshold search")

    # MHSA detail (implementation ambiguity)
    ap.add_argument("--mhsa_residual", action="store_true")

    ap.add_argument("--out_dir", type=str, default="runs/oraclead_paper_2d_full")
    ap.add_argument("--save_per_seed", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    lr = float(args.lr) if (args.lr and args.lr > 0) else (5e-5 if args.dataset=="PSM" else 5e-4)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    vus_omegas = [int(x.strip()) for x in args.vus_omegas.split(",") if x.strip()]

    if args.entities:
        wanted = [e.strip() for e in args.entities.split(",") if e.strip()]
        files = [os.path.join(args.input_dir, f"{e}.npz") for e in wanted]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))

    def safe_mean_std(arr):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std())

    rows=[]
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        data = np.load(f)
        train = data["train"].astype(np.float32)
        test  = data["test"].astype(np.float32)
        y = reduce_label(data["label"], test.shape[0])

        if train.ndim==1: train=train[:,None]
        if test.ndim==1: test=test[:,None]
        if train.shape[1] != test.shape[1]:
            print("[skip]", name, "N mismatch", flush=True)
            continue

        train_z, test_z, mu, sd = standardize_train_test(train, test)
        N = train_z.shape[1]
        if train_z.shape[0] < args.L + 1 or test_z.shape[0] < args.L + 1:
            print("[skip]", name, "too short", flush=True)
            continue

        print(f"\n=== {name} (Ttr={train_z.shape[0]}, Tte={test_z.shape[0]}, N={N}) ===", flush=True)
        print(f"lr={lr} AdamW(wd={args.weight_decay}) L={args.L} batch={args.batch} lam_dev={args.lam_dev} "
              f"enc/dec={args.enc_layers}/{args.dec_layers} rf1_bias={args.rf1_bias} "
              f"thr_mode={args.thr_mode} rf1_thr_mode={args.rf1_thr_mode} mhsa_residual={args.mhsa_residual}",
              flush=True)

        metrics=[]
        for seed in seeds:
            print(f"\n[seed {seed}] training ...", flush=True)
            set_seed(seed)

            model = OracleAD2D(
                N=N, L=args.L, d=args.d, heads=args.heads,
                enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                dropout=args.dropout, mhsa_residual=args.mhsa_residual
            ).to(device)

            train_one_seed(
                model, train_z, device,
                epochs=args.epochs, batch=args.batch, lr=lr, weight_decay=args.weight_decay,
                lam_recon=args.lam_recon, lam_dev=args.lam_dev, sls_ema=args.sls_ema
            )

            P_w, D_w, A_w = score_series(model, test_z, device, batch=args.batch)

            # align to time index; exclude first L-1 points (Appendix D / paper main text)
            Tt = test_z.shape[0]
            start = args.L - 1
            A_t = np.full((Tt,), np.nan, dtype=np.float32)
            A_t[start:] = A_w

            A_PR, A_ROC, F1, R_F1, Aff_F1, VUS_ROC, VUS_PR = evaluate_metrics(
                y, A_t, vus_omegas,
                rf1_alpha=args.rf1_alpha, rf1_bias=args.rf1_bias,
                thr_mode=args.thr_mode, rf1_thr_mode=args.rf1_thr_mode
            )

            metrics.append((A_PR, A_ROC, F1, R_F1, Aff_F1, VUS_ROC, VUS_PR))
            print(f"[seed {seed}]  A-PR={A_PR*100:.2f}  A-ROC={A_ROC*100:.2f}  "
                  f"F1={F1*100:.2f}  R-F1={R_F1*100:.2f}  "
                  f"Aff-F1={(Aff_F1*100) if np.isfinite(Aff_F1) else float('nan'):.2f}  "
                  f"VUS-ROC={VUS_ROC*100:.2f}  VUS-PR={VUS_PR*100:.2f}", flush=True)

            if args.save_per_seed:
                np.savez(os.path.join(args.out_dir, f"{name}_seed{seed}.npz"),
                         A_t=A_t, y=y,
                         sls=model.sls.detach().cpu().numpy(),
                         mu=mu, sd=sd)

        A_PR_m, A_PR_s = safe_mean_std([m[0] for m in metrics])
        A_ROC_m, A_ROC_s = safe_mean_std([m[1] for m in metrics])
        F1_m, F1_s = safe_mean_std([m[2] for m in metrics])
        R_F1_m, R_F1_s = safe_mean_std([m[3] for m in metrics])
        Aff_m, Aff_s = safe_mean_std([m[4] for m in metrics])
        VUS_ROC_m, VUS_ROC_s = safe_mean_std([m[5] for m in metrics])
        VUS_PR_m, VUS_PR_s = safe_mean_std([m[6] for m in metrics])

        print(f"\n[{name}] mean±std over {len(seeds)} seeds:", flush=True)
        print(f"  A-PR   {A_PR_m*100:.2f} ± {A_PR_s*100:.2f}", flush=True)
        print(f"  A-ROC  {A_ROC_m*100:.2f} ± {A_ROC_s*100:.2f}", flush=True)
        print(f"  F1     {F1_m*100:.2f} ± {F1_s*100:.2f}", flush=True)
        print(f"  R-F1   {R_F1_m*100:.2f} ± {R_F1_s*100:.2f}", flush=True)
        print(f"  Aff-F1 {(Aff_m*100) if np.isfinite(Aff_m) else float('nan'):.2f} ± {(Aff_s*100) if np.isfinite(Aff_s) else float('nan'):.2f}", flush=True)
        print(f"  VUS-ROC {VUS_ROC_m*100:.2f} ± {VUS_ROC_s*100:.2f}", flush=True)
        print(f"  VUS-PR  {VUS_PR_m*100:.2f} ± {VUS_PR_s*100:.2f}", flush=True)

        rows.append((name, A_PR_m, A_ROC_m, F1_m, R_F1_m, Aff_m, VUS_ROC_m, VUS_PR_m))

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["entity","A_PR","A_ROC","F1","R_F1","Aff_F1","VUS_ROC","VUS_PR"])
        df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
        print("\nSaved summary:", os.path.join(args.out_dir, "summary.csv"), flush=True)


if __name__ == "__main__":
    main()

    