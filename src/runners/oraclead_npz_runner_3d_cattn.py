import argparse, os, glob, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# utils
# -------------------------
def standardize_train_test(train, test, eps=1e-3, clip=10.0):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    train_z = (train - mu) / sd
    test_z  = (test  - mu) / sd
    if clip is not None and clip > 0:
        train_z = np.clip(train_z, -clip, clip)
        test_z  = np.clip(test_z,  -clip, clip)
    train_z = np.nan_to_num(train_z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test_z  = np.nan_to_num(test_z,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return train_z, test_z, mu.astype(np.float32), sd.astype(np.float32)

def reduce_label(y, T):
    y = np.asarray(y)
    if y.ndim == 2:
        y = (y.sum(axis=1) > 0).astype(np.int32)
    else:
        y = y.astype(np.int32)
    if len(y) != T:
        raise ValueError(f"label length mismatch: {len(y)} != {T}")
    return y

def auc_pr(y_true, score):
    mask = ~np.isnan(score)
    y_true = y_true[mask].astype(np.int32)
    score  = score[mask].astype(np.float64)
    npos = (y_true == 1).sum()
    if npos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    prec = tps / np.maximum(tps + fps, 1)
    rec  = tps / npos
    prec = np.concatenate([[1.0], prec])
    rec  = np.concatenate([[0.0], rec])
    return float(np.trapz(prec, rec))

def segments_from_labels(y):
    segs=[]; in_seg=False; s=0
    for i,v in enumerate(y):
        if v==1 and not in_seg:
            in_seg=True; s=i
        elif v==0 and in_seg:
            in_seg=False; segs.append((s,i-1))
    if in_seg:
        segs.append((s,len(y)-1))
    return segs

def point_adjust_preds(y_pred, y_true):
    y_adj = y_pred.copy()
    for s,e in segments_from_labels(y_true):
        if y_pred[s:e+1].sum() > 0:
            y_adj[s:e+1] = 1
    return y_adj

def f1(y_true, y_pred):
    tp = ((y_true==1)&(y_pred==1)).sum()
    fp = ((y_true==0)&(y_pred==1)).sum()
    fn = ((y_true==1)&(y_pred==0)).sum()
    if tp==0: return 0.0
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def best_f1(y_true, score, point_adjust=False):
    """
    빠른 BestF1:
      - point_adjust=False: 정확 + 빠름(정렬/누적)
      - point_adjust=True : 샘플링(200개 quantile)로 근사(속도용)
    """
    mask = ~np.isnan(score)
    y = y_true[mask].astype(np.int32)
    s = score[mask].astype(np.float64)

    if y.sum() == 0:
        return 0.0, float("nan")

    if point_adjust:
        qs = np.linspace(0.0, 1.0, 200)
        thr_list = np.unique(np.quantile(s, qs))[::-1]
        best = -1.0
        best_thr = float(thr_list[0])
        for thr in thr_list:
            y_pred = (s >= thr).astype(np.int32)
            y_pred = point_adjust_preds(y_pred, y)
            val = f1(y, y_pred)
            if val > best:
                best = val
                best_thr = float(thr)
        return float(best), float(best_thr)

    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    npos = tp[-1]
    prec = tp / np.maximum(tp + fp, 1)
    rec  = tp / npos
    f1s  = (2 * prec * rec) / np.maximum(prec + rec, 1e-12)

    change = np.r_[True, s[1:] != s[:-1]]
    f1_u = f1s[change]
    thr_u = s[change]

    j = int(np.argmax(f1_u)) if len(f1_u) else 0
    return float(f1_u[j]) if len(f1_u) else 0.0, float(thr_u[j]) if len(thr_u) else float("nan")


# -------------------------
# dataset
# -------------------------
class SlidingWindowDataset(Dataset):
    def __init__(self, series_TN: np.ndarray, L: int):
        self.x = series_TN.astype(np.float32)
        self.L = L
        self.T, self.N = self.x.shape
        if self.T < L:
            raise ValueError(f"T={self.T} < L={L}")
    def __len__(self):
        return self.T - self.L + 1
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx:idx+self.L])  # (L,N)


# -------------------------
# OracleAD modules
# -------------------------
class TemporalAttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1, bias=True)
    def forward(self, H):  # (B, L-1, d)
        a = torch.softmax(self.score(H).squeeze(-1), dim=1)  # (B,L-1)
        c = (H * a.unsqueeze(-1)).sum(dim=1)                 # (B,d)
        return c

class PerVarEncoder(nn.Module):
    def __init__(self, d: int, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(1, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.pool = TemporalAttnPool(d)
    def forward(self, x):  # (B,L-1,1)
        H,_ = self.lstm(x)
        H = self.drop(H)
        return self.pool(H)

class PerVarDecoder(nn.Module):
    """
    all-zero z, length L output:
      recon = first L-1
      pred  = last
    """
    def __init__(self, d: int, L: int, dropout=0.0):
        super().__init__()
        self.L = L
        self.init_h = nn.Linear(d, d)
        self.init_c = nn.Linear(d, d)
        self.lstm = nn.LSTM(1, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 1)
    def forward(self, c_star):  # (B,d)
        B,d = c_star.shape
        z = torch.zeros(B, self.L, 1, device=c_star.device, dtype=c_star.dtype)
        h0 = torch.tanh(self.init_h(c_star)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(c_star)).unsqueeze(0)
        Y,_ = self.lstm(z, (h0, c0))
        Y = self.drop(Y)
        O = self.out(Y).squeeze(-1)  # (B,L)
        recon = O[:, :self.L-1]
        pred  = O[:, self.L-1]
        return recon, pred

def pairwise_sq_l2(A, B):
    """
    squared L2 via GEMM:
      A,B: (B,N,d) -> (B,N,N)
      ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    """
    A2 = (A*A).sum(dim=2)                       # (B,N)
    B2 = (B*B).sum(dim=2)                       # (B,N)
    G  = torch.bmm(A, B.transpose(1,2))         # (B,N,N)
    D  = A2.unsqueeze(2) + B2.unsqueeze(1) - 2.0 * G
    return torch.clamp(D, min=0.0)

class OracleAD3D_CAttn(nn.Module):
    """
    A 방향:
      - 윈도우 내 변수간 MHSA(동시점 self-attn) 제거
      - 과거 윈도우 임베딩(K,V) -> 현재 윈도우 임베딩(Q) cross-attention으로 문맥 반영
      - lag τ=1..tau_max에 대해 cross-attn output을 집계(mean/decay)해서 현재 임베딩을 갱신
    """
    def __init__(self, N: int, L: int, d=64, heads=4, dropout=0.0, tau_max=5,
                 attn_agg="mean", attn_decay_alpha=0.7, residual=True):
        super().__init__()
        self.N=N; self.L=L; self.d=d; self.tau_max=tau_max
        self.attn_agg = attn_agg
        self.attn_decay_alpha = attn_decay_alpha
        self.residual = residual

        self.encoders = nn.ModuleList([PerVarEncoder(d, dropout) for _ in range(N)])
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.decoders = nn.ModuleList([PerVarDecoder(d, L, dropout) for _ in range(N)])

        self.register_buffer("sls3d", torch.zeros(tau_max, N, N), persistent=True)
        self.has_sls = False

    def reset_sls(self):
        self.sls3d.zero_()
        self.has_sls = False

    def encode(self, X):  # X: (B,L,N) -> C: (B,N,d)
        B,L,N = X.shape
        c_list=[]
        for i in range(N):
            ci = self.encoders[i](X[:, :L-1, i].unsqueeze(-1))  # (B,d)
            c_list.append(ci)
        C = torch.stack(c_list, dim=1)  # (B,N,d)
        return C

    def causal_cross_attend_over_time(self, C):
        """
        C: (B,N,d) where B is "time-ordered windows" (must be sequential in batch)
        output: C_star (B,N,d)
          for each tau:
            Q = C[tau:] , K/V = C[:-tau]
            out_tau -> align to indices [tau: ]
          aggregate over tau by mean or exp-decay
        """
        B,N,d = C.shape
        tau_max = min(self.tau_max, B-1)  # if B too small
        if tau_max <= 0:
            return C

        out_sum = torch.zeros_like(C)
        w_sum   = torch.zeros((B,1,1), device=C.device, dtype=C.dtype)

        for tau in range(1, tau_max+1):
            Q = C[tau:]     # (B-tau,N,d)
            KV = C[:-tau]   # (B-tau,N,d)
            out_tau, _ = self.cross_attn(Q, KV, KV, need_weights=False)  # (B-tau,N,d)

            if self.attn_agg == "decay":
                w = float(np.exp(-self.attn_decay_alpha * (tau-1)))
            else:
                w = 1.0

            out_sum[tau:] += out_tau * w
            w_sum[tau:]   += w

        out = out_sum / torch.clamp(w_sum, min=1e-8)

        if self.residual:
            return C + out
        return out

    def decode(self, C_star):  # (B,N,d) -> recon(B,L-1,N), pred(B,N)
        B,N,d = C_star.shape
        recon_list=[]; pred_list=[]
        for i in range(N):
            r,p = self.decoders[i](C_star[:, i, :])
            recon_list.append(r); pred_list.append(p)
        recon = torch.stack(recon_list, dim=-1)  # (B,L-1,N)
        pred  = torch.stack(pred_list,  dim=-1)  # (B,N)
        return recon, pred

    def forward(self, X):  # (B,L,N)
        C = self.encode(X)                                # (B,N,d)
        C_star = self.causal_cross_attend_over_time(C)    # (B,N,d)
        recon, pred = self.decode(C_star)
        return recon, pred, C_star


# -------------------------
# Train (3D SLS)
# -------------------------
def train_3d(model: OracleAD3D_CAttn, train_TN, device,
             epochs=10, batch=1024, lr=1e-4,
             lam_recon=0.1, lam_dev=0.3, sls_ema=0.9):
    """
    중요: shuffle=False (배치가 연속 윈도우여야 causal cross-attn / lag pairing이 의미 있음)
    epoch1: SLS 없어서 dev loss 생략
    epoch end: τ별 평균 SLS3D + EMA 업데이트
    """
    model.reset_sls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = SlidingWindowDataset(train_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=True, num_workers=0)

    tau_max = model.tau_max
    N = model.N

    for ep in range(1, epochs+1):
        model.train()
        sls_sum = torch.zeros(tau_max, N, N, device=device)
        sls_cnt = torch.zeros(tau_max, device=device)

        lp=lrn=ld=0.0; steps=0

        for X in loader:
            X = X.to(device)
            recon, pred, C_star = model(X)  # C_star: (B,N,d) time-ordered windows in batch

            xL = X[:, -1, :]
            xpast = X[:, :model.L-1, :]

            loss_pred  = torch.mean((xL - pred) ** 2)
            loss_recon = torch.mean((xpast - recon) ** 2)

            dev_terms=[]
            for tau in range(1, tau_max+1):
                if C_star.shape[0] <= tau:
                    continue
                A = C_star[:-tau]
                B = C_star[tau:]
                D_tau = pairwise_sq_l2(A, B)              # (B-tau,N,N)
                sls_sum[tau-1] += D_tau.mean(dim=0).detach()
                sls_cnt[tau-1] += 1

                if model.has_sls:
                    diff = D_tau - model.sls3d[tau-1].unsqueeze(0)
                    dev_terms.append(torch.mean(diff**2))

            if model.has_sls and len(dev_terms)>0:
                loss_dev = torch.stack(dev_terms).mean()
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
            epoch_sls = model.sls3d.clone()
            for tau in range(1, tau_max+1):
                if sls_cnt[tau-1] > 0:
                    epoch_sls[tau-1] = sls_sum[tau-1] / sls_cnt[tau-1]
            if not model.has_sls:
                model.sls3d.copy_(epoch_sls)
                model.has_sls = True
            else:
                beta=float(sls_ema)
                model.sls3d.mul_(beta).add_(epoch_sls * (1.0 - beta))

        print(f"  [ep {ep:02d}] pred={lp/steps:.6f} recon={lrn/steps:.6f} dev={ld/steps:.3e} has_sls={model.has_sls}", flush=True)


# -------------------------
# Score (3D) with progress (chunked, keeps lag context)
# -------------------------
@torch.no_grad()
def encode_all_windows(model: OracleAD3D_CAttn, series_TN, device, batch=1024):
    """
    모든 윈도우의 raw embedding C(b)을 먼저 쭉 뽑음 (윈도우 독립)
    return:
      C_raw: (W,N,d) CPU float32
      xL   : (W,N)   CPU float32
    """
    model.eval()
    ds = SlidingWindowDataset(series_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    C_list=[]; xL_list=[]
    for X in loader:
        X = X.to(device)
        C = model.encode(X)              # (B,N,d)
        xL = X[:, -1, :]                 # (B,N)
        C_list.append(C.detach().cpu())
        xL_list.append(xL.detach().cpu())
    C_raw = torch.cat(C_list, dim=0).numpy().astype(np.float32)
    xL    = torch.cat(xL_list, dim=0).numpy().astype(np.float32)
    return C_raw, xL

@torch.no_grad()
def score_3d(model: OracleAD3D_CAttn, test_TN, device,
             batch=1024, time_chunk=8192,
             agg="mean", score_chunk=32768, progress_every=10):
    """
    window-end score:
      P_w = mean_i |xL - pred|
      D_w = agg_tau ||D_tau(w) - SLS_tau||_F
      A_w = P_w * D_w

    여기서는 "A 방향 cross-attn"도 time_chunk 단위로 적용.
    - raw C(b)을 전부 뽑고
    - 각 time_chunk마다 앞쪽 tau_max만큼 컨텍스트를 포함해서 cross-attn
    - 그 결과로 pred, D를 계산
    """
    print("[score] encoding all windows ...", flush=True)
    C_raw_cpu, xL_cpu = encode_all_windows(model, test_TN, device, batch=batch)  # CPU numpy
    W, N, d = C_raw_cpu.shape
    tau_max = model.tau_max
    sls3d = model.sls3d.detach().to(device)

    P_w = np.full((W,), np.nan, dtype=np.float32)
    if agg == "max":
        D_w = np.zeros((W,), dtype=np.float32)
    else:
        D_sum = np.zeros((W,), dtype=np.float32)
        D_cnt = np.zeros((W,), dtype=np.float32)

    total_chunks = (W + time_chunk - 1) // time_chunk
    t0 = time.time()
    print(f"[score] start: W={W}, N={N}, tau_max={tau_max}, time_chunk={time_chunk}, total_chunks={total_chunks}", flush=True)

    for ci, s in enumerate(range(0, W, time_chunk), start=1):
        e = min(W, s + time_chunk)
        ext_s = max(0, s - tau_max)     # lag 컨텍스트
        ext_e = e

        C_ext = torch.from_numpy(C_raw_cpu[ext_s:ext_e]).to(device)  # (Bext,N,d)
        C_star_ext = model.causal_cross_attend_over_time(C_ext)      # (Bext,N,d)

        # pred 계산(현재 chunk 부분만)
        off = s - ext_s
        C_star_chunk = C_star_ext[off:off + (e - s)]                 # (Bchunk,N,d)
        recon, pred = model.decode(C_star_chunk)                     # pred: (Bchunk,N)
        pred_np = pred.detach().cpu().numpy().astype(np.float32)

        xL_chunk = xL_cpu[s:e]                                       # (Bchunk,N)
        P_w[s:e] = np.mean(np.abs(xL_chunk - pred_np), axis=1).astype(np.float32)

        # D 계산: ext 안에서 τ별 D_tau를 계산한 뒤, idx_global in [s,e)만 업데이트
        Bext = ext_e - ext_s
        for tau in range(1, tau_max+1):
            if Bext <= tau:
                continue
            A = C_star_ext[:-tau]
            B = C_star_ext[tau:]
            D_tau = pairwise_sq_l2(A, B)  # (Bext-tau,N,N)
            diff = D_tau - sls3d[tau-1].unsqueeze(0)
            fro = torch.linalg.norm(diff, ord="fro", dim=(1,2)).detach().cpu().numpy().astype(np.float32)

            idx_global = np.arange(ext_s + tau, ext_e, dtype=np.int64)  # 길이 Bext-tau
            mask = (idx_global >= s) & (idx_global < e)
            idx_sel = idx_global[mask]
            fro_sel = fro[mask]

            if agg == "max":
                D_w[idx_sel] = np.maximum(D_w[idx_sel], fro_sel)
            else:
                D_sum[idx_sel] += fro_sel
                D_cnt[idx_sel] += 1.0

        if progress_every > 0 and (ci % progress_every == 0 or ci == total_chunks):
            elapsed = time.time() - t0
            rate = ci / max(elapsed, 1e-6)
            eta = (total_chunks - ci) / max(rate, 1e-6)
            print(f"[score] chunk {ci}/{total_chunks}  done={100*ci/total_chunks:.1f}%  elapsed={elapsed/60:.2f}m  eta={eta/60:.2f}m", flush=True)

    if agg != "max":
        D_w = (D_sum / np.maximum(D_cnt, 1.0)).astype(np.float32)

    A_w = (P_w * D_w).astype(np.float32)
    print("[score] done.", flush=True)
    return P_w, D_w, A_w


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="dir with *.npz (train/test/label)")
    ap.add_argument("--entities", type=str, default="", help="comma list without .npz; empty=all")
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--tau_max", type=int, default=5)

    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam_recon", type=float, default=0.1)
    ap.add_argument("--lam_dev", type=float, default=0.3)
    ap.add_argument("--sls_ema", type=float, default=0.9)

    ap.add_argument("--attn_agg", type=str, default="mean", choices=["mean","decay"])
    ap.add_argument("--attn_decay_alpha", type=float, default=0.7)
    ap.add_argument("--no_residual", action="store_true")

    ap.add_argument("--agg", type=str, default="mean", choices=["max","mean"])
    ap.add_argument("--time_chunk", type=int, default=8192, help="scoring time chunk (windows) for causal cross-attn")
    ap.add_argument("--score_chunk", type=int, default=32768, help="(kept for compatibility; not used directly here)")
    ap.add_argument("--progress_every", type=int, default=10)

    ap.add_argument("--out_dir", type=str, default="runs_oraclead_npz_3d_cattn")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    if args.entities:
        wanted = [e.strip() for e in args.entities.split(",") if e.strip()]
        files = [os.path.join(args.input_dir, f"{e}.npz") for e in wanted]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))

    rows=[]
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)
        train = d["train"].astype(np.float32)
        test  = d["test"].astype(np.float32)
        y = reduce_label(d["label"], test.shape[0])

        if train.ndim==1: train=train[:,None]
        if test.ndim==1: test=test[:,None]
        if train.shape[1] != test.shape[1]:
            print("[skip]", name, "N mismatch", flush=True)
            continue

        train_z, test_z, mu, sd = standardize_train_test(train, test, eps=1e-3, clip=10.0)
        N = train_z.shape[1]
        if train_z.shape[0] < args.L + args.tau_max + 1 or test_z.shape[0] < args.L + args.tau_max + 1:
            print("[skip]", name, "too short for tau_max", flush=True)
            continue

        print(f"\n=== {name} (Ttr={train_z.shape[0]}, Tte={test_z.shape[0]}, N={N}) ===", flush=True)

        model = OracleAD3D_CAttn(
            N=N, L=args.L, d=args.d, heads=args.heads, dropout=args.dropout, tau_max=args.tau_max,
            attn_agg=args.attn_agg, attn_decay_alpha=args.attn_decay_alpha,
            residual=(not args.no_residual)
        ).to(device)

        train_3d(model, train_z, device,
                 epochs=args.epochs, batch=args.batch, lr=args.lr,
                 lam_recon=args.lam_recon, lam_dev=args.lam_dev, sls_ema=args.sls_ema)

        P_w, D_w, A_w = score_3d(model, test_z, device,
                                 batch=args.batch, time_chunk=args.time_chunk,
                                 agg=args.agg, progress_every=args.progress_every)

        # align to time index
        Tt = test_z.shape[0]
        start = args.L - 1
        A_t = np.full((Tt,), np.nan, dtype=np.float32)
        P_t = np.full((Tt,), np.nan, dtype=np.float32)
        D_t = np.full((Tt,), np.nan, dtype=np.float32)
        A_t[start:] = A_w
        P_t[start:] = P_w
        D_t[start:] = D_w

        pr = auc_pr(y, A_t)
        f1p,_  = best_f1(y, A_t, point_adjust=False)
        f1pa,_ = best_f1(y, A_t, point_adjust=True)

        print(f"  AUC-PR={pr:.4f} | BestF1={f1p:.4f} | BestPAF1={f1pa:.4f}", flush=True)

        np.savez(os.path.join(args.out_dir, f"{name}.npz"),
                 A_t=A_t, P_t=P_t, D_t=D_t, y=y,
                 sls3d=model.sls3d.detach().cpu().numpy(),
                 mu=mu, sd=sd)
        rows.append((name, pr, f1p, f1pa))

    if rows:
        import pandas as pd
        pd.DataFrame(rows, columns=["entity","AUC_PR","BestF1","BestPAF1"]).to_csv(
            os.path.join(args.out_dir, "summary.csv"), index=False
        )
        print("\nSaved summary:", os.path.join(args.out_dir, "summary.csv"), flush=True)

if __name__ == "__main__":
    main()