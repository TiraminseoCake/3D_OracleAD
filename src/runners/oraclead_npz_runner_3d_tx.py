import argparse, os, glob, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Metrics (same as 2D)
# =========================
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

def best_f1_point(y_true, score, n_q=400):
    y, s = _mask_valid(y_true, score)
    if int(y.sum())==0:
        return 0.0, float("nan")
    qs = np.linspace(0.0, 1.0, n_q)
    thr_list = np.unique(np.quantile(s, qs))[::-1]
    best=-1.0; best_thr=float(thr_list[0])
    for thr in thr_list:
        yp = (s >= thr).astype(np.int32)
        val = f1_point(y, yp)
        if val>best:
            best=val; best_thr=float(thr)
    return float(best), float(best_thr)

def point_adjust_preds(y_pred, y_true):
    y_adj = y_pred.copy()
    for s,e in _segments(y_true):
        if y_pred[s:e+1].sum() > 0:
            y_adj[s:e+1] = 1
    return y_adj

def best_paf1(y_true, score, n_q=200):
    y, s = _mask_valid(y_true, score)
    if int(y.sum())==0:
        return 0.0, float("nan")
    qs = np.linspace(0.0, 1.0, n_q)
    thr_list = np.unique(np.quantile(s, qs))[::-1]
    best=-1.0; best_thr=float(thr_list[0])
    for thr in thr_list:
        yp = (s >= thr).astype(np.int32)
        yp = point_adjust_preds(yp, y)
        val = f1_point(y, yp)
        if val>best:
            best=val; best_thr=float(thr)
    return float(best), float(best_thr)

def range_precision_recall(y_true, y_pred):
    gt = _segments(y_true)
    pr = _segments(y_pred)
    if len(gt)==0:
        return float("nan"), float("nan")
    if len(pr)==0:
        return 0.0, 0.0

    rec_list=[]
    for g in gt:
        ovs = [p for p in pr if _overlap_len(g,p)>0]
        if not ovs:
            rec_list.append(0.0); continue
        omega = sum(_overlap_len(g,p) for p in ovs) / (g[1]-g[0]+1)
        gamma = 1.0 if len(ovs)<=1 else 1.0/len(ovs)
        rec_list.append(float(omega*gamma))
    R = float(np.mean(rec_list))

    prec_list=[]
    for p in pr:
        ovs = [g for g in gt if _overlap_len(g,p)>0]
        if not ovs:
            prec_list.append(0.0); continue
        omega = sum(_overlap_len(p,g) for g in ovs) / (p[1]-p[0]+1)
        gamma = 1.0 if len(ovs)<=1 else 1.0/len(ovs)
        prec_list.append(float(omega*gamma))
    P = float(np.mean(prec_list))
    return P, R

def range_f1(y_true, y_pred):
    P,R = range_precision_recall(y_true, y_pred)
    if np.isnan(P) or np.isnan(R) or (P+R)==0:
        return 0.0
    return float(2*P*R/(P+R))

def best_range_f1(y_true, score, n_q=400):
    y, s = _mask_valid(y_true, score)
    if int(y.sum())==0:
        return 0.0, float("nan")
    qs = np.linspace(0.0, 1.0, n_q)
    thr_list = np.unique(np.quantile(s, qs))[::-1]
    best=-1.0; best_thr=float(thr_list[0])
    for thr in thr_list:
        yp = (s >= thr).astype(np.int32)
        val = range_f1(y, yp)
        if val>best:
            best=val; best_thr=float(thr)
    return float(best), float(best_thr)

def affiliation_f1_optional(y_true, score):
    return float("nan"), float("nan")

def vus_optional(y_true, score):
    return float("nan"), float("nan")


# =========================
# Data utils
# =========================
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


# =========================
# Model blocks
# =========================
class TemporalAttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1, bias=True)
    def forward(self, H):  # (B, L-1, d)
        a = torch.softmax(self.score(H).squeeze(-1), dim=1)
        return (H * a.unsqueeze(-1)).sum(dim=1)

class PerVarEncoder(nn.Module):
    def __init__(self, d: int, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.pool = TemporalAttnPool(d)
    def forward(self, x):
        H,_ = self.lstm(x)
        H = self.drop(H)
        return self.pool(H)

class PerVarDecoder(nn.Module):
    def __init__(self, d: int, L: int, dropout=0.1):
        super().__init__()
        self.L = L
        self.init_h = nn.Linear(d, d)
        self.init_c = nn.Linear(d, d)
        self.lstm = nn.LSTM(1, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 1)
    def forward(self, c_star):
        B,d = c_star.shape
        z = torch.zeros(B, self.L, 1, device=c_star.device, dtype=c_star.dtype)
        h0 = torch.tanh(self.init_h(c_star)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(c_star)).unsqueeze(0)
        Y,_ = self.lstm(z, (h0, c0))
        Y = self.drop(Y)
        O = self.out(Y).squeeze(-1)
        recon = O[:, :self.L-1]
        pred  = O[:, self.L-1]
        return recon, pred

class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ff_mult * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d, d),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        h = self.ln1(x)
        a,_ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(a)
        h2 = self.ln2(x)
        x = x + self.ffn(h2)
        return x

def pairwise_sq_l2(A, B):
    A2 = (A*A).sum(dim=2)
    B2 = (B*B).sum(dim=2)
    G  = torch.bmm(A, B.transpose(1,2))
    D  = A2.unsqueeze(2) + B2.unsqueeze(1) - 2.0 * G
    return torch.clamp(D, min=0.0)

class OracleAD3D_TX(nn.Module):
    def __init__(self, N: int, L: int, d=128, heads=8, dropout=0.1, layers=2, ff_mult=4, tau_max=5):
        super().__init__()
        self.N=N; self.L=L; self.d=d; self.tau_max=tau_max
        self.encoders = nn.ModuleList([PerVarEncoder(d, dropout) for _ in range(N)])
        self.rel_blocks = nn.ModuleList([TransformerBlock(d, heads, dropout, ff_mult) for _ in range(layers)])
        self.decoders = nn.ModuleList([PerVarDecoder(d, L, dropout) for _ in range(N)])
        self.register_buffer("sls3d", torch.zeros(tau_max, N, N), persistent=True)
        self.has_sls = False
    def reset_sls(self):
        self.sls3d.zero_()
        self.has_sls = False
    def forward(self, X):  # (B,L,N)
        B,L,N = X.shape
        c_list=[]
        for i in range(N):
            ci = self.encoders[i](X[:, :L-1, i].unsqueeze(-1))
            c_list.append(ci)
        C = torch.stack(c_list, dim=1)  # (B,N,d)
        for blk in self.rel_blocks:
            C = blk(C)
        C_star = C
        recon_list=[]; pred_list=[]
        for i in range(N):
            r,p = self.decoders[i](C_star[:, i, :])
            recon_list.append(r); pred_list.append(p)
        recon = torch.stack(recon_list, dim=-1)  # (B,L-1,N)
        pred  = torch.stack(pred_list,  dim=-1)  # (B,N)
        return recon, pred, C_star


# =========================
# Train / Score (3D)
# =========================
def train_3d(model, train_TN, device,
             epochs=50, batch=1024, lr=1e-4, weight_decay=0.0,
             lam_recon=0.1, lam_dev=0.3, sls_ema=0.97,
             dev_warmup=5, dist_norm="ln", sls_clip=0.0):
    model.reset_sls()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = SlidingWindowDataset(train_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=True, num_workers=0)

    ln_dist = nn.LayerNorm(model.d).to(device) if dist_norm == "ln" else None
    tau_max = model.tau_max
    N = model.N

    for ep in range(1, epochs+1):
        model.train()
        sls_sum = torch.zeros(tau_max, N, N, device=device)
        sls_cnt = torch.zeros(tau_max, device=device)

        if dev_warmup <= 0:
            lam_dev_eff = lam_dev
        else:
            lam_dev_eff = lam_dev * min(1.0, max(0.0, (ep-1)/float(dev_warmup)))

        lp=lrn=ld=0.0; steps=0
        for X in loader:
            X = X.to(device)
            recon, pred, C_star = model(X)

            if ln_dist is not None:
                C_star_d = ln_dist(C_star)
            else:
                C_star_d = C_star

            xL = X[:, -1, :]
            xpast = X[:, :model.L-1, :]

            loss_pred  = torch.mean((xL - pred) ** 2)
            loss_recon = torch.mean((xpast - recon) ** 2)

            dev_terms=[]
            for tau in range(1, tau_max+1):
                if C_star_d.shape[0] <= tau:
                    continue
                A = C_star_d[:-tau]
                B = C_star_d[tau:]
                D_tau = pairwise_sq_l2(A, B)  # (B-tau,N,N)
                if sls_clip and sls_clip > 0:
                    D_tau = torch.clamp(D_tau, 0.0, float(sls_clip))

                sls_sum[tau-1] += D_tau.mean(dim=0).detach()
                sls_cnt[tau-1] += 1

                if model.has_sls:
                    diff = D_tau - model.sls3d[tau-1].unsqueeze(0)
                    dev_terms.append(torch.mean(diff**2))

            if model.has_sls and len(dev_terms)>0:
                loss_dev = torch.stack(dev_terms).mean()
                loss = loss_pred + lam_recon*loss_recon + lam_dev_eff*loss_dev
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
            if sls_clip and sls_clip > 0:
                epoch_sls = torch.clamp(epoch_sls, 0.0, float(sls_clip))

            if not model.has_sls:
                model.sls3d.copy_(epoch_sls)
                model.has_sls = True
            else:
                beta=float(sls_ema)
                model.sls3d.mul_(beta).add_(epoch_sls * (1.0 - beta))
                if sls_clip and sls_clip > 0:
                    model.sls3d.copy_(torch.clamp(model.sls3d, 0.0, float(sls_clip)))

        print(f"  [ep {ep:02d}] pred={lp/steps:.6f} recon={lrn/steps:.6f} dev={ld/steps:.3e} "
              f"lam_dev_eff={lam_dev_eff:.3f} has_sls={model.has_sls}", flush=True)

@torch.no_grad()
def compute_embeddings(model, series_TN, device, batch=1024):
    model.eval()
    ds = SlidingWindowDataset(series_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)
    C_list=[]; pred_list=[]; xL_list=[]
    for X in loader:
        X = X.to(device)
        recon, pred, C = model(X)
        xL = X[:, -1, :]
        C_list.append(C.detach().cpu())
        pred_list.append(pred.detach().cpu())
        xL_list.append(xL.detach().cpu())
    return torch.cat(C_list, dim=0), torch.cat(pred_list, dim=0), torch.cat(xL_list, dim=0)

@torch.no_grad()
def score_3d(model, test_TN, device, batch=1024,
             agg="mean", chunk=32768, progress_every=10,
             dist_norm="ln", sls_clip=0.0):
    print("[score] computing embeddings ...", flush=True)
    C_cpu, pred_cpu, xL_cpu = compute_embeddings(model, test_TN, device, batch=batch)
    C = C_cpu.to(device)
    pred = pred_cpu.to(device)
    xL = xL_cpu.to(device)

    ln_dist = nn.LayerNorm(model.d).to(device) if dist_norm == "ln" else None
    if ln_dist is not None:
        C = ln_dist(C)

    W, N, d = C.shape
    tau_max = model.tau_max
    sls3d = model.sls3d.detach()

    P_w = (xL - pred).abs().mean(dim=1)  # (W,)

    if agg == "max":
        D_w = torch.zeros(W, device=device)
    else:
        D_sum = torch.zeros(W, device=device)
        D_cnt = torch.zeros(W, device=device)

    total_steps = 0
    for tau in range(1, tau_max+1):
        if W <= tau:
            continue
        end = W - tau
        total_steps += (end + chunk - 1) // chunk

    done_steps = 0
    t0 = time.time()
    print(f"[score] start: W={W}, N={N}, tau_max={tau_max}, chunk={chunk}, total_steps={total_steps}, agg={agg}", flush=True)

    for tau in range(1, tau_max+1):
        if W <= tau:
            continue
        end = W - tau
        for s in range(0, end, chunk):
            e = min(end, s+chunk)
            A = C[s:e]
            B = C[s+tau:e+tau]
            D_tau = pairwise_sq_l2(A, B)
            if sls_clip and sls_clip > 0:
                D_tau = torch.clamp(D_tau, 0.0, float(sls_clip))
            diff = D_tau - sls3d[tau-1].unsqueeze(0)
            fro = torch.linalg.norm(diff, ord="fro", dim=(1,2))

            idx = torch.arange(s+tau, e+tau, device=device)
            if agg == "max":
                D_w[idx] = torch.maximum(D_w[idx], fro)
            else:
                D_sum[idx] += fro
                D_cnt[idx] += 1

            done_steps += 1
            if progress_every > 0 and (done_steps % progress_every == 0 or done_steps == total_steps):
                elapsed = time.time() - t0
                rate = done_steps / max(elapsed, 1e-6)
                eta = (total_steps - done_steps) / max(rate, 1e-6)
                pct = 100.0 * done_steps / max(total_steps, 1)
                print(f"[score] {pct:5.1f}%  tau={tau}/{tau_max}  chunk={s}:{e}/{end}  "
                      f"step={done_steps}/{total_steps}  elapsed={elapsed/60:.2f}m  eta={eta/60:.2f}m", flush=True)

    if agg != "max":
        D_w = D_sum / torch.clamp(D_cnt, min=1.0)

    A_w = P_w * D_w
    print("[score] done.", flush=True)
    return P_w.detach().cpu().numpy(), D_w.detach().cpu().numpy(), A_w.detach().cpu().numpy()


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--entities", type=str, default="")
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--tau_max", type=int, default=5)

    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn_layers", type=int, default=2)
    ap.add_argument("--ff_mult", type=int, default=4)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--lam_recon", type=float, default=0.1)
    ap.add_argument("--lam_dev", type=float, default=0.3)
    ap.add_argument("--sls_ema", type=float, default=0.97)

    ap.add_argument("--dev_warmup", type=int, default=5)
    ap.add_argument("--dist_norm", type=str, default="ln", choices=["none","ln"])
    ap.add_argument("--sls_clip", type=float, default=0.0)

    ap.add_argument("--agg", type=str, default="mean", choices=["max","mean"])
    ap.add_argument("--score_chunk", type=int, default=32768)
    ap.add_argument("--progress_every", type=int, default=10)

    ap.add_argument("--out_dir", type=str, default="runs_oraclead_npz_3d_tx_metrics")
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
        data = np.load(f)
        train = data["train"].astype(np.float32)
        test  = data["test"].astype(np.float32)
        y = reduce_label(data["label"], test.shape[0])

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

        model = OracleAD3D_TX(N=N, L=args.L, d=args.d, heads=args.heads, dropout=args.dropout,
                              layers=args.attn_layers, ff_mult=args.ff_mult, tau_max=args.tau_max).to(device)

        train_3d(model, train_z, device,
                 epochs=args.epochs, batch=args.batch, lr=args.lr, weight_decay=args.weight_decay,
                 lam_recon=args.lam_recon, lam_dev=args.lam_dev, sls_ema=args.sls_ema,
                 dev_warmup=args.dev_warmup, dist_norm=args.dist_norm, sls_clip=args.sls_clip)

        P_w, D_w, A_w = score_3d(model, test_z, device, batch=args.batch,
                                 agg=args.agg, chunk=args.score_chunk, progress_every=args.progress_every,
                                 dist_norm=args.dist_norm, sls_clip=args.sls_clip)

        Tt = test_z.shape[0]
        start = args.L - 1
        A_t = np.full((Tt,), np.nan, dtype=np.float32)
        P_t = np.full((Tt,), np.nan, dtype=np.float32)
        D_t = np.full((Tt,), np.nan, dtype=np.float32)
        A_t[start:] = A_w.astype(np.float32)
        P_t[start:] = P_w.astype(np.float32)
        D_t[start:] = D_w.astype(np.float32)

        A_PR  = auc_pr(y, A_t)
        A_ROC = auc_roc(y, A_t)
        F1, _ = best_f1_point(y, A_t, n_q=400)
        R_F1,_= best_range_f1(y, A_t, n_q=400)
        Aff_F1,_ = affiliation_f1_optional(y, A_t)
        VUS_ROC, VUS_PR = vus_optional(y, A_t)
        PAF1,_ = best_paf1(y, A_t, n_q=200)

        print(f"  A-PR={A_PR:.4f} | A-ROC={A_ROC:.4f} | F1={F1:.4f} | R-F1={R_F1:.4f} | "
              f"Aff-F1={Aff_F1:.4f} | VUS-ROC={VUS_ROC:.4f} | VUS-PR={VUS_PR:.4f} | PAF1={PAF1:.4f}", flush=True)

        np.savez(os.path.join(args.out_dir, f"{name}.npz"),
                 A_t=A_t, P_t=P_t, D_t=D_t, y=y,
                 sls3d=model.sls3d.detach().cpu().numpy(),
                 mu=mu, sd=sd)

        rows.append((name, A_PR, A_ROC, F1, R_F1, Aff_F1, VUS_ROC, VUS_PR, PAF1))

    if rows:
        import pandas as pd
        pd.DataFrame(rows, columns=["entity","A_PR","A_ROC","F1","R_F1","Aff_F1","VUS_ROC","VUS_PR","PAF1"]).to_csv(
            os.path.join(args.out_dir, "summary.csv"), index=False
        )
        print("\nSaved summary:", os.path.join(args.out_dir, "summary.csv"), flush=True)

if __name__ == "__main__":
    main()