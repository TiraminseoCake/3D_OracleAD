import argparse, os, glob
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
    train_z = np.clip(train_z, -clip, clip)
    test_z  = np.clip(test_z,  -clip, clip)
    train_z = np.nan_to_num(train_z, nan=0.0, posinf=0.0, neginf=0.0)
    test_z  = np.nan_to_num(test_z,  nan=0.0, posinf=0.0, neginf=0.0)
    return train_z.astype(np.float32), test_z.astype(np.float32), mu, sd

def auc_pr(y_true, score):
    mask = ~np.isnan(score)
    y_true = y_true[mask].astype(np.int32)
    score = score[mask].astype(np.float64)
    npos = (y_true==1).sum()
    if npos==0: return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tps = np.cumsum(y==1)
    fps = np.cumsum(y==0)
    prec = tps/np.maximum(tps+fps,1)
    rec  = tps/npos
    prec = np.concatenate([[1.0], prec])
    rec  = np.concatenate([[0.0], rec])
    return float(np.trapz(prec, rec))

def segments_from_labels(y):
    segs=[]; in_seg=False; s=0
    for i,v in enumerate(y):
        if v==1 and not in_seg: in_seg=True; s=i
        elif v==0 and in_seg: in_seg=False; segs.append((s,i-1))
    if in_seg: segs.append((s,len(y)-1))
    return segs

def point_adjust_preds(y_pred, y_true):
    y_adj=y_pred.copy()
    for s,e in segments_from_labels(y_true):
        if y_pred[s:e+1].sum()>0:
            y_adj[s:e+1]=1
    return y_adj

def f1(y_true,y_pred):
    tp=((y_true==1)&(y_pred==1)).sum()
    fp=((y_true==0)&(y_pred==1)).sum()
    fn=((y_true==1)&(y_pred==0)).sum()
    if tp==0: return 0.0
    p=tp/(tp+fp) if (tp+fp)>0 else 0.0
    r=tp/(tp+fn) if (tp+fn)>0 else 0.0
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def best_f1(y_true, score, point_adjust=False):
    mask = ~np.isnan(score)
    y_true = y_true[mask].astype(np.int32)
    score = score[mask].astype(np.float64)
    thr_list = np.unique(score)
    thr_list = np.sort(thr_list)[::-1]
    best=-1.0; best_thr=float(thr_list[0])
    for thr in thr_list:
        y_pred = (score>=thr).astype(np.int32)
        if point_adjust:
            y_pred = point_adjust_preds(y_pred, y_true)
        val = f1(y_true, y_pred)
        if val>best:
            best=val; best_thr=float(thr)
    return float(best), float(best_thr)

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
# model
# -------------------------
class TemporalAttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)
    def forward(self, H):
        a = torch.softmax(self.score(H).squeeze(-1), dim=1)
        c = (H * a.unsqueeze(-1)).sum(dim=1)
        return c

class PerVarEncoder(nn.Module):
    def __init__(self, d: int, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(1, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.pool = TemporalAttnPool(d)
    def forward(self, x):
        H,_ = self.lstm(x)
        H = self.drop(H)
        return self.pool(H)

class PerVarDecoder(nn.Module):
    def __init__(self, d: int, L: int, dropout=0.0):
        super().__init__()
        self.L=L
        self.init_h = nn.Linear(d,d)
        self.init_c = nn.Linear(d,d)
        self.lstm = nn.LSTM(1,d,batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d,1)
    def forward(self, c_star):
        B,d = c_star.shape
        z = torch.zeros(B,self.L,1, device=c_star.device, dtype=c_star.dtype)
        h0 = torch.tanh(self.init_h(c_star)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(c_star)).unsqueeze(0)
        Y,_ = self.lstm(z,(h0,c0))
        Y = self.drop(Y)
        O = self.out(Y).squeeze(-1)        # (B,L)
        recon = O[:,:self.L-1]
        pred  = O[:, self.L-1]
        return recon, pred

class OracleAD(nn.Module):
    def __init__(self, N, L, d=64, heads=4, dropout=0.0):
        super().__init__()
        self.N=N; self.L=L; self.d=d
        self.encoders = nn.ModuleList([PerVarEncoder(d,dropout) for _ in range(N)])
        self.mhsa = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.decoders = nn.ModuleList([PerVarDecoder(d,L,dropout) for _ in range(N)])

        self.register_buffer("sls2d", torch.zeros(N,N), persistent=True)
        self.has_sls=False

    def reset_sls(self):
        self.sls2d.zero_(); self.has_sls=False

    def forward(self, X):
        B,L,N = X.shape
        c_list=[]
        for i in range(N):
            ci = self.encoders[i](X[:,:L-1,i].unsqueeze(-1))
            c_list.append(ci)
        C = torch.stack(c_list, dim=1)                 # (B,N,d)
        C_star,_ = self.mhsa(C,C,C, need_weights=False)
        recon_list=[]; pred_list=[]
        for i in range(N):
            r,p = self.decoders[i](C_star[:,i,:])
            recon_list.append(r); pred_list.append(p)
        recon = torch.stack(recon_list, dim=-1)        # (B,L-1,N)
        pred  = torch.stack(pred_list,  dim=-1)        # (B,N)
        return recon, pred, C_star

    @staticmethod
    def cdist2(A,B):
        return torch.cdist(A,B,p=2)**2

# -------------------------
# 2D train/score
# -------------------------
@torch.no_grad()
def compute_embeddings(model, series_TN, device, batch=256):
    model.eval()
    ds = SlidingWindowDataset(series_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False)
    C_list=[]; pred_list=[]; xL_list=[]
    for X in loader:
        X=X.to(device)
        recon,pred,C = model(X)
        xL = X[:,-1,:]
        C_list.append(C.detach().cpu())
        pred_list.append(pred.detach().cpu())
        xL_list.append(xL.detach().cpu())
    return torch.cat(C_list), torch.cat(pred_list), torch.cat(xL_list)  # (W,N,d),(W,N),(W,N)

def train_2d(model, train_TN, device, epochs, batch, lr, lam_recon, lam_dev, sls_ema):
    model.reset_sls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = SlidingWindowDataset(train_TN, model.L)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

    for ep in range(1, epochs+1):
        model.train()
        sls_sum = torch.zeros(model.N, model.N, device=device)
        sls_cnt = 0
        lp=lrn=ld=0.0; steps=0

        for X in loader:
            X=X.to(device)
            recon,pred,C = model(X)
            xL = X[:,-1,:]
            xpast = X[:,:model.L-1,:]
            loss_pred = torch.mean((xL-pred)**2)
            loss_recon= torch.mean((xpast-recon)**2)

            D = OracleAD.cdist2(C,C)               # (B,N,N)
            sls_sum += D.mean(dim=0).detach(); sls_cnt += 1

            if model.has_sls:
                loss_dev = torch.mean((D - model.sls2d.unsqueeze(0))**2)
                loss = loss_pred + lam_recon*loss_recon + lam_dev*loss_dev
            else:
                loss_dev = torch.tensor(0.0, device=device)
                loss = loss_pred + lam_recon*loss_recon

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            lp += float(loss_pred.detach().cpu())
            lrn+= float(loss_recon.detach().cpu())
            ld += float(loss_dev.detach().cpu())
            steps += 1

        with torch.no_grad():
            epoch_sls = sls_sum / max(sls_cnt,1)
            if not model.has_sls:
                model.sls2d.copy_(epoch_sls); model.has_sls=True
            else:
                beta=float(sls_ema)
                model.sls2d.mul_(beta).add_(epoch_sls*(1.0-beta))

        print(f"  [ep {ep:02d}] pred={lp/steps:.6f} recon={lrn/steps:.6f} dev={ld/steps:.3e} has_sls={model.has_sls}")

@torch.no_grad()
def score_2d(model, test_TN, device, batch=256):
    C_cpu, pred_cpu, xL_cpu = compute_embeddings(model, test_TN, device, batch=batch)
    C = C_cpu.to(device); pred=pred_cpu.to(device); xL=xL_cpu.to(device)
    W,N,d = C.shape
    P = (xL-pred).abs().mean(dim=1)                # (W,)
    Dmat = OracleAD.cdist2(C,C)                    # (W,N,N)
    Dev = torch.linalg.norm(Dmat - model.sls2d.unsqueeze(0), ord="fro", dim=(1,2))
    A = P*Dev
    return P.cpu().numpy(), Dev.cpu().numpy(), A.cpu().numpy()

# -------------------------
# 3D score (using existing embeddings)
# -------------------------
@torch.no_grad()
def score_3d(model, test_TN, device, tau_max, agg="mean", batch=256):
    C_cpu, pred_cpu, xL_cpu = compute_embeddings(model, test_TN, device, batch=batch)
    C = C_cpu.to(device); pred=pred_cpu.to(device); xL=xL_cpu.to(device)
    W,N,d = C.shape
    P = (xL-pred).abs().mean(dim=1)

    # build SLS3D from test embeddings is NOT correct; we need from train.
    # For runner simplicity, we will compute SLS3D from TRAIN embeddings and cache outside.
    raise RuntimeError("3D mode is implemented in the dedicated 3D runner; use oraclead_smap_runner_3d_full.py-style for now.")

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="directory containing *.npz with train/test/label")
    ap.add_argument("--entities", type=str, default="", help="comma list; empty=all npz files")
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam_recon", type=float, default=0.1)
    ap.add_argument("--lam_dev", type=float, default=3.0)
    ap.add_argument("--sls_ema", type=float, default=0.9)

    ap.add_argument("--out_dir", type=str, default="runs_oraclead_npz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.entities:
        wanted = set([e.strip() for e in args.entities.split(",") if e.strip()])
        files = [os.path.join(args.input_dir, f"{e}.npz") if not e.endswith(".npz") else os.path.join(args.input_dir, e) for e in wanted]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))

    rows = []

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        data = np.load(f)
        train = data["train"].astype(np.float32)
        test  = data["test"].astype(np.float32)
        y     = data["label"].astype(np.int32)
        if train.ndim==1: train=train[:,None]
        if test.ndim==1: test=test[:,None]
        if train.shape[1] != test.shape[1]:
            print("[skip]", name, "N mismatch", train.shape, test.shape); continue

        train_z, test_z, mu, sd = standardize_train_test(train, test)

        N = train_z.shape[1]
        if train_z.shape[0] < args.L or test_z.shape[0] < args.L:
            print("[skip]", name, "too short"); continue

        print(f"\n=== {name} (Ttr={train_z.shape[0]}, Tte={test_z.shape[0]}, N={N}) ===")
        model = OracleAD(N=N, L=args.L, d=args.d, heads=args.heads, dropout=args.dropout).to(device)

        train_2d(model, train_z, device, args.epochs, args.batch, args.lr, args.lam_recon, args.lam_dev, args.sls_ema)
        P_w, D_w, A_w = score_2d(model, test_z, device, batch=args.batch)

        # align to time index
        Tt = test_z.shape[0]
        A_t = np.full((Tt,), np.nan, dtype=np.float32)
        start = args.L - 1
        A_t[start:] = A_w.astype(np.float32)

        pr = auc_pr(y, A_t)
        f1p,_  = best_f1(y, A_t, point_adjust=False)
        f1pa,_ = best_f1(y, A_t, point_adjust=True)

        print(f"  AUC-PR={pr:.4f} | BestF1={f1p:.4f} | BestPAF1={f1pa:.4f}")

        np.savez(os.path.join(args.out_dir, f"{name}.npz"), A_t=A_t, y=y, mu=mu, sd=sd)
        rows.append((name, pr, f1p, f1pa))

    if rows:
        import pandas as pd
        pd.DataFrame(rows, columns=["entity","AUC_PR","BestF1","BestPAF1"]).to_csv(
            os.path.join(args.out_dir, "summary.csv"), index=False
        )
        print("\nSaved summary:", os.path.join(args.out_dir, "summary.csv"))

if __name__ == "__main__":
    main()
