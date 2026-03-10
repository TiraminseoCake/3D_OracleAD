import numpy as np

# ---------- helpers ----------
def _mask_valid(y, score):
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(score).astype(np.float64)
    m = ~np.isnan(s)
    return y[m], s[m]

def _segments(y01):
    segs=[]
    in_seg=False
    s=0
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

# ---------- AUCs ----------
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
    # rank-based ROC-AUC (handles ties)
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    # ranks 1..n
    ranks = np.arange(1, len(y_sorted)+1, dtype=np.float64)
    sum_ranks_pos = ranks[y_sorted==1].sum()
    auc = (sum_ranks_pos - npos*(npos+1)/2.0) / (npos*nneg)
    return float(auc)

# ---------- point F1 ----------
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

# ---------- Range-F1 (segment overlap + cardinality penalty) ----------
def range_precision_recall(y_true, y_pred):
    """
    Range-based Precision/Recall (simple & common):
      - segment overlap ratio * cardinality penalty (reciprocal)
      - This is a widely used practical version; exact R-F1 implementations may vary by paper.
    """
    gt = _segments(y_true)
    pr = _segments(y_pred)

    if len(gt)==0:
        return float("nan"), float("nan")
    if len(pr)==0:
        return 0.0, 0.0

    # Recall: per GT segment
    rec_list=[]
    for g in gt:
        ovs = [p for p in pr if _overlap_len(g,p)>0]
        if not ovs:
            rec_list.append(0.0); continue
        omega = sum(_overlap_len(g,p) for p in ovs) / (g[1]-g[0]+1)
        gamma = 1.0 if len(ovs)<=1 else 1.0/len(ovs)
        rec_list.append(float(omega*gamma))
    R = float(np.mean(rec_list))

    # Precision: per Pred segment
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

# ---------- Optional: Aff-F1 / VUS ----------
def affiliation_f1_optional(y_true, score):
    """
    If an affiliation-metric implementation is available, compute best Aff-F1 over thresholds.
    Otherwise returns (nan, nan).
    """
    try:
        # 여러 구현 패키지 호환 시도
        from affiliation.metrics import pr_from_events  # type: ignore
    except Exception:
        return float("nan"), float("nan")

    y, s = _mask_valid(y_true, score)
    if int(y.sum())==0:
        return 0.0, float("nan")

    # events
    gt = _segments(y)
    if len(gt)==0:
        return float("nan"), float("nan")

    qs = np.linspace(0.0, 1.0, 200)
    thr_list = np.unique(np.quantile(s, qs))[::-1]
    best=-1.0; best_thr=float(thr_list[0])
    for thr in thr_list:
        yp = (s >= thr).astype(np.int32)
        pr = _segments(yp)
        # pr_from_events returns precision, recall (affiliation)
        P,R = pr_from_events(pr, gt)  # NOTE: depends on package API
        val = 0.0 if (P+R)==0 else float(2*P*R/(P+R))
        if val>best:
            best=val; best_thr=float(thr)
    return float(best), float(best_thr)

def vus_optional(y_true, score):
    """
    Placeholder for VUS-ROC / VUS-PR. Different libraries define APIs differently.
    Returns (nan, nan) if not available.
    """
    return float("nan"), float("nan")
