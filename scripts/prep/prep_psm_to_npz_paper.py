import os, argparse, re
import numpy as np
import pandas as pd

DROP_PAT = re.compile(r"(time|timestamp|date|index|idx)", re.I)

def read_numeric_csv_paper(path: str) -> np.ndarray:
    df = pd.read_csv(path)

    # timestamp/time류 제거 -> PSM N=25 맞추기
    drop_cols = [c for c in df.columns if DROP_PAT.search(str(c))]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)

    # 논문에는 명시 없지만 StandardScaler 전 NaN 처리 필수
    # -> time-wise interpolate + ffill/bfill
    df = df.interpolate(axis=0, limit_direction="both")
    df = df.ffill().bfill().fillna(0.0)

    x = df.values.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def read_label_csv(path: str, T: int, label_cols: str) -> np.ndarray:
    df = pd.read_csv(path)
    if len(df) != T:
        raise ValueError(f"Label length mismatch: labels={len(df)} vs test={T}")

    cols = [c.strip() for c in label_cols.split(",") if c.strip()]
    sub = df[cols].select_dtypes(include=[np.number])
    if sub.shape[1] == 0:
        raise ValueError("Selected label cols are not numeric.")
    y = (sub.sum(axis=1) > 0).astype(np.int32).values
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", default="PSM")
    ap.add_argument("--label_cols", required=True)
    args = ap.parse_args()

    train = read_numeric_csv_paper(os.path.join(args.raw_dir, "train.csv"))
    test  = read_numeric_csv_paper(os.path.join(args.raw_dir, "test.csv"))
    y     = read_label_csv(os.path.join(args.raw_dir, "test_label.csv"), test.shape[0], args.label_cols)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.name}.npz")
    np.savez(out, train=train, test=test, label=y)

    print("Saved:", out)
    print("train shape:", train.shape, "test shape:", test.shape)
    print("label pos:", int(y.sum()), "ratio:", float(y.mean()), "unique:", np.unique(y))

if __name__ == "__main__":
    main()
