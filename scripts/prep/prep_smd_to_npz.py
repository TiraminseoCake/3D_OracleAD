import os, argparse
import numpy as np

def load_matrix(path: str) -> np.ndarray:
    # SMD는 보통 txt. delimiter가 ',' 또는 None일 수 있어서 유연하게 처리.
    try:
        x = np.loadtxt(path, delimiter=",").astype(np.float32)
        if x.ndim == 1:
            x = x[:, None]
        return x
    except Exception:
        x = np.loadtxt(path).astype(np.float32)
        if x.ndim == 1:
            x = x[:, None]
        return x

def load_label(path: str, T: int) -> np.ndarray:
    try:
        y = np.loadtxt(path, delimiter=",").astype(np.int32)
    except Exception:
        y = np.loadtxt(path).astype(np.int32)
    if y.ndim > 1:
        y = (y.sum(axis=1) > 0).astype(np.int32)
    if len(y) != T:
        raise ValueError(f"{path}: label length {len(y)} != test length {T}")
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True, help="OmniAnomaly/ServerMachineDataset")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    train_dir = os.path.join(args.raw_dir, "train")
    test_dir  = os.path.join(args.raw_dir, "test")
    lab_dir   = os.path.join(args.raw_dir, "test_label")

    os.makedirs(args.out_dir, exist_ok=True)

    # train 디렉토리 파일 기준으로 엔티티 목록 구성
    train_files = sorted([f for f in os.listdir(train_dir) if not f.startswith(".")])
    saved = 0

    for f in train_files:
        tr_path = os.path.join(train_dir, f)
        te_path = os.path.join(test_dir, f)
        lb_path = os.path.join(lab_dir, f)

        if not (os.path.exists(te_path) and os.path.exists(lb_path)):
            # 파일명이 완전히 동일하다는 가정이 대부분 맞지만, 혹시 다르면 여기서 skip됨
            continue

        train = load_matrix(tr_path)
        test  = load_matrix(te_path)
        if train.shape[1] != test.shape[1]:
            continue

        y = load_label(lb_path, test.shape[0])

        name = os.path.splitext(f)[0]
        out = os.path.join(args.out_dir, f"{name}.npz")
        np.savez(out, train=train, test=test, label=y)
        saved += 1

    print(f"Saved {saved} entities to {args.out_dir}")

if __name__ == "__main__":
    main()
