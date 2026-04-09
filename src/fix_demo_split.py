import os
import shutil
import pickle
import numpy as np

BASE = "../data/MIMIC_III"
BACKUP = "../data/MIMIC_III_demo_backup"
SEED = 3407
TEST_RATIO = 0.2

FILES_TO_BACKUP = [
    "binary_train_codes_x.pkl",
    "binary_test_codes_x.pkl",
    "train_codes_y.npy",
    "test_codes_y.npy",
    "train_visit_lens.npy",
    "test_visit_lens.npy",
    "train_pids.npy",
    "test_pids.npy",
]

def p(name: str) -> str:
    return os.path.join(BASE, name)

def load_pkl(name: str):
    with open(p(name), "rb") as f:
        return pickle.load(f)

def save_pkl(name: str, obj):
    with open(p(name), "wb") as f:
        pickle.dump(obj, f)

def take_list(xs, idx):
    return [xs[int(i)] for i in idx]

def take_arr(xs, idx):
    return xs[idx]

def main():
    os.makedirs(BACKUP, exist_ok=True)
    for fn in FILES_TO_BACKUP:
        src = p(fn)
        dst = os.path.join(BACKUP, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    train_x = load_pkl("binary_train_codes_x.pkl")
    test_x = load_pkl("binary_test_codes_x.pkl")
    train_y = np.load(p("train_codes_y.npy"))
    test_y = np.load(p("test_codes_y.npy"))
    train_visit_lens = np.load(p("train_visit_lens.npy"))
    test_visit_lens = np.load(p("test_visit_lens.npy"))
    train_pids = np.load(p("train_pids.npy"))
    test_pids = np.load(p("test_pids.npy"))

    print(f"before -> train_x: {len(train_x)}, test_x: {len(test_x)}")
    print(f"before -> train_y: {len(train_y)}, test_y: {len(test_y)}")

    if len(test_x) > 0:
        print("test set is already non-empty. Nothing to fix.")
        return

    n = len(train_x)
    if n == 0:
        raise RuntimeError("train set is also empty. Demo preprocessing is too aggressive to continue.")
    
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)

    if n == 1:
        # 극단적 fallback: 같은 샘플을 train/test 둘 다 사용
        train_idx = idx
        test_idx = idx
    else:
        n_test = max(1, int(round(n * TEST_RATIO)))
        n_test = min(n_test, n - 1)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    new_train_x = take_list(train_x, train_idx)
    new_test_x = take_list(train_x, test_idx)

    new_train_y = take_arr(train_y, train_idx)
    new_test_y = take_arr(train_y, test_idx)

    new_train_visit_lens = take_arr(train_visit_lens, train_idx)
    new_test_visit_lens = take_arr(train_visit_lens, test_idx)

    new_train_pids = take_arr(train_pids, train_idx)
    new_test_pids = take_arr(train_pids, test_idx)

    save_pkl("binary_train_codes_x.pkl", new_train_x)
    save_pkl("binary_test_codes_x.pkl", new_test_x)
    np.save(p("train_codes_y.npy"), new_train_y)
    np.save(p("test_codes_y.npy"), new_test_y)
    np.save(p("train_visit_lens.npy"), new_train_visit_lens)
    np.save(p("test_visit_lens.npy"), new_test_visit_lens)
    np.save(p("train_pids.npy"), new_train_pids)
    np.save(p("test_pids.npy"), new_test_pids)

    print(f"after  -> train_x: {len(new_train_x)}, test_x: {len(new_test_x)}")
    print(f"after  -> train_y: {len(new_train_y)}, test_y: {len(new_test_y)}")
    print("Demo split fix completed.")

if __name__ == "__main__":
    main()