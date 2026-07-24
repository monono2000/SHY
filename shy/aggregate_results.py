import pickle
import numpy as np
from collections import defaultdict

if __package__:
    from .runtime_paths import TRAINING_LOGS_DIR
else:
    from runtime_paths import TRAINING_LOGS_DIR

BASE = TRAINING_LOGS_DIR

REQUIRED_FILES = [
    "r2_list.pkl",
    "r4_list.pkl",
    "n2_list.pkl",
    "n4_list.pkl",
    "train_average_loss_per_epoch.pkl",
    "test_loss_per_epoch.pkl",
    "prediction_loss_per_epoch.pkl",
]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def detect_dataset(folder_name: str) -> str:
    name = folder_name.upper()
    if "MIMIC_III" in name:
        return "MIMIC_III"
    if "MIMIC_IV" in name:
        return "MIMIC_IV"
    return "UNKNOWN"


def detect_epoch_group(length: int) -> str:
    if length == 50:
        return "50E"
    if length == 450:
        return "450E"
    return f"{length}E_OTHER"


def summarize_rows(rows, title):
    print(f"\n===== {title} =====")
    if not rows:
        print("No runs found in this group.")
        return

    print("\n[Per-run results]")
    for r in rows:
        print(f"- run: {r['run']}")
        print(
            f"  best  | R@10={r['best_r2']:.6f}, R@20={r['best_r4']:.6f}, "
            f"nDCG@10={r['best_n2']:.6f}, nDCG@20={r['best_n4']:.6f}"
        )
        print(
            f"  final | R@10={r['last_r2']:.6f}, R@20={r['last_r4']:.6f}, "
            f"nDCG@10={r['last_n2']:.6f}, nDCG@20={r['last_n4']:.6f}"
        )
        print(
            f"  loss  | best_test={r['best_test_loss']:.6f}, "
            f"final_train={r['final_train_loss']:.6f}, "
            f"final_test={r['final_test_loss']:.6f}, "
            f"final_pred={r['final_pred_loss']:.6f}"
        )

    metric_keys = [
        "best_r2", "best_r4", "best_n2", "best_n4",
        "last_r2", "last_r4", "last_n2", "last_n4",
        "best_test_loss", "final_train_loss", "final_test_loss", "final_pred_loss",
    ]

    print("\n[Mean +/- Std]")
    for key in metric_keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        print(f"{key}: {mean:.6f} +/- {std:.6f}")


def main():
    if not BASE.is_dir():
        print(f"training_logs directory was not found: {BASE}")
        return

    all_dirs = sorted([p.name for p in BASE.iterdir() if p.is_dir()])

    grouped = defaultdict(list)
    skipped = []

    for d in all_dirs:
        run_dir = BASE / d

        missing = [f for f in REQUIRED_FILES if not (run_dir / f).exists()]
        if missing:
            skipped.append((d, f"missing required files: {missing}"))
            continue

        try:
            r2 = load_pkl(run_dir / "r2_list.pkl")
            r4 = load_pkl(run_dir / "r4_list.pkl")
            n2 = load_pkl(run_dir / "n2_list.pkl")
            n4 = load_pkl(run_dir / "n4_list.pkl")
            train_loss = load_pkl(run_dir / "train_average_loss_per_epoch.pkl")
            test_loss = load_pkl(run_dir / "test_loss_per_epoch.pkl")
            pred_loss = load_pkl(run_dir / "prediction_loss_per_epoch.pkl")
        except Exception as e:
            skipped.append((d, f"failed to load files: {e}"))
            continue

        epoch_len = len(r2)
        if not (len(r4) == len(n2) == len(n4) == epoch_len):
            skipped.append((d, "metric lengths do not match"))
            continue

        dataset = detect_dataset(d)
        epoch_group = detect_epoch_group(epoch_len)

        row = {
            "run": d,
            "dataset": dataset,
            "epoch_group": epoch_group,
            "best_r2": float(np.max(r2)),
            "best_r4": float(np.max(r4)),
            "best_n2": float(np.max(n2)),
            "best_n4": float(np.max(n4)),
            "last_r2": float(r2[-1]),
            "last_r4": float(r4[-1]),
            "last_n2": float(n2[-1]),
            "last_n4": float(n4[-1]),
            "best_test_loss": float(np.min(test_loss)),
            "final_train_loss": float(train_loss[-1]),
            "final_test_loss": float(test_loss[-1]),
            "final_pred_loss": float(pred_loss[-1]),
        }

        grouped[(dataset, epoch_group)].append(row)

    ordered_keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    for dataset, epoch_group in ordered_keys:
        summarize_rows(grouped[(dataset, epoch_group)], f"{dataset} | {epoch_group}")

    print("\n===== Skipped Folders =====")
    if not skipped:
        print("None")
    else:
        for folder, reason in skipped:
            print(f"- {folder}: {reason}")


if __name__ == "__main__":
    main()
