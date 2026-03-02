from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.decider_features import DECIDER_FEATURE_NAMES
from utils.decider_model import DECIDER_CLASSES, TrainedDeciderModel, save_decider_model


@dataclass
class TrainConfig:
    train_csv: Path
    model_out: Path
    label_col: str
    val_split: float
    seed: int
    epochs: int
    lr: float
    l2: float
    min_confidence: float


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train fusion decider (snatching + fight/weapon).")
    parser.add_argument("--train-csv", required=True, type=Path, help="CSV with DECIDER_FEATURE_NAMES + target label.")
    parser.add_argument("--model-out", default=Path("models/incident_decider_v1.json"), type=Path)
    parser.add_argument("--label-col", default="target", type=str)
    parser.add_argument("--val-split", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=2500, type=int)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--l2", default=1e-4, type=float)
    parser.add_argument("--min-confidence", default=0.45, type=float)
    args = parser.parse_args()
    return TrainConfig(
        train_csv=args.train_csv,
        model_out=args.model_out,
        label_col=args.label_col,
        val_split=float(max(0.0, min(0.5, args.val_split))),
        seed=args.seed,
        epochs=max(1, int(args.epochs)),
        lr=max(1e-5, float(args.lr)),
        l2=max(0.0, float(args.l2)),
        min_confidence=float(max(0.2, min(0.95, args.min_confidence))),
    )


def _softmax(z: np.ndarray) -> np.ndarray:
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z_shift)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    if y_true.size == 0:
        return 0.0
    f1s: List[float] = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall <= 1e-12:
            f1s.append(0.0)
        else:
            f1s.append(float(2 * precision * recall / (precision + recall)))
    return float(np.mean(f1s))


def _build_xy(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    missing = [c for c in DECIDER_FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    for col in DECIDER_FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    target_raw = df[label_col].astype(str).str.strip().str.lower()
    valid = set(DECIDER_CLASSES)
    bad_labels = sorted(set(v for v in target_raw.unique() if v not in valid))
    if bad_labels:
        raise ValueError(f"Invalid labels in {label_col}: {bad_labels}. Allowed: {DECIDER_CLASSES}")

    x = df[DECIDER_FEATURE_NAMES].to_numpy(dtype=float)
    class_to_idx = {c: i for i, c in enumerate(DECIDER_CLASSES)}
    y = np.array([class_to_idx[v] for v in target_raw], dtype=int)
    return x, y


def _split_indices(n: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_split))
    if n >= 10:
        n_val = max(1, min(n - 1, n_val))
    else:
        n_val = 0
    return idx[n_val:], idx[:n_val]


def _train_softmax(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    epochs: int,
    lr: float,
    l2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = x_train.shape
    w = np.zeros((n_classes, n_features), dtype=float)
    b = np.zeros((n_classes,), dtype=float)

    class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
    class_weights = np.ones((n_classes,), dtype=float)
    non_zero = class_counts > 0
    if np.any(non_zero):
        mean_count = float(np.mean(class_counts[non_zero]))
        class_weights[non_zero] = mean_count / class_counts[non_zero]
    sample_weights = class_weights[y_train]
    y_one_hot = _one_hot(y_train, n_classes=n_classes)

    for _ in range(epochs):
        logits = np.matmul(x_train, w.T) + b
        probs = _softmax(logits)
        err = (probs - y_one_hot) * sample_weights[:, None]
        grad_w = np.matmul(err.T, x_train) / max(1.0, float(np.sum(sample_weights))) + l2 * w
        grad_b = np.sum(err, axis=0) / max(1.0, float(np.sum(sample_weights)))

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def _print_metrics(split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": _accuracy(y_true, y_pred),
        "macro_f1": _macro_f1(y_true, y_pred, len(DECIDER_CLASSES)),
        "samples": float(y_true.size),
    }
    print(
        f"[{split_name}] samples={int(metrics['samples'])} "
        f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )
    return metrics


def main() -> None:
    cfg = _parse_args()
    df = pd.read_csv(cfg.train_csv)
    x, y = _build_xy(df, cfg.label_col)
    if x.shape[0] < 4:
        raise ValueError("Need at least 4 labeled rows to train a decider model.")

    train_idx, val_idx = _split_indices(x.shape[0], cfg.val_split, cfg.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    x_train_n = (x_train - means) / scales
    x_val_n = (x_val - means) / scales if x_val.size else x_val

    w, b = _train_softmax(
        x_train=x_train_n,
        y_train=y_train,
        n_classes=len(DECIDER_CLASSES),
        epochs=cfg.epochs,
        lr=cfg.lr,
        l2=cfg.l2,
    )

    train_pred = np.argmax(np.matmul(x_train_n, w.T) + b, axis=1)
    train_metrics = _print_metrics("train", y_train, train_pred)

    val_metrics: Dict[str, float] | None = None
    if x_val_n.size:
        val_pred = np.argmax(np.matmul(x_val_n, w.T) + b, axis=1)
        val_metrics = _print_metrics("val", y_val, val_pred)
    else:
        print("[val] skipped (dataset too small)")

    model = TrainedDeciderModel(
        version="1.0",
        feature_names=list(DECIDER_FEATURE_NAMES),
        classes=list(DECIDER_CLASSES),
        means=means.astype(float),
        scales=scales.astype(float),
        weights=w.astype(float),
        bias=b.astype(float),
        min_confidence=cfg.min_confidence,
    )
    save_decider_model(model, cfg.model_out)
    print(f"Saved model: {cfg.model_out}")

    metrics_path = cfg.model_out.with_suffix(".metrics.json")
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_csv": str(cfg.train_csv),
        "label_col": cfg.label_col,
        "feature_names": DECIDER_FEATURE_NAMES,
        "classes": DECIDER_CLASSES,
        "hyperparams": {
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "l2": cfg.l2,
            "val_split": cfg.val_split,
            "seed": cfg.seed,
            "min_confidence": cfg.min_confidence,
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
        },
    }
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
