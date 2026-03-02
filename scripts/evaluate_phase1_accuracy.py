from __future__ import annotations

import argparse
import json
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
from utils.decider_model import load_decider_model, predict_decider_class


NEGATIVE_LABELS = {"none", "no_incident", "normal", "negative", "0", "false"}
POSITIVE_LABELS = {"incident", "positive", "1", "true", "snatching", "fight_weapon", "both"}


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate phase-1 binary incident accuracy.")
    parser.add_argument("--csv", required=True, type=Path, help="Labeled CSV with feature columns.")
    parser.add_argument("--model", default=Path("models/incident_decider_v1.json"), type=Path)
    parser.add_argument("--label-col", default="target", type=str)
    parser.add_argument("--threshold", default=0.55, type=float)
    parser.add_argument("--min-accuracy", default=0.70, type=float)
    parser.add_argument("--report-out", default=Path("models/phase1_eval_report.json"), type=Path)
    return parser.parse_args()


def _to_binary_label(raw: str) -> int:
    v = str(raw).strip().lower()
    if v in NEGATIVE_LABELS:
        return 0
    if v in POSITIVE_LABELS:
        return 1
    # fallback: treat any unknown non-empty label as incident
    return 1 if v else 0


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / max(1, y_true.size)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) <= 1e-12 else (2 * precision * recall / (precision + recall))
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _sweep_best_threshold(y_true: np.ndarray, incident_probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.55
    best_metrics = {"accuracy": -1.0}
    for thr in np.linspace(0.30, 0.90, 61):
        pred = (incident_probs >= thr).astype(int)
        m = _binary_metrics(y_true, pred)
        if m["accuracy"] > best_metrics["accuracy"]:
            best_metrics = m
            best_thr = float(thr)
    return best_thr, best_metrics


def main() -> int:
    args = _parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    df = pd.read_csv(args.csv)
    missing = [c for c in DECIDER_FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    for c in DECIDER_FEATURE_NAMES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    y_true = np.array([_to_binary_label(v) for v in df[args.label_col].tolist()], dtype=int)
    model = load_decider_model(args.model)

    incident_probs: List[float] = []
    for _, row in df.iterrows():
        feats = {name: float(row[name]) for name in DECIDER_FEATURE_NAMES}
        _, _, probs = predict_decider_class(model, feats)
        incident_probs.append(float(1.0 - probs.get("none", 0.0)))

    probs_arr = np.array(incident_probs, dtype=float)
    threshold = float(max(0.05, min(0.95, args.threshold)))
    y_pred = (probs_arr >= threshold).astype(int)
    metrics = _binary_metrics(y_true, y_pred)

    best_thr, best_metrics = _sweep_best_threshold(y_true, probs_arr)
    passed = metrics["accuracy"] >= float(args.min_accuracy)

    print(f"Samples: {y_true.size}")
    print(f"Threshold: {threshold:.3f}")
    print(
        f"Accuracy={metrics['accuracy']:.4f} Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
    )
    print(f"Target accuracy >= {args.min_accuracy:.2f}: {'PASS' if passed else 'FAIL'}")
    print(f"Best threshold by accuracy: {best_thr:.3f} (acc={best_metrics['accuracy']:.4f})")

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "csv": str(args.csv),
        "model": str(args.model),
        "label_col": args.label_col,
        "threshold": threshold,
        "metrics": metrics,
        "min_accuracy_target": float(args.min_accuracy),
        "pass": bool(passed),
        "best_threshold_by_accuracy": float(best_thr),
        "best_metrics_by_accuracy": best_metrics,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved: {args.report_out}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
