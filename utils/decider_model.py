from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


DECIDER_CLASSES = ["none", "snatching", "fight_weapon", "both"]


@dataclass(frozen=True)
class TrainedDeciderModel:
    version: str
    feature_names: List[str]
    classes: List[str]
    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    min_confidence: float = 0.45


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _to_float_array(values: Sequence[float], expected_len: int, fallback: float = 0.0) -> np.ndarray:
    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size == expected_len:
        return arr
    out = np.full((expected_len,), float(fallback), dtype=float)
    size = min(expected_len, arr.size)
    if size > 0:
        out[:size] = arr[:size]
    return out


def model_to_json_dict(model: TrainedDeciderModel) -> Dict[str, object]:
    return {
        "version": model.version,
        "feature_names": model.feature_names,
        "classes": model.classes,
        "means": model.means.tolist(),
        "scales": model.scales.tolist(),
        "weights": model.weights.tolist(),
        "bias": model.bias.tolist(),
        "min_confidence": float(model.min_confidence),
    }


def save_decider_model(model: TrainedDeciderModel, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model_to_json_dict(model), ensure_ascii=False, indent=2), encoding="utf-8")


def load_decider_model(path: str | Path) -> TrainedDeciderModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    feature_names = [str(v) for v in payload.get("feature_names", [])]
    classes = [str(v) for v in payload.get("classes", [])]
    means = _to_float_array(payload.get("means", []), len(feature_names), fallback=0.0)
    scales = _to_float_array(payload.get("scales", []), len(feature_names), fallback=1.0)
    scales = np.where(np.abs(scales) < 1e-9, 1.0, scales)

    raw_weights = np.array(payload.get("weights", []), dtype=float)
    if raw_weights.ndim == 1:
        raw_weights = raw_weights.reshape(1, -1)
    if raw_weights.shape != (len(classes), len(feature_names)):
        fixed = np.zeros((len(classes), len(feature_names)), dtype=float)
        rows = min(fixed.shape[0], raw_weights.shape[0] if raw_weights.ndim == 2 else 0)
        cols = min(fixed.shape[1], raw_weights.shape[1] if raw_weights.ndim == 2 else 0)
        if rows > 0 and cols > 0:
            fixed[:rows, :cols] = raw_weights[:rows, :cols]
        raw_weights = fixed

    bias = _to_float_array(payload.get("bias", []), len(classes), fallback=0.0)
    min_confidence = float(payload.get("min_confidence", 0.45))

    return TrainedDeciderModel(
        version=str(payload.get("version", "1.0")),
        feature_names=feature_names,
        classes=classes,
        means=means,
        scales=scales,
        weights=raw_weights,
        bias=bias,
        min_confidence=min_confidence,
    )


def predict_decider_class(
    model: TrainedDeciderModel,
    features: Dict[str, float],
) -> tuple[str, float, Dict[str, float]]:
    x = np.array([float(features.get(name, 0.0)) for name in model.feature_names], dtype=float)
    x_norm = (x - model.means) / model.scales
    logits = np.matmul(model.weights, x_norm) + model.bias
    probs = _softmax(logits)

    best_idx = int(np.argmax(probs))
    pred_class = model.classes[best_idx]
    confidence = float(probs[best_idx])
    probabilities = {label: float(probs[idx]) for idx, label in enumerate(model.classes)}
    return pred_class, confidence, probabilities
