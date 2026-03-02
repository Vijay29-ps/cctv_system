from __future__ import annotations

import os
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from pipelines.common import PipelineResult
from .decider_features import DECIDER_FEATURE_NAMES, build_decider_features
from .decider_model import load_decider_model, predict_decider_class


@dataclass(frozen=True)
class DecisionResult:
    snatching_found: bool
    fight_found: bool
    weapon_found: bool
    incident_found: bool
    reason: str
    incident_confidence: float = 0.0
    snatching_confidence: float = 0.0
    fight_weapon_confidence: float = 0.0
    decider_class: str | None = None
    decider_confidence: float = 0.0
    decider_probabilities: Dict[str, float] | None = None
    decider_features: Dict[str, float] | None = None


def _path_from_env(name: str, default_path: str) -> Path:
    raw = (os.getenv(name) or default_path).strip()
    return Path(raw)


@lru_cache(maxsize=1)
def _load_trained_decider_model():
    model_path = _path_from_env("DECIDER_MODEL_PATH", "models/incident_decider_v1.json")
    if not model_path.exists():
        return None
    try:
        return load_decider_model(model_path)
    except Exception:
        return None


def _dominant_output_enabled() -> bool:
    mode = (os.getenv("DECIDER_OUTPUT_MODE") or "dominant").strip().lower()
    return mode != "multi"


def _snatching_priority_enabled() -> bool:
    mode = (os.getenv("DECIDER_SNATCHING_PRIORITY") or "true").strip().lower()
    return mode not in {"0", "false", "no", "off"}


def _phase1_incident_threshold() -> float:
    raw = (os.getenv("DECIDER_PHASE1_INCIDENT_THRESHOLD") or "0.55").strip()
    try:
        val = float(raw)
    except ValueError:
        val = 0.55
    return min(0.95, max(0.05, val))


def _apply_dominant_output(
    snatching_found: bool,
    fight_found: bool,
    weapon_found: bool,
    snatching_confidence: float,
    fight_weapon_confidence: float,
    reason: str,
    snatching_priority: bool = False,
) -> tuple[bool, bool, bool, str]:
    if snatching_found and (fight_found or weapon_found) and snatching_priority and _snatching_priority_enabled():
        return snatching_found, fight_found, weapon_found, f"{reason}|snatching-priority"

    if not _dominant_output_enabled():
        return snatching_found, fight_found, weapon_found, reason

    if snatching_found and (fight_found or weapon_found):
        if snatching_confidence >= fight_weapon_confidence:
            return True, False, False, f"{reason}|dominant:snatching"
        return False, True, True, f"{reason}|dominant:fight_weapon"
    return snatching_found, fight_found, weapon_found, reason


def _heuristic_decision(snatching_res: PipelineResult, fight_weapon_res: PipelineResult) -> DecisionResult:
    sn_meta: Dict[str, Any] = snatching_res.metadata or {}
    fw_meta: Dict[str, Any] = fight_weapon_res.metadata or {}

    sn_locked = bool(sn_meta.get("snatching_locked", False))
    sn_confirmed = int(sn_meta.get("confirmed_incidents", 0) or 0)
    sn_candidates = int(sn_meta.get("candidate_hits", 0) or 0)
    sn_rejected = int(sn_meta.get("rejected_hits", 0) or 0)

    fight_events = int(fw_meta.get("fight_events", 0) or 0)
    weapon_frames = int(fw_meta.get("weapon_frames", 0) or 0)
    pose_verified = int(fw_meta.get("pose_verified_fight_frames", 0) or 0)
    pose_rejected = int(fw_meta.get("pose_rejected_fight_boxes", 0) or 0)
    fight_pose_accept_ratio = pose_verified / max(1.0, float(pose_verified + pose_rejected))

    # Conservative fight gate to reduce one-box edge false positives.
    fight_found = fight_events >= 1 and (pose_verified >= 2 or fight_pose_accept_ratio >= 0.55)
    weapon_found = weapon_frames >= 1

    # Snatching logic remains strict and context-aware.
    if sn_confirmed >= 2:
        snatching_found = True
        reason = "heuristic:snatching-confirmed-2plus"
    elif sn_confirmed == 1:
        if fight_events > 0:
            snatching_found = True
            reason = "heuristic:snatching-confirmed-then-fight"
        elif weapon_frames > 0 and fight_events == 0:
            snatching_found = False
            reason = "heuristic:weapon-only-overruled-snatching"
        else:
            snatching_found = sn_candidates >= 8 and sn_candidates > sn_rejected
            reason = "heuristic:snatching-only-candidate-check"
    else:
        snatching_found = sn_locked and sn_candidates >= 6 and sn_candidates > sn_rejected
        reason = "heuristic:no-confirmed-snatching"

    sn_conf_meta = float(sn_meta.get("incident_confidence", 0.0) or 0.0)
    fw_conf_meta = float(fw_meta.get("incident_confidence", 0.0) or 0.0)
    snatching_confidence_calc = min(
        1.0,
        max(
            0.0,
            (0.35 if sn_locked else 0.0)
            + (0.22 * min(sn_confirmed, 3))
            + (0.015 * sn_candidates)
            - (0.01 * sn_rejected),
        ),
    )
    fight_weapon_confidence_calc = min(
        1.0,
        max(
            0.0,
            (0.18 * fight_events)
            + (0.12 * weapon_frames)
            + (0.25 * fight_pose_accept_ratio),
        ),
    )
    snatching_confidence = max(sn_conf_meta, snatching_confidence_calc)
    fight_weapon_confidence = max(fw_conf_meta, fight_weapon_confidence_calc)

    snatching_found, fight_found, weapon_found, reason = _apply_dominant_output(
        snatching_found=snatching_found,
        fight_found=fight_found,
        weapon_found=weapon_found,
        snatching_confidence=snatching_confidence,
        fight_weapon_confidence=fight_weapon_confidence,
        reason=reason,
        snatching_priority=(sn_confirmed >= 1 or sn_locked),
    )

    incident_found = snatching_found or fight_found or weapon_found
    incident_confidence = max(
        (snatching_confidence if snatching_found else 0.0),
        (fight_weapon_confidence if (fight_found or weapon_found) else 0.0),
    )
    return DecisionResult(
        snatching_found=snatching_found,
        fight_found=fight_found,
        weapon_found=weapon_found,
        incident_found=incident_found,
        reason=reason,
        incident_confidence=incident_confidence,
        snatching_confidence=snatching_confidence,
        fight_weapon_confidence=fight_weapon_confidence,
    )


def decide_incidents(snatching_res: PipelineResult, fight_weapon_res: PipelineResult) -> DecisionResult:
    sn_meta: Dict[str, Any] = snatching_res.metadata or {}
    fw_meta: Dict[str, Any] = fight_weapon_res.metadata or {}
    sn_confirmed = int(sn_meta.get("confirmed_incidents", 0) or 0)
    sn_locked = bool(sn_meta.get("snatching_locked", False))
    sn_candidates = int(sn_meta.get("candidate_hits", 0) or 0)
    sn_rejected = int(sn_meta.get("rejected_hits", 0) or 0)
    sn_explicit_evidence = sn_confirmed >= 1 or sn_locked
    fight_events = int(fw_meta.get("fight_events", 0) or 0)
    weapon_frames = int(fw_meta.get("weapon_frames", 0) or 0)
    fight_explicit_evidence = fight_events > 0
    weapon_explicit_evidence = weapon_frames > 0

    features = build_decider_features(snatching_res, fight_weapon_res)
    model = _load_trained_decider_model()
    fallback = _heuristic_decision(snatching_res, fight_weapon_res)
    if model is None:
        return DecisionResult(
            snatching_found=fallback.snatching_found,
            fight_found=fallback.fight_found,
            weapon_found=fallback.weapon_found,
            incident_found=fallback.incident_found,
            reason=f"{fallback.reason}|model:missing",
            incident_confidence=fallback.incident_confidence,
            snatching_confidence=fallback.snatching_confidence,
            fight_weapon_confidence=fallback.fight_weapon_confidence,
            decider_features={k: float(features.get(k, 0.0)) for k in DECIDER_FEATURE_NAMES},
        )

    pred_class, confidence, probabilities = predict_decider_class(model, features)
    min_conf = float(os.getenv("DECIDER_MIN_CONFIDENCE", model.min_confidence))
    if confidence < min_conf:
        return DecisionResult(
            snatching_found=fallback.snatching_found,
            fight_found=fallback.fight_found,
            weapon_found=fallback.weapon_found,
            incident_found=fallback.incident_found,
            reason=f"{fallback.reason}|model:low-confidence:{pred_class}:{confidence:.3f}",
            incident_confidence=fallback.incident_confidence,
            snatching_confidence=fallback.snatching_confidence,
            fight_weapon_confidence=fallback.fight_weapon_confidence,
            decider_class=pred_class,
            decider_confidence=confidence,
            decider_probabilities=probabilities,
            decider_features={k: float(features.get(k, 0.0)) for k in DECIDER_FEATURE_NAMES},
        )

    if pred_class == "both":
        snatching_found = True
        fight_found = True
        weapon_found = True
    elif pred_class == "snatching":
        snatching_found = True
        fight_found = False
        weapon_found = False
    elif pred_class == "fight_weapon":
        snatching_found = False
        fight_found = True
        weapon_found = True
    else:
        snatching_found = False
        fight_found = False
        weapon_found = False

    fw_evidence_override = False
    # Evidence-first: include fight/weapon when branch has explicit evidence.
    if fight_explicit_evidence:
        if not fight_found:
            fw_evidence_override = True
        fight_found = True
    else:
        fight_found = False
    if weapon_explicit_evidence:
        if not weapon_found:
            fw_evidence_override = True
        weapon_found = True
    else:
        weapon_found = False

    fight_inferred_from_model_with_weapon = False
    if (
        pred_class == "fight_weapon"
        and confidence >= 0.80
        and weapon_explicit_evidence
        and not fight_explicit_evidence
    ):
        fight_found = True
        fight_inferred_from_model_with_weapon = True

    noisy_snatching_with_fw = (
        (not sn_explicit_evidence)
        and (fight_explicit_evidence or weapon_explicit_evidence)
        and (sn_confirmed < 2)
        and (sn_rejected > sn_candidates)
    )
    if noisy_snatching_with_fw and snatching_found:
        snatching_found = False

    snatching_evidence_override = False
    # Preserve snatching when explicit evidence exists.
    if _snatching_priority_enabled() and sn_explicit_evidence and not snatching_found:
        snatching_found = True
        snatching_evidence_override = True

    # Preserve snatching in mixed scenes.
    if (
        _snatching_priority_enabled()
        and sn_explicit_evidence
        and (fight_found or weapon_found)
    ):
        snatching_found = True

    snatching_confidence = float(probabilities.get("snatching", 0.0) + probabilities.get("both", 0.0))
    fight_weapon_confidence = float(probabilities.get("fight_weapon", 0.0) + probabilities.get("both", 0.0))

    snatching_found, fight_found, weapon_found, reason = _apply_dominant_output(
        snatching_found=snatching_found,
        fight_found=fight_found,
        weapon_found=weapon_found,
        snatching_confidence=snatching_confidence,
        fight_weapon_confidence=fight_weapon_confidence,
        reason=f"model:{pred_class}:{confidence:.3f}",
        snatching_priority=(sn_confirmed >= 1 or sn_locked),
    )
    if snatching_evidence_override:
        reason = f"{reason}|snatching-evidence-override"
    if fw_evidence_override:
        reason = f"{reason}|fight-weapon-evidence-override"
    if fight_inferred_from_model_with_weapon:
        reason = f"{reason}|fight-inferred-from-model-with-weapon"
    if noisy_snatching_with_fw:
        reason = f"{reason}|snatching-suppressed-noisy-with-fight-weapon"

    incident_confidence = float(1.0 - probabilities.get("none", 0.0))
    phase1_thr = _phase1_incident_threshold()
    incident_found = snatching_found or fight_found or weapon_found
    if incident_found and incident_confidence < phase1_thr and not sn_explicit_evidence:
        snatching_found = False
        fight_found = False
        weapon_found = False
        incident_found = False
        reason = f"{reason}|phase1:below-threshold:{incident_confidence:.3f}<{phase1_thr:.3f}"
    elif incident_found and incident_confidence < phase1_thr and sn_explicit_evidence:
        reason = f"{reason}|phase1:bypass-snatching-evidence"

    return DecisionResult(
        snatching_found=snatching_found,
        fight_found=fight_found,
        weapon_found=weapon_found,
        incident_found=incident_found,
        reason=reason,
        incident_confidence=incident_confidence,
        snatching_confidence=snatching_confidence,
        fight_weapon_confidence=fight_weapon_confidence,
        decider_class=pred_class,
        decider_confidence=confidence,
        decider_probabilities=probabilities,
        decider_features={k: float(features.get(k, 0.0)) for k in DECIDER_FEATURE_NAMES},
    )
