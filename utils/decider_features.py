from __future__ import annotations

from typing import Any, Dict

from pipelines.common import PipelineResult


DECIDER_FEATURE_NAMES = [
    "sn_score",
    "sn_locked",
    "sn_confirmed",
    "sn_candidates",
    "sn_rejected",
    "sn_candidate_margin",
    "sn_accept_ratio",
    "fw_score",
    "fight_events",
    "weapon_frames",
    "pose_verified_fight_frames",
    "pose_rejected_fight_boxes",
    "fight_pose_accept_ratio",
    "weapon_to_fight_ratio",
]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_decider_features(snatching_res: PipelineResult, fight_weapon_res: PipelineResult) -> Dict[str, float]:
    sn_meta: Dict[str, Any] = snatching_res.metadata or {}
    fw_meta: Dict[str, Any] = fight_weapon_res.metadata or {}

    sn_score = _safe_float(snatching_res.incident_score)
    sn_locked = 1.0 if bool(sn_meta.get("snatching_locked", False)) else 0.0
    sn_confirmed = _safe_int(sn_meta.get("confirmed_incidents", 0))
    sn_candidates = _safe_int(sn_meta.get("candidate_hits", 0))
    sn_rejected = _safe_int(sn_meta.get("rejected_hits", 0))
    sn_candidate_margin = sn_candidates - sn_rejected
    sn_accept_ratio = sn_candidates / max(1.0, float(sn_candidates + sn_rejected))

    fw_score = _safe_float(fight_weapon_res.incident_score)
    fight_events = _safe_int(fw_meta.get("fight_events", 0))
    weapon_frames = _safe_int(fw_meta.get("weapon_frames", 0))
    pose_verified_fight_frames = _safe_int(fw_meta.get("pose_verified_fight_frames", 0))
    pose_rejected_fight_boxes = _safe_int(fw_meta.get("pose_rejected_fight_boxes", 0))
    total_pose_boxes = pose_verified_fight_frames + pose_rejected_fight_boxes
    fight_pose_accept_ratio = pose_verified_fight_frames / max(1.0, float(total_pose_boxes))
    weapon_to_fight_ratio = weapon_frames / max(1.0, float(fight_events))

    return {
        "sn_score": sn_score,
        "sn_locked": sn_locked,
        "sn_confirmed": float(sn_confirmed),
        "sn_candidates": float(sn_candidates),
        "sn_rejected": float(sn_rejected),
        "sn_candidate_margin": float(sn_candidate_margin),
        "sn_accept_ratio": float(sn_accept_ratio),
        "fw_score": fw_score,
        "fight_events": float(fight_events),
        "weapon_frames": float(weapon_frames),
        "pose_verified_fight_frames": float(pose_verified_fight_frames),
        "pose_rejected_fight_boxes": float(pose_rejected_fight_boxes),
        "fight_pose_accept_ratio": float(fight_pose_accept_ratio),
        "weapon_to_fight_ratio": float(weapon_to_fight_ratio),
    }
