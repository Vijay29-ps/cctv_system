from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

Point = Tuple[float, float]
Box = Tuple[int, int, int, int]

# COCO keypoint indices (Ultralytics pose model)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


def load_pose_model(model_name: str = "yolov8n-pose.pt") -> YOLO:
    return YOLO(model_name)


def detect_pose_people(model: YOLO, frame: np.ndarray, conf: float = 0.25) -> List[Dict[str, object]]:
    results = model(frame, conf=conf, verbose=False)
    if not results:
        return []

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return []

    boxes = res.boxes.xyxy.cpu().numpy()
    kxy = res.keypoints.xy.cpu().numpy() if res.keypoints.xy is not None else None
    kconf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
    if kxy is None or kconf is None:
        return []

    out: List[Dict[str, object]] = []
    n = min(len(boxes), len(kxy), len(kconf))
    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        out.append(
            {
                "box": (x1, y1, x2, y2),
                "kxy": kxy[i],
                "kconf": kconf[i],
            }
        )
    return out


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    aa = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    ba = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    denom = aa + ba - inter
    return 0.0 if denom <= 0 else inter / denom


def best_pose_for_box(
    poses: List[Dict[str, object]],
    target_box: Box,
    min_iou: float = 0.05,
) -> Optional[Dict[str, object]]:
    best_pose: Optional[Dict[str, object]] = None
    best_iou = min_iou
    for p in poses:
        pbox = p.get("box")
        if not isinstance(pbox, tuple):
            continue
        ov = _iou(target_box, pbox)  # type: ignore[arg-type]
        if ov >= best_iou:
            best_iou = ov
            best_pose = p
    return best_pose


def _get_kpt(pose: Dict[str, object], idx: int, min_conf: float = 0.25) -> Optional[Point]:
    kxy = pose.get("kxy")
    kconf = pose.get("kconf")
    if not isinstance(kxy, np.ndarray) or not isinstance(kconf, np.ndarray):
        return None
    if idx >= len(kxy) or idx >= len(kconf):
        return None
    if float(kconf[idx]) < min_conf:
        return None
    x, y = float(kxy[idx][0]), float(kxy[idx][1])
    return (x, y)


def wrist_center(pose: Dict[str, object], min_conf: float = 0.25) -> Optional[Point]:
    lw = _get_kpt(pose, LEFT_WRIST, min_conf=min_conf)
    rw = _get_kpt(pose, RIGHT_WRIST, min_conf=min_conf)
    if lw and rw:
        return ((lw[0] + rw[0]) / 2.0, (lw[1] + rw[1]) / 2.0)
    return lw or rw


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _arm_extension_ratio(pose: Dict[str, object], min_conf: float = 0.25) -> float:
    ratios: List[float] = []
    for s_idx, e_idx, w_idx in [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    ]:
        s = _get_kpt(pose, s_idx, min_conf=min_conf)
        e = _get_kpt(pose, e_idx, min_conf=min_conf)
        w = _get_kpt(pose, w_idx, min_conf=min_conf)
        if not (s and e and w):
            continue
        upper = max(_dist(s, e), 1.0)
        full = _dist(s, w)
        ratios.append(full / upper)
    if not ratios:
        return 0.0
    return max(ratios)


def pose_aggression_score(
    pose: Optional[Dict[str, object]],
    prev_wrist: Optional[Point],
    fps: float,
    min_conf: float = 0.25,
) -> Tuple[float, Optional[Point]]:
    if pose is None:
        return 0.0, None

    score = 0.0
    wrist = wrist_center(pose, min_conf=min_conf)
    if wrist and prev_wrist:
        speed = _dist(wrist, prev_wrist) * max(fps, 1.0)
        if speed >= 140:
            score += 1.0

    ext = _arm_extension_ratio(pose, min_conf=min_conf)
    if ext >= 1.8:
        score += 0.7

    ls = _get_kpt(pose, LEFT_SHOULDER, min_conf=min_conf)
    rs = _get_kpt(pose, RIGHT_SHOULDER, min_conf=min_conf)
    lw = _get_kpt(pose, LEFT_WRIST, min_conf=min_conf)
    rw = _get_kpt(pose, RIGHT_WRIST, min_conf=min_conf)
    if (ls and lw and lw[1] < ls[1] - 8) or (rs and rw and rw[1] < rs[1] - 8):
        score += 0.4

    return score, wrist


def snatching_pose_score(
    suspect_pose: Optional[Dict[str, object]],
    victim_box: Optional[Box],
    prev_suspect_wrist: Optional[Point],
    fps: float,
    min_conf: float = 0.25,
) -> Tuple[float, Optional[Point]]:
    if suspect_pose is None or victim_box is None:
        return 0.0, None

    score = 0.0
    wrist = wrist_center(suspect_pose, min_conf=min_conf)
    if wrist is None:
        return 0.0, None

    if prev_suspect_wrist:
        speed = _dist(wrist, prev_suspect_wrist) * max(fps, 1.0)
        if speed >= 160:
            score += 1.0

    ext = _arm_extension_ratio(suspect_pose, min_conf=min_conf)
    if ext >= 1.8:
        score += 0.6

    vx1, vy1, vx2, vy2 = victim_box
    vc = ((vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0)
    vdiag = math.hypot(max(1, vx2 - vx1), max(1, vy2 - vy1))
    max_reach = max(90.0, 0.65 * vdiag)
    if _dist(wrist, vc) <= max_reach:
        score += 1.0

    return score, wrist
