import cv2
import time
import math
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque
from ultralytics import YOLO

from .common import PipelinePaths, PipelineResult
from utils.config import get_config
from utils.hf_model import get_weapon_weights_path
from utils.io import ensure_dir, write_jsonl
from utils.pose_utils import (
    best_pose_for_box,
    detect_pose_people,
    load_pose_model,
    pose_aggression_score,
)
from utils.video import get_video_info, make_writer


def _download_weapon_model_from_hf() -> str:
    """
    Hugging Face repo: https://huggingface.co/psv12/weapon
    File: "All_weapon .pt"  (NOTE: space before .pt)
    """
    config = get_config()
    return get_weapon_weights_path(config)


def run_fight_weapon(video_path: str, out_dir: str) -> PipelineResult:
    config = get_config()
    fight_model_path = Path(config.fight_model_path)
    if not fight_model_path.exists():
        raise RuntimeError(f"FIGHT_MODEL_PATH does not exist: {fight_model_path}")

    WEAPON_CONF = 0.35
    FIGHT_CONF = 0.4
    EVENT_COOLDOWN_SEC = 5
    FIGHT_WINDOW_SEC = EVENT_COOLDOWN_SEC
    MAX_TRACK_DIST = 80
    FIGHT_FRAME_SKIP = 5
    FIGHT_POSE_SCORE_MIN = 0.6
    FIGHT_STRONG_CONF = 0.72
    FIGHT_MIN_TRACK_HITS = 2
    FIGHT_MIN_BOX_AREA_RATIO = 0.002
    BORDER_MARGIN_PX = 8
    MIN_FIGHTERS_FOR_EVENT = 2
    MIN_FIGHT_EVENT_CONF = 0.62
    POSE_FALLBACK_AGGR_MIN = 0.9
    POSE_FALLBACK_PAIR_DIST = 220
    POSE_FALLBACK_STREAK_FRAMES = 3
    POSE_FALLBACK_MIN_EVENT_CONF = 0.58

    out_dir = ensure_dir(Path(out_dir))
    snapshots_dir = ensure_dir(out_dir / "snapshots")

    paths = PipelinePaths(
        out_dir=out_dir,
        annotated_video=out_dir / "fight_weapon_annotated.mp4",
        annotations_jsonl=out_dir / "fight_weapon.jsonl",
        evidence_dir=snapshots_dir,
        snapshots_dir=snapshots_dir,
        events_csv=out_dir / "events.csv",
    )

    info = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    writer = make_writer(str(paths.annotated_video), info.w, info.h, info.fps)

    # ---- Weapon model from Hugging Face ----
    fight_model = YOLO(str(fight_model_path))
    weapon_model_path = _download_weapon_model_from_hf()
    weapon_model = YOLO(weapon_model_path)
    pose_model = load_pose_model(config.pose_model_name)
    pose_conf = config.pose_confidence

    raw_class_ids = (os.getenv("FIGHT_CLASS_IDS") or "0").strip()
    allowed_class_ids = set()
    for part in raw_class_ids.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            allowed_class_ids.add(int(p))
        except ValueError:
            pass
    if not allowed_class_ids:
        allowed_class_ids = {0}

    raw_class_names = (os.getenv("FIGHT_CLASS_NAMES") or "fight,violence,violent,assault,attack,aggression").strip()
    allowed_class_name_tokens = tuple(
        token.strip().lower()
        for token in raw_class_names.split(",")
        if token.strip()
    )

    def is_fight_like_label(raw_label: str, class_id: int) -> bool:
        if class_id in allowed_class_ids:
            return True
        label = (raw_label or "").strip().lower()
        if not label:
            return False
        weapon_keywords = ("weapon", "gun", "knife", "pistol", "rifle")
        if any(k in label for k in weapon_keywords):
            return False
        if allowed_class_name_tokens:
            return any(k in label for k in allowed_class_name_tokens)
        return False

    def local_fight_infer(frame):
        result = fight_model(frame, conf=FIGHT_CONF, verbose=False)[0]
        if result.boxes is None:
            return []
        names = getattr(result, "names", None)
        if names is None:
            names = getattr(fight_model, "names", {})
        preds = []
        for b in result.boxes:
            cls_idx = int(b.cls[0])
            if isinstance(names, dict):
                cls_name = str(names.get(cls_idx, cls_idx))
            elif isinstance(names, list) and 0 <= cls_idx < len(names):
                cls_name = str(names[cls_idx])
            else:
                cls_name = str(cls_idx)
            conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = x1 + (w / 2.0)
            cy = y1 + (h / 2.0)
            preds.append(
                {
                    "class": cls_name,
                    "class_id": cls_idx,
                    "confidence": conf,
                    "x": cx,
                    "y": cy,
                    "width": w,
                    "height": h,
                }
            )
        return preds

    # ---- simple tracking ----
    tracks = {}
    next_id = 1

    def centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def assign_ids(detections):
        nonlocal tracks, next_id
        new_tracks = {}
        for det in detections:
            box = det["box"]
            conf = float(det.get("conf", 0.0))
            c = centroid(box)
            best_id, best_d = None, MAX_TRACK_DIST
            for tid, prev in tracks.items():
                d = distance(c, prev["centroid"])
                if d < best_d:
                    best_d = d
                    best_id = tid
            if best_id is None:
                tid = f"F{next_id}"
                next_id += 1
                hits = 1
            else:
                tid = best_id
                hits = int(tracks.get(tid, {}).get("hits", 1)) + 1
            new_tracks[tid] = {"box": box, "centroid": c, "conf": conf, "hits": hits}
        tracks = new_tracks
        return new_tracks

    records = []
    annotations_rows = []
    frame_id = 0
    last_fight_time = 0.0
    active_fighter_ids = set()
    active_window_start = None
    active_max_simultaneous = 0
    wrist_history_by_fighter = {}
    pose_verified_fight_frames = 0
    pose_rejected_fight_boxes = 0
    pose_fallback_fight_frames = 0
    pose_fallback_streak = 0
    pose_fallback_conf_samples = []
    weapon_frames = 0
    weapon_conf_samples = []
    fight_conf_samples = []
    active_fight_conf_window = deque()
    fps = info.fps if info.fps and info.fps > 0 else 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        now = time.time()

        frame_ann = {"frame_id": frame_id, "items": []}
        fight_detections = []

        # ---- Fight detection every N frames ----
        if frame_id % FIGHT_FRAME_SKIP == 0:
            try:
                predictions = local_fight_infer(frame)
                for p in predictions:
                    cls_label = str(p.get("class", "") or p.get("class_name", "") or "")
                    cls_id = int(p.get("class_id", -1) or -1)
                    if not is_fight_like_label(cls_label, cls_id):
                        continue
                    conf = float(p.get("confidence", 0) or 0.0)
                    if conf >= FIGHT_CONF:
                        x1 = int(p["x"] - p["width"] / 2)
                        y1 = int(p["y"] - p["height"] / 2)
                        x2 = int(p["x"] + p["width"] / 2)
                        y2 = int(p["y"] + p["height"] / 2)

                        x1 = max(0, min(info.w - 1, x1))
                        y1 = max(0, min(info.h - 1, y1))
                        x2 = max(0, min(info.w - 1, x2))
                        y2 = max(0, min(info.h - 1, y2))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        bw = x2 - x1
                        bh = y2 - y1
                        box_area_ratio = (bw * bh) / max(1.0, float(info.w * info.h))
                        if box_area_ratio < FIGHT_MIN_BOX_AREA_RATIO:
                            continue
                        if (
                            x1 <= BORDER_MARGIN_PX
                            or y1 <= BORDER_MARGIN_PX
                            or x2 >= (info.w - BORDER_MARGIN_PX)
                            or y2 >= (info.h - BORDER_MARGIN_PX)
                        ):
                            # Border-touching boxes are frequent false positives.
                            continue

                        fight_detections.append({"box": (x1, y1, x2, y2), "conf": conf})
            except Exception as e:
                print("Fight model infer error:", e)

        tracked = assign_ids(fight_detections)
        pose_people = detect_pose_people(pose_model, frame, conf=pose_conf) if tracked else []
        verified_tracked = {}
        for tid, data in tracked.items():
            box = data["box"]
            det_conf = float(data.get("conf", 0.0))
            hit_streak = int(data.get("hits", 1))
            pose = best_pose_for_box(pose_people, box, min_iou=0.05)
            if pose is None:
                # If pose is unavailable, keep only stronger persistent tracks.
                if det_conf >= FIGHT_STRONG_CONF or hit_streak >= FIGHT_MIN_TRACK_HITS:
                    verified_tracked[tid] = data
                    fight_conf_samples.append(det_conf)
                continue
            score, wrist = pose_aggression_score(
                pose,
                wrist_history_by_fighter.get(tid),
                fps=fps,
                min_conf=pose_conf,
            )
            if wrist is not None:
                wrist_history_by_fighter[tid] = wrist
            if score >= FIGHT_POSE_SCORE_MIN:
                verified_tracked[tid] = data
                combined_conf = min(1.0, max(0.0, (0.65 * det_conf) + (0.35 * min(score, 1.0))))
                fight_conf_samples.append(combined_conf)
            else:
                pose_rejected_fight_boxes += 1
        tracked = verified_tracked
        if tracked:
            pose_verified_fight_frames += 1
            pose_fallback_streak = 0

        # Pose-only fallback when fight boxes are missed.
        if not tracked:
            fallback_people = detect_pose_people(pose_model, frame, conf=pose_conf)
            fallback_candidates = []
            for pp in fallback_people:
                score, _ = pose_aggression_score(pp, None, fps=fps, min_conf=pose_conf)
                if score >= POSE_FALLBACK_AGGR_MIN:
                    box = pp.get("box")
                    if isinstance(box, tuple) and len(box) == 4:
                        x1, y1, x2, y2 = box
                        fallback_candidates.append((box, float(min(1.0, max(0.0, score / 1.8)))))

            fallback_trigger = False
            fallback_conf = 0.0
            if len(fallback_candidates) >= 2:
                best_pair_conf = 0.0
                best_pair = None
                for i in range(len(fallback_candidates)):
                    for j in range(i + 1, len(fallback_candidates)):
                        (a_box, a_conf) = fallback_candidates[i]
                        (b_box, b_conf) = fallback_candidates[j]
                        ac = centroid(a_box)
                        bc = centroid(b_box)
                        if distance(ac, bc) <= POSE_FALLBACK_PAIR_DIST:
                            conf = (a_conf + b_conf) / 2.0
                            if conf > best_pair_conf:
                                best_pair_conf = conf
                                best_pair = (a_box, b_box)
                if best_pair is not None:
                    fallback_trigger = True
                    fallback_conf = best_pair_conf
                    pose_fallback_fight_frames += 1
                    pose_fallback_conf_samples.append(fallback_conf)
                    for idx, box in enumerate(best_pair, start=1):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(
                            frame,
                            f"FIGHT_POSE P{idx}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2,
                        )
                        frame_ann["items"].append(
                            {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "label": "FIGHT_POSE",
                                "conf": float(fallback_conf),
                                "source": "fight_weapon",
                            }
                        )

            if fallback_trigger:
                pose_fallback_streak += 1
            else:
                pose_fallback_streak = max(0, pose_fallback_streak - 1)

            if (
                pose_fallback_streak >= POSE_FALLBACK_STREAK_FRAMES
                and fallback_conf >= POSE_FALLBACK_MIN_EVENT_CONF
                and (now - last_fight_time) > EVENT_COOLDOWN_SEC
            ):
                snap_name = f"fight_pose_frame{frame_id}_{int(now)}.jpg"
                snap_path = snapshots_dir / snap_name
                cv2.imwrite(str(snap_path), frame)
                records.append(
                    {
                        "event": "fight",
                        "frame_id": frame_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "num_fighters": 2,
                        "fighter_ids": "POSE_P1,POSE_P2",
                        "event_confidence": float(fallback_conf),
                        "snapshot_path": str(snap_path),
                    }
                )
                last_fight_time = now
                pose_fallback_streak = 0

        # Draw + annotate fight boxes
        for tid, data in tracked.items():
            x1, y1, x2, y2 = data["box"]
            det_conf = float(data.get("conf", 0.0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"FIGHTER {tid}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            frame_ann["items"].append(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": f"FIGHT {tid}", "conf": det_conf, "source": "fight_weapon"}
            )

        # Track fighters in the window
        if tracked:
            if active_window_start is None:
                active_window_start = now
            active_max_simultaneous = max(active_max_simultaneous, len(tracked))
            for tid, data in tracked.items():
                active_fighter_ids.add(tid)
                active_fight_conf_window.append(float(data.get("conf", 0.0)))

        # Snapshot logging logic
        should_log = False
        if tracked and (now - last_fight_time) > EVENT_COOLDOWN_SEC:
            should_log = True
        if (not tracked) and active_window_start is not None and (now - active_window_start) > FIGHT_WINDOW_SEC:
            should_log = True

        if should_log and active_window_start is not None and len(active_fighter_ids) > 0:
            snap_name = f"fight_frame{frame_id}_{int(now)}.jpg"
            snap_path = snapshots_dir / snap_name
            cv2.imwrite(str(snap_path), frame)
            event_conf = (
                sum(active_fight_conf_window) / max(1, len(active_fight_conf_window))
                if active_fight_conf_window
                else 0.0
            )
            # Fight event requires multi-person evidence and minimum confidence.
            if (
                len(active_fighter_ids) >= MIN_FIGHTERS_FOR_EVENT
                and active_max_simultaneous >= MIN_FIGHTERS_FOR_EVENT
                and event_conf >= MIN_FIGHT_EVENT_CONF
            ):
                records.append(
                    {
                        "event": "fight",
                        "frame_id": frame_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "num_fighters": len(active_fighter_ids),
                        "fighter_ids": ",".join(sorted(active_fighter_ids)),
                        "event_confidence": float(event_conf),
                        "snapshot_path": str(snap_path),
                    }
                )

            last_fight_time = now
            active_fighter_ids.clear()
            active_fight_conf_window.clear()
            active_window_start = None
            active_max_simultaneous = 0

        # ---- Weapon detection (HF YOLO weights) ----
        weapon_results = weapon_model(frame, conf=WEAPON_CONF, verbose=False)[0]
        weapon_detected_this_frame = False
        for b in weapon_results.boxes:
            label = weapon_model.names[int(b.cls[0])]
            if str(label).lower() == "person":
                continue
            conf = float(b.conf[0]) if hasattr(b, "conf") else 1.0

            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                str(label),
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            frame_ann["items"].append(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": f"WEAPON:{label}", "conf": conf, "source": "fight_weapon"}
            )
            weapon_detected_this_frame = True
            weapon_conf_samples.append(conf)

        if weapon_detected_this_frame:
            weapon_frames += 1

        if active_window_start is not None:
            cv2.putText(
                frame,
                f"FIGHTERS INVOLVED: {len(active_fighter_ids)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        writer.write(frame)
        annotations_rows.append(frame_ann)

    cap.release()
    writer.release()

    pd.DataFrame(records).to_csv(paths.events_csv, index=False)
    write_jsonl(paths.annotations_jsonl, annotations_rows)

    fight_events = len(records)
    fight_confidence = (
        (sum(float(r.get("event_confidence", 0.0)) for r in records) / max(1, fight_events))
        if records
        else (
            max(
                (sum(fight_conf_samples) / max(1, len(fight_conf_samples)) if fight_conf_samples else 0.0),
                (sum(pose_fallback_conf_samples) / max(1, len(pose_fallback_conf_samples)) if pose_fallback_conf_samples else 0.0),
            )
        )
    )
    weapon_confidence = sum(weapon_conf_samples) / max(1, len(weapon_conf_samples)) if weapon_conf_samples else 0.0
    incident_confidence = min(
        1.0,
        max(
            0.0,
            (0.55 * fight_confidence)
            + (0.45 * weapon_confidence)
            + (0.04 * min(4, fight_events))
            + (0.02 * min(8, weapon_frames)),
        ),
    )

    incident_found = (fight_events > 0) or (weapon_frames > 0)
    return PipelineResult(
        name="fight_weapon",
        incident_found=incident_found,
        incident_score=float(fight_events + weapon_frames),
        annotated_video_path=str(paths.annotated_video),
        annotations_jsonl_path=str(paths.annotations_jsonl),
        evidence_dir=str(paths.snapshots_dir) if incident_found else None,
        events_csv_path=str(paths.events_csv) if fight_events > 0 else None,
        metadata={
            "fight_events": fight_events,
            "weapon_frames": weapon_frames,
            "pose_verified_fight_frames": pose_verified_fight_frames,
            "pose_rejected_fight_boxes": pose_rejected_fight_boxes,
            "pose_fallback_fight_frames": pose_fallback_fight_frames,
            "fight_confidence": float(fight_confidence),
            "weapon_confidence": float(weapon_confidence),
            "incident_confidence": float(incident_confidence),
        },
    )
