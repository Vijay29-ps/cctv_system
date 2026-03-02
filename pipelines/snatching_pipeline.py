import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
from typing import Optional

from .common import PipelinePaths, PipelineResult
from utils.config import get_config
from utils.hf_model import get_snatching_weights_path
from utils.io import ensure_dir, write_jsonl
from utils.pose_utils import (
    best_pose_for_box,
    detect_pose_people,
    load_pose_model,
    pose_aggression_score,
    snatching_pose_score,
)
from utils.video import get_video_info, make_writer


def run_snatching(
    video_path: str,
    out_dir: str,
    evidence_dir: Optional[str] = None,
    camera_id: str = "Unknown Camera",
) -> PipelineResult:
    ACC_THRESHOLD = 2.5
    ALERT_FRAMES = 40
    HISTORY = 5
    RF_CONF = 0.5
    RF_PROBE_COOLDOWN_FRAMES = 8
    SNATCH_CONF_WITH_MOTORCYCLE = 0.55
    SNATCH_CONF_WITHOUT_MOTORCYCLE = 0.62
    SNATCH_GLOBAL_RF_CONF = 0.68
    SNATCH_SINGLE_HIT_STRONG_CONF = 0.78
    SNATCH_CONFIRM_MIN_HITS = 2
    SNATCH_CONFIRM_WINDOW_SEC = 2.5
    CONFIRMED_INCIDENT_COOLDOWN_SEC = 8.0
    NEARBY_VICTIM_MAX_DIST_PX = 220
    NEARBY_MOTORCYCLE_MAX_DIST_PX = 260
    SNATCH_POSE_MIN_SCORE = 1.0
    FIGHT_LIKE_POSE_MIN_SCORE = 1.0
    GLOBAL_RF_PROBE_COOLDOWN_FRAMES = 6

    config = get_config()
    pose_conf = config.pose_confidence
    snatching_model_path = Path(config.snatching_model_path)
    if not snatching_model_path.exists():
        try:
            snatching_model_path = Path(get_snatching_weights_path(config))
        except Exception as exc:
            raise RuntimeError(
                "SNATCHING_MODEL_PATH does not exist and Hugging Face fallback failed: "
                f"{snatching_model_path} | error={exc}"
            ) from exc

    out_dir_path = ensure_dir(Path(out_dir))
    if evidence_dir:
        evidence_dir_path = ensure_dir(Path(evidence_dir))
    else:
        evidence_dir_path = ensure_dir(out_dir_path / "evidence")

    paths = PipelinePaths(
        out_dir=out_dir_path,
        annotated_video=out_dir_path / "snatching_annotated.mp4",
        annotations_jsonl=out_dir_path / "snatching.jsonl",
        evidence_dir=evidence_dir_path,
        snapshots_dir=evidence_dir_path,
    )

    info = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    writer = make_writer(str(paths.annotated_video), info.w, info.h, info.fps)

    yolo = YOLO("yolov8n.pt")
    pose_model = load_pose_model(config.pose_model_name)

    snatch_model = YOLO(str(snatching_model_path))

    track_history = defaultdict(lambda: deque(maxlen=HISTORY))
    candidate_hit_times = defaultdict(lambda: deque())
    wrist_history_by_tid = {}
    alert_conf_by_tid = {}
    alert_state = {}
    rf_checked = {}
    global_rf_checked = -999999
    global_candidate_hit_times = deque()
    global_alert_frames = 0
    global_alert_conf = 0.0
    global_alert_box = None
    last_confirmed_incident_sec = -1e9
    snatching_locked = False
    frame_id = 0

    annotations_rows = []
    incident_count = 0
    raw_confirmed_incidents = 0
    candidate_hits = 0
    rejected_hits = 0
    candidate_conf_samples = []
    confirmed_conf_samples = []
    confirmed_suspect_ids = set()
    confirmed_victim_ids = set()
    fps = info.fps if info.fps and info.fps > 0 else 30.0
    is_low_frame_input = int(info.frames or 0) <= 3

    raw_class_ids = (os.getenv("SNATCHING_CLASS_IDS") or "0").strip()
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

    raw_class_names = (os.getenv("SNATCHING_CLASS_NAMES") or "snatching,chain_snatching,snatch,robbery,theft,steal").strip()
    allowed_class_name_tokens = tuple(
        token.strip().lower()
        for token in raw_class_names.split(",")
        if token.strip()
    )

    def get_dominant_color(image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return "Unknown"

        img_small = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        hsv_image = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

        color_ranges = {
            "BLACK": ([0, 0, 0], [180, 255, 50]),
            "WHITE": ([0, 0, 200], [180, 30, 255]),
            "GRAY": ([0, 0, 51], [180, 30, 199]),
            "RED": ([0, 70, 50], [10, 255, 255]),
            "RED2": ([170, 70, 50], [180, 255, 255]),
            "GREEN": ([36, 50, 50], [89, 255, 255]),
            "BLUE": ([90, 50, 50], [130, 255, 255]),
            "YELLOW": ([20, 100, 100], [35, 255, 255]),
        }

        color_pixel_counts = defaultdict(int)
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(hsv_image, lower, upper)
            color_pixel_counts[color_name.replace("2", "")] += cv2.countNonZero(mask)

        if not color_pixel_counts:
            return "Unknown"

        dominant_color = max(color_pixel_counts, key=color_pixel_counts.get)

        total_pixels = img_small.shape[0] * img_small.shape[1]
        if color_pixel_counts[dominant_color] / total_pixels < 0.1:
            return "Mixed/Other"

        return dominant_color.capitalize()

    def get_direction(points: deque) -> str:
        if len(points) < 2:
            return "Unknown"
        start_pt, end_pt = points[0], points[-1]
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]

        if abs(dx) < 10 and abs(dy) < 10:
            return "Stationary"

        angle = np.rad2deg(np.arctan2(-dy, dx))
        if -22.5 <= angle < 22.5:
            return "East"
        if 22.5 <= angle < 67.5:
            return "Northeast"
        if 67.5 <= angle < 112.5:
            return "North"
        if 112.5 <= angle < 157.5:
            return "Northwest"
        if angle >= 157.5 or angle < -157.5:
            return "West"
        if -157.5 <= angle < -112.5:
            return "Southwest"
        if -112.5 <= angle < -67.5:
            return "South"
        if -67.5 <= angle < -22.5:
            return "Southeast"
        return "Unknown"

    def _is_snatching_class(raw_label: str, class_id: int) -> bool:
        if class_id in allowed_class_ids:
            return True
        cls_name = (raw_label or "").strip().lower()
        if not cls_name:
            return False
        if cls_name.isdigit():
            try:
                return int(cls_name) in allowed_class_ids
            except ValueError:
                return False
        if allowed_class_name_tokens:
            return any(token in cls_name for token in allowed_class_name_tokens)
        return False

    def local_snatching_infer(image: np.ndarray, conf: float) -> list[dict]:
        result = snatch_model(image, conf=conf, verbose=False)[0]
        if result.boxes is None:
            return []

        names = getattr(result, "names", None)
        if names is None:
            names = getattr(snatch_model, "names", {})

        preds = []
        for b in result.boxes:
            cls_idx = int(b.cls[0])
            if isinstance(names, dict):
                cls_name = str(names.get(cls_idx, cls_idx))
            elif isinstance(names, list) and 0 <= cls_idx < len(names):
                cls_name = str(names[cls_idx])
            else:
                cls_name = str(cls_idx)

            conf_v = float(b.conf[0]) if hasattr(b, "conf") else 0.0
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = x1 + (w / 2.0)
            cy = y1 + (h / 2.0)
            preds.append(
                {
                    "class": cls_name,
                    "class_id": cls_idx,
                    "confidence": conf_v,
                    "x": cx,
                    "y": cy,
                    "width": w,
                    "height": h,
                }
            )
        return preds

    def snatching_confidence(preds: list) -> float:
        best_conf = 0.0
        for p in preds:
            conf = float(p.get("confidence", 0) or 0)
            cls_name = str(p.get("class", "")).strip().lower()
            cls_id = int(p.get("class_id", -1) or -1)
            if not _is_snatching_class(cls_name, cls_id):
                continue
            if conf > best_conf:
                best_conf = conf
        return best_conf

    def best_snatching_prediction(preds: list) -> tuple[float, Optional[tuple[int, int, int, int]]]:
        best_conf = 0.0
        best_box = None
        for p in preds:
            conf = float(p.get("confidence", 0) or 0)
            cls_name = str(p.get("class", "")).strip().lower()
            cls_id = int(p.get("class_id", -1) or -1)
            if not _is_snatching_class(cls_name, cls_id):
                continue
            if conf <= best_conf:
                continue
            px = float(p.get("x", 0.0))
            py = float(p.get("y", 0.0))
            pw = float(p.get("width", 0.0))
            ph = float(p.get("height", 0.0))
            x1 = int(max(0, px - pw / 2))
            y1 = int(max(0, py - ph / 2))
            x2 = int(max(x1 + 1, px + pw / 2))
            y2 = int(max(y1 + 1, py + ph / 2))
            best_conf = conf
            best_box = (x1, y1, x2, y2)
        return best_conf, best_box

    def write_confirmed_incident(
        frame: np.ndarray,
        tid: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        victim: Optional[tuple],
        nearest_vehicle: Optional[tuple],
        direction: str,
    ) -> None:
        nonlocal incident_count

        incident_count += 1
        incident_folder_name = f"snatching_{incident_count:03d}"
        incident_path = ensure_dir(paths.evidence_dir / incident_folder_name)

        full_frame_path = incident_path / "full_frame.jpg"
        offender_path = incident_path / "offender.jpg"
        cv2.imwrite(str(full_frame_path), frame)
        cv2.imwrite(str(offender_path), frame[y1:y2, x1:x2])

        victim_path = None
        if victim:
            vx1, vy1, vx2, vy2 = victim
            victim_path = incident_path / "victim.jpg"
            cv2.imwrite(str(victim_path), frame[vy1:vy2, vx1:vx2])

        vehicle_color, vehicle_path, vehicle_type = "N/A", None, "N/A"
        if nearest_vehicle:
            vehicle_type = "Motorcycle (YOLO Class 3)"
            mx1, my1, mx2, my2 = nearest_vehicle
            vehicle_crop = frame[my1:my2, mx1:mx2]
            vehicle_path = incident_path / "vehicle.jpg"
            cv2.imwrite(str(vehicle_path), vehicle_crop)
            vehicle_color = get_dominant_color(vehicle_crop)

        report_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base_output_dir = paths.out_dir.parent
        report_lines = [
            "--- INCIDENT REPORT ---",
            f"Timestamp: {report_ts}",
            "Event: CHAIN SNATCHING DETECTED",
            f"Location: {camera_id}",
            "",
            "VEHICLE PROFILE:",
            f"- Type: {vehicle_type}",
            f"- Color: {vehicle_color} (HSV Analysis)",
            f"- Suspect ID: {tid if tid >= 0 else 'Unknown'}",
            f"- Direction: Heading {direction} (Velocity Tracking)",
            "",
            "EVIDENCE:",
        ]

        evidence_map = {
            "Full Scene": full_frame_path,
            "Suspect Crop": offender_path,
            "Victim Crop": victim_path,
            "Vehicle Crop": vehicle_path,
        }
        for name, pth in evidence_map.items():
            if pth:
                report_lines.append(f"- {name}: {pth.relative_to(base_output_dir)}")

        report_file_path = incident_path / "incident_report.txt"
        report_file_path.write_text("\n".join(report_lines))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        frame_ann = {"frame_id": frame_id, "items": []}

        results = yolo.track(
            frame,
            persist=True,
            conf=0.4,
            classes=[0, 3],  # person, motorcycle
            verbose=False,
        )

        persons = []
        vehicles = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, tid, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                if int(cls) == 0:
                    persons.append((int(tid), (x1, y1, x2, y2)))
                elif int(cls) == 3:
                    vehicles.append((int(tid), (x1, y1, x2, y2)))

        pose_people = detect_pose_people(pose_model, frame, conf=pose_conf) if persons else []

        for tid, (x1, y1, x2, y2) in persons:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            track_history[tid].append((cx, cy))

            # Context gates reduce cross-labeling with fight-only scenes.
            victim_tid, victim, victim_dist_sq = None, None, 1e12
            for v_tid, (vx1, vy1, vx2, vy2) in persons:
                if v_tid == tid:
                    continue
                vcx, vcy = (vx1 + vx2) // 2, (vy1 + vy2) // 2
                d_sq = (vcx - cx) ** 2 + (vcy - cy) ** 2
                if d_sq < victim_dist_sq:
                    victim_dist_sq = d_sq
                    victim_tid = v_tid
                    victim = (vx1, vy1, vx2, vy2)
            has_nearby_victim = (
                victim is not None
                and victim_dist_sq <= (NEARBY_VICTIM_MAX_DIST_PX ** 2)
            )

            nearest_vehicle, vehicle_dist_sq = None, 1e12
            for _, (mx1, my1, mx2, my2) in vehicles:
                mcx, mcy = (mx1 + mx2) // 2, (my1 + my2) // 2
                d_sq = (mcx - cx) ** 2 + (mcy - cy) ** 2
                if d_sq < vehicle_dist_sq:
                    vehicle_dist_sq = d_sq
                    nearest_vehicle = (mx1, my1, mx2, my2)
            has_nearby_motorcycle = (
                nearest_vehicle is not None
                and vehicle_dist_sq <= (NEARBY_MOTORCYCLE_MAX_DIST_PX ** 2)
            )

            triggered = False
            if len(track_history[tid]) >= 3:
                pts = list(track_history[tid])
                (x0, y0), (x1p, y1p), (x2p, y2p) = pts[-3:]
                v1 = np.hypot(x1p - x0, y1p - y0)
                v2 = np.hypot(x2p - x1p, y2p - y1p)
                acc = abs(v2 - v1)
                if acc > ACC_THRESHOLD:
                    triggered = True

            suspect_pose = best_pose_for_box(pose_people, (x1, y1, x2, y2), min_iou=0.05)
            victim_pose = (
                best_pose_for_box(pose_people, victim, min_iou=0.05)
                if victim is not None
                else None
            )

            prev_suspect_wrist = wrist_history_by_tid.get(tid)
            suspect_aggr, suspect_wrist = pose_aggression_score(
                suspect_pose,
                prev_suspect_wrist,
                fps=fps,
                min_conf=pose_conf,
            )
            if suspect_wrist is not None:
                wrist_history_by_tid[tid] = suspect_wrist

            victim_aggr = 0.0
            if victim_tid is not None:
                prev_victim_wrist = wrist_history_by_tid.get(victim_tid)
                victim_aggr, victim_wrist = pose_aggression_score(
                    victim_pose,
                    prev_victim_wrist,
                    fps=fps,
                    min_conf=pose_conf,
                )
                if victim_wrist is not None:
                    wrist_history_by_tid[victim_tid] = victim_wrist

            snatch_pose, _ = snatching_pose_score(
                suspect_pose,
                victim,
                prev_suspect_wrist,
                fps=fps,
                min_conf=pose_conf,
            )
            fight_like_pose = (
                suspect_aggr >= FIGHT_LIKE_POSE_MIN_SCORE
                and victim_aggr >= FIGHT_LIKE_POSE_MIN_SCORE
            )
            pose_supports_snatching = snatch_pose >= SNATCH_POSE_MIN_SCORE and not fight_like_pose

            # Precision-first probing: only if victim context exists and either
            # movement looks abrupt or a motorcycle is nearby.
            has_victim_candidate = victim is not None
            should_probe_rf = has_victim_candidate and (triggered or has_nearby_motorcycle or has_nearby_victim)

            if should_probe_rf:
                last = rf_checked.get(tid, -999999)
                if frame_id - last >= RF_PROBE_COOLDOWN_FRAMES:
                    rf_checked[tid] = frame_id
                    if has_nearby_motorcycle:
                        crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                    else:
                        crop = frame

                    try:
                        preds = local_snatching_infer(crop, conf=RF_CONF)
                        best_conf = snatching_confidence(preds)
                        required_conf = (
                            SNATCH_CONF_WITH_MOTORCYCLE
                            if has_nearby_motorcycle
                            else SNATCH_CONF_WITHOUT_MOTORCYCLE
                        )
                        if suspect_pose is None:
                            pose_gate_pass = best_conf >= (required_conf + 0.12)
                        else:
                            pose_gate_pass = pose_supports_snatching or (
                                (not fight_like_pose) and best_conf >= (required_conf + 0.08)
                            )

                        strong_rf_snatch = best_conf >= (required_conf + 0.20)
                        if best_conf >= required_conf and (pose_gate_pass or strong_rf_snatch):
                            candidate_hits += 1
                            candidate_conf_samples.append(best_conf)
                            now_sec = frame_id / fps
                            hits = candidate_hit_times[tid]
                            hits.append((now_sec, best_conf))
                            while hits and (now_sec - hits[0][0]) > SNATCH_CONFIRM_WINDOW_SEC:
                                hits.popleft()

                            single_strong_hit = best_conf >= SNATCH_SINGLE_HIT_STRONG_CONF
                            single_frame_confirm = is_low_frame_input and len(hits) >= 1 and best_conf >= (required_conf + 0.05)
                            if len(hits) >= SNATCH_CONFIRM_MIN_HITS or single_strong_hit or single_frame_confirm:
                                confirm_conf = float(max(v for _, v in hits))
                                confirmed_conf_samples.append(confirm_conf)
                                if (now_sec - last_confirmed_incident_sec) >= CONFIRMED_INCIDENT_COOLDOWN_SEC:
                                    raw_confirmed_incidents += 1
                                    confirmed_suspect_ids.add(int(tid))
                                    if victim_tid is not None:
                                        confirmed_victim_ids.add(int(victim_tid))
                                    if not snatching_locked:
                                        write_confirmed_incident(
                                            frame=frame,
                                            tid=tid,
                                            x1=x1,
                                            y1=y1,
                                            x2=x2,
                                            y2=y2,
                                            victim=victim,
                                            nearest_vehicle=nearest_vehicle,
                                            direction=get_direction(track_history[tid]),
                                        )
                                        snatching_locked = True
                                        alert_state[tid] = ALERT_FRAMES
                                        alert_conf_by_tid[tid] = confirm_conf
                                    last_confirmed_incident_sec = now_sec
                                hits.clear()
                        else:
                            rejected_hits += 1

                    except Exception as e:
                        print("Snatching model infer error:", e)

            if tid in alert_state:
                alert_state[tid] -= 1
                if alert_state[tid] <= 0:
                    del alert_state[tid]
                    alert_conf_by_tid.pop(tid, None)

            if tid in alert_state:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    "SNATCHING",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                )
                frame_ann["items"].append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "label": "SNATCHING",
                        "conf": float(alert_conf_by_tid.get(tid, 1.0)),
                        "source": "snatching",
                    }
                )
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Full-frame RF fallback for low-light / weak tracking scenes.
        if frame_id - global_rf_checked >= GLOBAL_RF_PROBE_COOLDOWN_FRAMES:
            global_rf_checked = frame_id
            try:
                preds = local_snatching_infer(frame, conf=RF_CONF)
                best_conf, best_box = best_snatching_prediction(preds)
                if best_conf >= SNATCH_GLOBAL_RF_CONF and best_box is not None:
                    candidate_hits += 1
                    candidate_conf_samples.append(best_conf)
                    now_sec = frame_id / fps
                    global_candidate_hit_times.append((now_sec, best_conf, best_box))
                    while global_candidate_hit_times and (now_sec - global_candidate_hit_times[0][0]) > SNATCH_CONFIRM_WINDOW_SEC:
                        global_candidate_hit_times.popleft()

                    single_strong_hit = best_conf >= SNATCH_SINGLE_HIT_STRONG_CONF
                    single_frame_confirm = is_low_frame_input and best_conf >= SNATCH_GLOBAL_RF_CONF
                    if len(global_candidate_hit_times) >= SNATCH_CONFIRM_MIN_HITS or single_strong_hit or single_frame_confirm:
                        confirm_conf = float(max(v for _, v, _ in global_candidate_hit_times))
                        best_confirm_box = max(global_candidate_hit_times, key=lambda x: x[1])[2]
                        confirmed_conf_samples.append(confirm_conf)
                        if (now_sec - last_confirmed_incident_sec) >= CONFIRMED_INCIDENT_COOLDOWN_SEC:
                            raw_confirmed_incidents += 1
                            if not snatching_locked:
                                bx1, by1, bx2, by2 = best_confirm_box
                                write_confirmed_incident(
                                    frame=frame,
                                    tid=-1,
                                    x1=bx1,
                                    y1=by1,
                                    x2=bx2,
                                    y2=by2,
                                    victim=None,
                                    nearest_vehicle=None,
                                    direction="Unknown",
                                )
                                snatching_locked = True
                                global_alert_frames = ALERT_FRAMES
                                global_alert_conf = confirm_conf
                                global_alert_box = best_confirm_box
                            last_confirmed_incident_sec = now_sec
                        global_candidate_hit_times.clear()
            except Exception as e:
                print("Snatching global infer error:", e)

        if global_alert_frames > 0 and global_alert_box is not None:
            bx1, by1, bx2, by2 = global_alert_box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(
                frame,
                "SNATCHING",
                (bx1, max(0, by1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
            )
            frame_ann["items"].append(
                {
                    "x1": int(bx1),
                    "y1": int(by1),
                    "x2": int(bx2),
                    "y2": int(by2),
                    "label": "SNATCHING",
                    "conf": float(global_alert_conf),
                    "source": "snatching",
                }
            )
            global_alert_frames -= 1

        writer.write(frame)
        annotations_rows.append(frame_ann)

    cap.release()
    writer.release()

    write_jsonl(paths.annotations_jsonl, annotations_rows)

    incident_found = snatching_locked
    candidate_conf_mean = float(np.mean(candidate_conf_samples)) if candidate_conf_samples else 0.0
    confirmed_conf_mean = float(np.mean(confirmed_conf_samples)) if confirmed_conf_samples else 0.0
    support_bonus = min(0.2, 0.015 * candidate_hits)
    reject_penalty = min(0.25, 0.015 * rejected_hits)
    base_conf = confirmed_conf_mean if confirmed_conf_samples else candidate_conf_mean
    incident_confidence = min(
        1.0,
        max(
            0.0,
            base_conf + (0.2 if snatching_locked else 0.0) + support_bonus - reject_penalty,
        ),
    )
    return PipelineResult(
        name="snatching",
        incident_found=incident_found,
        incident_score=float(incident_count if incident_count > 0 else (1 if snatching_locked else 0)),
        annotated_video_path=str(paths.annotated_video),
        annotations_jsonl_path=str(paths.annotations_jsonl),
        evidence_dir=str(paths.evidence_dir) if incident_found else None,
        metadata={
            "incidents": incident_count,
            "confirmed_incidents": raw_confirmed_incidents,
            "candidate_hits": candidate_hits,
            "rejected_hits": rejected_hits,
            "snatching_locked": snatching_locked,
            "candidate_conf_mean": candidate_conf_mean,
            "confirmed_conf_mean": confirmed_conf_mean,
            "incident_confidence": float(incident_confidence),
            "confirmed_suspect_ids": sorted(confirmed_suspect_ids),
            "confirmed_victim_ids": sorted(confirmed_victim_ids),
        },
    )
