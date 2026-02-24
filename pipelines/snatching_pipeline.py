import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
from roboflow import Roboflow  # type: ignore
from typing import Optional

from .common import PipelinePaths, PipelineResult
from utils.io import ensure_dir, write_jsonl
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
    RF_COOLDOWN = 30

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing. Export it or put it in a .env and load it in main.py.")

    out_dir_path = ensure_dir(Path(out_dir))
    tmp_dir = ensure_dir(out_dir_path / "tmp")

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

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("sunfibo").project("chain_snatching-qlv3z")
    version = project.version(3)
    rf_model = version.model

    track_history = defaultdict(lambda: deque(maxlen=HISTORY))
    alert_state = {}
    rf_checked = {}
    frame_id = 0

    annotations_rows = []
    incident_count = 0

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

        if not results or results[0].boxes.id is None:
            writer.write(frame)
            annotations_rows.append(frame_ann)
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        persons = []
        vehicles = []

        for box, tid, cls in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)
            if int(cls) == 0:
                persons.append((int(tid), (x1, y1, x2, y2)))
            elif int(cls) == 3:
                vehicles.append((int(tid), (x1, y1, x2, y2)))

        for tid, (x1, y1, x2, y2) in persons:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            track_history[tid].append((cx, cy))

            triggered = False
            if len(track_history[tid]) >= 3:
                pts = list(track_history[tid])
                (x0, y0), (x1p, y1p), (x2p, y2p) = pts[-3:]
                v1 = np.hypot(x1p - x0, y1p - y0)
                v2 = np.hypot(x2p - x1p, y2p - y1p)
                acc = abs(v2 - v1)
                if acc > ACC_THRESHOLD:
                    triggered = True

            if triggered:
                last = rf_checked.get(tid, -999999)
                if frame_id - last > RF_COOLDOWN:
                    crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                    img_path = tmp_dir / f"check_{frame_id}_{tid}.jpg"
                    cv2.imwrite(str(img_path), crop)

                    try:
                        rf_result = rf_model.predict(str(img_path), confidence=RF_CONF).json()
                        preds = rf_result.get("predictions", [])

                        if any(p.get("confidence", 0) >= RF_CONF for p in preds):
                            incident_count += 1
                            incident_folder_name = f"snatching_{incident_count:03d}"
                            incident_path = ensure_dir(paths.evidence_dir / incident_folder_name)

                            # Save evidence images
                            full_frame_path = incident_path / "full_frame.jpg"
                            offender_path = incident_path / "offender.jpg"
                            cv2.imwrite(str(full_frame_path), frame)
                            cv2.imwrite(str(offender_path), frame[y1:y2, x1:x2])

                            # Find and save victim image
                            victim, best_d = None, 1e9
                            for v_tid, (vx1, vy1, vx2, vy2) in persons:
                                if v_tid == tid:
                                    continue
                                vcx, vcy = (vx1 + vx2) // 2, (vy1 + vy2) // 2
                                d = (vcx - cx) ** 2 + (vcy - cy) ** 2
                                if d < best_d:
                                    best_d, victim = d, (vx1, vy1, vx2, vy2)

                            victim_path = None
                            if victim:
                                vx1, vy1, vx2, vy2 = victim
                                victim_path = incident_path / "victim.jpg"
                                cv2.imwrite(str(victim_path), frame[vy1:vy2, vx1:vx2])

                            # Vehicle analysis
                            vehicle_color, vehicle_path, vehicle_type = "N/A", None, "N/A"
                            if vehicles:
                                vehicle_type = "Motorcycle (YOLO Class 3)"
                                mx1, my1, mx2, my2 = vehicles[0][1]
                                vehicle_crop = frame[my1:my2, mx1:mx2]
                                vehicle_path = incident_path / "vehicle.jpg"
                                cv2.imwrite(str(vehicle_path), vehicle_crop)
                                vehicle_color = get_dominant_color(vehicle_crop)

                            suspect_direction = get_direction(track_history[tid])

                            # Generate incident report
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
                                f"- Suspect ID: {tid}",
                                f"- Direction: Heading {suspect_direction} (Velocity Tracking)",
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

                            alert_state[tid] = ALERT_FRAMES

                    except Exception as e:
                        print("Roboflow error:", e)

                    rf_checked[tid] = frame_id

            if tid in alert_state:
                alert_state[tid] -= 1
                if alert_state[tid] <= 0:
                    del alert_state[tid]

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
                        "conf": 1.0,
                        "source": "snatching",
                    }
                )
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        writer.write(frame)
        annotations_rows.append(frame_ann)

    cap.release()
    writer.release()

    write_jsonl(paths.annotations_jsonl, annotations_rows)

    incident_found = incident_count > 0
    return PipelineResult(
        name="snatching",
        incident_found=incident_found,
        incident_score=float(incident_count),
        annotated_video_path=str(paths.annotated_video),
        annotations_jsonl_path=str(paths.annotations_jsonl),
        evidence_dir=str(paths.evidence_dir) if incident_found else None,
        metadata={"incidents": incident_count},
    )
