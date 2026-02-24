import cv2
import time
import math
import base64
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from .common import PipelinePaths, PipelineResult
from utils.config import get_config
from utils.io import ensure_dir, write_jsonl
from utils.video import get_video_info, make_writer


def _download_weapon_model_from_hf() -> str:
    """
    Hugging Face repo: https://huggingface.co/psv12/weapon
    File: "All_weapon .pt"  (NOTE: space before .pt)
    """
    config = get_config()
    return hf_hub_download(
        repo_id=config.hf_weapon_repo_id,
        filename=config.hf_weapon_filename,
        revision=config.hf_weapon_revision,
        token=config.hf_token,
    )


def run_fight_weapon(video_path: str, out_dir: str) -> PipelineResult:
    # Roboflow hosted fight model (Hosted API)
    ROBOFLOW_MODEL_ID = "fight-9uyg7/1"
    config = get_config(require_roboflow=True)
    api_key = config.roboflow_api_key

    WEAPON_CONF = 0.35
    FIGHT_CONF = 0.4
    EVENT_COOLDOWN_SEC = 5
    FIGHT_WINDOW_SEC = EVENT_COOLDOWN_SEC
    MAX_TRACK_DIST = 80
    RF_FRAME_SKIP = 5

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
    weapon_model_path = _download_weapon_model_from_hf()
    weapon_model = YOLO(weapon_model_path)

    # ---- Roboflow fight detection via Hosted API (base64 body) ----
    def roboflow_infer(frame):
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            return {}

        img_b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")

        url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"
        resp = requests.post(
            url,
            params={"api_key": api_key, "confidence": str(FIGHT_CONF)},
            data=img_b64,  # IMPORTANT: base64 string body
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )

        if resp.status_code != 200:
            # show exact roboflow error response
            raise RuntimeError(f"Roboflow {resp.status_code}: {resp.text}")

        return resp.json()

    # ---- simple tracking ----
    tracks = {}
    next_id = 1

    def centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def assign_ids(boxes):
        nonlocal tracks, next_id
        new_tracks = {}
        for box in boxes:
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
            else:
                tid = best_id
            new_tracks[tid] = {"box": box, "centroid": c}
        tracks = new_tracks
        return new_tracks

    records = []
    annotations_rows = []
    frame_id = 0
    last_fight_time = 0.0
    active_fighter_ids = set()
    active_window_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        now = time.time()

        frame_ann = {"frame_id": frame_id, "items": []}
        fight_boxes = []

        # ---- Fight detection every N frames ----
        if frame_id % RF_FRAME_SKIP == 0:
            try:
                rf_result = roboflow_infer(frame)
                for p in rf_result.get("predictions", []):
                    # Roboflow returns already thresholded by confidence param,
                    # but we keep this as an extra guard
                    if p.get("confidence", 0) >= FIGHT_CONF:
                        x1 = int(p["x"] - p["width"] / 2)
                        y1 = int(p["y"] - p["height"] / 2)
                        x2 = int(p["x"] + p["width"] / 2)
                        y2 = int(p["y"] + p["height"] / 2)
                        fight_boxes.append((x1, y1, x2, y2))
            except Exception as e:
                print("Roboflow fight infer error:", e)

        tracked = assign_ids(fight_boxes)

        # Draw + annotate fight boxes
        for tid, data in tracked.items():
            x1, y1, x2, y2 = data["box"]
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
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": f"FIGHT {tid}", "conf": 1.0, "source": "fight_weapon"}
            )

        # Track fighters in the window
        if fight_boxes:
            if active_window_start is None:
                active_window_start = now
            for tid in tracked.keys():
                active_fighter_ids.add(tid)

        # Snapshot logging logic
        should_log = False
        if fight_boxes and (now - last_fight_time) > EVENT_COOLDOWN_SEC:
            should_log = True
        if (not fight_boxes) and active_window_start is not None and (now - active_window_start) > FIGHT_WINDOW_SEC:
            should_log = True

        if should_log and active_window_start is not None and len(active_fighter_ids) > 0:
            snap_name = f"fight_frame{frame_id}_{int(now)}.jpg"
            snap_path = snapshots_dir / snap_name
            cv2.imwrite(str(snap_path), frame)

            records.append(
                {
                    "event": "fight",
                    "frame_id": frame_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "num_fighters": len(active_fighter_ids),
                    "fighter_ids": ",".join(sorted(active_fighter_ids)),
                    "snapshot_path": str(snap_path),
                }
            )

            last_fight_time = now
            active_fighter_ids.clear()
            active_window_start = None

        # ---- Weapon detection (HF YOLO weights) ----
        weapon_results = weapon_model(frame, conf=WEAPON_CONF, verbose=False)[0]
        for b in weapon_results.boxes:
            label = weapon_model.names[int(b.cls[0])]
            if str(label).lower() == "person":
                continue

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
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": f"WEAPON:{label}", "conf": 1.0, "source": "fight_weapon"}
            )

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

    incident_found = len(records) > 0
    return PipelineResult(
        name="fight_weapon",
        incident_found=incident_found,
        incident_score=float(len(records)),
        annotated_video_path=str(paths.annotated_video),
        annotations_jsonl_path=str(paths.annotations_jsonl),
        evidence_dir=str(paths.snapshots_dir) if incident_found else None,
        events_csv_path=str(paths.events_csv) if incident_found else None,
        metadata={"fight_events": len(records)},
    )
