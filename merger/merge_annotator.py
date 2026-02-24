import cv2
from pathlib import Path
from typing import Dict, List

from utils.io import read_jsonl
from utils.video import get_video_info, make_writer

def _load_by_frame(jsonl_path: str) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for row in read_jsonl(Path(jsonl_path)):
        fid = int(row["frame_id"])
        out[fid] = row.get("items", [])
    return out

def merge_annotate_video(video_path: str, snatching_jsonl: str, fight_weapon_jsonl: str, out_path: str) -> str:
    sn = _load_by_frame(snatching_jsonl)
    fw = _load_by_frame(fight_weapon_jsonl)

    info = get_video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    writer = make_writer(out_path, info.w, info.h, info.fps)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        items = []
        items.extend(sn.get(frame_id, []))
        items.extend(fw.get(frame_id, []))

        for it in items:
            x1, y1, x2, y2 = it["x1"], it["y1"], it["x2"], it["y2"]
            label = it.get("label", "EVENT")
            source = it.get("source", "")

            if source == "snatching":
                color = (0, 0, 255)
            elif "WEAPON" in label:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    return out_path
