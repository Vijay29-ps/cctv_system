import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.io import read_jsonl
from utils.video import get_video_info, make_writer


def _load_by_frame(jsonl_path: str) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for row in read_jsonl(Path(jsonl_path)):
        fid = int(row["frame_id"])
        out[fid] = row.get("items", [])
    return out


def _banner_text(items: List[dict]) -> str:
    has_snatching = False
    has_fight = False
    has_weapon = False
    for it in items:
        label = str(it.get("label", "")).upper()
        source = str(it.get("source", "")).lower()
        if source == "snatching" or "SNATCH" in label:
            has_snatching = True
        if "WEAPON" in label:
            has_weapon = True
        if "FIGHT" in label:
            has_fight = True

    if has_snatching and has_fight and has_weapon:
        return "SNATCHING, FIGHT, AND WEAPON ACTIVITY DETECTED"
    if has_snatching and has_fight:
        return "SNATCHING HAPPENING; FIGHT ALSO TAKING PLACE"
    if has_snatching and has_weapon:
        return "SNATCHING WITH WEAPON ACTIVITY HAPPENING"
    if has_snatching:
        return "SNATCHING HAPPENING"
    if has_fight and has_weapon:
        return "FIGHT AND WEAPON ACTIVITY HAPPENING"
    if has_fight:
        return "FIGHT ACTIVITY HAPPENING"
    if has_weapon:
        return "WEAPON ACTIVITY HAPPENING"
    return ""


def _first_event_frames(
    sn_by_frame: Dict[int, List[dict]],
    fw_by_frame: Dict[int, List[dict]],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    first_snatch: Optional[int] = None
    first_fight: Optional[int] = None
    first_weapon: Optional[int] = None

    for fid in sorted(sn_by_frame.keys()):
        for it in sn_by_frame.get(fid, []):
            label = str(it.get("label", "")).upper()
            if "SNATCH" in label:
                first_snatch = fid
                break
        if first_snatch is not None:
            break

    for fid in sorted(fw_by_frame.keys()):
        for it in fw_by_frame.get(fid, []):
            label = str(it.get("label", "")).upper()
            if "FIGHT" in label and first_fight is None:
                first_fight = fid
            if "WEAPON" in label and first_weapon is None:
                first_weapon = fid
        if first_fight is not None and first_weapon is not None:
            break

    return first_snatch, first_fight, first_weapon


def _banner_text_with_order(
    items: List[dict],
    first_snatch: Optional[int],
    first_fight: Optional[int],
) -> str:
    base = _banner_text(items)
    if not base:
        return ""

    has_snatching = any(
        (str(it.get("source", "")).lower() == "snatching")
        or ("SNATCH" in str(it.get("label", "")).upper())
        for it in items
    )
    has_fight = any("FIGHT" in str(it.get("label", "")).upper() for it in items)

    if has_snatching and has_fight and first_snatch is not None and first_fight is not None:
        if first_snatch < first_fight:
            return "SNATCHING FIRST, FIGHT FOLLOWING"
        if first_fight < first_snatch:
            return "FIGHT FIRST, SNATCHING FOLLOWING"
    return base


def merge_annotate_video(video_path: str, snatching_jsonl: str, fight_weapon_jsonl: str, out_path: str) -> str:
    sn = _load_by_frame(snatching_jsonl)
    fw = _load_by_frame(fight_weapon_jsonl)
    first_snatch, first_fight, _ = _first_event_frames(sn, fw)

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

        # Draw fight/weapon first and snatching last so snatching stays visually dominant.
        items_sorted = sorted(
            items,
            key=lambda it: 1
            if (
                str(it.get("source", "")).lower() == "snatching"
                or "SNATCH" in str(it.get("label", "")).upper()
            )
            else 0,
        )
        for it in items_sorted:
            x1, y1, x2, y2 = it["x1"], it["y1"], it["x2"], it["y2"]
            label = it.get("label", "EVENT")
            source = it.get("source", "")

            label_upper = str(label).upper()
            if source == "snatching" or "SNATCH" in label_upper:
                color = (0, 0, 255)
                thickness = 4
            elif "WEAPON" in label_upper:
                color = (255, 0, 0)
                thickness = 2
            elif "FIGHT" in label_upper:
                color = (0, 165, 255)
                thickness = 2
            else:
                color = (0, 255, 255)
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness)

        banner = _banner_text_with_order(items, first_snatch, first_fight)
        if banner:
            cv2.rectangle(frame, (0, 0), (info.w, 36), (0, 0, 0), -1)
            cv2.putText(
                frame,
                banner,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )

        writer.write(frame)

    cap.release()
    writer.release()
    return out_path
