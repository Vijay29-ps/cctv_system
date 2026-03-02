from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .fir import generate_fir
from .io import ensure_dir


def _incident_types(snatching_found: bool, fight_found: bool, weapon_found: bool) -> List[str]:
    out: List[str] = []
    if snatching_found:
        out.append("Snatching")
    if fight_found:
        out.append("Fight")
    if weapon_found:
        out.append("Weapon")
    return out


def _existing_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.exists() else None


def _read_fight_snapshots_from_csv(events_csv_path: Optional[Path]) -> List[Path]:
    if not events_csv_path or not events_csv_path.exists():
        return []

    out: List[Path] = []
    with events_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            snap_raw = (row.get("snapshot_path") or "").strip()
            if not snap_raw:
                continue
            snap = Path(snap_raw)
            if snap.exists():
                out.append(snap)
    return out


def _collect_snatching_candidates(evidence_dir: Optional[Path]) -> List[Path]:
    if not evidence_dir or not evidence_dir.exists():
        return []

    out: List[Path] = []
    preferred = ["full_frame.jpg", "offender.jpg", "victim.jpg", "vehicle.jpg"]
    for incident_dir in sorted(p for p in evidence_dir.iterdir() if p.is_dir()):
        for name in preferred:
            p = incident_dir / name
            if p.exists():
                out.append(p)
    return out


def _collect_fight_candidates(evidence_dir: Optional[Path], events_csv_path: Optional[Path]) -> List[Path]:
    from_csv = _read_fight_snapshots_from_csv(events_csv_path)
    if from_csv:
        return from_csv

    if not evidence_dir or not evidence_dir.exists():
        return []
    return sorted(evidence_dir.glob("*.jpg"))


def _pick_first_by_name(paths: List[Path], name: str) -> Optional[Path]:
    for p in paths:
        if p.name.lower() == name.lower():
            return p
    return None


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            out.append(p)
    return out


def _select_collage_images(snatching_paths: List[Path], fight_paths: List[Path]) -> List[Path]:
    selected: List[Path] = []
    key_snapping = [
        _pick_first_by_name(snatching_paths, "full_frame.jpg"),
        _pick_first_by_name(snatching_paths, "offender.jpg"),
        _pick_first_by_name(snatching_paths, "victim.jpg"),
    ]
    for p in key_snapping:
        if p is not None:
            selected.append(p)

    if fight_paths:
        selected.append(fight_paths[0])

    for p in snatching_paths + fight_paths:
        if len(selected) >= 4:
            break
        selected.append(p)

    return _dedupe_paths(selected)[:4]


def _label_for_image(path: Path) -> str:
    lower = path.name.lower()
    if lower == "full_frame.jpg":
        return "SNATCHING FULL FRAME"
    if lower == "offender.jpg":
        return "SNATCHING OFFENDER"
    if lower == "victim.jpg":
        return "SNATCHING VICTIM"
    if lower == "vehicle.jpg":
        return "SNATCHING VEHICLE"
    return f"FIGHT SNAPSHOT {path.stem}"


def _draw_label(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        label[:60],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _extract_first_frame(video_path: Path, out_path: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open final video for snapshot fallback: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame for snapshot fallback: {video_path}")

    cv2.imwrite(str(out_path), frame)
    return out_path


def build_evidence_collage(
    snatching_evidence_dir: Optional[str],
    fight_evidence_dir: Optional[str],
    events_csv_path: Optional[str],
    final_video_path: str,
    out_path: str,
) -> str:
    out = Path(out_path)
    ensure_dir(out.parent)

    snatching_paths = _collect_snatching_candidates(_existing_path(snatching_evidence_dir))
    fight_paths = _collect_fight_candidates(
        _existing_path(fight_evidence_dir),
        _existing_path(events_csv_path),
    )
    selected = _select_collage_images(snatching_paths, fight_paths)

    valid_images: List[np.ndarray] = []
    labels: List[str] = []
    for p in selected:
        img = cv2.imread(str(p))
        if img is None:
            continue
        valid_images.append(img)
        labels.append(_label_for_image(p))

    if not valid_images:
        return str(_extract_first_frame(Path(final_video_path), out))

    if len(valid_images) == 1:
        single = _draw_label(valid_images[0], labels[0])
        cv2.imwrite(str(out), single)
        return str(out)

    tile_w, tile_h = 640, 360
    canvas = np.zeros((tile_h * 2, tile_w * 2, 3), dtype=np.uint8)

    for idx, (img, label) in enumerate(zip(valid_images[:4], labels[:4])):
        resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        labeled = _draw_label(resized, label)
        r = idx // 2
        c = idx % 2
        y1, y2 = r * tile_h, (r + 1) * tile_h
        x1, x2 = c * tile_w, (c + 1) * tile_w
        canvas[y1:y2, x1:x2] = labeled

    cv2.imwrite(str(out), canvas)
    return str(out)


def write_summary_csv(
    out_path: str,
    run_id: str,
    camera_id: str,
    incident_found: bool,
    incident_label: str,
    incident_types: List[str],
    snatching_score: Any,
    fight_weapon_score: Any,
    final_video_path: str,
    snapshot_path: str,
    fir_path: str,
    events_csv_path: Optional[str],
) -> str:
    out = Path(out_path)
    ensure_dir(out.parent)

    row = {
        "run_id": run_id,
        "camera_id": camera_id,
        "incident_found": incident_found,
        "incident_label": incident_label,
        "incident_types": ", ".join(incident_types) if incident_types else "None",
        "snatching_score": snatching_score,
        "fight_weapon_score": fight_weapon_score,
        "final_video_path": final_video_path,
        "snapshot_path": snapshot_path,
        "fir_path": fir_path,
        "events_csv_path": events_csv_path or "",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    fields = list(row.keys())

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    return str(out)


def write_summary_json(
    out_path: str,
    run_id: str,
    camera_id: str,
    incident_found: bool,
    incident_label: str,
    incident_types: List[str],
    snatching_score: Any,
    fight_weapon_score: Any,
    final_video_path: str,
    snapshot_path: str,
    fir_path: str,
    summary_csv_path: str,
    decider: Optional[Dict[str, Any]] = None,
) -> str:
    out = Path(out_path)
    ensure_dir(out.parent)

    payload = {
        "run_id": run_id,
        "camera_id": camera_id,
        "incident_found": incident_found,
        "incident_label": incident_label,
        "incident_types": incident_types,
        "snatching_score": snatching_score,
        "fight_weapon_score": fight_weapon_score,
        "video_path": final_video_path,
        "snapshot_path": snapshot_path,
        "fir_path": fir_path,
        "summary_csv_path": summary_csv_path,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if decider:
        payload["decider"] = decider
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def build_official_outputs(
    run_id: str,
    camera_id: str,
    base_output_dir: str,
    final_video_path: str,
    incident_found: bool,
    incident_label: Optional[str],
    snatching: Dict[str, Any],
    fight_weapon: Dict[str, Any],
    decider: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = Path(base_output_dir)
    official_dir = ensure_dir(base / "official")

    events_csv_candidate = (
        _existing_path(fight_weapon.get("events_csv_path"))
        or _existing_path(str(base / "fight_weapon" / "events.csv"))
    )
    events_csv_str = str(events_csv_candidate) if events_csv_candidate else None

    snapshot_path = str(official_dir / "evidence_snapshot.jpg")
    snapshot_path = build_evidence_collage(
        snatching_evidence_dir=snatching.get("evidence_dir"),
        fight_evidence_dir=fight_weapon.get("evidence_dir"),
        events_csv_path=events_csv_str,
        final_video_path=final_video_path,
        out_path=snapshot_path,
    )

    sn_found = bool(snatching.get("incident_found", False))
    fw_meta = fight_weapon.get("metadata") or {}
    if "decision_fight_found" in fw_meta or "decision_weapon_found" in fw_meta:
        fight_found = bool(fw_meta.get("decision_fight_found", False))
        weapon_found = bool(fw_meta.get("decision_weapon_found", False))
    else:
        fight_found = bool((fw_meta.get("fight_events") or 0) > 0)
        weapon_found = bool((fw_meta.get("weapon_frames") or 0) > 0)
        # Backward-compatible fallback if older metadata is missing.
        if not (fight_found or weapon_found):
            fight_found = bool(fight_weapon.get("incident_found", False))
    incident_types = _incident_types(sn_found, fight_found, weapon_found)

    fir_path = str(official_dir / f"AUTO-FIR-{run_id}.txt")
    generate_fir(
        run_id=run_id,
        camera_id=camera_id,
        incident_found=incident_found,
        incident_types=incident_types,
        snatching_score=snatching.get("score"),
        fight_weapon_score=fight_weapon.get("score"),
        final_video_path=Path(final_video_path),
        snapshot_path=Path(snapshot_path),
        events_csv_path=Path(events_csv_str) if events_csv_str else None,
        summary_csv_path=Path(official_dir / "summary.csv"),
        out_path=Path(fir_path),
    )

    summary_csv_path = write_summary_csv(
        out_path=str(official_dir / "summary.csv"),
        run_id=run_id,
        camera_id=camera_id,
        incident_found=incident_found,
        incident_label=incident_label or ("INCIDENT" if incident_found else "NO_INCIDENT"),
        incident_types=incident_types,
        snatching_score=snatching.get("score"),
        fight_weapon_score=fight_weapon.get("score"),
        final_video_path=final_video_path,
        snapshot_path=snapshot_path,
        fir_path=fir_path,
        events_csv_path=events_csv_str,
    )

    summary_json_path = write_summary_json(
        out_path=str(official_dir / "result.json"),
        run_id=run_id,
        camera_id=camera_id,
        incident_found=incident_found,
        incident_label=incident_label or ("INCIDENT" if incident_found else "NO_INCIDENT"),
        incident_types=incident_types,
        snatching_score=snatching.get("score"),
        fight_weapon_score=fight_weapon.get("score"),
        final_video_path=final_video_path,
        snapshot_path=snapshot_path,
        fir_path=fir_path,
        summary_csv_path=summary_csv_path,
        decider=decider,
    )

    return {
        "video_path": final_video_path,
        "json_path": summary_json_path,
        "csv_path": summary_csv_path,
        "snapshot_path": snapshot_path,
        "fir_path": fir_path,
        "csv_paths": {
            "events_csv": events_csv_str,
            "summary_csv": summary_csv_path,
        },
    }
