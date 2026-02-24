from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import ensure_dir


def _incident_types(snatching_found: bool, fight_weapon_found: bool) -> List[str]:
    out: List[str] = []
    if snatching_found:
        out.append("Snatching")
    if fight_weapon_found:
        out.append("Fight/Weapon")
    return out


def _read_fight_snapshots_from_csv(events_csv_path: Optional[str]) -> List[str]:
    if not events_csv_path:
        return []

    p = Path(events_csv_path)
    if not p.exists():
        return []

    out: List[str] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            snap = (row.get("snapshot_path") or "").strip()
            if snap and Path(snap).exists():
                out.append(snap)
    return out


def _collect_snatching_snapshots(evidence_dir: Optional[str]) -> List[str]:
    if not evidence_dir:
        return []

    base = Path(evidence_dir)
    if not base.exists():
        return []

    out: List[str] = []
    preferred = ["full_frame.jpg", "offender.jpg", "victim.jpg", "vehicle.jpg"]
    for folder in sorted([p for p in base.iterdir() if p.is_dir()]):
        for name in preferred:
            fp = folder / name
            if fp.exists():
                out.append(str(fp))
    return out


def _collect_fight_snapshots(evidence_dir: Optional[str], events_csv_path: Optional[str]) -> List[str]:
    from_csv = _read_fight_snapshots_from_csv(events_csv_path)
    if from_csv:
        return from_csv

    if not evidence_dir:
        return []
    base = Path(evidence_dir)
    if not base.exists():
        return []
    return [str(p) for p in sorted(base.glob("*.jpg"))]


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def generate_fir(
    run_id: str,
    camera_id: str,
    base_output_dir: Path,
    final_video_path: Path,
    snatching: Dict[str, Any],
    fight_weapon: Dict[str, Any],
    incident_found: bool,
) -> Dict[str, Any]:
    if not incident_found:
        return {
            "generated": False,
            "reason": "No incident detected",
            "path": None,
            "fir_number": None,
            "snapshot_paths": [],
        }

    fir_dir = ensure_dir(base_output_dir / "fir")
    fir_number = f"AUTO-FIR-{run_id}"
    fir_path = fir_dir / f"{fir_number}.txt"

    sn_found = bool(snatching.get("incident_found", False))
    fw_found = bool(fight_weapon.get("incident_found", False))
    incident_types = _incident_types(sn_found, fw_found)

    sn_shots = _collect_snatching_snapshots(snatching.get("evidence_dir"))
    fw_shots = _collect_fight_snapshots(
        fight_weapon.get("evidence_dir"),
        fight_weapon.get("events_csv_path"),
    )
    snapshot_paths = _dedupe(sn_shots + fw_shots)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    incident_type_line = ", ".join(incident_types) if incident_types else "Unknown"

    lines = [
        "FIRST INFORMATION REPORT (AUTO-GENERATED CCTV DRAFT)",
        "------------------------------------------------------------------",
        f"1. FIR Number: {fir_number}",
        f"2. Date & Time of Registration: {now}",
        "3. Police Station: To be filled by Duty Officer",
        "4. District: To be filled by Duty Officer",
        "5. Type of Information: Electronic CCTV Alert",
        f"6. Date & Time of Occurrence: {now}",
        f"7. Place of Occurrence (Camera/Location): {camera_id}",
        "8. Complainant/Informant: Automated CCTV Monitoring System",
        "9. Suspect/Accused: Unknown (identity to be verified from footage)",
        f"10. Nature of Offence: {incident_type_line}",
        "11. Brief Facts of the Case:",
        (
            "    CCTV analytics detected suspicious activity consistent with "
            f"{incident_type_line}. This draft is system-generated and must be "
            "verified by investigating officer before formal registration."
        ),
        "12. Digital Evidence Attached:",
        f"    - Final Output Video: {final_video_path}",
    ]

    if snapshot_paths:
        lines.append("    - Snapshot Evidence:")
        for idx, path in enumerate(snapshot_paths, start=1):
            lines.append(f"      {idx}. {path}")
    else:
        lines.append("    - Snapshot Evidence: None found")

    lines.extend(
        [
            "13. Analytics Summary:",
            f"    - Snatching Detected: {sn_found} (score={snatching.get('score')})",
            f"    - Fight/Weapon Detected: {fw_found} (score={fight_weapon.get('score')})",
            f"14. System Run ID: {run_id}",
            "15. Action Taken: Immediate review required by control room/police desk.",
            "",
            "Disclaimer: This is an auto-generated FIR draft for operational use.",
            "Legal sections and final FIR registration details must be confirmed by police.",
        ]
    )

    fir_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "generated": True,
        "path": str(fir_path),
        "fir_number": fir_number,
        "incident_types": incident_types,
        "snapshot_paths": snapshot_paths,
        "final_video_path": str(final_video_path),
    }
