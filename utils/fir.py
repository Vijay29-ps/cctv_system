from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import ensure_dir


def _brief_case_facts(incident_found: bool, incident_types: List[str]) -> str:
    if not incident_found:
        return "No cognizable incident detected in this run."

    lower_types = [t.lower() for t in incident_types]
    has_snatching = any("snatch" in t for t in lower_types)
    has_fight_weapon = any(("fight" in t) or ("weapon" in t) for t in lower_types)

    if has_snatching and has_fight_weapon:
        return (
            "CCTV analytics indicates chain snatching is happening, and a fight/weapon "
            "confrontation is also taking place. This is an auto-generated draft and "
            "requires officer verification."
        )
    if has_snatching:
        return (
            "CCTV analytics indicates chain snatching activity. No fight/weapon activity "
            "was flagged in this run. This is an auto-generated draft and requires officer "
            "verification."
        )
    if has_fight_weapon:
        return (
            "CCTV analytics indicates fight/weapon activity. No snatching activity was "
            "flagged in this run. This is an auto-generated draft and requires officer "
            "verification."
        )

    return (
        "CCTV analytics detected suspicious activity. This is an auto-generated draft and "
        "requires officer verification."
    )


def generate_fir(
    run_id: str,
    camera_id: str,
    incident_found: bool,
    incident_types: List[str],
    snatching_score: Any,
    fight_weapon_score: Any,
    final_video_path: Path,
    snapshot_path: Optional[Path],
    events_csv_path: Optional[Path],
    summary_csv_path: Optional[Path],
    out_path: Path,
) -> Dict[str, Any]:
    ensure_dir(out_path.parent)

    fir_number = f"AUTO-FIR-{run_id}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    incident_type_line = ", ".join(incident_types) if incident_types else "None"

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
    ]

    lines.append(f"    {_brief_case_facts(incident_found, incident_types)}")

    lines.extend(
        [
            "12. Digital Evidence Attached:",
            f"    - Final Output Video: {final_video_path}",
            f"    - Evidence Snapshot: {snapshot_path if snapshot_path else 'Not available'}",
            f"    - Events CSV: {events_csv_path if events_csv_path else 'Not available'}",
            f"    - Summary CSV: {summary_csv_path if summary_csv_path else 'Not available'}",
            "13. Analytics Summary:",
            f"    - Snatching Score: {snatching_score}",
            f"    - Fight/Weapon Score: {fight_weapon_score}",
            f"14. System Run ID: {run_id}",
            "15. Action Taken: Immediate review required by control room/police desk.",
            "",
            "Disclaimer: This is an auto-generated FIR draft for operational use.",
            "Legal sections and final FIR registration details must be confirmed by police.",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "generated": True,
        "path": str(out_path),
        "fir_number": fir_number,
    }
