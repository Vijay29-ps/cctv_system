from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from pipelines.snatching_pipeline import run_snatching
from pipelines.fight_weapon_pipeline import run_fight_weapon
from merger.merge_annotator import merge_annotate_video
from utils.io import ensure_dir
from utils.fir import generate_fir


def process_video(video_path: str, camera_id: str = "Cam-01 (Default)") -> Dict[str, Any]:
    load_dotenv()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = ensure_dir(Path("outputs") / f"run_{run_id}")

    evidence_dir = ensure_dir(base / "evidence")
    sn_dir = ensure_dir(base / "snatching")
    fw_dir = ensure_dir(base / "fight_weapon")
    merged_dir = ensure_dir(base / "merged")
    incident_dir = ensure_dir(base / "incident")
    silent_dir = ensure_dir(base / "silent")

    sn_res = run_snatching(video_path, str(sn_dir), evidence_dir=str(evidence_dir), camera_id=camera_id)
    fw_res = run_fight_weapon(video_path, str(fw_dir))

    merged_out = merged_dir / "merged_annotated.mp4"
    merge_annotate_video(
        video_path=video_path,
        snatching_jsonl=sn_res.annotations_jsonl_path,
        fight_weapon_jsonl=fw_res.annotations_jsonl_path,
        out_path=str(merged_out),
    )

    incident_found = bool(sn_res.incident_found or fw_res.incident_found)
    final_path = (incident_dir / "incident_merged.mp4") if incident_found else (silent_dir / "silent_merged.mp4")
    final_path.write_bytes(merged_out.read_bytes())

    snatching_payload = {
        "incident_found": sn_res.incident_found,
        "score": sn_res.incident_score,
        "metadata": sn_res.metadata,
        "evidence_dir": sn_res.evidence_dir,
    }
    fight_weapon_payload = {
        "incident_found": fw_res.incident_found,
        "score": fw_res.incident_score,
        "metadata": fw_res.metadata,
        "evidence_dir": fw_res.evidence_dir,
        "events_csv_path": fw_res.events_csv_path,
    }

    try:
        fir_payload = generate_fir(
            run_id=run_id,
            camera_id=camera_id,
            base_output_dir=base,
            final_video_path=final_path,
            snatching=snatching_payload,
            fight_weapon=fight_weapon_payload,
            incident_found=incident_found,
        )
    except Exception as e:
        fir_payload = {
            "generated": False,
            "error": str(e),
            "path": None,
            "fir_number": None,
            "snapshot_paths": [],
        }

    return {
        "run_id": run_id,
        "camera_id": camera_id,
        "incident_found": incident_found,
        "local_final_output": str(final_path),
        "snatching": snatching_payload,
        "fight_weapon": fight_weapon_payload,
        "cdn": {
            "enabled": False,
            "response": None,
            "error": None,
        },
        "fir": fir_payload,
    }


def main(video_path: str, camera_id: str = "Cam-01 (Default)"):
    result = process_video(video_path, camera_id=camera_id)
    print(result)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python main.py <video_path> [camera_id]")
    camera_id = sys.argv[2] if len(sys.argv) > 2 else "Cam-01 (Default)"
    main(sys.argv[1], camera_id=camera_id)
