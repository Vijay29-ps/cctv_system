from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Iterable
import os
import cv2

from dotenv import load_dotenv
load_dotenv()

from pipelines.common import PipelineResult
from pipelines.snatching_pipeline import run_snatching
from pipelines.fight_weapon_pipeline import run_fight_weapon
from merger.merge_annotator import merge_annotate_video
from utils.io import ensure_dir
from utils.main_decider import decide_incidents
from utils.output_bundle import build_official_outputs
from utils.video import make_writer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _prune_run_outputs(base_dir: Path, keep_paths: Iterable[Path]) -> None:
    base_resolved = base_dir.resolve()

    keep = {base_resolved}
    for raw in keep_paths:
        p = raw if raw.is_absolute() else (Path.cwd() / raw)
        rp = p.resolve()
        keep.add(rp)

        cur = rp.parent
        while True:
            keep.add(cur)
            if cur == base_resolved:
                break
            if base_resolved not in cur.parents:
                break
            cur = cur.parent

    for p in sorted(base_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        rp = p.resolve()
        if rp in keep:
            continue
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
            continue
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass


def _pipeline_mode() -> str:
    mode = (os.getenv("PIPELINE_MODE") or "all").strip().lower()
    valid = {"all", "snatching_only", "fight_weapon_only"}
    return mode if mode in valid else "all"


def _prepare_input_video(input_path: str, base_dir: Path) -> str:
    src = Path(input_path)
    ext = src.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        return input_path

    if not src.exists():
        raise RuntimeError(f"Input image not found: {input_path}")

    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"Cannot read input image: {input_path}")

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise RuntimeError(f"Invalid image shape for input: {input_path}")

    input_dir = ensure_dir(base_dir / "input")
    out_video = input_dir / "input_from_image.mp4"
    writer = make_writer(str(out_video), w, h, 3.0)
    for _ in range(3):
        writer.write(img)
    writer.release()
    return str(out_video)


def _empty_snatching_result(sn_dir: Path) -> PipelineResult:
    sn_jsonl = sn_dir / "snatching.jsonl"
    sn_jsonl.write_text("", encoding="utf-8")
    return PipelineResult(
        name="snatching",
        incident_found=False,
        incident_score=0.0,
        annotated_video_path=str(sn_dir / "snatching_annotated.mp4"),
        annotations_jsonl_path=str(sn_jsonl),
        evidence_dir=None,
        events_csv_path=None,
        metadata={
            "confirmed_incidents": 0,
            "candidate_hits": 0,
            "rejected_hits": 0,
            "snatching_locked": False,
        },
    )


def _empty_fight_weapon_result(fw_dir: Path) -> PipelineResult:
    fw_jsonl = fw_dir / "fight_weapon.jsonl"
    fw_jsonl.write_text("", encoding="utf-8")
    return PipelineResult(
        name="fight_weapon",
        incident_found=False,
        incident_score=0.0,
        annotated_video_path=str(fw_dir / "fight_weapon_annotated.mp4"),
        annotations_jsonl_path=str(fw_jsonl),
        evidence_dir=None,
        events_csv_path=None,
        metadata={
            "fight_events": 0,
            "weapon_frames": 0,
            "pose_verified_fight_frames": 0,
            "pose_rejected_fight_boxes": 0,
        },
    )


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
    prepared_video_path = _prepare_input_video(video_path, base)

    mode = _pipeline_mode()
    if mode == "fight_weapon_only":
        sn_res = _empty_snatching_result(sn_dir)
        fw_res = run_fight_weapon(prepared_video_path, str(fw_dir))
    elif mode == "snatching_only":
        sn_res = run_snatching(prepared_video_path, str(sn_dir), evidence_dir=str(evidence_dir), camera_id=camera_id)
        fw_res = _empty_fight_weapon_result(fw_dir)
    else:
        sn_res = run_snatching(prepared_video_path, str(sn_dir), evidence_dir=str(evidence_dir), camera_id=camera_id)
        fw_res = run_fight_weapon(prepared_video_path, str(fw_dir))

    merged_out = merged_dir / "merged_annotated.mp4"
    merge_annotate_video(
        video_path=prepared_video_path,
        snatching_jsonl=sn_res.annotations_jsonl_path,
        fight_weapon_jsonl=fw_res.annotations_jsonl_path,
        out_path=str(merged_out),
    )

    decision = decide_incidents(sn_res, fw_res)
    incident_found = decision.incident_found
    final_path = (incident_dir / "incident_merged.mp4") if incident_found else (silent_dir / "silent_merged.mp4")
    final_path.write_bytes(merged_out.read_bytes())

    snatching_payload = {
        "incident_found": decision.snatching_found,
        "score": (
            float((sn_res.metadata or {}).get("incident_confidence", sn_res.incident_score))
            if decision.snatching_found
            else 0.0
        ),
        "metadata": {
            **(sn_res.metadata or {}),
            "decision_reason": decision.reason,
            "raw_score": sn_res.incident_score,
        },
        "evidence_dir": sn_res.evidence_dir,
    }
    fight_weapon_payload = {
        "incident_found": bool(decision.fight_found or decision.weapon_found),
        "score": (
            float((fw_res.metadata or {}).get("incident_confidence", fw_res.incident_score))
            if bool(decision.fight_found or decision.weapon_found)
            else 0.0
        ),
        "metadata": {
            **(fw_res.metadata or {}),
            "decision_fight_found": bool(decision.fight_found),
            "decision_weapon_found": bool(decision.weapon_found),
        },
        "evidence_dir": fw_res.evidence_dir,
        "events_csv_path": fw_res.events_csv_path,
    }

    outputs_payload = build_official_outputs(
        run_id=run_id,
        camera_id=camera_id,
        base_output_dir=str(base),
        final_video_path=str(final_path),
        incident_found=incident_found,
        incident_label=("INCIDENT" if incident_found else "NO_INCIDENT"),
        snatching=snatching_payload,
        fight_weapon=fight_weapon_payload,
        decider={
            "reason": decision.reason,
            "predicted_class": decision.decider_class,
            "confidence": decision.decider_confidence,
            "incident_confidence": decision.incident_confidence,
            "snatching_confidence": decision.snatching_confidence,
            "fight_weapon_confidence": decision.fight_weapon_confidence,
            "probabilities": decision.decider_probabilities,
            "features": decision.decider_features,
            "pipeline_mode": mode,
        },
    )

    keep_paths = [
        Path(outputs_payload["video_path"]),
        Path(outputs_payload["json_path"]),
        Path(outputs_payload["csv_path"]),
        Path(outputs_payload["fir_path"]),
        Path(outputs_payload["snapshot_path"]),
    ]
    events_csv = outputs_payload.get("csv_paths", {}).get("events_csv")
    if events_csv:
        keep_paths.append(Path(events_csv))
    _prune_run_outputs(base, keep_paths)

    return {
        "run_id": run_id,
        "camera_id": camera_id,
        "incident_found": incident_found,
        "outputs": outputs_payload,
    }


def main(video_path: str, camera_id: str = "Cam-01 (Default)"):
    result = process_video(video_path, camera_id=camera_id)
    print(result)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python main.py <input_path(video_or_image)> [camera_id]")
    camera_id = sys.argv[2] if len(sys.argv) > 2 else "Cam-01 (Default)"
    main(sys.argv[1], camera_id=camera_id)
