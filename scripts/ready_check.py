from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv


def _check_imports(modules: List[str]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            errors.append(f"import {mod}: {exc}")
    return len(errors) == 0, errors


def _check_file(path: Path, label: str) -> Tuple[bool, str]:
    if path.exists():
        return True, f"{label}: OK ({path})"
    return False, f"{label}: MISSING ({path})"


def main() -> int:
    load_dotenv()
    repo_root = Path(__file__).resolve().parents[1]
    mode = (os.getenv("PIPELINE_MODE", "all") or "all").strip().lower()

    required_imports = [
        "cv2",
        "numpy",
        "pandas",
        "ultralytics",
        "fastapi",
        "uvicorn",
        "huggingface_hub",
    ]
    ok_imports, import_errors = _check_imports(required_imports)

    checks: List[Tuple[bool, str]] = []

    decider_model = Path(os.getenv("DECIDER_MODEL_PATH", "models/incident_decider_v1.json"))
    if not decider_model.is_absolute():
        decider_model = repo_root / decider_model
    checks.append(_check_file(decider_model, "Decider model"))

    phase1 = os.getenv("DECIDER_PHASE1_INCIDENT_THRESHOLD", "0.55")
    try:
        phase1_v = float(phase1)
        phase1_ok = 0.05 <= phase1_v <= 0.95
    except ValueError:
        phase1_ok = False
    checks.append((phase1_ok, f"DECIDER_PHASE1_INCIDENT_THRESHOLD={phase1}"))

    checks.append((mode in {"all", "snatching_only", "fight_weapon_only"}, f"PIPELINE_MODE={mode}"))

    snatching_model = Path(os.getenv("SNATCHING_MODEL_PATH", r"C:\dataset\best_model.pt"))
    if not snatching_model.is_absolute():
        snatching_model = repo_root / snatching_model
    if mode in {"all", "snatching_only"}:
        if snatching_model.exists():
            checks.append(_check_file(snatching_model, "Snatching model (.pt)"))
        else:
            hf_sn_repo = (os.getenv("HF_SNATCHING_REPO_ID") or "psv12/weapon").strip()
            hf_sn_file = (os.getenv("HF_SNATCHING_FILENAME") or "best_model.pt").strip()
            checks.append(
                (
                    bool(hf_sn_repo and hf_sn_file),
                    f"Snatching model via HF fallback ({hf_sn_repo}/{hf_sn_file})",
                )
            )

    fight_model = Path(os.getenv("FIGHT_MODEL_PATH", r"C:\dataset\fight_detection_best.pt"))
    if not fight_model.is_absolute():
        fight_model = repo_root / fight_model
    if mode in {"all", "fight_weapon_only"}:
        checks.append(_check_file(fight_model, "Fight model (.pt)"))
        hf_weapon_repo = (os.getenv("HF_WEAPON_REPO_ID") or "psv12/weapon").strip()
        hf_weapon_file = (os.getenv("HF_WEAPON_FILENAME") or "All_weapon .pt").strip()
        checks.append(
            (
                bool(hf_weapon_repo and hf_weapon_file),
                f"Weapon model via HF ({hf_weapon_repo}/{hf_weapon_file})",
            )
        )

    env_file = repo_root / ".env"
    checks.append((env_file.exists(), f".env file exists ({env_file})"))

    all_checks_ok = ok_imports and all(ok for ok, _ in checks)

    print("=== CCTV READY CHECK ===")
    print(f"repo: {repo_root}")
    print("")
    print("Imports:")
    if ok_imports:
        print("  OK")
    else:
        for err in import_errors:
            print(f"  FAIL: {err}")

    print("")
    print("Config/Files:")
    for ok, msg in checks:
        prefix = "OK" if ok else "FAIL"
        print(f"  {prefix}: {msg}")

    print("")
    print("Run commands (PowerShell):")
    print("  python main.py <input_path(video_or_image)> \"Cam-01\"")
    print("  python server.py")
    print("")

    report = {
        "ready": all_checks_ok,
        "imports_ok": ok_imports,
        "import_errors": import_errors,
        "checks": [{"ok": ok, "message": msg} for ok, msg in checks],
    }
    report_path = repo_root / "outputs" / "ready_check_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {report_path}")

    return 0 if all_checks_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
