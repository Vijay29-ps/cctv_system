import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    roboflow_api_url: str
    roboflow_api_key: str
    roboflow_fight_model_id: str
    snatching_model_path: str
    fight_model_path: str
    hf_token: Optional[str]
    hf_weapon_repo_id: str
    hf_weapon_filename: str
    hf_weapon_revision: str
    hf_snatching_repo_id: str
    hf_snatching_filename: str
    hf_snatching_revision: str
    snatching_min_confirmed_events_for_output: int
    pose_model_name: str
    pose_confidence: float


def get_config(require_roboflow: bool = False) -> AppConfig:
    # Load local .env for development; production should use real env vars.
    load_dotenv()
    raw_min_confirmed = (os.getenv("SNATCHING_MIN_CONFIRMED_EVENTS_FOR_OUTPUT") or "2").strip()
    raw_pose_conf = (os.getenv("POSE_CONFIDENCE") or "0.25").strip()
    try:
        min_confirmed = max(1, int(raw_min_confirmed))
    except ValueError:
        min_confirmed = 2
    try:
        pose_conf = min(1.0, max(0.05, float(raw_pose_conf)))
    except ValueError:
        pose_conf = 0.25

    config = AppConfig(
        roboflow_api_url=(os.getenv("ROBOFLOW_API_URL") or "https://serverless.roboflow.com").strip(),
        roboflow_api_key=os.getenv("ROBOFLOW_API_KEY", "").strip(),
        roboflow_fight_model_id=(os.getenv("ROBOFLOW_FIGHT_MODEL_ID") or "violence-weapon-detection/1").strip(),
        snatching_model_path=(os.getenv("SNATCHING_MODEL_PATH") or r"C:\dataset\best_model.pt").strip(),
        fight_model_path=(os.getenv("FIGHT_MODEL_PATH") or r"C:\dataset\fight_detection_best.pt").strip(),
        hf_token=(os.getenv("HF_TOKEN") or "").strip() or None,
        hf_weapon_repo_id=os.getenv("HF_WEAPON_REPO_ID", "psv12/weapon"),
        hf_weapon_filename=os.getenv("HF_WEAPON_FILENAME", "All_weapon .pt"),
        hf_weapon_revision=os.getenv("HF_WEAPON_REVISION", "main"),
        hf_snatching_repo_id=os.getenv("HF_SNATCHING_REPO_ID", "psv12/weapon"),
        hf_snatching_filename=os.getenv("HF_SNATCHING_FILENAME", "best_model.pt"),
        hf_snatching_revision=os.getenv("HF_SNATCHING_REVISION", "main"),
        snatching_min_confirmed_events_for_output=min_confirmed,
        pose_model_name=os.getenv("POSE_MODEL_NAME", "yolov8n-pose.pt").strip() or "yolov8n-pose.pt",
        pose_confidence=pose_conf,
    )

    if require_roboflow and not config.roboflow_api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing. Configure it in environment or local .env")

    return config
