import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    roboflow_api_key: str
    hf_token: Optional[str]
    hf_weapon_repo_id: str
    hf_weapon_filename: str
    hf_weapon_revision: str


def get_config(require_roboflow: bool = False) -> AppConfig:
    # Load local .env for development; production should use real env vars.
    load_dotenv()

    config = AppConfig(
        roboflow_api_key=os.getenv("ROBOFLOW_API_KEY", "").strip(),
        hf_token=(os.getenv("HF_TOKEN") or "").strip() or None,
        hf_weapon_repo_id=os.getenv("HF_WEAPON_REPO_ID", "psv12/weapon"),
        hf_weapon_filename=os.getenv("HF_WEAPON_FILENAME", "All_weapon .pt"),
        hf_weapon_revision=os.getenv("HF_WEAPON_REVISION", "main"),
    )

    if require_roboflow and not config.roboflow_api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing. Configure it in environment or local .env")

    return config
