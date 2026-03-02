import os
from huggingface_hub import hf_hub_download
from utils.config import AppConfig

def _download_hf_weights(
    repo_id: str,
    filename: str,
    revision: str,
    token: str | None,
) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )


def get_weapon_weights_path(config: AppConfig | None = None) -> str:
    """
    Downloads weapon weights from Hugging Face and returns local cached file path.

    Defaults are set for:
      repo: psv12/weapon
      file: "All_weapon .pt"   (NOTE the space before .pt)

    Override via env vars:
      HF_WEAPON_REPO_ID
      HF_WEAPON_FILENAME
      HF_WEAPON_REVISION
      HF_TOKEN (only if HF repo is private)
    """
    if config is None:
        repo_id = os.getenv("HF_WEAPON_REPO_ID", "psv12/weapon")
        filename = os.getenv("HF_WEAPON_FILENAME", "All_weapon .pt")
        revision = os.getenv("HF_WEAPON_REVISION", "main")
        token = os.getenv("HF_TOKEN")  # optional
    else:
        repo_id = config.hf_weapon_repo_id
        filename = config.hf_weapon_filename
        revision = config.hf_weapon_revision
        token = config.hf_token

    return _download_hf_weights(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )


def get_snatching_weights_path(config: AppConfig | None = None) -> str:
    """
    Downloads snatching weights from Hugging Face and returns local cached file path.

    Defaults are set for:
      repo: psv12/weapon
      file: "best_model.pt"

    Override via env vars:
      HF_SNATCHING_REPO_ID
      HF_SNATCHING_FILENAME
      HF_SNATCHING_REVISION
      HF_TOKEN (only if HF repo is private)
    """
    if config is None:
        repo_id = os.getenv("HF_SNATCHING_REPO_ID", "psv12/weapon")
        filename = os.getenv("HF_SNATCHING_FILENAME", "best_model.pt")
        revision = os.getenv("HF_SNATCHING_REVISION", "main")
        token = os.getenv("HF_TOKEN")  # optional
    else:
        repo_id = config.hf_snatching_repo_id
        filename = config.hf_snatching_filename
        revision = config.hf_snatching_revision
        token = config.hf_token

    return _download_hf_weights(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )
