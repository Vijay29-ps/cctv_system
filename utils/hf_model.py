import os
from huggingface_hub import hf_hub_download

def get_weapon_weights_path() -> str:
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
    repo_id = os.getenv("HF_WEAPON_REPO_ID", "psv12/weapon")
    filename = os.getenv("HF_WEAPON_FILENAME", "All_weapon .pt")
    revision = os.getenv("HF_WEAPON_REVISION", "main")
    token = os.getenv("HF_TOKEN")  # optional

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )
