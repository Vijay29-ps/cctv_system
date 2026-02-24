import os
import subprocess
from pathlib import Path
from typing import Dict, Optional


def _parse_headers(headers_raw: str) -> Dict[str, str]:
    """
    CDN_HEADERS format (in .env):
      CDN_HEADERS="Authorization: Bearer XXX|x-tenant-id: ABC|x-api-key: KKK"
    """
    headers: Dict[str, str] = {}
    if not headers_raw:
        return headers

    parts = [p.strip() for p in headers_raw.split("|") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        headers[k.strip()] = v.strip()
    return headers


def upload_video_to_cdn(
    file_path: str,
    cdn_url: Optional[str] = None,
    remote_folder: str = "CMS",
    form_field: str = "file",
    timeout_sec: int = 600,
) -> str:
    """
    Upload a video to Mobius content service using curl (multipart form).
    Returns stdout (often JSON).

    Required .env:
      CDN_URL="https://ig.gov-cloud.ai/mobius-content-service/v1.0/content/upload?filePath=CMS"
      CDN_HEADERS="Header1: Value1|Header2: Value2"

    Notes:
    - If your API expects a different form field, change form_field (default: "file").
    - If URL contains filePath=CMS, we replace it with filePath=<remote_folder>.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    cdn_url = cdn_url or os.getenv(
        "CDN_URL",
        "https://ig.gov-cloud.ai/mobius-content-service/v1.0/content/upload?filePath=CMS",
    )

    if "filePath=CMS" in cdn_url and remote_folder:
        cdn_url = cdn_url.replace("filePath=CMS", f"filePath={remote_folder}")

    headers = _parse_headers(os.getenv("CDN_HEADERS", ""))

    cmd = [
        "curl",
        "--location",
        cdn_url,
        "--fail",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout_sec),
    ]

    for k, v in headers.items():
        cmd += ["--header", f"{k}: {v}"]

    cmd += ["--form", f"{form_field}=@{str(p)}"]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"CDN upload failed (exit={res.returncode}).\n"
            f"STDERR:\n{res.stderr}\n"
            f"STDOUT:\n{res.stdout}"
        )

    return res.stdout.strip()
