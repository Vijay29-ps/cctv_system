from pathlib import Path
import re
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from main import process_video
from utils.io import ensure_dir

app = FastAPI(title="CCTV Processing API")

UPLOAD_DIR = Path("uploads")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ALLOWED_EXT = VIDEO_EXT | IMAGE_EXT


def safe_filename(filename: str) -> str:
    base = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def _validate_extension(path_like: str) -> Optional[str]:
    ext = Path(path_like).suffix.lower()
    if ext not in ALLOWED_EXT:
        return f"Unsupported file type {ext}. Allowed: {sorted(ALLOWED_EXT)}"
    return None


@app.post("/process-video")
async def process_video_api(
    file: Optional[UploadFile] = File(None),
    input_path: Optional[str] = Form(None),
    camera_id: str = Form("Cam-01 (Default)"),
) -> Any:
    local_input_path: Optional[Path] = None

    if file is not None:
        if not file.filename:
            return JSONResponse(status_code=400, content={"error": "Empty filename"})
        ext_error = _validate_extension(file.filename)
        if ext_error:
            return JSONResponse(status_code=400, content={"error": ext_error})
        upload_path = ensure_dir(UPLOAD_DIR)
        local_input_path = upload_path / safe_filename(file.filename)
        data = await file.read()
        local_input_path.write_bytes(data)
    elif input_path:
        ext_error = _validate_extension(input_path)
        if ext_error:
            return JSONResponse(status_code=400, content={"error": ext_error})
        local_input_path = Path(input_path)
        if not local_input_path.exists():
            return JSONResponse(status_code=400, content={"error": f"Input path not found: {input_path}"})
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide either form-data 'file' upload or 'input_path'"},
        )

    try:
        result = process_video(str(local_input_path), camera_id=camera_id)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
