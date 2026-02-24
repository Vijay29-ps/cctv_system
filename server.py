from pathlib import Path
import re
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from main import process_video
from utils.io import ensure_dir

app = FastAPI(title="CCTV Processing API")

UPLOAD_DIR = Path("uploads")
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def safe_filename(filename: str) -> str:
    base = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


@app.post("/process-video")
async def process_video_api(
    file: Optional[UploadFile] = File(None),
    camera_id: str = Form("Cam-01 (Default)"),
) -> Any:
    if file is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Upload file using form-data key 'file'"},
        )

    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "Empty filename"})

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return JSONResponse(status_code=400, content={"error": f"Unsupported file type {ext}"})

    upload_path = ensure_dir(UPLOAD_DIR)
    input_path = upload_path / safe_filename(file.filename)

    data = await file.read()
    input_path.write_bytes(data)

    try:
        result = process_video(str(input_path), camera_id=camera_id)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
