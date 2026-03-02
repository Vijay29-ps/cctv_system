# Ready To Test (Windows PowerShell)

## 1) Install dependencies
```powershell
pip install -r requirements.txt
```

## 2) Configure `.env`
Create/update `.env` in repo root:

```env
SNATCHING_MODEL_PATH=C:\dataset\best_model.pt
SNATCHING_CLASS_IDS=0
SNATCHING_CLASS_NAMES=snatching,chain_snatching,snatch,robbery,theft,steal
FIGHT_MODEL_PATH=C:\dataset\fight_detection_best.pt
FIGHT_CLASS_IDS=0
FIGHT_CLASS_NAMES=fight,violence,violent,assault,attack,aggression
PIPELINE_MODE=all
DECIDER_MODEL_PATH=models/incident_decider_v1.json
DECIDER_MIN_CONFIDENCE=0.45
DECIDER_OUTPUT_MODE=dominant
DECIDER_SNATCHING_PRIORITY=true
DECIDER_PHASE1_INCIDENT_THRESHOLD=0.55
POSE_MODEL_NAME=yolov8n-pose.pt
POSE_CONFIDENCE=0.25
```

## 3) Run preflight check
```powershell
python scripts/ready_check.py
```

Expected: `ready: true` in `outputs/ready_check_report.json`.

## 4) Local video test (direct pipeline)
```powershell
python main.py "C:\dataset\cctv_system-main\outputs\run_20260226_123549\incident\incident_merged.mp4" "Cam-04"
```

Output folder:
- `outputs/run_YYYYmmdd_HHMMSS/official/result.json`
- `outputs/run_YYYYmmdd_HHMMSS/official/summary.csv`
- `outputs/run_YYYYmmdd_HHMMSS/official/AUTO-FIR-*.txt`
- `outputs/run_YYYYmmdd_HHMMSS/incident/incident_merged.mp4` (or `silent/silent_merged.mp4`)

## 5) API test (`server.py`)
Start server:
```powershell
python server.py
```

Send request from another terminal:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/process-video" `
  -F "camera_id=Cam-04" `
  -F "file=@C:\dataset\cctv_system-main\outputs\run_20260226_123549\incident\incident_merged.mp4"
```

## 6) Validate decision fields
Check `official/result.json` includes:
- `incident_label` (`INCIDENT` / `NO_INCIDENT`)
- `incident_types`
- `decider.predicted_class`
- `decider.confidence`
- `decider.incident_confidence`
- `decider.reason`

## 7) Quick mode checks
Snatching only:
```powershell
$env:PIPELINE_MODE = "snatching_only"
python main.py "<video_path>" "Cam-04"
```

Fight + weapon only:
```powershell
$env:PIPELINE_MODE = "fight_weapon_only"
python main.py "<video_path>" "Cam-04"
```

All enabled:
```powershell
$env:PIPELINE_MODE = "all"
python main.py "<video_path>" "Cam-04"
```
