# cctv_system

## Quick Start

1. Install dependencies:
```powershell
pip install -r requirements.txt
```

2. Configure `.env` from `.env.example`.

3. Run preflight:
```powershell
python scripts/ready_check.py
```

4. Test pipeline:
```powershell
python main.py "<video_path>" "Cam-01"
```

Detailed test/run instructions: [docs/READY_TO_TEST.md](/c:/dataset/cctv_system-main/docs/READY_TO_TEST.md)
