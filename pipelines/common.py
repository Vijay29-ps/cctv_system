from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class PipelinePaths:
    out_dir: Path
    annotated_video: Path
    annotations_jsonl: Path
    evidence_dir: Path
    snapshots_dir: Path
    events_csv: Optional[Path] = None


@dataclass
class PipelineResult:
    name: str
    incident_found: bool
    incident_score: float
    annotated_video_path: str
    annotations_jsonl_path: str
    evidence_dir: Optional[str] = None
    events_csv_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
