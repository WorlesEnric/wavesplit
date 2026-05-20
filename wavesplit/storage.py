from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .config import AppConfig
from .models import JobPaths

JOB_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9]{4,12}$")


def make_job_id(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    return f"{current:%Y%m%d-%H%M%S}-{uuid4().hex[:4]}"


def validate_job_id(job_id: str) -> str:
    if not JOB_ID_RE.match(job_id):
        raise ValueError("Invalid job id.")
    return job_id


def job_dir_for(config: AppConfig, job_id: str) -> Path:
    return Path(config.storage_dir) / "jobs" / validate_job_id(job_id)


def paths_for_job(job_dir: str | Path) -> JobPaths:
    root = Path(job_dir)
    return JobPaths(
        job_dir=str(root),
        input_dir=str(root / "input"),
        normalized_dir=str(root / "normalized"),
        alignment_dir=str(root / "alignment"),
        clips_dir=str(root / "clips"),
        qa_dir=str(root / "qa"),
        logs_dir=str(root / "logs"),
        output_dir=str(root / "output"),
        status_path=str(root / "status.json"),
        original_audio_path=str(root / "input" / "original.wav"),
        transcript_path=str(root / "input" / "transcript.txt"),
    )


def create_job_layout(job_dir: str | Path) -> JobPaths:
    paths = paths_for_job(job_dir)
    for directory in [
        paths.input_dir,
        paths.normalized_dir,
        paths.alignment_dir,
        paths.clips_dir,
        paths.qa_dir,
        paths.logs_dir,
        paths.output_dir,
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    return paths


def copy_inputs(audio_path: str | Path, transcript_path: str | Path, paths: JobPaths) -> None:
    shutil.copy2(audio_path, paths.original_audio_path)
    shutil.copy2(transcript_path, paths.transcript_path)
