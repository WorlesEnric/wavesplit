from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import STAGE_ORDER, STAGE_WEIGHTS


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def compute_progress(stage: str, stage_fraction: float = 0.0) -> float:
    progress = 0.0
    stage_fraction = max(0.0, min(1.0, stage_fraction))
    for item in STAGE_ORDER:
        if item == stage:
            progress += STAGE_WEIGHTS[item] * stage_fraction
            break
        progress += STAGE_WEIGHTS[item]
    return round(min(progress, 1.0), 4)


class JobStatusStore:
    def __init__(self, status_path: str | Path):
        self.path = Path(status_path)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def initialize(self, job_id: str, audio_filename: str, text_filename: str) -> dict[str, Any]:
        now = utc_now_iso()
        data = {
            "job_id": job_id,
            "state": "queued",
            "stage": "upload_saved",
            "progress": 0.0,
            "message": "Upload saved",
            "created_at": now,
            "updated_at": now,
            "input": {
                "audio_filename": audio_filename,
                "text_filename": text_filename,
                "line_count": None,
                "audio_duration_sec": None,
            },
            "counts": {
                "total": 0,
                "aligned": 0,
                "cut": 0,
                "qa_pass": 0,
                "qa_review": 0,
                "qa_fail": 0,
            },
            "error": None,
        }
        self.write(data)
        return data

    def update(self, **changes: Any) -> dict[str, Any]:
        data = self.read()
        for key, value in changes.items():
            if key == "input" and isinstance(value, dict):
                data.setdefault("input", {}).update(value)
            elif key == "counts" and isinstance(value, dict):
                data.setdefault("counts", {}).update(value)
            else:
                data[key] = value
        data["updated_at"] = utc_now_iso()
        self.write(data)
        return data

    def stage(
        self,
        stage: str,
        message: str,
        *,
        stage_fraction: float = 0.0,
        counts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.update(
            state="running",
            stage=stage,
            progress=compute_progress(stage, stage_fraction),
            message=message,
            counts=counts or {},
        )

    def fail(self, stage: str, message: str) -> dict[str, Any]:
        return self.update(
            state="failed",
            stage=stage,
            progress=compute_progress(stage, 1.0),
            message=message,
            error=message,
        )

    def succeed(self, message: str = "Done") -> dict[str, Any]:
        return self.update(
            state="succeeded",
            stage="done",
            progress=1.0,
            message=message,
            error=None,
        )
