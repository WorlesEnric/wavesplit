from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AppConfig
from .models import AudioInfo, ClipRecord, dataclass_to_dict


MANIFEST_FIELDS = [
    "clip_id",
    "line_index",
    "original_text",
    "normalized_text",
    "output_file",
    "start_sec",
    "end_sec",
    "duration_sec",
    "duplicate_index",
    "alignment_score_mean",
    "confidence",
    "status",
    "asr_text",
    "asr_normalized_text",
    "similarity",
    "wer",
    "leading_silence_ms",
    "trailing_silence_ms",
    "flags",
]

QA_FIELDS = MANIFEST_FIELDS + ["peak_dbfs", "rms_dbfs"]


def _row(record: ClipRecord, fields: list[str]) -> dict[str, Any]:
    data = asdict(record)
    if isinstance(data.get("flags"), list):
        data["flags"] = "|".join(data["flags"])
    return {field: data.get(field) for field in fields}


def write_csv(path: str | Path, records: list[ClipRecord], fields: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(_row(record, fields))


def write_json(path: str | Path, payload: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def report_payload(job_id: str, records: list[ClipRecord], audio_info: AudioInfo, config: AppConfig) -> dict[str, object]:
    counts = Counter(record.status for record in records)
    return {
        "job_id": job_id,
        "summary": {
            "total": len(records),
            "pass": counts.get("pass", 0),
            "review": counts.get("review", 0),
            "fail": counts.get("fail", 0),
            "missing_audio": counts.get("missing_audio", 0),
            "duration_sec": audio_info.duration_sec,
            "alignment_engine": config.alignment.engine,
            "asr_engine": config.qa.asr_engine if config.qa.enabled else "disabled",
            "asr_model": config.qa.asr_model if config.qa.enabled else None,
        },
        "clips": [dataclass_to_dict(record) for record in records],
    }


def write_readme(
    path: str | Path,
    *,
    audio_filename: str,
    transcript_filename: str,
    config: AppConfig,
    counts: Counter,
) -> None:
    content = f"""WaveSplit output

Input audio: {audio_filename}
Input transcript: {transcript_filename}
Alignment engine: {config.alignment.engine}
Alignment model: {config.alignment.model}
Pre padding: {config.alignment.pre_padding_ms} ms
Post padding: {config.alignment.post_padding_ms} ms
ASR QA: {config.qa.asr_engine if config.qa.enabled else 'disabled'}
ASR model: {config.qa.asr_model if config.qa.enabled else 'n/a'}

QA status counts:
pass: {counts.get('pass', 0)}
review: {counts.get('review', 0)}
fail: {counts.get('fail', 0)}
missing_audio: {counts.get('missing_audio', 0)}

Status meanings:
pass   - no automated warning was found.
review - one or more boundary, duration, ASR, or diagnostic flags should be checked by a person.
fail   - the clip is highly suspicious and should be corrected or regenerated.
missing_audio - the transcript row has no detected matching speech segment; timing and file fields are null.
"""
    Path(path).write_text(content, encoding="utf-8")


def package_outputs(
    *,
    job_id: str,
    paths: Any,
    records: list[ClipRecord],
    audio_info: AudioInfo,
    config: AppConfig,
    audio_filename: str,
    transcript_filename: str,
) -> dict[str, object]:
    output_dir = Path(paths.output_dir)
    qa_dir = Path(paths.qa_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "manifest.csv", records, MANIFEST_FIELDS)
    write_json(output_dir / "manifest.json", [dataclass_to_dict(record) for record in records])
    write_csv(qa_dir / "qa_report.csv", records, QA_FIELDS)
    payload = report_payload(job_id, records, audio_info, config)
    write_json(qa_dir / "qa_report.json", payload)
    write_json(qa_dir / "asr_results.json", [dataclass_to_dict(record) for record in records])
    counts = Counter(record.status for record in records)
    write_readme(
        output_dir / "README.txt",
        audio_filename=audio_filename,
        transcript_filename=transcript_filename,
        config=config,
        counts=counts,
    )

    zip_path = output_dir / "clips.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in records:
            if not record.output_file:
                continue
            clip_path = Path(paths.clips_dir) / record.output_file
            archive.write(clip_path, f"clips/{record.output_file}")
        archive.write(output_dir / "manifest.csv", "manifest.csv")
        archive.write(output_dir / "manifest.json", "manifest.json")
        archive.write(qa_dir / "qa_report.csv", "qa_report.csv")
        archive.write(qa_dir / "qa_report.json", "qa_report.json")
        archive.write(output_dir / "README.txt", "README.txt")
    return payload


def build_diagnostics_zip(paths: Any) -> Path:
    output_path = Path(paths.output_dir) / "diagnostics.zip"
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in ["status.json", "alignment/raw_alignment.json", "alignment/line_alignment.json"]:
            path = Path(paths.job_dir) / relative
            if path.exists():
                archive.write(path, relative)
        logs_dir = Path(paths.logs_dir)
        if logs_dir.exists():
            for path in logs_dir.glob("*"):
                if path.is_file():
                    archive.write(path, f"logs/{path.name}")
    return output_path
