from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


JOB_STATES = {"queued", "running", "succeeded", "failed", "cancelled"}

STAGE_WEIGHTS: dict[str, float] = {
    "validating": 0.05,
    "normalizing_text": 0.05,
    "aligning": 0.35,
    "building_segments": 0.05,
    "cutting": 0.20,
    "qa_asr": 0.14,
    "qa_scoring": 0.06,
    "packaging": 0.10,
}

STAGE_ORDER = list(STAGE_WEIGHTS)


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {k: dataclass_to_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [dataclass_to_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: dataclass_to_dict(v) for k, v in value.items()}
    return value


@dataclass
class AudioInfo:
    codec_name: str
    sample_rate: int
    channels: int
    duration_sec: float
    format_name: str = "wav"


@dataclass
class LineRecord:
    line_index: int
    original_text: str
    normalized_text: str
    tokens: list[str]
    token_start_index: int
    token_end_index: int


@dataclass
class LineAlignment:
    clip_id: str
    line_index: int
    original_text: str
    normalized_text: str
    raw_start_sec: float | None
    raw_end_sec: float | None
    start_sec: float | None
    end_sec: float | None
    alignment_score_mean: float | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        if self.start_sec is None or self.end_sec is None:
            return 0.0
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class ClipRecord:
    clip_id: str
    line_index: int
    original_text: str
    normalized_text: str
    output_file: str | None
    duplicate_index: int
    start_sec: float | None
    end_sec: float | None
    duration_sec: float | None
    alignment_score_mean: float | None
    confidence: int
    status: str
    asr_text: str
    asr_normalized_text: str
    similarity: float | None
    wer: float | None
    leading_silence_ms: int | None
    trailing_silence_ms: int | None
    peak_dbfs: float | None
    rms_dbfs: float | None
    flags: list[str] = field(default_factory=list)


@dataclass
class JobPaths:
    job_dir: str
    input_dir: str
    normalized_dir: str
    alignment_dir: str
    clips_dir: str
    qa_dir: str
    logs_dir: str
    output_dir: str
    status_path: str
    original_audio_path: str
    transcript_path: str
