from __future__ import annotations

import os
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AlignmentConfig:
    engine: str = "auto"
    model: str = "storage/models/ctc_forced_aligner.onnx"
    language: str = "en"
    ctc_language: str = "eng"
    ctc_batch_size: int = 2
    ctc_device: str = "cpu"
    pre_padding_ms: int = 80
    post_padding_ms: int = 120
    vad_frame_ms: int = 10
    min_active_ms: int = 80
    min_silence_ms: int = 350
    max_silence_ms: int = 1200


@dataclass
class QAConfig:
    enabled: bool = True
    asr_engine: str = "faster_whisper"
    asr_model: str = "tiny.en"
    device: str = "cpu"
    compute_type: str = "int8"
    similarity_pass: float = 92.0
    similarity_review: float = 80.0
    wer_pass: float = 0.25
    wer_review: float = 0.50
    leading_silence_review_ms: int = 800
    trailing_silence_review_ms: int = 1000
    aliases: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class WorkerConfig:
    queue_name: str = "wavesplit"
    concurrency: int = 1
    redis_url: str = "redis://localhost:6379/0"


@dataclass
class AuthUserConfig:
    username: str = "admin"
    password: str = "change-me-now"


@dataclass
class AuthConfig:
    enabled: bool = True
    session_secret: str = "change-this-session-secret-before-cloudflare"
    session_cookie: str = "wavesplit_session"
    session_max_age_sec: int = 86400
    cookie_secure: bool = False
    users: list[AuthUserConfig] = field(default_factory=lambda: [AuthUserConfig()])


@dataclass
class AppConfig:
    storage_dir: str = "./storage"
    max_upload_mb: int = 1024
    job_retention_days: int = 7
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)


def _apply_mapping(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_mapping(current, value)
        elif isinstance(current, list) and current and is_dataclass(current[0]) and isinstance(value, list):
            item_type = type(current[0])
            converted = []
            for item in value:
                if isinstance(item, dict):
                    next_item = item_type()
                    _apply_mapping(next_item, item)
                    converted.append(next_item)
                else:
                    converted.append(item)
            setattr(obj, key, converted)
        else:
            setattr(obj, key, value)


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    candidate = Path(path) if path else Path("config.yaml")
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, dict):
            _apply_mapping(config, data)

    env_map = {
        "WAVESPLIT_STORAGE_DIR": ("storage_dir", str),
        "WAVESPLIT_MAX_UPLOAD_MB": ("max_upload_mb", int),
        "WAVESPLIT_ALIGNMENT_ENGINE": ("alignment.engine", str),
        "WAVESPLIT_CTC_DEVICE": ("alignment.ctc_device", str),
        "WAVESPLIT_PRE_PADDING_MS": ("alignment.pre_padding_ms", int),
        "WAVESPLIT_POST_PADDING_MS": ("alignment.post_padding_ms", int),
        "WAVESPLIT_QA_ENABLED": ("qa.enabled", _env_bool),
        "WAVESPLIT_ASR_ENGINE": ("qa.asr_engine", str),
        "WAVESPLIT_ASR_MODEL": ("qa.asr_model", str),
        "WAVESPLIT_ASR_DEVICE": ("qa.device", str),
        "WAVESPLIT_ASR_COMPUTE_TYPE": ("qa.compute_type", str),
        "WAVESPLIT_REDIS_URL": ("worker.redis_url", str),
        "WAVESPLIT_AUTH_ENABLED": ("auth.enabled", _env_bool),
        "WAVESPLIT_AUTH_SESSION_SECRET": ("auth.session_secret", str),
        "WAVESPLIT_AUTH_COOKIE_SECURE": ("auth.cookie_secure", _env_bool),
    }
    for env_name, (path_name, caster) in env_map.items():
        if env_name not in os.environ:
            continue
        target = config
        parts = path_name.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], caster(os.environ[env_name]))
    return config
