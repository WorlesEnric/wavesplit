from __future__ import annotations

from pathlib import Path
from typing import Any

from .audio import detect_voice_intervals, probe_audio
from .config import AppConfig, load_config

DEFAULT_MARGIN_SEC = 1.0


def build_timestamp_segments(
    intervals: list[tuple[float, float]],
    *,
    audio_duration_sec: float,
    margin_sec: float = DEFAULT_MARGIN_SEC,
) -> list[dict[str, float]]:
    cleaned = sorted(
        (max(0.0, start), min(audio_duration_sec, end))
        for start, end in intervals
        if end > start
    )
    segments = [
        {
            "raw_start_sec": start,
            "raw_end_sec": end,
            "start_sec": max(0.0, start - margin_sec),
            "end_sec": min(audio_duration_sec, end + margin_sec),
        }
        for start, end in cleaned
    ]

    for index, ((_, current_raw_end), (next_raw_start, _)) in enumerate(zip(cleaned, cleaned[1:])):
        gap = next_raw_start - current_raw_end
        if gap >= margin_sec * 2:
            current_end = current_raw_end + margin_sec
            next_start = next_raw_start - margin_sec
        elif gap > 0:
            short_margin = gap / 4
            current_end = current_raw_end + short_margin
            next_start = next_raw_start - short_margin
        else:
            boundary = (current_raw_end + next_raw_start) / 2
            current_end = boundary
            next_start = boundary

        segments[index]["end_sec"] = min(audio_duration_sec, max(segments[index]["start_sec"], current_end))
        segments[index + 1]["start_sec"] = max(0.0, min(segments[index + 1]["end_sec"], next_start))

    return [
        {key: round(value, 3) for key, value in segment.items()}
        for segment in segments
        if segment["end_sec"] > segment["start_sec"]
    ]


def format_timestamp_txt(segments: list[dict[str, float]]) -> str:
    return "".join(f"{segment['start_sec']:.3f} {segment['end_sec']:.3f}\n" for segment in segments)


def generate_timestamp_payload(
    audio_path: str | Path,
    *,
    reference_text: str = "",
    config: AppConfig | None = None,
    margin_sec: float = DEFAULT_MARGIN_SEC,
) -> dict[str, Any]:
    config = config or load_config()
    reference_text = reference_text.strip()
    audio_info = probe_audio(audio_path)
    intervals, debug = detect_voice_intervals(
        audio_path,
        audio_duration_sec=audio_info.duration_sec,
        config=config.alignment,
    )
    segments = build_timestamp_segments(
        intervals,
        audio_duration_sec=audio_info.duration_sec,
        margin_sec=margin_sec,
    )
    return {
        "summary": {
            "total": len(segments),
            "duration_sec": round(audio_info.duration_sec, 3),
            "margin_sec": margin_sec,
            "reference_text": reference_text,
        },
        "raw_intervals": [[round(start, 3), round(end, 3)] for start, end in intervals],
        "segments": segments,
        "debug": debug,
        "text": format_timestamp_txt(segments),
    }
