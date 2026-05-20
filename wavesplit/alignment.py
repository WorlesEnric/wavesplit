from __future__ import annotations

import json
import re
from pathlib import Path

from .audio import choose_silence_split_points, detect_speech_intervals, detect_speech_intervals_in_window
from .config import AppConfig
from .errors import AlignmentError
from .models import AudioInfo, LineAlignment, LineRecord, dataclass_to_dict
from .segments import apply_padding_and_fix_overlaps


MISSING_AUDIO_FLAG = "missing_audio_segment"


SRT_TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


def align_transcript(
    audio_path: str | Path,
    lines: list[LineRecord],
    audio_info: AudioInfo,
    config: AppConfig,
    alignment_dir: str | Path,
) -> tuple[list[LineAlignment], dict[str, object]]:
    engine = config.alignment.engine
    if engine not in {"ctc_forced_aligner", "energy_ordered", "auto"}:
        raise AlignmentError(
            f"Alignment engine '{engine}' is not available in this local build. "
            "Set alignment.engine to 'auto', 'ctc_forced_aligner', or 'energy_ordered'."
        )

    if engine in {"ctc_forced_aligner", "auto"}:
        try:
            return _align_with_ctc(audio_path, lines, audio_info, config, alignment_dir)
        except Exception as exc:
            if engine == "ctc_forced_aligner":
                raise AlignmentError(f"CTC forced alignment failed: {exc}") from exc
            fallback_note = str(exc)
        else:
            fallback_note = ""
    else:
        fallback_note = ""

    return _align_with_energy(
        audio_path,
        lines,
        audio_info,
        config,
        alignment_dir,
        fallback_note=fallback_note,
    )


def _align_with_ctc(
    audio_path: str | Path,
    lines: list[LineRecord],
    audio_info: AudioInfo,
    config: AppConfig,
    alignment_dir: str | Path,
) -> tuple[list[LineAlignment], dict[str, object]]:
    import ctc_forced_aligner
    import onnxruntime as ort
    from ctc_forced_aligner import AlignmentSingleton

    model_path = Path(config.alignment.model)
    if not model_path.exists():
        raise AlignmentError(f"CTC ONNX model not found at {model_path}.")

    Path(alignment_dir).mkdir(parents=True, exist_ok=True)
    srt_path = Path(alignment_dir) / "ctc_line_alignment.srt"
    device = config.alignment.ctc_device.lower()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise AlignmentError("CTC CUDA was requested, but ONNX Runtime CUDAExecutionProvider is not available.")
    aligner = _load_ctc_aligner(AlignmentSingleton, ctc_forced_aligner, str(model_path), providers)
    ok = aligner.generate_srt(
        str(audio_path),
        _line_text_path(alignment_dir, lines),
        str(srt_path),
        language=config.alignment.ctc_language,
        batch_size=config.alignment.ctc_batch_size,
    )
    if not ok or not srt_path.exists():
        raise AlignmentError("CTC aligner did not produce an SRT file.")

    srt_entries = _parse_srt(srt_path)
    word_timestamps = list(aligner.word_timestamps or [])
    entries = _line_entries_from_word_timestamps(word_timestamps, lines)

    alignments: list[LineAlignment] = []
    for line, entry in zip(lines, entries):
        flags: list[str] = []
        if entry["end"] <= entry["start"]:
            flags.append("boundary_invalid_repaired")
        alignments.append(
            LineAlignment(
                clip_id=f"clip-{line.line_index:06d}",
                line_index=line.line_index,
                original_text=line.original_text,
                normalized_text=line.normalized_text,
                raw_start_sec=round(float(entry["start"]), 3),
                raw_end_sec=round(float(entry["end"]), 3),
                start_sec=round(float(entry["start"]), 3),
                end_sec=round(float(entry["end"]), 3),
                alignment_score_mean=_line_score(word_timestamps, line.token_start_index, line.token_end_index),
                flags=flags,
            )
        )

    repeated_refinements = _refine_repeated_text_runs_with_vad(
        alignments,
        audio_path=audio_path,
        audio_duration_sec=audio_info.duration_sec,
        config=config,
    )
    present_alignments = [item for item in alignments if not _is_missing_audio(item)]
    split_points = choose_silence_split_points(
        audio_path,
        [(float(item.raw_start_sec), float(item.raw_end_sec)) for item in present_alignments],
        audio_duration_sec=audio_info.duration_sec,
        config=config.alignment,
    )
    apply_padding_and_fix_overlaps(
        present_alignments,
        audio_duration_sec=audio_info.duration_sec,
        config=config.alignment,
        split_points=split_points,
        include_edge_silence=True,
    )
    _round_alignments(alignments)
    raw_debug = {
        "engine": "ctc_forced_aligner",
        "model": str(model_path),
        "language": config.alignment.ctc_language,
        "batch_size": config.alignment.ctc_batch_size,
        "device": device,
        "providers": aligner.alignment_model.get_providers(),
        "word_timestamp_count": len(word_timestamps),
        "line_timestamp_count": len(entries),
        "srt_line_timestamp_count": len(srt_entries),
        "line_timestamp_source": "word_timestamps",
        "segments": [[item.raw_start_sec, item.raw_end_sec] for item in alignments],
        "silence_split_points": split_points,
        "repeated_text_refinements": repeated_refinements,
        "word_timestamps": word_timestamps,
    }
    (Path(alignment_dir) / "raw_alignment.json").write_text(
        json.dumps(raw_debug, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (Path(alignment_dir) / "line_alignment.json").write_text(
        json.dumps([dataclass_to_dict(item) for item in alignments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return alignments, raw_debug


def _load_ctc_aligner(alignment_singleton: object, ctc_module: object, model_path: str, providers: list[str]) -> object:
    instance = getattr(alignment_singleton, "_instance", None)
    if instance is not None:
        loaded_model = getattr(instance, "model_path", None)
        loaded_providers = getattr(getattr(instance, "alignment_model", None), "get_providers", lambda: [])()
        if loaded_model != model_path or list(loaded_providers) != providers:
            setattr(alignment_singleton, "_instance", None)

    original_session = ctc_module.onnxruntime.InferenceSession

    def inference_session(path: str, *args: object, **kwargs: object) -> object:
        kwargs.setdefault("providers", providers)
        return original_session(path, *args, **kwargs)

    ctc_module.onnxruntime.InferenceSession = inference_session
    try:
        return alignment_singleton(model_path=model_path)
    finally:
        ctc_module.onnxruntime.InferenceSession = original_session


def _align_with_energy(
    audio_path: str | Path,
    lines: list[LineRecord],
    audio_info: AudioInfo,
    config: AppConfig,
    alignment_dir: str | Path,
    *,
    fallback_note: str = "",
) -> tuple[list[LineAlignment], dict[str, object]]:
    Path(alignment_dir).mkdir(parents=True, exist_ok=True)

    intervals, raw_debug = detect_speech_intervals(
        audio_path,
        target_count=len(lines),
        audio_duration_sec=audio_info.duration_sec,
        config=config.alignment,
    )
    split_points = choose_silence_split_points(
        audio_path,
        intervals,
        audio_duration_sec=audio_info.duration_sec,
        config=config.alignment,
    )
    raw_debug["silence_split_points"] = split_points
    if fallback_note:
        raw_debug["fallback_from_ctc_error"] = fallback_note
    (Path(alignment_dir) / "raw_alignment.json").write_text(
        json.dumps(raw_debug | {"segments": intervals}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    alignments: list[LineAlignment] = []
    base_score = 96.0 if raw_debug.get("adjustment") == "none" else 88.0
    for line, (start, end) in zip(lines, intervals):
        flags: list[str] = []
        if raw_debug.get("adjustment") != "none":
            flags.append(f"energy_segments_{raw_debug['adjustment']}")
        alignments.append(
            LineAlignment(
                clip_id=f"clip-{line.line_index:06d}",
                line_index=line.line_index,
                original_text=line.original_text,
                normalized_text=line.normalized_text,
                raw_start_sec=round(start, 3),
                raw_end_sec=round(end, 3),
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                alignment_score_mean=base_score,
                flags=flags,
            )
        )
    apply_padding_and_fix_overlaps(
        alignments,
        audio_duration_sec=audio_info.duration_sec,
        config=config.alignment,
        split_points=split_points,
        include_edge_silence=True,
    )
    _round_alignments(alignments)

    (Path(alignment_dir) / "line_alignment.json").write_text(
        json.dumps([dataclass_to_dict(item) for item in alignments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return alignments, raw_debug


def _line_text_path(alignment_dir: str | Path, lines: list[LineRecord]) -> str:
    path = Path(alignment_dir) / "ctc_transcript.txt"
    path.write_text("\n".join(line.original_text for line in lines), encoding="utf-8")
    return str(path)


def _parse_srt_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def _parse_srt(path: str | Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    blocks = Path(path).read_text(encoding="utf-8").strip().split("\n\n")
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        match = SRT_TIME_RE.search(lines[1])
        if not match:
            continue
        entries.append(
            {
                "index": int(lines[0]) if lines[0].isdigit() else len(entries) + 1,
                "start": _parse_srt_time(match.group("start")),
                "end": _parse_srt_time(match.group("end")),
                "text": " ".join(lines[2:]),
            }
        )
    return entries


def _line_entries_from_word_timestamps(
    word_timestamps: list[dict[str, object]],
    lines: list[LineRecord],
) -> list[dict[str, object]]:
    expected_word_count = sum(len(line.tokens) for line in lines)
    if len(word_timestamps) != expected_word_count:
        raise AlignmentError(
            f"CTC aligner produced {len(word_timestamps)} word timestamps for "
            f"{expected_word_count} transcript tokens."
        )

    entries: list[dict[str, object]] = []
    cursor = 0
    for line in lines:
        token_count = len(line.tokens)
        span = word_timestamps[cursor : cursor + token_count]
        cursor += token_count
        if not span:
            raise AlignmentError(f"Transcript line {line.line_index} has no CTC word timestamps.")
        entries.append(
            {
                "index": line.line_index,
                "start": _word_timestamp_time(span[0], "start", line.line_index),
                "end": _word_timestamp_time(span[-1], "end", line.line_index),
                "text": " ".join(str(word.get("text", "")).strip() for word in span).strip(),
            }
        )
    return entries


def _word_timestamp_time(word: dict[str, object], key: str, line_index: int) -> float:
    value = word.get(key)
    if not isinstance(value, (int, float)):
        raise AlignmentError(f"CTC word timestamp for line {line_index} is missing numeric '{key}'.")
    return float(value)


def _line_score(word_timestamps: list[dict[str, object]], start_index: int, end_index: int) -> float | None:
    scores = []
    for word in word_timestamps[start_index:end_index]:
        score = word.get("score") if isinstance(word, dict) else None
        if isinstance(score, (int, float)):
            scores.append(float(score))
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def _refine_repeated_text_runs_with_vad(
    alignments: list[LineAlignment],
    *,
    audio_path: str | Path,
    audio_duration_sec: float,
    config: AppConfig,
) -> list[dict[str, object]]:
    refinements: list[dict[str, object]] = []
    start = 0
    while start < len(alignments):
        end = start + 1
        while end < len(alignments) and alignments[end].normalized_text == alignments[start].normalized_text:
            end += 1
        count = end - start
        if count >= 3:
            refinements.extend(
                _refine_repeated_run(
                    alignments,
                    start,
                    end,
                    audio_path=audio_path,
                    audio_duration_sec=audio_duration_sec,
                    config=config,
                )
            )
        start = end
    return refinements


def _refine_repeated_run(
    alignments: list[LineAlignment],
    start: int,
    end: int,
    *,
    audio_path: str | Path,
    audio_duration_sec: float,
    config: AppConfig,
) -> list[dict[str, object]]:
    first = alignments[start]
    last = alignments[end - 1]
    if first.raw_start_sec is None or last.raw_end_sec is None:
        return []
    margin = 0.5
    window_start = max(0.0, first.raw_start_sec - margin)
    window_end = min(audio_duration_sec, last.raw_end_sec + margin)
    if start > 0:
        previous = alignments[start - 1]
        if previous.raw_end_sec is not None:
            window_start = max((previous.raw_end_sec + first.raw_start_sec) / 2, window_start)
    if end < len(alignments):
        nxt = alignments[end]
        if nxt.raw_start_sec is not None:
            window_end = min((last.raw_end_sec + nxt.raw_start_sec) / 2, window_end)
    if window_end <= window_start:
        return []

    target_count = end - start
    try:
        intervals, debug = detect_speech_intervals_in_window(
            audio_path,
            target_count=target_count,
            audio_duration_sec=audio_duration_sec,
            config=config.alignment,
            start_sec=window_start,
            end_sec=window_end,
            prefer_balanced=True,
            adjust_to_target=False,
        )
    except Exception as exc:
        return [
            {
                "line_start": first.line_index,
                "line_end": last.line_index,
                "status": "failed",
                "reason": str(exc),
            }
        ]

    if len(intervals) != target_count:
        try:
            intervals, debug = detect_speech_intervals_in_window(
                audio_path,
                target_count=target_count,
                audio_duration_sec=audio_duration_sec,
                config=config.alignment,
                start_sec=window_start,
                end_sec=window_end,
                prefer_balanced=True,
                adjust_to_target=True,
            )
        except Exception as exc:
            return [
                {
                    "line_start": first.line_index,
                    "line_end": last.line_index,
                    "status": "failed",
                    "reason": str(exc),
                }
            ]

    if len(intervals) != target_count:
        return [
            {
                "line_start": first.line_index,
                "line_end": last.line_index,
                "status": "skipped",
                "initial_segment_count": debug.get("initial_segment_count"),
                "adjustment": debug.get("adjustment"),
            }
        ]
    if debug.get("adjustment") == "split":
        initial_intervals = _debug_intervals(debug.get("initial_intervals"))
        if 0 < len(initial_intervals) < end - start:
            for alignment, (new_start, new_end) in zip(alignments[start : start + len(initial_intervals)], initial_intervals):
                alignment.raw_start_sec = round(new_start, 3)
                alignment.raw_end_sec = round(new_end, 3)
                alignment.start_sec = alignment.raw_start_sec
                alignment.end_sec = alignment.raw_end_sec
                alignment.flags.append("repeated_text_vad_refined")
                alignment.flags.append("repeated_text_count_mismatch")

            for alignment in alignments[start + len(initial_intervals) : end]:
                alignment.raw_start_sec = None
                alignment.raw_end_sec = None
                alignment.start_sec = None
                alignment.end_sec = None
                alignment.flags.append(MISSING_AUDIO_FLAG)
                alignment.flags.append("repeated_text_count_mismatch")
                alignment.flags.append("unmatched_indices_assumed_tail")

            return [
                {
                    "line_start": first.line_index,
                    "line_end": last.line_index,
                    "status": "applied_missing_tail",
                    "text": first.normalized_text,
                    "expected_segment_count": end - start,
                    "detected_segment_count": len(initial_intervals),
                    "missing_line_start": alignments[start + len(initial_intervals)].line_index,
                    "missing_line_end": last.line_index,
                    "assumption": "unmatched_indices_assumed_tail",
                    "window_start_sec": debug.get("window_start_sec"),
                    "window_end_sec": debug.get("window_end_sec"),
                    "threshold": debug.get("threshold"),
                    "max_gap_ms": debug.get("max_gap_ms"),
                    "initial_segment_count": debug.get("initial_segment_count"),
                    "adjustment": debug.get("adjustment"),
                }
            ]
        return [
            {
                "line_start": first.line_index,
                "line_end": last.line_index,
                "status": "skipped",
                "initial_segment_count": debug.get("initial_segment_count"),
                "adjustment": debug.get("adjustment"),
                "reason": "refuse_artificial_split",
            }
        ]
    if not _balanced_repeated_intervals(intervals):
        return [
            {
                "line_start": first.line_index,
                "line_end": last.line_index,
                "status": "skipped",
                "initial_segment_count": debug.get("initial_segment_count"),
                "adjustment": debug.get("adjustment"),
                "reason": "duration_outlier",
            }
        ]

    for alignment, (new_start, new_end) in zip(alignments[start:end], intervals):
        alignment.raw_start_sec = round(new_start, 3)
        alignment.raw_end_sec = round(new_end, 3)
        alignment.start_sec = alignment.raw_start_sec
        alignment.end_sec = alignment.raw_end_sec
        alignment.flags.append("repeated_text_vad_refined")

    return [
        {
            "line_start": first.line_index,
            "line_end": last.line_index,
            "status": "applied",
            "text": first.normalized_text,
            "window_start_sec": debug.get("window_start_sec"),
            "window_end_sec": debug.get("window_end_sec"),
            "threshold": debug.get("threshold"),
            "max_gap_ms": debug.get("max_gap_ms"),
            "initial_segment_count": debug.get("initial_segment_count"),
            "adjustment": debug.get("adjustment"),
        }
    ]


def _balanced_repeated_intervals(intervals: list[tuple[float, float]]) -> bool:
    if len(intervals) < 3:
        return True
    durations = sorted(max(0.0, end - start) for start, end in intervals)
    median = durations[len(durations) // 2]
    if median <= 0:
        return False
    return durations[-1] <= max(2.4, median * 2.4) and durations[0] >= max(0.12, median * 0.25)


def _debug_intervals(value: object) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    intervals: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        start, end = item
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
            intervals.append((float(start), float(end)))
    return intervals


def _is_missing_audio(alignment: LineAlignment) -> bool:
    return MISSING_AUDIO_FLAG in alignment.flags or alignment.start_sec is None or alignment.end_sec is None


def _round_alignments(alignments: list[LineAlignment]) -> None:
    for item in alignments:
        if _is_missing_audio(item):
            item.raw_start_sec = None
            item.raw_end_sec = None
            item.start_sec = None
            item.end_sec = None
            continue
        item.start_sec = round(item.start_sec, 3)
        item.end_sec = round(item.end_sec, 3)
        item.raw_start_sec = round(item.raw_start_sec, 3)
        item.raw_end_sec = round(item.raw_end_sec, 3)
