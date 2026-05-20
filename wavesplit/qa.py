from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from jiwer import wer as jiwer_wer
from rapidfuzz import fuzz

from .audio import analyze_clip_energy
from .config import QAConfig
from .models import ClipRecord
from .text import normalize_text


class FasterWhisperTranscriber:
    def __init__(self, config: QAConfig):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(config.asr_model, device=config.device, compute_type=config.compute_type)

    def transcribe(self, clip_path: str | Path) -> str:
        segments, _info = self.model.transcribe(
            str(clip_path),
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def build_alias_map(aliases: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, values in aliases.items():
        normalized_canonical = normalize_text(canonical)
        mapping[normalized_canonical] = normalized_canonical
        for value in values:
            mapping[normalize_text(value)] = normalized_canonical
    return mapping


def normalize_for_qa(text: str, aliases: dict[str, list[str]] | None = None) -> str:
    normalized = normalize_text(text)
    if not aliases:
        return normalized
    alias_map = build_alias_map(aliases)
    return " ".join(alias_map.get(token, token) for token in normalized.split())


def safe_wer(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    return float(jiwer_wer(reference, hypothesis))


def duration_ratios_by_text(rows: Iterable[dict[str, object]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("duration_sec") is None:
            continue
        grouped[str(row["normalized_text"])].append(float(row["duration_sec"]))
    return grouped


def score_clip(
    *,
    normalized_text: str,
    asr_text: str,
    duration_sec: float,
    metrics: dict[str, object],
    config: QAConfig,
    base_flags: list[str],
    same_text_median: float | None,
    asr_error: str | None = None,
) -> dict[str, object]:
    flags = list(dict.fromkeys(base_flags))
    qa_reference = normalize_for_qa(normalized_text, config.aliases)
    asr_normalized = normalize_for_qa(asr_text, config.aliases)
    similarity: float | None = None
    word_error_rate: float | None = None
    exact_match = False

    if asr_error:
        flags.append("asr_unavailable")
    else:
        exact_match = qa_reference == asr_normalized
        similarity = float(fuzz.ratio(qa_reference, asr_normalized))
        word_error_rate = safe_wer(qa_reference, asr_normalized)

    leading = int(metrics.get("leading_silence_ms") or 0)
    trailing = int(metrics.get("trailing_silence_ms") or 0)
    if bool(metrics.get("start_energy_flag")):
        flags.append("start_maybe_clipped")
    if bool(metrics.get("end_energy_flag")):
        flags.append("end_maybe_clipped")
    if leading > config.leading_silence_review_ms:
        flags.append("leading_silence_long")
    if trailing > config.trailing_silence_review_ms:
        flags.append("trailing_silence_long")
    if duration_sec < 0.25:
        flags.append("duration_too_short")
    if duration_sec > 8.0:
        flags.append("duration_long")
    if same_text_median and same_text_median > 0:
        ratio = duration_sec / same_text_median
        if ratio > 2.5:
            flags.append("same_text_duration_high")
        elif ratio < 0.4:
            flags.append("same_text_duration_low")

    severe_text_mismatch = False
    review_text_mismatch = False
    if not asr_error:
        severe_text_mismatch = bool(similarity is not None and word_error_rate is not None) and (
            similarity < config.similarity_review or word_error_rate > config.wer_review
        )
        review_text_mismatch = bool(similarity is not None and word_error_rate is not None) and not (
            exact_match or (similarity >= config.similarity_pass and word_error_rate <= config.wer_pass)
        )
        if severe_text_mismatch:
            flags.append("text_mismatch_fail")
        elif review_text_mismatch:
            flags.append("text_mismatch_review")

    if "duration_too_short" in flags or severe_text_mismatch:
        status = "fail"
    elif flags or review_text_mismatch:
        status = "review"
    else:
        status = "pass"

    confidence = 100
    if asr_error:
        confidence -= 12
    if severe_text_mismatch:
        confidence -= 45
    elif review_text_mismatch:
        confidence -= 18
    confidence -= 10 * sum(flag in flags for flag in ["start_maybe_clipped", "end_maybe_clipped"])
    confidence -= 8 * sum(flag in flags for flag in ["leading_silence_long", "trailing_silence_long"])
    confidence -= 20 * sum(flag.startswith("duration_") for flag in flags)
    confidence -= 10 * sum(flag.startswith("same_text_duration_") for flag in flags)
    confidence -= 5 * sum(flag in flags for flag in ["padding_overlap_adjusted", "boundary_overlap"])
    confidence = max(0, min(100, int(round(confidence))))

    return {
        "asr_normalized_text": asr_normalized,
        "similarity": similarity,
        "wer": word_error_rate,
        "status": status,
        "confidence": confidence,
        "flags": list(dict.fromkeys(flags)),
    }


def run_qa(
    rows: list[dict[str, object]],
    *,
    clips_dir: str | Path,
    config: QAConfig,
    progress: callable | None = None,
) -> list[ClipRecord]:
    medians: dict[str, float] = {}
    grouped = duration_ratios_by_text(rows)
    for text, durations in grouped.items():
        sorted_durations = sorted(durations)
        mid = len(sorted_durations) // 2
        medians[text] = (
            sorted_durations[mid]
            if len(sorted_durations) % 2 == 1
            else (sorted_durations[mid - 1] + sorted_durations[mid]) / 2
        )

    transcriber = None
    asr_error: str | None = None
    if config.enabled and config.asr_engine == "faster_whisper":
        try:
            transcriber = FasterWhisperTranscriber(config)
        except Exception as exc:  # pragma: no cover - depends on local model setup
            asr_error = str(exc)
    elif config.enabled and config.asr_engine not in {"none", "disabled"}:
        asr_error = f"ASR engine '{config.asr_engine}' is not implemented."

    records: list[ClipRecord] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        if row.get("output_file") is None or row.get("start_sec") is None or row.get("end_sec") is None:
            flags = list(dict.fromkeys(list(row.get("flags") or []) + ["missing_audio_segment"]))
            records.append(
                ClipRecord(
                    clip_id=str(row["clip_id"]),
                    line_index=int(row["line_index"]),
                    original_text=str(row["original_text"]),
                    normalized_text=str(row["normalized_text"]),
                    output_file=None,
                    duplicate_index=int(row["duplicate_index"]),
                    start_sec=None,
                    end_sec=None,
                    duration_sec=None,
                    alignment_score_mean=row.get("alignment_score_mean"),  # type: ignore[arg-type]
                    confidence=0,
                    status="missing_audio",
                    asr_text="",
                    asr_normalized_text="",
                    similarity=None,
                    wer=None,
                    leading_silence_ms=None,
                    trailing_silence_ms=None,
                    peak_dbfs=None,
                    rms_dbfs=None,
                    flags=flags,
                )
            )
            if progress:
                progress(index, total, records[-1])
            continue
        clip_path = Path(clips_dir) / str(row["output_file"])
        metrics = analyze_clip_energy(clip_path)
        asr_text = ""
        item_error = asr_error
        if transcriber is not None:
            try:
                asr_text = transcriber.transcribe(clip_path)
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                item_error = str(exc)
        elif not config.enabled:
            item_error = "ASR QA disabled by configuration."

        score = score_clip(
            normalized_text=str(row["normalized_text"]),
            asr_text=asr_text,
            duration_sec=float(row["duration_sec"]),
            metrics=metrics,
            config=config,
            base_flags=list(row.get("flags") or []),
            same_text_median=medians.get(str(row["normalized_text"])),
            asr_error=item_error,
        )
        records.append(
            ClipRecord(
                clip_id=str(row["clip_id"]),
                line_index=int(row["line_index"]),
                original_text=str(row["original_text"]),
                normalized_text=str(row["normalized_text"]),
                output_file=str(row["output_file"]),
                duplicate_index=int(row["duplicate_index"]),
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                duration_sec=float(row["duration_sec"]),
                alignment_score_mean=row.get("alignment_score_mean"),  # type: ignore[arg-type]
                confidence=int(score["confidence"]),
                status=str(score["status"]),
                asr_text=asr_text,
                asr_normalized_text=str(score["asr_normalized_text"]),
                similarity=score["similarity"],  # type: ignore[arg-type]
                wer=score["wer"],  # type: ignore[arg-type]
                leading_silence_ms=int(metrics.get("leading_silence_ms") or 0),
                trailing_silence_ms=int(metrics.get("trailing_silence_ms") or 0),
                peak_dbfs=float(metrics.get("peak_dbfs") or -120.0),
                rms_dbfs=float(metrics.get("rms_dbfs") or -120.0),
                flags=list(score["flags"]),  # type: ignore[arg-type]
            )
        )
        if progress:
            progress(index, total, records[-1])
    return records
