from __future__ import annotations

import json
import math
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .alignment import align_transcript
from .audio import cut_audio_clip, detect_speech_intervals_in_window, probe_audio
from .config import AppConfig, load_config
from .errors import WaveSplitError
from .models import ClipRecord, LineAlignment, dataclass_to_dict
from .naming import assign_output_names
from .packaging import package_outputs
from .qa import FasterWhisperTranscriber, normalize_for_qa, run_qa
from .status import JobStatusStore
from .storage import copy_inputs, create_job_layout, paths_for_job
from .text import build_line_manifest, read_transcript, write_line_manifest


MISSING_AUDIO_FLAG = "missing_audio_segment"


def prepare_job_inputs(
    *,
    audio_path: str | Path,
    transcript_path: str | Path,
    job_dir: str | Path,
    job_id: str | None = None,
) -> Any:
    paths = create_job_layout(job_dir)
    copy_inputs(audio_path, transcript_path, paths)
    status = JobStatusStore(paths.status_path)
    status.initialize(
        job_id or Path(job_dir).name,
        Path(audio_path).name,
        Path(transcript_path).name,
    )
    return paths


def _base_clip_rows(alignments: list[LineAlignment], output_names: dict[int, tuple[str, int]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for alignment in alignments:
        assigned_output_file, duplicate_index = output_names[alignment.line_index]
        missing_audio = MISSING_AUDIO_FLAG in alignment.flags or alignment.start_sec is None or alignment.end_sec is None
        output_file: str | None = None if missing_audio else assigned_output_file
        duration_sec = None
        if alignment.start_sec is not None and alignment.end_sec is not None:
            duration_sec = round(alignment.end_sec - alignment.start_sec, 3)
        rows.append(
            {
                "clip_id": alignment.clip_id,
                "line_index": alignment.line_index,
                "original_text": alignment.original_text,
                "normalized_text": alignment.normalized_text,
                "output_file": output_file,
                "duplicate_index": duplicate_index,
                "start_sec": alignment.start_sec,
                "end_sec": alignment.end_sec,
                "duration_sec": duration_sec,
                "alignment_score_mean": alignment.alignment_score_mean,
                "flags": list(alignment.flags),
            }
        )
    return rows


def _phrase_occurrences(reference: str, hypothesis: str) -> int:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words or not hypothesis_words or len(hypothesis_words) < len(reference_words):
        return 0
    count = 0
    index = 0
    width = len(reference_words)
    while index <= len(hypothesis_words) - width:
        if hypothesis_words[index : index + width] == reference_words:
            count += 1
            index += width
        else:
            index += 1
    return count


def _candidate_score(reference: str, hypothesis: str) -> float:
    if not hypothesis:
        return 0.0
    if hypothesis == reference:
        return 1.0
    if _phrase_occurrences(reference, hypothesis) > 0:
        return 0.9
    return SequenceMatcher(None, reference, hypothesis).ratio()


def _candidate_count(reference: str, hypothesis: str, occurrences: int) -> int:
    reference_words = max(1, len(reference.split()))
    hypothesis_words = len(hypothesis.split())
    by_length = int(math.ceil(hypothesis_words / reference_words)) if hypothesis_words else 2
    return max(2, min(6, max(occurrences, by_length)))


def _select_repair_candidate(
    *,
    intervals: list[tuple[float, float]],
    reference: str,
    audio_path: str | Path,
    logs_dir: str | Path,
    config: AppConfig,
    transcriber: FasterWhisperTranscriber | None,
    row_index: int,
    anchor_start_sec: float | None = None,
    anchor_end_sec: float | None = None,
) -> tuple[int, dict[str, object]]:
    candidate_dir = Path(logs_dir) / "repair_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    best_index = 0
    best_rank = (-1.0, -1.0, -float("inf"))
    best: dict[str, object] = {
        "score": -1.0,
        "asr_text": "",
        "asr_normalized_text": "",
    }

    for candidate_index, (start_sec, end_sec) in enumerate(intervals):
        asr_text = ""
        asr_normalized = ""
        score = 1.0 if transcriber is None and candidate_index == 0 else 0.0
        if transcriber is not None:
            candidate_path = candidate_dir / f"row-{row_index + 1:04d}-{candidate_index + 1}.wav"
            cut_audio_clip(audio_path, candidate_path, start_sec, end_sec)
            try:
                asr_text = transcriber.transcribe(candidate_path)
                asr_normalized = normalize_for_qa(asr_text, config.qa.aliases)
                score = _candidate_score(reference, asr_normalized)
            finally:
                try:
                    candidate_path.unlink()
                except FileNotFoundError:
                    pass

        overlap = 0.0
        center_score = 0.0
        if anchor_start_sec is not None and anchor_end_sec is not None:
            overlap_start = max(anchor_start_sec, start_sec)
            overlap_end = min(anchor_end_sec, end_sec)
            overlap = max(0.0, overlap_end - overlap_start) / max(0.001, end_sec - start_sec)
            anchor_center = (anchor_start_sec + anchor_end_sec) / 2
            center_score = -abs(((start_sec + end_sec) / 2) - anchor_center)
        rank = (score, overlap, center_score)
        if rank > best_rank:
            best_rank = rank
            best_index = candidate_index
            best = {
                "score": round(score, 4),
                "asr_text": asr_text,
                "asr_normalized_text": asr_normalized,
            }

    return best_index, best


def _is_repeated_context(records: list[ClipRecord], index: int) -> bool:
    text = records[index].normalized_text
    return (index > 0 and records[index - 1].normalized_text == text) or (
        index + 1 < len(records) and records[index + 1].normalized_text == text
    )


def _candidate_repair_bounds(
    *,
    intervals: list[tuple[float, float]],
    selected_index: int,
    row_start: float,
    row_end: float,
    config: AppConfig,
) -> tuple[float, float]:
    speech_start, speech_end = intervals[selected_index]
    pre = config.alignment.pre_padding_ms / 1000.0
    post = config.alignment.post_padding_ms / 1000.0
    start_sec = max(row_start, speech_start - pre)
    end_sec = min(row_end, speech_end + post)
    if selected_index > 0:
        start_sec = max(start_sec, (intervals[selected_index - 1][1] + speech_start) / 2)
    if selected_index + 1 < len(intervals):
        end_sec = min(end_sec, (speech_end + intervals[selected_index + 1][0]) / 2)
    return round(start_sec, 3), round(end_sec, 3)


def _repair_duplicate_snippets(
    rows: list[dict[str, object]],
    records: list[ClipRecord],
    *,
    audio_path: str | Path,
    clips_dir: str | Path,
    logs_dir: str | Path,
    audio_duration_sec: float,
    config: AppConfig,
) -> list[dict[str, object]]:
    transcriber: FasterWhisperTranscriber | None = None
    repairs: list[dict[str, object]] = []
    for row_index, (row, record) in enumerate(zip(rows, records)):
        flags = list(row.get("flags") or [])
        if "qa_duplicate_repaired" in flags:
            continue

        reference = normalize_for_qa(record.normalized_text, config.qa.aliases)
        hypothesis = record.asr_normalized_text or normalize_for_qa(record.asr_text, config.qa.aliases)
        if row.get("start_sec") is None or row.get("end_sec") is None or row.get("output_file") is None:
            continue
        occurrences = _phrase_occurrences(reference, hypothesis)
        reference_seen = occurrences > 0
        word_ratio = len(hypothesis.split()) / max(1, len(reference.split()))
        fragment_like = _is_repeated_context(records, row_index) and (
            record.status == "fail"
            or "same_text_duration_low" in record.flags
            or "start_maybe_clipped" in record.flags
            or "end_maybe_clipped" in record.flags
        ) and (hypothesis != reference or "same_text_duration_low" in record.flags)
        long_multi_snippet = (
            ("duration_long" in record.flags or "same_text_duration_high" in record.flags)
            and reference_seen
            and word_ratio >= 2.0
        )
        if occurrences < 2 and not long_multi_snippet and not fragment_like:
            continue

        target_count = 3 if fragment_like and occurrences < 2 and not long_multi_snippet else _candidate_count(reference, hypothesis, occurrences)
        search_start = float(row["start_sec"])
        search_end = float(row["end_sec"])
        adjust_to_target = True
        if fragment_like and occurrences < 2 and not long_multi_snippet:
            margin = 1.2
            search_start = max(0.0, search_start - margin)
            search_end = min(audio_duration_sec, search_end + margin)
            adjust_to_target = False
        try:
            intervals, debug = detect_speech_intervals_in_window(
                audio_path,
                target_count=target_count,
                audio_duration_sec=audio_duration_sec,
                config=config.alignment,
                start_sec=search_start,
                end_sec=search_end,
                prefer_balanced=True,
                adjust_to_target=adjust_to_target,
            )
        except Exception as exc:
            repairs.append(
                {
                    "line_index": record.line_index,
                    "output_file": record.output_file,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
            continue

        if not intervals or (len(intervals) < 2 and not fragment_like):
            continue
        if transcriber is None and config.qa.enabled and config.qa.asr_engine == "faster_whisper":
            try:
                transcriber = FasterWhisperTranscriber(config.qa)
            except Exception:
                transcriber = None

        selected_index, candidate = _select_repair_candidate(
            intervals=intervals,
            reference=reference,
            audio_path=audio_path,
            logs_dir=logs_dir,
            config=config,
            transcriber=transcriber,
            row_index=row_index,
            anchor_start_sec=float(row["start_sec"]),
            anchor_end_sec=float(row["end_sec"]),
        )
        min_score = 0.9 if fragment_like and occurrences < 2 and not long_multi_snippet else 0.75
        if transcriber is not None and float(candidate["score"]) < min_score:
            repairs.append(
                {
                    "line_index": record.line_index,
                    "output_file": record.output_file,
                    "status": "skipped",
                    "reason": "no_matching_candidate",
                    "candidate": candidate,
                }
            )
            continue

        new_start, new_end = _candidate_repair_bounds(
            intervals=intervals,
            selected_index=selected_index,
            row_start=search_start if fragment_like and occurrences < 2 and not long_multi_snippet else float(row["start_sec"]),
            row_end=search_end if fragment_like and occurrences < 2 and not long_multi_snippet else float(row["end_sec"]),
            config=config,
        )
        if new_end - new_start < 0.25:
            continue

        row["start_sec"] = new_start
        row["end_sec"] = new_end
        row["duration_sec"] = round(new_end - new_start, 3)
        flags.append("qa_duplicate_repaired" if occurrences >= 2 else "qa_long_snippet_repaired")
        if fragment_like and occurrences < 2 and not long_multi_snippet:
            flags[-1] = "qa_fragment_repaired"
        row["flags"] = list(dict.fromkeys(flags))
        cut_audio_clip(
            audio_path,
            Path(clips_dir) / record.output_file,
            new_start,
            new_end,
        )
        repairs.append(
            {
                "line_index": record.line_index,
                "output_file": record.output_file,
                "status": "applied",
                "reason": "duplicate_asr"
                if occurrences >= 2
                else "fragment_asr"
                if fragment_like
                else "long_multi_snippet",
                "old_start_sec": record.start_sec,
                "old_end_sec": record.end_sec,
                "new_start_sec": new_start,
                "new_end_sec": new_end,
                "candidate_count": len(intervals),
                "selected_candidate": selected_index + 1,
                "candidate": candidate,
                "vad_debug": {
                    "initial_segment_count": debug.get("initial_segment_count"),
                    "adjustment": debug.get("adjustment"),
                    "threshold": debug.get("threshold"),
                    "max_gap_ms": debug.get("max_gap_ms"),
                },
            }
        )
    return repairs


def run_pipeline(job_dir: str | Path, config: AppConfig | None = None) -> dict[str, object]:
    config = config or load_config()
    paths = paths_for_job(job_dir)
    status = JobStatusStore(paths.status_path)
    existing_status = status.read()
    job_id = existing_status.get("job_id") or Path(job_dir).name
    audio_filename = existing_status.get("input", {}).get("audio_filename") or "original.wav"
    transcript_filename = existing_status.get("input", {}).get("text_filename") or "transcript.txt"

    try:
        status.stage("validating", "Validating input files", stage_fraction=0.0)
        audio_info = probe_audio(paths.original_audio_path)
        lines = read_transcript(paths.transcript_path)
        status.stage(
            "validating",
            "Input files validated",
            stage_fraction=1.0,
            counts={"total": len(lines)},
        )
        status.update(
            input={
                "line_count": len(lines),
                "audio_duration_sec": round(audio_info.duration_sec, 3),
            }
        )

        status.stage("normalizing_text", "Normalizing transcript", stage_fraction=0.0)
        line_records = build_line_manifest(lines)
        write_line_manifest(line_records, Path(paths.normalized_dir) / "transcript.normalized.json")
        status.stage("normalizing_text", "Transcript normalized", stage_fraction=1.0)

        status.stage("aligning", "Aligning transcript to audio", stage_fraction=0.0)
        alignments, _raw_debug = align_transcript(
            paths.original_audio_path,
            line_records,
            audio_info,
            config,
            paths.alignment_dir,
        )
        status.stage(
            "aligning",
            f"Aligned {len(alignments)}/{len(line_records)} lines",
            stage_fraction=1.0,
            counts={"aligned": len(alignments)},
        )

        status.stage("building_segments", "Building output segment manifest", stage_fraction=0.0)
        output_names = assign_output_names(line_records)
        rows = _base_clip_rows(alignments, output_names)
        status.stage("building_segments", "Output segment manifest built", stage_fraction=1.0)

        cut_rows: list[dict[str, object]] = []
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            status.stage(
                "cutting",
                f"Cutting clip {index}/{total}",
                stage_fraction=index / total,
                counts={"cut": index},
            )
            if row["output_file"] is None or row["start_sec"] is None or row["end_sec"] is None:
                cut_rows.append(row)
                continue
            cut_audio_clip(
                paths.original_audio_path,
                Path(paths.clips_dir) / str(row["output_file"]),
                float(row["start_sec"]),
                float(row["end_sec"]),
            )
            cut_rows.append(row)

        def qa_progress(index: int, total_count: int, _record: ClipRecord) -> None:
            status.stage(
                "qa_asr",
                f"Running ASR QA for clip {index}/{total_count}",
                stage_fraction=index / max(1, total_count),
            )

        status.stage("qa_asr", "Starting ASR QA", stage_fraction=0.0)
        records = run_qa(cut_rows, clips_dir=paths.clips_dir, config=config.qa, progress=qa_progress)
        repair_log: list[dict[str, object]] = []
        for repair_pass in range(1, 3):
            repairs = _repair_duplicate_snippets(
                cut_rows,
                records,
                audio_path=paths.original_audio_path,
                clips_dir=paths.clips_dir,
                logs_dir=paths.logs_dir,
                audio_duration_sec=audio_info.duration_sec,
                config=config,
            )
            applied_repairs = [repair for repair in repairs if repair.get("status") == "applied"]
            if repairs:
                repair_log.extend({"pass": repair_pass, **repair} for repair in repairs)
            if not applied_repairs:
                break
            status.stage(
                "qa_asr",
                f"Re-running ASR QA after repairing {len(applied_repairs)} duplicate clips",
                stage_fraction=0.0,
                counts={"qa_repairs": len(applied_repairs)},
            )
            records = run_qa(cut_rows, clips_dir=paths.clips_dir, config=config.qa, progress=qa_progress)
        if repair_log:
            Path(paths.qa_dir).mkdir(parents=True, exist_ok=True)
            (Path(paths.qa_dir) / "repair_report.json").write_text(
                json.dumps(repair_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        counts = Counter(record.status for record in records)
        status.stage(
            "qa_scoring",
            "QA scoring complete",
            stage_fraction=1.0,
            counts={
                "qa_pass": counts.get("pass", 0),
                "qa_review": counts.get("review", 0),
                "qa_fail": counts.get("fail", 0),
                "qa_missing_audio": counts.get("missing_audio", 0),
            },
        )

        status.stage("packaging", "Packaging zip and reports", stage_fraction=0.0)
        payload = package_outputs(
            job_id=str(job_id),
            paths=paths,
            records=records,
            audio_info=audio_info,
            config=config,
            audio_filename=str(audio_filename),
            transcript_filename=str(transcript_filename),
        )
        status.stage("packaging", "Packaging complete", stage_fraction=1.0)
        status.succeed(f"Generated {sum(1 for record in records if record.output_file)} clips")
        return payload
    except WaveSplitError as exc:
        status.fail(status.read().get("stage", "error"), str(exc))
        raise
    except Exception as exc:
        status.fail(status.read().get("stage", "error"), f"Unexpected error: {exc}")
        raise


def process_inputs(
    *,
    audio_path: str | Path,
    transcript_path: str | Path,
    out_dir: str | Path,
    config: AppConfig | None = None,
    job_id: str | None = None,
) -> dict[str, object]:
    config = config or load_config()
    paths = prepare_job_inputs(
        audio_path=audio_path,
        transcript_path=transcript_path,
        job_dir=out_dir,
        job_id=job_id or Path(out_dir).name,
    )
    return run_pipeline(paths.job_dir, config=config)
