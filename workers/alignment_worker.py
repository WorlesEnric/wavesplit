"""Alignment worker — serves the ``align-audio`` queue.

The "audio↔TXT alignment as a single self-contained unit process": for each
leased job it reads a LONG WAV artifact + a TXT (one sentence per line) from the
job, runs forced alignment (``wavesplit.alignment`` CTC + energy fallback) +
segment padding + clip cutting (``wavesplit.audio``), then for EACH line PUTs the
cut clip back as an artifact and completes with the per-line manifest::

    { manifest: [ { line_index, text, artifact_id, start_sec, end_sec, score,
                    flags, missing }, ... ],
      line_count: int, clip_count: int }

The whole upload→align→store flow happens inside the worker (REFACTOR-PLAN D2).

Job input contract (the assumed ``StandardizePacket`` / ``wav_txt`` envelope)
-----------------------------------------------------------------------------
``envelope.input`` carries the resolved packet role. The long WAV is read from
the role's ``_artifacts`` under any name in :data:`AUDIO_ARTIFACT_FIELDS`; the
transcript from a ``_artifacts`` TXT (:data:`TRANSCRIPT_ARTIFACT_FIELDS`) or an
inline ``transcript`` / ``raw_payload`` text field. Confirm the exact field
names with the DSL — see ``workers/README.md``.

The aligner is dependency-injected (``align_fn``) so the handler is unit-testable
WITHOUT torch / onnxruntime / ctc_forced_aligner / ffmpeg.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from wavesplit.config import AppConfig, load_config

from .gateway_client import (
    GatewayArtifacts,
    GatewayJobQueue,
    WorkerEnv,
    decode_audio_value,
    role_artifacts,
    role_fields,
    run_worker_loop,
)


LOGGER = logging.getLogger("wavesplit.workers.alignment")

DEFAULT_QUEUE = "align-audio"

PACKET_ROLES = ("packet", "source_packet", "sourcePacket", "target", "self")
AUDIO_ARTIFACT_FIELDS = ("audio", "long_audio", "wav", "source_audio", "raw_audio")
TRANSCRIPT_ARTIFACT_FIELDS = ("transcript", "txt", "raw_payload", "text")
TRANSCRIPT_TEXT_FIELDS = ("transcript", "raw_payload", "text")


# align_fn(audio_path, line_records, audio_info, config, alignment_dir)
#   -> (list[LineAlignment], debug_dict)   (defaults to wavesplit.alignment.align_transcript)
AlignFn = Callable[..., Any]


def _resolve_packet(envelope: Dict[str, Any]):
    input_map = envelope.get("input") or {}
    for role in PACKET_ROLES:
        if role in input_map:
            return role_fields(envelope, role), role_artifacts(envelope, role)
    if input_map:
        role = next(iter(input_map))
        return role_fields(envelope, role), role_artifacts(envelope, role)
    return {}, {}


def _artifact_id(artifacts: Dict[str, Dict[str, Any]], fields) -> Optional[str]:
    for field in fields:
        ref = artifacts.get(field)
        if isinstance(ref, dict) and ref.get("artifactId"):
            return str(ref["artifactId"])
    return None


def _read_transcript_lines(
    fields: Dict[str, Any],
    packet_artifacts: Dict[str, Dict[str, Any]],
    envelope: Dict[str, Any],
    artifacts: GatewayArtifacts,
) -> List[str]:
    """Resolve transcript text (artifact or inline field) into raw lines."""
    text: Optional[str] = None
    txt_id = _artifact_id(packet_artifacts, TRANSCRIPT_ARTIFACT_FIELDS)
    if txt_id:
        value = artifacts.artifact_get(txt_id)
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            if value.get("encoding") == "base64":
                text = decode_audio_value(value).decode("utf-8", "replace")
            else:
                text = value.get("text") or value.get("value") or value.get("raw")
    if text is None:
        for key in TRANSCRIPT_TEXT_FIELDS:
            if fields.get(key):
                text = str(fields[key])
                break
    if text is None:
        ext = envelope.get("externalInput")
        if isinstance(ext, str):
            text = ext
        elif isinstance(ext, dict):
            for key in TRANSCRIPT_TEXT_FIELDS:
                if ext.get(key):
                    text = str(ext[key])
                    break
    if text is None:
        raise ValueError("Alignment: no transcript artifact or inline text on the packet input")
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    if not lines:
        raise ValueError("Alignment: transcript has no non-empty lines")
    return lines


def align_and_cut(
    *,
    audio_path: str,
    lines: List[str],
    clips_dir: str,
    config: AppConfig,
    align_fn: Optional[AlignFn] = None,
) -> List[Dict[str, Any]]:
    """Run align → pad → cut for one long WAV + its transcript lines.

    Returns one record per line:
      { line_index, text, output_file, start_sec, end_sec, score, flags, missing }
    A ``missing`` record has no cut clip (alignment dropped it). ``align_fn`` is
    injectable for testing; defaults to :func:`wavesplit.alignment.align_transcript`.
    """
    # Lazy imports keep this module importable without numpy / ffmpeg.
    from wavesplit.audio import cut_audio_clip, probe_audio
    from wavesplit.naming import assign_output_names
    from wavesplit.text import build_line_manifest

    aligner = align_fn
    if aligner is None:
        from wavesplit.alignment import align_transcript

        aligner = align_transcript

    audio_info = probe_audio(audio_path)
    line_records = build_line_manifest(lines)

    alignment_dir = str(Path(clips_dir) / "_alignment")
    Path(alignment_dir).mkdir(parents=True, exist_ok=True)
    alignments, _debug = aligner(audio_path, line_records, audio_info, config, alignment_dir)

    output_names = assign_output_names(line_records)
    records: List[Dict[str, Any]] = []
    for alignment in alignments:
        line_index = alignment.line_index
        assigned_name, _dup = output_names[line_index]
        missing = (
            "missing_audio_segment" in alignment.flags
            or alignment.start_sec is None
            or alignment.end_sec is None
        )
        record: Dict[str, Any] = {
            "line_index": line_index,
            "text": alignment.original_text,
            "output_file": None,
            "start_sec": None if missing else round(float(alignment.start_sec), 3),
            "end_sec": None if missing else round(float(alignment.end_sec), 3),
            "score": alignment.alignment_score_mean,
            "flags": list(alignment.flags),
            "missing": missing,
        }
        if not missing:
            out_path = Path(clips_dir) / assigned_name
            cut_audio_clip(audio_path, out_path, float(alignment.start_sec), float(alignment.end_sec))
            record["output_file"] = assigned_name
        records.append(record)
    return records


def make_handler(config: AppConfig, *, align_fn: Optional[AlignFn] = None):
    """Build a run-loop handler bound to an app config (+ optional injected aligner)."""

    def handler(
        envelope: Dict[str, Any],
        jobs: GatewayJobQueue,
        artifacts: GatewayArtifacts,
    ) -> Dict[str, Any]:
        fields, packet_artifacts = _resolve_packet(envelope)
        audio_id = _artifact_id(packet_artifacts, AUDIO_ARTIFACT_FIELDS)
        if not audio_id:
            # explicit id field fallback.
            for key in ("audio_artifact_id", "long_audio_id"):
                if fields.get(key):
                    audio_id = str(fields[key])
                    break
        if not audio_id:
            raise ValueError("Alignment: no long-audio artifact on the packet input")

        audio_value = artifacts.artifact_get(audio_id)
        if audio_value is None:
            raise ValueError("Alignment: audio artifact {} not found".format(audio_id))
        audio_bytes = decode_audio_value(audio_value)
        lines = _read_transcript_lines(fields, packet_artifacts, envelope, artifacts)

        job_id = envelope.get("jobId")
        with tempfile.TemporaryDirectory(prefix="wavesplit-align-") as tmp:
            audio_path = str(Path(tmp) / "source.wav")
            with open(audio_path, "wb") as fh:
                fh.write(audio_bytes)
            clips_dir = str(Path(tmp) / "clips")
            Path(clips_dir).mkdir(parents=True, exist_ok=True)

            records = align_and_cut(
                audio_path=audio_path,
                lines=lines,
                clips_dir=clips_dir,
                config=config,
                align_fn=align_fn,
            )

            manifest: List[Dict[str, Any]] = []
            clip_count = 0
            total = len(records) or 1
            for index, record in enumerate(records, start=1):
                artifact_id: Optional[str] = None
                if record["output_file"] and not record["missing"]:
                    clip_path = Path(clips_dir) / record["output_file"]
                    with open(clip_path, "rb") as fh:
                        ref = artifacts.put_bytes(fh.read(), content_type="audio/wav")
                    artifact_id = ref["artifactId"]
                    clip_count += 1
                manifest.append(
                    {
                        "line_index": record["line_index"],
                        "text": record["text"],
                        "artifact_id": artifact_id,
                        "start_sec": record["start_sec"],
                        "end_sec": record["end_sec"],
                        "score": record["score"],
                        "flags": record["flags"],
                        "missing": record["missing"],
                    }
                )
                if job_id:
                    try:
                        jobs.job_progress(job_id, index / total)
                    except Exception:  # noqa: BLE001 - progress is best-effort
                        pass

        LOGGER.info("align-audio lines=%d clips=%d", len(manifest), clip_count)
        return {
            "output": {
                "manifest": manifest,
                "line_count": len(manifest),
                "clip_count": clip_count,
            }
        }

    return handler


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    env = WorkerEnv(DEFAULT_QUEUE)
    app_config = load_config(os.environ.get("WAVESPLIT_CONFIG"))
    handler = make_handler(app_config)
    jobs = env.job_queue()
    artifacts = env.artifacts()
    LOGGER.info("Alignment worker serving queue %r at %s", env.queue, env.gateway_url)
    run_worker_loop(
        env.queue,
        handler,
        jobs=jobs,
        artifacts=artifacts,
        idle_delay=env.poll_interval,
    )


if __name__ == "__main__":
    main()
