"""QC worker — serves the ``auto-qc`` queue.

Replaces the TS stub ``autoQcCollectedHandler`` (which returns
``{passed:true, quality_score:0.95}``). For each leased ``AutoQcCollected`` job
it reads the SINGLE audio clip artifact attached to the data point + the
expected text, runs wavesplit's automated QC on that one clip (NO splitting —
the app already stores clips individually), and completes with::

    { passed: bool, quality_score: float, flags: [str, ...] }

QC reuses :func:`wavesplit.qa.score_clip` (ASR similarity / WER, energy /
silence, duration) and :class:`wavesplit.qa.FasterWhisperTranscriber`. The
transcriber and the energy analyzer are dependency-injected so the handler is
unit-testable WITHOUT faster-whisper / numpy / ffmpeg.

Job input contract (the assumed ``AutoQcCollected`` envelope)
-------------------------------------------------------------
``envelope.input`` carries the resolved data-point role. We look for the clip
artifact under the role's ``_artifacts`` (any of the field names in
:data:`AUDIO_ARTIFACT_FIELDS`) or, failing that, an ``audio_artifact_id`` /
``clip_artifact_id`` field or ``inputArtifacts``. The expected text comes from
the role fields (any of :data:`EXPECTED_TEXT_FIELDS`) or ``externalInput``.
See ``workers/README.md`` for the exact field names to confirm with the DSL.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from wavesplit.config import QAConfig, load_config

from .gateway_client import (
    GatewayArtifacts,
    GatewayJobQueue,
    WorkerEnv,
    decode_audio_value,
    role_artifacts,
    role_fields,
    run_worker_loop,
)


LOGGER = logging.getLogger("wavesplit.workers.qc")

DEFAULT_QUEUE = "auto-qc"

# Role aliases the AutoQcCollected envelope might key the data point under.
POINT_ROLES = ("point", "datapoint", "data_point", "dataPoint", "target", "self")
# Artifact field names that may hold the collected clip.
AUDIO_ARTIFACT_FIELDS = ("audio", "clip", "audio_clip", "collected_audio", "recording")
# Field names that may carry the expected/target text.
EXPECTED_TEXT_FIELDS = ("text", "expected_text", "normalized_text", "original_text", "transcript")


# A transcriber is anything with ``transcribe(path) -> str``.
Transcriber = Any
# An energy analyzer is ``analyze(path) -> metrics dict``.
EnergyAnalyzer = Callable[[Any], Dict[str, Any]]


def _first(d: Dict[str, Any], keys, default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def _resolve_point(envelope: Dict[str, Any]):
    """Return (fields, artifacts) for whichever role holds the data point."""
    input_map = envelope.get("input") or {}
    for role in POINT_ROLES:
        if role in input_map:
            return role_fields(envelope, role), role_artifacts(envelope, role)
    # fall back to the first/only role in the input map.
    if input_map:
        role = next(iter(input_map))
        return role_fields(envelope, role), role_artifacts(envelope, role)
    return {}, {}


def _find_clip_artifact_id(
    fields: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
    envelope: Dict[str, Any],
) -> Optional[str]:
    for field in AUDIO_ARTIFACT_FIELDS:
        ref = artifacts.get(field)
        if isinstance(ref, dict) and ref.get("artifactId"):
            return str(ref["artifactId"])
    # explicit id fields on the resolved input.
    for key in ("audio_artifact_id", "clip_artifact_id", "audio_id"):
        if fields.get(key):
            return str(fields[key])
    # last resort: the single entry of inputArtifacts ({artifactId: contentType}).
    input_artifacts = envelope.get("inputArtifacts") or {}
    if len(input_artifacts) == 1:
        return next(iter(input_artifacts))
    return None


def score_single_clip(
    *,
    clip_path: str,
    expected_text: str,
    config: QAConfig,
    transcriber: Optional[Transcriber] = None,
    energy_analyzer: Optional[EnergyAnalyzer] = None,
) -> Dict[str, Any]:
    """Run wavesplit QC on ONE clip against its expected text.

    A thin adapter over :func:`wavesplit.qa.score_clip` for the single-clip case
    (no manifest, no same-text median). Returns the raw ``score_clip`` dict
    enriched with ``passed`` / ``quality_score`` for the gateway contract.

    ``transcriber`` / ``energy_analyzer`` are injectable for testing; in
    production they default to the real FasterWhisper + ffmpeg-backed analyzer.
    """
    # Imported lazily so the module imports without numpy/ffmpeg present.
    from wavesplit.qa import score_clip

    analyzer = energy_analyzer
    if analyzer is None:
        from wavesplit.audio import analyze_clip_energy

        analyzer = analyze_clip_energy

    metrics = analyzer(clip_path)
    duration_sec = float(metrics.get("duration_sec") or 0.0)

    asr_text = ""
    asr_error: Optional[str] = None
    if transcriber is not None:
        try:
            asr_text = transcriber.transcribe(clip_path)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            asr_error = str(exc)
    elif config.enabled and config.asr_engine == "faster_whisper":
        # No injected transcriber and ASR is on → try to build the real one.
        from wavesplit.qa import FasterWhisperTranscriber

        try:
            transcriber = FasterWhisperTranscriber(config)
            asr_text = transcriber.transcribe(clip_path)
        except Exception as exc:  # pragma: no cover - depends on local model
            asr_error = str(exc)
    else:
        asr_error = "ASR QA disabled by configuration."

    score = score_clip(
        normalized_text=expected_text,
        asr_text=asr_text,
        duration_sec=duration_sec,
        metrics=metrics,
        config=config,
        base_flags=[],
        same_text_median=None,
        asr_error=asr_error,
    )
    # status is one of pass / review / fail. "review" is borderline; treat only
    # "pass" as auto-passed so the DSL routes review/fail back for re-collect.
    status = str(score.get("status"))
    passed = status == "pass"
    quality_score = float(score.get("confidence", 0)) / 100.0
    flags: List[str] = list(score.get("flags") or [])
    return {
        "passed": passed,
        "quality_score": round(quality_score, 4),
        "status": status,
        "flags": flags,
        "asr_text": asr_text,
        "similarity": score.get("similarity"),
        "wer": score.get("wer"),
    }


def make_handler(
    config: QAConfig,
    *,
    transcriber: Optional[Transcriber] = None,
    energy_analyzer: Optional[EnergyAnalyzer] = None,
):
    """Build a run-loop handler bound to a QA config (+ optional injected deps)."""

    def handler(
        envelope: Dict[str, Any],
        jobs: GatewayJobQueue,
        artifacts: GatewayArtifacts,
    ) -> Dict[str, Any]:
        fields, point_artifacts = _resolve_point(envelope)
        expected_text = _first(fields, EXPECTED_TEXT_FIELDS, default="")
        if not expected_text:
            ext = envelope.get("externalInput")
            if isinstance(ext, dict):
                expected_text = _first(ext, EXPECTED_TEXT_FIELDS, default="")
        artifact_id = _find_clip_artifact_id(fields, point_artifacts, envelope)
        if not artifact_id:
            raise ValueError("AutoQc: no audio clip artifact on the data-point input")

        value = artifacts.artifact_get(artifact_id)
        if value is None:
            raise ValueError("AutoQc: clip artifact {} not found".format(artifact_id))
        clip_bytes = decode_audio_value(value)

        with tempfile.TemporaryDirectory(prefix="wavesplit-qc-") as tmp:
            clip_path = str(Path(tmp) / "clip.wav")
            with open(clip_path, "wb") as fh:
                fh.write(clip_bytes)
            result = score_single_clip(
                clip_path=clip_path,
                expected_text=str(expected_text),
                config=config,
                transcriber=transcriber,
                energy_analyzer=energy_analyzer,
            )
        LOGGER.info(
            "auto-qc clip=%s passed=%s score=%.3f flags=%s",
            artifact_id,
            result["passed"],
            result["quality_score"],
            result["flags"],
        )
        return {
            "output": {
                "passed": result["passed"],
                "quality_score": result["quality_score"],
                "flags": result["flags"],
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
    handler = make_handler(app_config.qa)
    jobs = env.job_queue()
    artifacts = env.artifacts()
    LOGGER.info("QC worker serving queue %r at %s", env.queue, env.gateway_url)
    run_worker_loop(
        env.queue,
        handler,
        jobs=jobs,
        artifacts=artifacts,
        idle_delay=env.poll_interval,
    )


if __name__ == "__main__":
    main()
