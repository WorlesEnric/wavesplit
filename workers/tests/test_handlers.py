"""Handler tests with INJECTED transcriber/analyzer/aligner — no ML deps.

These import ``score_clip`` / ``build_line_manifest`` / ``assign_output_names``
from wavesplit (pure-python; no numpy/torch at import) and stub the audio I/O
(ffmpeg) + ASR + alignment so the worker handlers run end-to-end against the
FakeGateway from test_gateway_client.
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from workers.gateway_client import GatewayArtifacts, GatewayJobQueue  # noqa: E402
from workers.tests.test_gateway_client import FakeGateway  # noqa: E402


def _clients():
    gw = FakeGateway()
    return gw, GatewayJobQueue("http://gw", "t", session=gw), GatewayArtifacts("http://gw", "t", session=gw)


# ── QC handler ─────────────────────────────────────────────────────────────────


def test_qc_handler_scores_single_clip_with_injected_deps():
    from wavesplit.config import QAConfig
    from workers import qc_worker

    gw, jobs, artifacts = _clients()
    # stage a clip artifact the way the alignment worker would.
    clip_ref = artifacts.put_bytes(b"RIFF....fake-clip", "audio/wav")

    fake_metrics = {
        "duration_sec": 1.2,
        "peak_dbfs": -3.0,
        "rms_dbfs": -20.0,
        "leading_silence_ms": 0,
        "trailing_silence_ms": 0,
        "start_energy_flag": False,
        "end_energy_flag": False,
    }

    class FakeTranscriber:
        def transcribe(self, path):
            return "hello world"

    handler = qc_worker.make_handler(
        QAConfig(enabled=True, asr_engine="faster_whisper"),
        transcriber=FakeTranscriber(),
        energy_analyzer=lambda path: fake_metrics,
    )

    envelope = {
        "jobId": "j1",
        "process": "AutoQcCollected",
        "input": {"point": {"text": "hello world", "_artifacts": {"audio": {"artifactId": clip_ref["artifactId"]}}}},
    }
    result = handler(envelope, jobs, artifacts)
    out = result["output"]
    assert out["passed"] is True
    assert 0.0 <= out["quality_score"] <= 1.0
    assert out["flags"] == []


def test_qc_handler_flags_text_mismatch_as_not_passed():
    from wavesplit.config import QAConfig
    from workers import qc_worker

    gw, jobs, artifacts = _clients()
    clip_ref = artifacts.put_bytes(b"clip", "audio/wav")
    fake_metrics = {
        "duration_sec": 1.0,
        "peak_dbfs": -3.0,
        "rms_dbfs": -20.0,
        "leading_silence_ms": 0,
        "trailing_silence_ms": 0,
        "start_energy_flag": False,
        "end_energy_flag": False,
    }

    class WrongTranscriber:
        def transcribe(self, path):
            return "totally different words here nothing matches"

    handler = qc_worker.make_handler(
        QAConfig(enabled=True, asr_engine="faster_whisper"),
        transcriber=WrongTranscriber(),
        energy_analyzer=lambda path: fake_metrics,
    )
    envelope = {
        "jobId": "j2",
        "input": {"point": {"text": "hello world", "_artifacts": {"audio": {"artifactId": clip_ref["artifactId"]}}}},
    }
    out = handler(envelope, jobs, artifacts)["output"]
    assert out["passed"] is False
    assert any("text_mismatch" in f for f in out["flags"])


# ── alignment handler ───────────────────────────────────────────────────────────


def _install_fake_audio_module(monkeypatch_cut):
    """Inject a fake wavesplit.audio so probe_audio/cut_audio_clip don't shell out."""
    import wavesplit.audio as audio_mod
    from wavesplit.models import AudioInfo

    def fake_probe(path):
        return AudioInfo(codec_name="pcm_s16le", sample_rate=16000, channels=1, duration_sec=3.0)

    def fake_cut(input_path, output_path, start, end):
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"CLIP[%0.2f-%0.2f]" % (start, end))
        return ""

    audio_mod.probe_audio = fake_probe  # type: ignore[assignment]
    audio_mod.cut_audio_clip = fake_cut  # type: ignore[assignment]


def test_alignment_handler_aligns_cuts_and_uploads():
    from wavesplit.config import AppConfig
    from wavesplit.models import LineAlignment
    from workers import alignment_worker

    _install_fake_audio_module(None)

    gw, jobs, artifacts = _clients()
    audio_ref = artifacts.put_bytes(b"LONG-WAV-BYTES", "audio/wav")

    # fake aligner: one alignment per line, evenly spaced; last line "missing".
    def fake_align(audio_path, line_records, audio_info, config, alignment_dir):
        aligns = []
        for i, lr in enumerate(line_records):
            missing = i == len(line_records) - 1
            aligns.append(
                LineAlignment(
                    clip_id="clip-%06d" % lr.line_index,
                    line_index=lr.line_index,
                    original_text=lr.original_text,
                    normalized_text=lr.normalized_text,
                    raw_start_sec=None if missing else float(i),
                    raw_end_sec=None if missing else float(i) + 0.8,
                    start_sec=None if missing else float(i),
                    end_sec=None if missing else float(i) + 0.8,
                    alignment_score_mean=None if missing else 95.0,
                    flags=["missing_audio_segment"] if missing else [],
                )
            )
        return aligns, {"engine": "fake"}

    handler = alignment_worker.make_handler(AppConfig(), align_fn=fake_align)
    envelope = {
        "jobId": "ja",
        "process": "StandardizePacket",
        "input": {
            "packet": {
                "transcript": "first line\nsecond line\nthird line",
                "_artifacts": {"audio": {"artifactId": audio_ref["artifactId"]}},
            }
        },
    }
    out = handler(envelope, jobs, artifacts)["output"]
    assert out["line_count"] == 3
    assert out["clip_count"] == 2  # third is missing
    manifest = out["manifest"]
    assert manifest[0]["artifact_id"] and manifest[0]["start_sec"] == 0.0
    assert manifest[2]["artifact_id"] is None and manifest[2]["missing"] is True
    # uploaded clips are retrievable as bytes.
    assert artifacts.get_bytes(manifest[0]["artifact_id"]).startswith(b"CLIP")
