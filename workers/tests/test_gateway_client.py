"""Unit tests for the gateway client + run loop — NO ML deps required.

These exercise the HTTP protocol shape, the base64 audio envelope, the envelope
helpers, and the run-loop lease/complete/fail behavior using a fake in-memory
gateway. faster-whisper / torch / numpy / ffmpeg are NOT imported here.
"""

from __future__ import annotations

import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from workers.gateway_client import (  # noqa: E402
    AUDIO_ENVELOPE_KEY,
    OUTPUT_ARTIFACTS_KEY,
    GatewayArtifacts,
    GatewayJobQueue,
    decode_audio_value,
    encode_audio_value,
    envelope_of,
    role_artifacts,
    role_fields,
    run_worker_loop,
)


# ── fake gateway over the requests.Session surface ────────────────────────────


class _FakeResponse:
    def __init__(self, status_code, payload, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else ("" if payload is None else "{}")

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class FakeGateway:
    """Minimal in-memory gateway speaking /jobs/* and /artifact*."""

    def __init__(self):
        self.queues = {}  # queue -> list[job]
        self.jobs = {}
        self.artifacts = {}
        self.completed = {}
        self.failed = {}
        self.progress = []
        self._artifact_seq = 0

    def enqueue(self, job):
        self.jobs[job["jobId"]] = job
        self.queues.setdefault(job["queue"], []).append(job)

    # the requests.Session interface the client uses.
    def get(self, url, headers=None, timeout=None):
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        path = "/" + path
        if path.startswith("/artifact/"):
            aid = path[len("/artifact/"):]
            from urllib.parse import unquote

            aid = unquote(aid)
            if aid not in self.artifacts:
                return _FakeResponse(404, None, text="")
            return _FakeResponse(200, {"value": self.artifacts[aid]})
        return _FakeResponse(404, None, text="")

    def post(self, url, headers=None, json=None, timeout=None):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        if path == "/jobs/lease":
            queue = (json or {}).get("queue")
            pending = self.queues.get(queue, [])
            if not pending:
                return _FakeResponse(200, None, text="")
            return _FakeResponse(200, pending.pop(0))
        if path == "/artifact":
            self._artifact_seq += 1
            aid = "art-{}".format(self._artifact_seq)
            self.artifacts[aid] = (json or {})["value"]
            return _FakeResponse(200, {"artifactId": aid, "digest": "sha256:fake"})
        if path.endswith("/complete"):
            job_id = path.split("/")[2]
            self.completed[job_id] = (json or {}).get("output")
            return _FakeResponse(200, {})
        if path.endswith("/fail"):
            job_id = path.split("/")[2]
            self.failed[job_id] = (json or {}).get("error")
            return _FakeResponse(200, {})
        if path.endswith("/progress"):
            self.progress.append((path.split("/")[2], (json or {}).get("progress")))
            return _FakeResponse(200, {})
        if path.endswith("/heartbeat"):
            return _FakeResponse(200, {})
        return _FakeResponse(404, None, text="")


def _clients():
    gw = FakeGateway()
    jobs = GatewayJobQueue("http://gw", "tok", session=gw)
    artifacts = GatewayArtifacts("http://gw", "tok", session=gw)
    return gw, jobs, artifacts


# ── tests ──────────────────────────────────────────────────────────────────────


def test_audio_envelope_roundtrip():
    raw = b"\x00\x01RIFFfake-wav-bytes\xff"
    value = encode_audio_value(raw, "audio/wav")
    assert value[AUDIO_ENVELOPE_KEY] is True
    assert value["encoding"] == "base64"
    assert base64.b64decode(value["data"]) == raw
    assert decode_audio_value(value) == raw
    # bare base64 string is also accepted.
    assert decode_audio_value(base64.b64encode(raw).decode()) == raw


def test_artifact_put_get_bytes():
    gw, _jobs, artifacts = _clients()
    ref = artifacts.put_bytes(b"hello-wav", "audio/wav")
    assert ref["artifactId"]
    assert artifacts.get_bytes(ref["artifactId"]) == b"hello-wav"
    assert artifacts.has(ref["artifactId"]) is True
    assert artifacts.has("missing") is False


def test_envelope_helpers():
    job = {
        "jobId": "j1",
        "process": "AutoQcCollected",
        "queue": "auto-qc",
        "payload": {
            "process": "AutoQcCollected",
            "invocationId": "inv-1",
            "input": {
                "point": {
                    "text": "hello world",
                    "_artifacts": {"audio": {"artifactId": "art-9", "digest": "d"}},
                }
            },
            "externalInput": {"foo": "bar"},
            "inputArtifacts": {"art-9": "audio/wav"},
        },
    }
    env = envelope_of(job)
    assert env["process"] == "AutoQcCollected"
    assert env["invocationId"] == "inv-1"
    assert role_fields(env, "point")["text"] == "hello world"
    assert role_artifacts(env, "point")["audio"]["artifactId"] == "art-9"


def test_run_loop_success_carries_output_artifacts():
    gw, jobs, artifacts = _clients()
    gw.enqueue({"jobId": "j1", "process": "X", "queue": "q", "payload": {"process": "X", "input": {}}})

    def handler(envelope, jobs_, artifacts_):
        ref = artifacts_.put_bytes(b"clip", "audio/wav")
        return {"output": {"ok": True}, "output_artifacts": {"audio": ref["artifactId"]}}

    report = run_worker_loop("q", handler, jobs=jobs, artifacts=artifacts, max_iterations=2, idle_delay=0)
    assert report == {"leased": 1, "succeeded": 1, "failed": 0}
    out = gw.completed["j1"]
    assert out["ok"] is True
    assert out[OUTPUT_ARTIFACTS_KEY] == {"audio": "art-1"}


def test_run_loop_failure_marks_failed():
    gw, jobs, artifacts = _clients()
    gw.enqueue({"jobId": "j2", "process": "X", "queue": "q", "payload": {"process": "X", "input": {}}})

    def handler(envelope, jobs_, artifacts_):
        raise ValueError("boom")

    report = run_worker_loop("q", handler, jobs=jobs, artifacts=artifacts, max_iterations=2, idle_delay=0)
    assert report == {"leased": 1, "succeeded": 0, "failed": 1}
    assert gw.failed["j2"] == "boom"


def test_run_loop_empty_queue_drains_in_finite_mode():
    gw, jobs, artifacts = _clients()
    report = run_worker_loop(
        "q", lambda *a: {}, jobs=jobs, artifacts=artifacts, max_iterations=3, idle_delay=0
    )
    assert report == {"leased": 0, "succeeded": 0, "failed": 0}
