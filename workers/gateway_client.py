"""HTTP client for the EchoThink gateway's job + artifact surface.

Python mirror of the TypeScript reference transport
(``echothink-worker/src/gateway-transport.ts`` + the run loop in
``echothink-worker/src/run.ts``). It implements:

* :class:`GatewayJobQueue` — ``GatewayJobQueue`` over ``/jobs/*``:
  ``POST /jobs/lease``, ``/jobs/{id}/progress``, ``/jobs/{id}/heartbeat``,
  ``/jobs/{id}/complete``, ``/jobs/{id}/fail`` (plus create / get / cancel).
* :class:`GatewayArtifacts` — ``GatewayArtifacts`` over ``/artifact*``:
  ``GET /artifact/{id}`` → ``{ value }`` and ``POST /artifact`` →
  ``{ artifactId, digest }``.
* :func:`run_worker_loop` — the lease → run handler → complete/fail loop.

Every request carries ``Authorization: Bearer <service-account token>`` so the
gateway authorizes the pull path as the worker identity.

The wire contract (the leased ``Job.payload`` is a ``WorkerInvocationEnvelope``;
``markSucceeded(output)`` carries staged output-artifact ids under the magic
``__outputArtifacts`` key) is defined in
``echothink-sdk/src/worker/wire.ts`` / ``worker/index.ts`` and reproduced by the
helpers below.

Binary audio over the JSON ``/artifact`` API
--------------------------------------------
The gateway's ``/artifact`` endpoint stores an arbitrary JSON ``value`` (see
``echothink-gateway/src/http-server.ts``). It has no binary path today, so this
client encodes binary blobs (e.g. WAV clips) as a JSON envelope::

    {"__wavesplit_audio__": true, "encoding": "base64",
     "contentType": "audio/wav", "data": "<base64 bytes>"}

Use :func:`encode_audio_value` / :func:`decode_audio_value` (or the
:meth:`GatewayArtifacts.put_bytes` / :meth:`GatewayArtifacts.get_bytes`
helpers). Long WAVs are heavy as base64-over-JSON — see ``workers/README.md``
and REFACTOR-PLAN Phase D4 for the planned binary/chunked endpoint.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import requests


LOGGER = logging.getLogger("wavesplit.workers.gateway")

# Magic envelope marker for base64-wrapped binary artifacts (see module docstring).
AUDIO_ENVELOPE_KEY = "__wavesplit_audio__"
# Where the worker SDK stows staged output-artifact ids inside markSucceeded's
# output (see echothink-sdk/src/worker/index.ts line ~137).
OUTPUT_ARTIFACTS_KEY = "__outputArtifacts"
WARNINGS_KEY = "__warnings"


class GatewayError(RuntimeError):
    """A non-2xx (and non-404) response from the gateway."""


# ── binary audio <-> JSON value helpers ───────────────────────────────────────


def encode_audio_value(data: bytes, content_type: str = "audio/wav") -> Dict[str, Any]:
    """Wrap raw audio bytes into the base64 JSON envelope the /artifact API accepts."""
    return {
        AUDIO_ENVELOPE_KEY: True,
        "encoding": "base64",
        "contentType": content_type,
        "data": base64.b64encode(data).decode("ascii"),
    }


def decode_audio_value(value: Any) -> bytes:
    """Decode a value produced by :func:`encode_audio_value` (or a bare base64 str).

    Accepts:
      * the ``{__wavesplit_audio__: true, data: <b64>}`` envelope,
      * a bare base64 string, or
      * the generic ``{encoding: "base64", data: <b64>}`` shape.
    """
    if isinstance(value, str):
        return base64.b64decode(value)
    if isinstance(value, dict):
        if value.get("encoding") == "base64" and "data" in value:
            return base64.b64decode(value["data"])
        # tolerate a nested {value: ...} wrapper.
        if "data" in value:
            return base64.b64decode(value["data"])
    raise ValueError("artifact value is not a recognized base64 audio envelope")


# ── low-level HTTP ─────────────────────────────────────────────────────────────


class _GatewayHttp:
    def __init__(
        self,
        base_url: str,
        bearer: Optional[str] = None,
        *,
        timeout: float = 120.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base = base_url.rstrip("/")
        self.bearer = bearer
        self.timeout = timeout
        self.session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.bearer:
            headers["authorization"] = "Bearer " + self.bearer
        return headers

    def get(self, path: str) -> Optional[Any]:
        resp = self.session.get(self.base + path, headers=self._headers(), timeout=self.timeout)
        if resp.status_code == 404:
            return None
        if not resp.ok:
            raise GatewayError("GET {} -> {}: {}".format(path, resp.status_code, resp.text[:500]))
        return resp.json() if resp.text else None

    def post(self, path: str, body: Optional[Any] = None) -> Optional[Any]:
        resp = self.session.post(
            self.base + path,
            headers=self._headers(),
            json=body if body is not None else {},
            timeout=self.timeout,
        )
        if not resp.ok:
            raise GatewayError("POST {} -> {}: {}".format(path, resp.status_code, resp.text[:500]))
        return resp.json() if resp.text else None


# ── job queue ──────────────────────────────────────────────────────────────────


class GatewayJobQueue:
    """Durable job queue over the gateway ``/jobs/*`` endpoints."""

    def __init__(
        self,
        base_url: str,
        bearer: Optional[str] = None,
        *,
        timeout: float = 120.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.http = _GatewayHttp(base_url, bearer, timeout=timeout, session=session)

    def lease_job(self, queue: str) -> Optional[Dict[str, Any]]:
        """POST /jobs/lease — claim the next queued job for ``queue`` (or None)."""
        result = self.http.post("/jobs/lease", {"queue": queue})
        return result or None

    def job_progress(self, job_id: str, progress: float) -> Optional[Dict[str, Any]]:
        return self.http.post("/jobs/{}/progress".format(_enc(job_id)), {"progress": progress})

    def job_heartbeat(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.http.post("/jobs/{}/heartbeat".format(_enc(job_id)), {})

    def mark_succeeded(
        self,
        job_id: str,
        output: Optional[Dict[str, Any]] = None,
        *,
        output_artifacts: Optional[Dict[str, str]] = None,
        warnings: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """POST /jobs/{id}/complete.

        ``output_artifacts`` (result-field -> staged artifactId) is folded into
        the durable output under the ``__outputArtifacts`` key, exactly as the
        TS worker loop does (echothink-sdk/src/worker/index.ts).
        """
        payload: Dict[str, Any] = dict(output or {})
        if output_artifacts:
            payload[OUTPUT_ARTIFACTS_KEY] = output_artifacts
        if warnings:
            payload[WARNINGS_KEY] = warnings
        return self.http.post("/jobs/{}/complete".format(_enc(job_id)), {"output": payload})

    def mark_failed(self, job_id: str, error: str) -> Optional[Dict[str, Any]]:
        return self.http.post("/jobs/{}/fail".format(_enc(job_id)), {"error": error})

    def mark_cancelled(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.http.post("/jobs/{}/cancel".format(_enc(job_id)), {})

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.http.get("/jobs/{}".format(_enc(job_id)))

    def create_job(self, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.http.post("/jobs", args)


# ── artifacts ────────────────────────────────────────────────────────────────


class GatewayArtifacts:
    """Artifact reads/writes over the gateway ``/artifact*`` endpoints."""

    def __init__(
        self,
        base_url: str,
        bearer: Optional[str] = None,
        *,
        timeout: float = 300.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.http = _GatewayHttp(base_url, bearer, timeout=timeout, session=session)

    def artifact_get(self, artifact_id: str) -> Optional[Any]:
        """GET /artifact/{id} → the stored ``value`` (or None on 404)."""
        result = self.http.get("/artifact/{}".format(_enc(artifact_id)))
        if result is None:
            return None
        # endpoint wraps as { value } so null/false round-trips.
        if isinstance(result, dict) and "value" in result:
            return result["value"]
        return result

    def artifact_put(self, value: Any, content_type: Optional[str] = None) -> Dict[str, Any]:
        """POST /artifact → { artifactId, digest }."""
        body: Dict[str, Any] = {"value": value}
        if content_type is not None:
            body["contentType"] = content_type
        result = self.http.post("/artifact", body)
        if not isinstance(result, dict) or "artifactId" not in result:
            raise GatewayError("POST /artifact returned no artifactId: {!r}".format(result))
        return result

    # convenience binary helpers (base64 JSON envelope) -------------------------

    def put_bytes(self, data: bytes, content_type: str = "audio/wav") -> Dict[str, Any]:
        return self.artifact_put(encode_audio_value(data, content_type), content_type=content_type)

    def get_bytes(self, artifact_id: str) -> bytes:
        value = self.artifact_get(artifact_id)
        if value is None:
            raise GatewayError("artifact {} not found".format(artifact_id))
        return decode_audio_value(value)

    def has(self, artifact_id: str) -> bool:
        return self.artifact_get(artifact_id) is not None


def _enc(value: str) -> str:
    from urllib.parse import quote

    return quote(str(value), safe="")


# ── envelope helpers ───────────────────────────────────────────────────────────


def envelope_of(job: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the WorkerInvocationEnvelope from a leased ``Job``.

    The durable ``Job.payload`` is the envelope (process / invocationId / input /
    externalInput / inputArtifacts / ...). Mirrors envelopeOf() in
    echothink-sdk/src/worker/index.ts; falls back to top-level Job fields.
    """
    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "process": payload.get("process") or job.get("process"),
        "invocationId": payload.get("invocationId") or job.get("invocationId") or "",
        "jobId": job.get("jobId"),
        "queue": payload.get("queue") or job.get("queue"),
        "actor": payload.get("actor"),
        "input": payload.get("input") or {},
        "externalInput": payload.get("externalInput"),
        "inputArtifacts": payload.get("inputArtifacts") or {},
        "deadline": payload.get("deadline"),
        "traceContext": payload.get("traceContext"),
    }


def role_fields(envelope: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Read a resolved input role's flat fields.

    The runtime materializes a role as flat fields (+ ``_state`` / ``_artifacts``).
    Mirrors the TS ``packetInput`` helper.
    """
    raw = (envelope.get("input") or {}).get(role) or {}
    if not isinstance(raw, dict):
        return {}
    fields = raw.get("fields") if isinstance(raw.get("fields"), dict) else raw
    return fields or {}


def role_artifacts(envelope: Dict[str, Any], role: str) -> Dict[str, Dict[str, Any]]:
    """Read a resolved input role's attached artifacts map ``{field: {artifactId}}``."""
    raw = (envelope.get("input") or {}).get(role) or {}
    if not isinstance(raw, dict):
        return {}
    artifacts = raw.get("_artifacts") or {}
    return artifacts if isinstance(artifacts, dict) else {}


# ── generic run loop ───────────────────────────────────────────────────────────

# A handler receives (envelope, jobs, artifacts) and returns a result dict:
#   { "output": {...}, "output_artifacts": {field: artifactId}, "warnings": [...] }
# Any raised exception → markFailed.
WorkerHandler = Callable[
    [Dict[str, Any], "GatewayJobQueue", "GatewayArtifacts"],
    Optional[Dict[str, Any]],
]


def run_worker_loop(
    queue: str,
    handler: WorkerHandler,
    *,
    jobs: GatewayJobQueue,
    artifacts: GatewayArtifacts,
    idle_delay: float = 1.0,
    max_iterations: float = float("inf"),
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, int]:
    """Lease one job at a time from ``queue``, run ``handler``, complete/fail, loop.

    Returns a small report ``{leased, succeeded, failed}``. ``max_iterations`` is
    the number of LEASED jobs to process (``inf`` = run forever, finite = drain /
    test mode). ``idle_delay`` seconds are slept on an empty lease.
    """
    report = {"leased": 0, "succeeded": 0, "failed": 0}
    iterations = 0
    while iterations < max_iterations:
        try:
            job = jobs.lease_job(queue)
        except GatewayError as exc:
            LOGGER.warning("lease error on queue %s: %s", queue, exc)
            sleep(idle_delay)
            continue
        if not job:
            if max_iterations == float("inf"):
                sleep(idle_delay)
                continue
            break

        report["leased"] += 1
        iterations += 1
        job_id = job.get("jobId")
        envelope = envelope_of(job)
        try:
            result = handler(envelope, jobs, artifacts) or {}
            jobs.mark_succeeded(
                job_id,
                result.get("output") or {},
                output_artifacts=result.get("output_artifacts"),
                warnings=result.get("warnings"),
            )
            report["succeeded"] += 1
            LOGGER.info("job %s (%s) succeeded", job_id, envelope.get("process"))
        except Exception as exc:  # noqa: BLE001 - report all handler errors to gateway
            error = str(exc) or exc.__class__.__name__
            try:
                jobs.mark_failed(job_id, error)
            except GatewayError as fail_exc:
                LOGGER.error("could not mark job %s failed: %s", job_id, fail_exc)
            report["failed"] += 1
            LOGGER.exception("job %s (%s) failed: %s", job_id, envelope.get("process"), error)
    return report


# ── config from env ────────────────────────────────────────────────────────────


class WorkerEnv:
    """Worker connection config read from the environment.

    GATEWAY_URL    — gateway base url (default http://localhost:4500)
    WORKER_BEARER  — service-account session bearer (Authorization: Bearer ...)
    QUEUE          — queue to serve (overrides the per-worker default)
    POLL_INTERVAL  — idle poll seconds between empty leases (default 1.0)
    """

    def __init__(self, default_queue: str) -> None:
        self.gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4500")
        self.bearer = os.environ.get("WORKER_BEARER") or None
        self.queue = os.environ.get("QUEUE", default_queue)
        self.poll_interval = float(os.environ.get("POLL_INTERVAL", "1.0"))

    def job_queue(self, session: Optional[requests.Session] = None) -> GatewayJobQueue:
        return GatewayJobQueue(self.gateway_url, self.bearer, session=session)

    def artifacts(self, session: Optional[requests.Session] = None) -> GatewayArtifacts:
        return GatewayArtifacts(self.gateway_url, self.bearer, session=session)
