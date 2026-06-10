# wavesplit standalone HTTP workers

Two long-lived, pull-based workers that connect to the EchoThink gateway over
HTTP and run wavesplit's heavy ML logic out-of-process. They mirror the TypeScript
reference worker (`echothink-worker/src/gateway-transport.ts` + `run.ts`) so they
slot into the same durable Job Queue + Artifact surface — no gateway changes.

| Worker             | Queue         | Replaces                          | Heavy deps                                  |
|--------------------|---------------|-----------------------------------|---------------------------------------------|
| `qc_worker`        | `auto-qc`     | TS `autoQcCollectedHandler` stub  | faster-whisper                              |
| `alignment_worker` | `align-audio` | new `wav_txt` standardize branch  | torch / torchaudio / onnxruntime / ctc      |

Both also need the `ffmpeg`/`ffprobe` binaries on `PATH` (wavesplit shells out).

## Protocol

Each request carries `Authorization: Bearer <service-account token>`.

* Job queue (`gateway_client.GatewayJobQueue`): `POST /jobs/lease`,
  `/jobs/{id}/progress`, `/jobs/{id}/heartbeat`, `/jobs/{id}/complete`,
  `/jobs/{id}/fail`.
* Artifacts (`gateway_client.GatewayArtifacts`): `GET /artifact/{id}` → `{value}`,
  `POST /artifact {value, contentType}` → `{artifactId, digest}`.

A leased `Job.payload` is a `WorkerInvocationEnvelope` (see
`echothink-sdk/src/worker/wire.ts`): `process`, `invocationId`, `jobId`, `queue`,
`actor`, `input` (resolved roles, each with flat fields + `_artifacts`
`{field:{artifactId,digest}}`), `externalInput`, `inputArtifacts`.

`mark_succeeded(output, output_artifacts=…)` folds staged artifact ids into the
durable output under the magic key `__outputArtifacts` (exactly as the TS loop in
`echothink-sdk/src/worker/index.ts` does), so the host attaches them on commit.

### Binary audio over the JSON `/artifact` API

The gateway `/artifact` endpoint stores an arbitrary JSON `value` and has no
binary path today. This client base64-encodes binary audio into a JSON envelope:

```json
{"__wavesplit_audio__": true, "encoding": "base64",
 "contentType": "audio/wav", "data": "<base64 bytes>"}
```

`GatewayArtifacts.put_bytes(bytes)` / `get_bytes(id)` wrap/unwrap it (and
`get_bytes` also tolerates a bare base64 string or a generic `{encoding, data}`
shape). **Caveat (REFACTOR-PLAN D4):** a long WAV is tens of MB and base64-over-
JSON is ~33% heavier + buffered fully in memory. Fine for single clips and small
samples; for production long-audio volume the gateway should grow a
binary/chunked artifact path (kept app-agnostic). Until then the alignment worker
reads the whole long WAV and writes per-line clips through this JSON path.

## Job input/output contracts (ASSUMED — confirm against the DSL)

The workers probe several field-name aliases (see the constants in each module)
so they tolerate the exact DSL naming. The assumed shapes:

### `auto-qc` (`AutoQcCollected`)

* **Input**: the resolved data-point role (tried under `point`/`datapoint`/
  `target`/… or the sole input role) with:
  * a clip artifact in `_artifacts` under `audio`/`clip`/`audio_clip`/… **or** an
    `audio_artifact_id`/`clip_artifact_id` field, **or** the single entry of
    `inputArtifacts`;
  * expected text in a `text`/`expected_text`/`normalized_text`/… field (or
    `externalInput`).
* **Output**: `{ passed: bool, quality_score: float (0..1), flags: [str] }`.
  `passed` is true only when wavesplit QC status is `pass` (so `review`/`fail`
  route the point back for re-collect, matching `Collected → QcPassed | Pending`).

### `align-audio` (`StandardizePacket` `wav_txt` branch)

* **Input**: the resolved packet role (tried under `packet`/`source_packet`/
  `target`/…) with:
  * a long-WAV artifact in `_artifacts` under `audio`/`long_audio`/`wav`/… (or an
    `audio_artifact_id` field);
  * the transcript either as a TXT artifact in `_artifacts`
    (`transcript`/`txt`/`raw_payload`/`text`) or inline in a
    `transcript`/`raw_payload`/`text` field (or `externalInput`). One sentence
    per line.
* **Output**: per-line manifest
  ```json
  { "manifest": [ { "line_index": 1, "text": "...", "artifact_id": "art-…",
                    "start_sec": 0.0, "end_sec": 0.8, "score": 95.0,
                    "flags": [...], "missing": false }, ... ],
    "line_count": 3, "clip_count": 2 }
  ```
  Each non-missing line's cut clip is `put` back as its own audio artifact; the
  DSL then spawns one data point per manifest entry, attaching `artifact_id` and
  starting it `Collected` (skip human collection → straight to auto-QC).

#### Open contract questions for the DSL side

1. **Role + field names**: under what role key does the runtime materialize the
   data point (auto-qc) and the packet (align)? What are the canonical artifact
   field names for the clip / long-WAV / transcript? The workers guess from the
   alias lists above; pin them down so the guessing can be tightened.
2. **align output → data points**: how should the manifest map to spawned data
   points — does the DSL read `output.manifest` and `SpawnMany` over it with
   `artifact_id`/`start_sec`/`end_sec` attached, and what field does the clip
   artifact attach to on each point (so it matches the auto-qc input field)?
3. **`missing` lines**: should a missing-audio line still spawn a (flagged) point
   for manual handling, or be dropped? The worker reports both `clip_count` and
   per-line `missing`.
4. **quality threshold**: is `passed == (status=="pass")` the right gate, or
   should `review` also pass? The raw `status`/`similarity`/`wer` are computed
   but only `passed`/`quality_score`/`flags` are returned today.

## Running

Per-worker entrypoints (config via env):

```bash
# QC worker (own host)
GATEWAY_URL=http://gateway:4500 \
WORKER_BEARER=<auto-qc service-account token> \
QUEUE=auto-qc \
python -m workers.qc_worker

# Alignment worker (own, heavier host)
GATEWAY_URL=http://gateway:4500 \
WORKER_BEARER=<align-audio service-account token> \
QUEUE=align-audio \
WAVESPLIT_ALIGNMENT_ENGINE=auto \
python -m workers.alignment_worker
```

Env vars: `GATEWAY_URL` (default `http://localhost:4500`), `WORKER_BEARER`,
`QUEUE` (per-worker default `auto-qc`/`align-audio`), `POLL_INTERVAL` (idle poll
seconds, default `1.0`), `WAVESPLIT_CONFIG` (optional wavesplit `config.yaml`
path), `LOG_LEVEL`. The wavesplit ML knobs (`WAVESPLIT_ASR_MODEL`,
`WAVESPLIT_ALIGNMENT_ENGINE`, `WAVESPLIT_CTC_DEVICE`, …) are honored via
`wavesplit.config.load_config`.

The two workers run on **separate hosts** with **isolated** dependency sets:

```bash
# build from the wavesplit repo root
docker build -f workers/Dockerfile.qc        -t wavesplit-qc-worker .
docker build -f workers/Dockerfile.alignment -t wavesplit-alignment-worker .
```

`requirements-qc.txt` has faster-whisper but **no** torch/onnxruntime;
`requirements-alignment.txt` has torch/torchaudio/onnxruntime/ctc-forced-aligner
but **no** faster-whisper.

## What is / isn't verified

**Verified here:**
* `python -m py_compile` on every new file (syntax-clean on Python 3.9–3.11).
* Unit tests (`workers/tests/`, run with the lightweight deps only — `requests`,
  `numpy`, `rapidfuzz`, `jiwer`, `pytest`): **9 passed**. They cover the base64
  audio envelope round-trip, artifact put/get, envelope/role helpers, the
  lease→complete→fail run loop (incl. `__outputArtifacts`), and BOTH handlers
  end-to-end against a fake in-memory gateway with **injected** transcriber /
  energy-analyzer / aligner (real QC `score_clip` scoring + real
  `build_line_manifest`/`assign_output_names`/manifest+upload flow).

**NOT verified (heavy ML deps unavailable in this environment):**
* faster-whisper ASR transcription (`FasterWhisperTranscriber`).
* torch / onnxruntime / ctc_forced_aligner forced alignment + the energy
  fallback, and the ffmpeg-backed `probe_audio`/`cut_audio_clip`/
  `analyze_clip_energy` (these are dependency-injected in tests).
* A live end-to-end run against a real gateway with a real bearer token.

The handlers are structured so the gateway client + run loop + handler logic are
fully testable without the ML stack; only the injected transcriber / aligner /
ffmpeg audio I/O are unverified.
