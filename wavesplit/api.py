from __future__ import annotations

import asyncio
import io
import json
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .auth import configured_user, create_session_token, password_matches, username_from_request
from .config import AppConfig, load_config
from .errors import WaveSplitError
from .packaging import build_diagnostics_zip
from .pipeline import run_pipeline
from .status import JobStatusStore
from .storage import create_job_layout, job_dir_for, make_job_id, validate_job_id
from .timestamps import generate_timestamp_payload


CHUNK_UPLOAD_MAX_BYTES = 32 * 1024 * 1024


class LoginRequest(BaseModel):
    username: str
    password: str


class ChunkedUploadInitRequest(BaseModel):
    audio_filename: str
    transcript_filename: str
    audio_size: int
    transcript_size: int
    audio_chunks: int
    transcript_chunks: int


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_config()
    app = FastAPI(title="WaveSplit", version="0.1.0")
    app.state.config = config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def require_api_auth(request: Request, call_next: Any) -> Any:
        path = request.url.path
        if (
            not config.auth.enabled
            or not path.startswith("/api/")
            or path in {"/api/auth/session", "/api/auth/login", "/api/auth/logout"}
        ):
            return await call_next(request)
        if username_from_request(config, request):
            return await call_next(request)
        return JSONResponse({"detail": "Authentication required."}, status_code=401)

    @app.get("/api/auth/session")
    def auth_session(request: Request) -> dict[str, object]:
        if not config.auth.enabled:
            return {"authenticated": True, "username": "auth-disabled"}
        username = username_from_request(config, request)
        return {"authenticated": bool(username), "username": username}

    @app.post("/api/auth/login")
    def auth_login(payload: LoginRequest, response: Response) -> dict[str, object]:
        if not config.auth.enabled:
            return {"authenticated": True, "username": "auth-disabled"}
        user = configured_user(config, payload.username)
        if user is None or not password_matches(user, payload.password):
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        token = create_session_token(config, user.username)
        response.set_cookie(
            config.auth.session_cookie,
            token,
            max_age=config.auth.session_max_age_sec,
            httponly=True,
            secure=config.auth.cookie_secure,
            samesite="lax",
        )
        return {"authenticated": True, "username": user.username}

    @app.post("/api/auth/logout")
    def auth_logout(response: Response) -> dict[str, object]:
        response.delete_cookie(config.auth.session_cookie, httponly=True, secure=config.auth.cookie_secure, samesite="lax")
        return {"authenticated": False, "username": None}

    @app.get("/api/health")
    def health() -> dict[str, object]:
        ffmpeg_ok = _command_ok(["ffmpeg", "-version"])
        ffprobe_ok = _command_ok(["ffprobe", "-version"])
        redis_ok = _redis_ok(config)
        alignment_model_exists = Path(config.alignment.model).exists() if config.alignment.engine in {"auto", "ctc_forced_aligner"} else None
        checks = {
            "ffmpeg": ffmpeg_ok,
            "ffprobe": ffprobe_ok,
            "redis": redis_ok,
            "alignment_model": alignment_model_exists,
            "storage_dir": Path(config.storage_dir).exists(),
            "web_dist": Path("web/dist/index.html").exists(),
        }
        required = [ffmpeg_ok, ffprobe_ok, checks["storage_dir"], checks["web_dist"]]
        if alignment_model_exists is not None:
            required.append(alignment_model_exists)
        return {
            "ok": all(bool(item) for item in required),
            "storage_dir": config.storage_dir,
            "queue_mode": "redis" if redis_ok else "local_thread_fallback",
            "checks": checks,
        }

    @app.post("/api/jobs")
    async def create_job(audio: UploadFile = File(...), transcript: UploadFile = File(...)) -> dict[str, str]:
        if not audio.filename or not audio.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Please upload a .wav audio file.")
        if not transcript.filename or not transcript.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="Please upload a .txt transcript file.")

        job_id = make_job_id()
        paths = create_job_layout(job_dir_for(config, job_id))
        max_bytes = config.max_upload_mb * 1024 * 1024
        await _save_upload(audio, Path(paths.original_audio_path), max_bytes)
        await _save_upload(transcript, Path(paths.transcript_path), max_bytes)
        JobStatusStore(paths.status_path).initialize(job_id, audio.filename, transcript.filename)
        _enqueue_or_run_local(config, job_id)
        return {"job_id": job_id, "state": "queued"}

    @app.post("/api/timestamps")
    async def create_timestamps(reference_text: str = Form(...), audio: UploadFile = File(...)) -> Response:
        if not audio.filename or not audio.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Please upload a .wav audio file.")
        if not reference_text.strip():
            raise HTTPException(status_code=400, detail="Please provide reference text.")

        request_id = make_job_id()
        root = Path(config.storage_dir) / "timestamps" / request_id
        audio_path = root / "original.wav"
        try:
            await _save_upload(audio, audio_path, config.max_upload_mb * 1024 * 1024)
            payload = generate_timestamp_payload(audio_path, reference_text=reference_text, config=config)
        except WaveSplitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(root, ignore_errors=True)

        headers = {
            "Content-Disposition": 'attachment; filename="timestamps.txt"',
            "X-WaveSplit-Segment-Count": str(payload["summary"]["total"]),
        }
        return Response(content=str(payload["text"]), media_type="text/plain; charset=utf-8", headers=headers)

    @app.post("/api/timestamps/batch")
    async def create_batch_timestamps(audios: list[UploadFile] = File(...)) -> Response:
        if not audios:
            raise HTTPException(status_code=400, detail="Please upload at least one .wav audio file.")

        request_id = make_job_id()
        root = Path(config.storage_dir) / "timestamps" / request_id
        max_bytes = config.max_upload_mb * 1024 * 1024
        used_names: set[str] = set()
        output = io.BytesIO()
        processed = 0
        try:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for index, audio in enumerate(audios, start=1):
                    filename = _upload_basename(audio.filename or "")
                    if not filename.lower().endswith(".wav"):
                        raise HTTPException(status_code=400, detail="Please upload only .wav audio files.")
                    reference_text = Path(filename).stem.strip()
                    if not reference_text:
                        raise HTTPException(status_code=400, detail="WAV filename must contain reference text.")

                    audio_path = root / f"{index:06d}.wav"
                    await _save_upload(audio, audio_path, max_bytes)
                    payload = generate_timestamp_payload(audio_path, reference_text=reference_text, config=config)
                    archive.writestr(_unique_timestamp_txt_name(filename, used_names), str(payload["text"]))
                    processed += 1
        except HTTPException:
            raise
        except WaveSplitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(root, ignore_errors=True)

        headers = {
            "Content-Disposition": 'attachment; filename="timestamps.zip"',
            "X-WaveSplit-File-Count": str(processed),
        }
        return Response(content=output.getvalue(), media_type="application/zip", headers=headers)


    @app.post("/api/uploads")
    def init_chunked_upload(payload: ChunkedUploadInitRequest) -> dict[str, str]:
        if not payload.audio_filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Please upload a .wav audio file.")
        if not payload.transcript_filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="Please upload a .txt transcript file.")
        if payload.audio_size <= 0 or payload.transcript_size <= 0:
            raise HTTPException(status_code=400, detail="Upload files must not be empty.")
        if payload.audio_chunks <= 0 or payload.transcript_chunks <= 0:
            raise HTTPException(status_code=400, detail="Invalid upload chunk count.")
        max_bytes = config.max_upload_mb * 1024 * 1024
        if payload.audio_size > max_bytes or payload.transcript_size > max_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")

        upload_id = make_job_id()
        root = _upload_dir_for(config, upload_id)
        root.mkdir(parents=True, exist_ok=False)
        (root / "metadata.json").write_text(payload.model_dump_json(), encoding="utf-8")
        return {"upload_id": upload_id}

    @app.post("/api/uploads/{upload_id}/chunks")
    async def upload_chunk(
        upload_id: str,
        file_kind: str = Form(...),
        chunk_index: int = Form(...),
        chunk: UploadFile = File(...),
    ) -> dict[str, object]:
        metadata = _chunked_upload_metadata(config, upload_id)
        if file_kind not in {"audio", "transcript"}:
            raise HTTPException(status_code=400, detail="Invalid upload file kind.")
        expected_chunks = int(metadata[f"{file_kind}_chunks"])
        if chunk_index < 0 or chunk_index >= expected_chunks:
            raise HTTPException(status_code=400, detail="Invalid upload chunk index.")
        target = _upload_dir_for(config, upload_id) / file_kind / f"{chunk_index:06d}.part"
        await _save_upload(chunk, target, CHUNK_UPLOAD_MAX_BYTES)
        return {"ok": True, "file_kind": file_kind, "chunk_index": chunk_index}

    @app.post("/api/uploads/{upload_id}/complete")
    def complete_chunked_upload(upload_id: str) -> dict[str, str]:
        metadata = _chunked_upload_metadata(config, upload_id)
        job_id = make_job_id()
        paths = create_job_layout(job_dir_for(config, job_id))
        max_bytes = config.max_upload_mb * 1024 * 1024
        root = _upload_dir_for(config, upload_id)
        try:
            _assemble_chunked_file(root / "audio", int(metadata["audio_chunks"]), Path(paths.original_audio_path), max_bytes)
            _assemble_chunked_file(root / "transcript", int(metadata["transcript_chunks"]), Path(paths.transcript_path), max_bytes)
            JobStatusStore(paths.status_path).initialize(job_id, str(metadata["audio_filename"]), str(metadata["transcript_filename"]))
        except Exception:
            shutil.rmtree(paths.job_dir, ignore_errors=True)
            raise
        shutil.rmtree(root, ignore_errors=True)
        _enqueue_or_run_local(config, job_id)
        return {"job_id": job_id, "state": "queued"}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        paths = _paths_or_404(config, job_id)
        status = JobStatusStore(paths.status_path).read()
        if not status:
            raise HTTPException(status_code=404, detail="Job status not found.")
        return status

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        paths = _paths_or_404(config, job_id)

        async def stream() -> Any:
            last_payload = ""
            while True:
                if not Path(paths.status_path).exists():
                    yield "event: error\ndata: {\"error\":\"missing status\"}\n\n"
                    break
                payload = Path(paths.status_path).read_text(encoding="utf-8")
                if payload != last_payload:
                    last_payload = payload
                    yield f"event: progress\ndata: {payload}\n\n"
                    state = json.loads(payload).get("state")
                    if state in {"succeeded", "failed", "cancelled"}:
                        break
                await asyncio.sleep(1.0)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/jobs/{job_id}/report")
    def get_report(job_id: str) -> JSONResponse:
        paths = _paths_or_404(config, job_id)
        path = Path(paths.qa_dir) / "qa_report.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Report is not ready.")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/jobs/{job_id}/download")
    def download_zip(job_id: str) -> FileResponse:
        paths = _paths_or_404(config, job_id)
        path = Path(paths.output_dir) / "clips.zip"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Zip is not ready.")
        return FileResponse(path, filename="clips.zip", media_type="application/zip")

    @app.get("/api/jobs/{job_id}/manifest.csv")
    def manifest_csv(job_id: str) -> FileResponse:
        paths = _paths_or_404(config, job_id)
        path = Path(paths.output_dir) / "manifest.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Manifest is not ready.")
        return FileResponse(path, filename="manifest.csv", media_type="text/csv")

    @app.get("/api/jobs/{job_id}/qa_report.csv")
    def qa_report_csv(job_id: str) -> FileResponse:
        paths = _paths_or_404(config, job_id)
        path = Path(paths.qa_dir) / "qa_report.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="QA report is not ready.")
        return FileResponse(path, filename="qa_report.csv", media_type="text/csv")

    @app.get("/api/jobs/{job_id}/diagnostics.zip")
    def diagnostics_zip(job_id: str) -> FileResponse:
        paths = _paths_or_404(config, job_id)
        path = build_diagnostics_zip(paths)
        return FileResponse(path, filename="diagnostics.zip", media_type="application/zip")

    @app.get("/api/jobs/{job_id}/clips/{clip_id}")
    def clip(job_id: str, clip_id: str) -> FileResponse:
        paths = _paths_or_404(config, job_id)
        manifest_path = Path(paths.output_dir) / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Manifest is not ready.")
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = next((item for item in records if item.get("clip_id") == clip_id), None)
        if not record:
            raise HTTPException(status_code=404, detail="Clip id not found.")
        if not record.get("output_file"):
            raise HTTPException(status_code=404, detail="Clip has no audio segment.")
        clip_path = (Path(paths.clips_dir) / record["output_file"]).resolve()
        clips_root = Path(paths.clips_dir).resolve()
        if clips_root not in clip_path.parents and clip_path != clips_root:
            raise HTTPException(status_code=400, detail="Invalid clip path.")
        if not clip_path.exists():
            raise HTTPException(status_code=404, detail="Clip file is missing.")
        return FileResponse(clip_path, filename=record["output_file"], media_type="audio/wav")

    static_dir = Path("web/dist")
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")

    return app


async def _save_upload(upload: UploadFile, target: Path, max_bytes: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("wb") as fh:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
            fh.write(chunk)


def _upload_basename(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name


def _unique_timestamp_txt_name(filename: str, used_names: set[str]) -> str:
    stem = Path(_upload_basename(filename)).stem.strip() or "timestamps"
    candidate = f"{stem}.txt"
    suffix = 2
    while candidate in used_names:
        candidate = f"{stem}-{suffix}.txt"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _upload_dir_for(config: AppConfig, upload_id: str) -> Path:
    return Path(config.storage_dir) / "uploads" / validate_job_id(upload_id)


def _chunked_upload_metadata(config: AppConfig, upload_id: str) -> dict[str, Any]:
    try:
        path = _upload_dir_for(config, upload_id) / "metadata.json"
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload session not found.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid upload session metadata.")
    return payload


def _assemble_chunked_file(chunks_dir: Path, chunk_count: int, target: Path, max_bytes: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with target.open("wb") as output:
            for index in range(chunk_count):
                chunk_path = chunks_dir / f"{index:06d}.part"
                if not chunk_path.exists():
                    raise HTTPException(status_code=400, detail="Upload is missing one or more chunks.")
                written += chunk_path.stat().st_size
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
                with chunk_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def _paths_or_404(config: AppConfig, job_id: str) -> Any:
    try:
        paths = create_job_layout(job_dir_for(config, validate_job_id(job_id)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not Path(paths.job_dir).exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    return paths


def _enqueue_or_run_local(config: AppConfig, job_id: str) -> None:
    try:
        from redis import Redis
        from rq import Queue

        connection = Redis.from_url(config.worker.redis_url)
        connection.ping()
        queue = Queue(config.worker.queue_name, connection=connection)
        queue.enqueue("wavesplit.worker.run_job", job_id)
    except Exception:
        thread = threading.Thread(target=_run_local, args=(config, job_id), daemon=True)
        thread.start()


def _command_ok(command: list[str]) -> bool:
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return False
    return result.returncode == 0


def _redis_ok(config: AppConfig) -> bool:
    try:
        from redis import Redis

        connection = Redis.from_url(config.worker.redis_url)
        return bool(connection.ping())
    except Exception:
        return False


def _run_local(config: AppConfig, job_id: str) -> None:
    run_pipeline(job_dir_for(config, job_id), config=config)


app = create_app()
