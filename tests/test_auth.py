from fastapi.testclient import TestClient

import wavesplit.api as api_module
from wavesplit.api import create_app
from wavesplit.config import AppConfig, AuthUserConfig


def _client() -> TestClient:
    config = AppConfig()
    config.auth.session_secret = "test-secret"
    config.auth.users = [AuthUserConfig(username="admin", password="secret")]
    return TestClient(create_app(config))


def test_api_routes_require_authentication():
    client = _client()
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/auth/session").json() == {"authenticated": False, "username": None}


def test_login_sets_session_cookie_and_unlocks_api():
    client = _client()
    bad = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert bad.status_code == 401

    good = client.post("/api/auth/login", json={"username": "admin", "password": "secret"})
    assert good.status_code == 200
    assert good.json() == {"authenticated": True, "username": "admin"}
    assert client.get("/api/auth/session").json() == {"authenticated": True, "username": "admin"}
    assert client.get("/api/health").status_code == 200

    client.post("/api/auth/logout")
    assert client.get("/api/health").status_code == 401


def test_chunked_upload_assembles_files_and_queues_job(tmp_path, monkeypatch):
    config = AppConfig(storage_dir=str(tmp_path / "storage"))
    config.auth.enabled = False
    enqueued: list[str] = []
    monkeypatch.setattr(api_module, "_enqueue_or_run_local", lambda config, job_id: enqueued.append(job_id))
    client = TestClient(create_app(config))

    audio = b"wav-bytes-" * 1024
    transcript = b"line one\nline two\n"
    init = client.post(
        "/api/uploads",
        json={
            "audio_filename": "sample.wav",
            "transcript_filename": "sample.txt",
            "audio_size": len(audio),
            "transcript_size": len(transcript),
            "audio_chunks": 3,
            "transcript_chunks": 2,
        },
    )
    assert init.status_code == 200
    upload_id = init.json()["upload_id"]

    audio_parts = [audio[:100], audio[100:4000], audio[4000:]]
    transcript_parts = [transcript[:8], transcript[8:]]
    for index, payload in enumerate(audio_parts):
        response = client.post(
            f"/api/uploads/{upload_id}/chunks",
            data={"file_kind": "audio", "chunk_index": str(index)},
            files={"chunk": ("chunk.part", payload, "application/octet-stream")},
        )
        assert response.status_code == 200
    for index, payload in enumerate(transcript_parts):
        response = client.post(
            f"/api/uploads/{upload_id}/chunks",
            data={"file_kind": "transcript", "chunk_index": str(index)},
            files={"chunk": ("chunk.part", payload, "application/octet-stream")},
        )
        assert response.status_code == 200

    complete = client.post(f"/api/uploads/{upload_id}/complete")
    assert complete.status_code == 200
    job_id = complete.json()["job_id"]
    job_dir = tmp_path / "storage" / "jobs" / job_id
    assert (job_dir / "input" / "original.wav").read_bytes() == audio
    assert (job_dir / "input" / "transcript.txt").read_bytes() == transcript
    assert not (tmp_path / "storage" / "uploads" / upload_id).exists()
    assert enqueued == [job_id]
