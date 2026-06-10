import io
import math
import wave
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from wavesplit.api import create_app
from wavesplit.config import AppConfig
from wavesplit.timestamps import build_timestamp_segments, format_timestamp_txt, generate_timestamp_payload


def _write_repeated_tone_wav(path: Path) -> None:
    sample_rate = 16000
    chunks = []
    for duration, amp in [
        (0.25, 0.0),
        (0.45, 0.25),
        (0.8, 0.0),
        (0.45, 0.25),
        (2.2, 0.0),
        (0.45, 0.25),
        (0.25, 0.0),
    ]:
        count = int(sample_rate * duration)
        if amp == 0:
            chunks.append(np.zeros(count, dtype=np.float32))
        else:
            t = np.arange(count, dtype=np.float32) / sample_rate
            chunks.append((amp * np.sin(2 * math.pi * 440 * t)).astype(np.float32))
    samples = np.concatenate(chunks)
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_short_gaps_get_smaller_non_overlapping_margins():
    segments = build_timestamp_segments(
        [(1.0, 2.0), (2.8, 3.6), (6.0, 7.0)],
        audio_duration_sec=8.5,
        margin_sec=1.0,
    )

    assert segments[0]["start_sec"] == 0.0
    assert segments[0]["end_sec"] == 2.2
    assert segments[1]["start_sec"] == 2.6
    assert segments[1]["end_sec"] == 4.6
    assert segments[2]["start_sec"] == 5.0
    assert segments[2]["end_sec"] == 8.0
    assert format_timestamp_txt(segments) == "0.000 2.200\n2.600 4.600\n5.000 8.000\n"


def test_generate_timestamp_payload_detects_repeated_voice_segments(tmp_path):
    audio = tmp_path / "repeated.wav"
    _write_repeated_tone_wav(audio)
    config = AppConfig(storage_dir=str(tmp_path / "storage"))
    config.auth.enabled = False

    payload = generate_timestamp_payload(audio, config=config)

    assert payload["summary"]["total"] == 3
    segments = payload["segments"]
    assert segments[0]["end_sec"] < segments[1]["start_sec"]
    assert segments[1]["end_sec"] < segments[2]["start_sec"]
    assert len(payload["text"].strip().splitlines()) == 3


def test_timestamp_endpoint_returns_txt(tmp_path):
    audio = tmp_path / "repeated.wav"
    _write_repeated_tone_wav(audio)
    config = AppConfig(storage_dir=str(tmp_path / "storage"))
    config.auth.enabled = False
    client = TestClient(create_app(config))

    response = client.post(
        "/api/timestamps",
        data={"reference_text": "hello nissan"},
        files={"audio": ("repeated.wav", audio.read_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["x-wavesplit-segment-count"] == "3"
    lines = response.text.strip().splitlines()
    assert len(lines) == 3
    assert all(len(line.split()) == 2 for line in lines)


def test_batch_timestamp_endpoint_uses_wav_filename_as_reference_and_returns_zip(tmp_path):
    first = tmp_path / "hello nissan.wav"
    second = tmp_path / "start car.wav"
    _write_repeated_tone_wav(first)
    _write_repeated_tone_wav(second)
    config = AppConfig(storage_dir=str(tmp_path / "storage"))
    config.auth.enabled = False
    client = TestClient(create_app(config))

    response = client.post(
        "/api/timestamps/batch",
        files=[
            ("audios", ("hello nissan.wav", first.read_bytes(), "audio/wav")),
            ("audios", ("start car.wav", second.read_bytes(), "audio/wav")),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert response.headers["x-wavesplit-file-count"] == "2"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == ["hello nissan.txt", "start car.txt"]
        assert len(archive.read("hello nissan.txt").decode("utf-8").strip().splitlines()) == 3
        assert len(archive.read("start car.txt").decode("utf-8").strip().splitlines()) == 3
