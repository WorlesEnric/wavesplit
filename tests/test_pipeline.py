import math
import wave
from pathlib import Path

import numpy as np

from wavesplit.config import AppConfig
from wavesplit.pipeline import process_inputs


def _write_synthetic_wav(path: Path) -> None:
    sr = 16000
    chunks = []
    for duration, amp in [(0.4, 0.0), (0.65, 0.25), (0.85, 0.0), (0.7, 0.25), (0.35, 0.0)]:
        count = int(sr * duration)
        if amp == 0:
            chunks.append(np.zeros(count, dtype=np.float32))
        else:
            t = np.arange(count, dtype=np.float32) / sr
            chunks.append((amp * np.sin(2 * math.pi * 440 * t)).astype(np.float32))
    samples = np.concatenate(chunks)
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def test_process_inputs_generates_manifest_zip_and_clips(tmp_path):
    audio = tmp_path / "sample.wav"
    transcript = tmp_path / "sample.txt"
    out_dir = tmp_path / "job"
    _write_synthetic_wav(audio)
    transcript.write_text("hello nissan\nhello nissan\n", encoding="utf-8")

    config = AppConfig(storage_dir=str(tmp_path / "storage"))
    config.alignment.engine = "energy_ordered"
    config.qa.enabled = False
    payload = process_inputs(audio_path=audio, transcript_path=transcript, out_dir=out_dir, config=config, job_id="job")

    assert payload["summary"]["total"] == 2
    clips = payload["clips"]
    assert clips[0]["start_sec"] == 0.0
    assert math.isclose(clips[0]["end_sec"], clips[1]["start_sec"], abs_tol=0.001)
    assert math.isclose(clips[1]["end_sec"], payload["summary"]["duration_sec"], abs_tol=0.001)
    assert (out_dir / "clips" / "hello nissan.wav").exists()
    assert (out_dir / "clips" / "hello nissan-2.wav").exists()
    assert (out_dir / "output" / "manifest.csv").exists()
    assert (out_dir / "output" / "clips.zip").exists()
    assert (out_dir / "qa" / "qa_report.json").exists()
