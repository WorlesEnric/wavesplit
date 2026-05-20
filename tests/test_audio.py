import math
import wave
from pathlib import Path

import numpy as np

from wavesplit.audio import detect_speech_intervals_in_window
from wavesplit.config import AlignmentConfig


def _write_repeated_tone_wav(path: Path) -> None:
    sr = 16000
    chunks = []
    for duration, amp in [
        (0.25, 0.0),
        (0.45, 0.25),
        (0.9, 0.0),
        (0.45, 0.25),
        (0.9, 0.0),
        (0.45, 0.25),
        (0.25, 0.0),
    ]:
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


def test_windowed_speech_detection_finds_repeated_utterances(tmp_path):
    audio = tmp_path / "repeated.wav"
    _write_repeated_tone_wav(audio)

    intervals, debug = detect_speech_intervals_in_window(
        audio,
        target_count=3,
        audio_duration_sec=3.65,
        config=AlignmentConfig(),
        start_sec=0.0,
        end_sec=3.65,
    )

    assert debug["adjustment"] == "none"
    assert len(intervals) == 3
    assert intervals[0][0] < 0.3
    assert intervals[1][0] > intervals[0][1]
    assert intervals[2][0] > intervals[1][1]
