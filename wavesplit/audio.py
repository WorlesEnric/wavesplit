from __future__ import annotations

import json
import math
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .config import AlignmentConfig
from .errors import InputValidationError, PipelineError
from .models import AudioInfo


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def probe_audio(path: str | Path) -> AudioInfo:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise InputValidationError("Audio file cannot be decoded by ffmpeg.")
    payload = json.loads(result.stdout)
    streams = [stream for stream in payload.get("streams", []) if "sample_rate" in stream]
    if not streams:
        raise InputValidationError("Audio file must contain at least one audio stream.")
    duration = float(payload.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise InputValidationError("Audio duration must be greater than zero.")
    stream = streams[0]
    return AudioInfo(
        codec_name=stream.get("codec_name", ""),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        duration_sec=duration,
        format_name=payload.get("format", {}).get("format_name", ""),
    )


def load_audio_samples(path: str | Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise InputValidationError("Audio file cannot be decoded by ffmpeg.")
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def _intervals_from_active(active: np.ndarray, frame_sec: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    index = 0
    while index < len(active):
        if active[index]:
            end = index + 1
            while end < len(active) and active[end]:
                end += 1
            intervals.append((index * frame_sec, end * frame_sec))
            index = end
        else:
            index += 1
    return intervals


def _postprocess_active(
    active: np.ndarray,
    *,
    max_gap_frames: int,
    min_active_frames: int,
) -> np.ndarray:
    active = active.copy()

    def remove_short_active_regions() -> None:
        index = 0
        while index < len(active):
            if active[index]:
                end = index + 1
                while end < len(active) and active[end]:
                    end += 1
                if end - index < min_active_frames:
                    active[index:end] = False
                index = end
            else:
                index += 1

    # Remove tiny noise islands before filling gaps; otherwise a chain of short
    # bursts can bridge two utterances into one long interval.
    remove_short_active_regions()

    index = 0
    while index < len(active):
        if not active[index]:
            end = index + 1
            while end < len(active) and not active[end]:
                end += 1
            if index > 0 and end < len(active) and end - index <= max_gap_frames:
                active[index:end] = True
            index = end
        else:
            index += 1

    remove_short_active_regions()
    return active


def _reduce_to_target(intervals: list[tuple[float, float]], target_count: int) -> tuple[list[tuple[float, float]], str]:
    reduced = list(intervals)
    dropped = False
    merged = False
    while len(reduced) > target_count and len(reduced) > 1:
        durations = [end - start for start, end in reduced]
        median = float(np.median(durations)) if durations else 0.0
        shortest_index = min(range(len(reduced)), key=lambda item: durations[item])
        shortest = durations[shortest_index]
        if median > 0 and shortest <= max(0.35, median * 0.5):
            del reduced[shortest_index]
            dropped = True
            continue

        gaps = [(reduced[i + 1][0] - reduced[i][1], i) for i in range(len(reduced) - 1)]
        _, index = min(gaps, key=lambda item: item[0])
        reduced[index] = (reduced[index][0], reduced[index + 1][1])
        del reduced[index + 1]
        merged = True

    if dropped and merged:
        return reduced, "dropped_short_merged"
    if dropped:
        return reduced, "dropped_short"
    if merged:
        return reduced, "merged"
    return reduced, "none"


def _split_to_target(intervals: list[tuple[float, float]], target_count: int) -> list[tuple[float, float]]:
    split = list(intervals)
    while len(split) < target_count and split:
        index = max(range(len(split)), key=lambda item: split[item][1] - split[item][0])
        start, end = split[index]
        midpoint = (start + end) / 2
        split[index : index + 1] = [(start, midpoint), (midpoint, end)]
    return split


def _adjust_to_target(intervals: list[tuple[float, float]], target_count: int) -> tuple[list[tuple[float, float]], str]:
    if len(intervals) > target_count:
        return _reduce_to_target(intervals, target_count)
    if len(intervals) < target_count:
        return _split_to_target(intervals, target_count), "split"
    return list(intervals), "none"


def _duration_balance_penalty(intervals: list[tuple[float, float]]) -> float:
    if len(intervals) < 3:
        return 0.0
    durations = np.array([max(0.0, end - start) for start, end in intervals], dtype=np.float32)
    median = float(np.median(durations))
    if median <= 0:
        return 0.0
    max_ratio = float(np.max(durations) / median)
    min_ratio = float(np.min(durations) / median)
    cv = float(np.std(durations) / median)
    return max(0.0, max_ratio - 1.8) * 8.0 + max(0.0, 0.45 - min_ratio) * 4.0 + cv * 2.0


def choose_silence_split_points(
    path: str | Path,
    intervals: list[tuple[float, float]],
    *,
    audio_duration_sec: float,
    config: AlignmentConfig,
) -> list[dict[str, float | str]]:
    samples, sample_rate = load_audio_samples(path, sample_rate=16000)
    frame_size = max(1, int(sample_rate * config.vad_frame_ms / 1000))
    usable = samples[: len(samples) // frame_size * frame_size]
    if len(usable) == 0 or len(intervals) < 2:
        return []
    frames = usable.reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    frame_sec = frame_size / sample_rate

    split_points: list[dict[str, float | str]] = []
    for current, nxt in zip(intervals, intervals[1:]):
        gap_start = min(audio_duration_sec, max(0.0, current[1]))
        gap_end = min(audio_duration_sec, max(0.0, nxt[0]))
        if gap_end <= gap_start:
            split_points.append(
                {
                    "time": round(min(audio_duration_sec, max(0.0, (current[1] + nxt[0]) / 2)), 3),
                    "gap_start": round(gap_start, 3),
                    "gap_end": round(gap_end, 3),
                    "method": "midpoint",
                }
            )
            continue

        split_time, method = _low_energy_split_time(rms, frame_sec, gap_start, gap_end)
        split_points.append(
            {
                "time": round(split_time, 3),
                "gap_start": round(gap_start, 3),
                "gap_end": round(gap_end, 3),
                "method": method,
            }
        )
    return split_points


def _low_energy_split_time(rms: np.ndarray, frame_sec: float, gap_start: float, gap_end: float) -> tuple[float, str]:
    midpoint = (gap_start + gap_end) / 2
    gap_duration = gap_end - gap_start
    if gap_duration <= frame_sec:
        return midpoint, "midpoint"

    edge_guard = min(0.08, gap_duration / 4)
    search_start = gap_start + edge_guard
    search_end = gap_end - edge_guard
    if search_end <= search_start:
        search_start, search_end = gap_start, gap_end

    centers = (np.arange(len(rms), dtype=np.float32) + 0.5) * frame_sec
    candidate_indices = np.flatnonzero((centers >= search_start) & (centers <= search_end))
    if len(candidate_indices) == 0:
        return midpoint, "midpoint"

    window = max(1, int(round(0.05 / frame_sec)))
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / window
        energy = np.convolve(rms, kernel, mode="same")
    else:
        energy = rms

    candidate_energy = energy[candidate_indices]
    if float(candidate_energy.max() - candidate_energy.min()) < 1e-6:
        return midpoint, "midpoint"

    best_index = int(candidate_indices[int(np.argmin(candidate_energy))])
    return float(min(gap_end, max(gap_start, centers[best_index]))), "min_energy"


def detect_speech_intervals(
    path: str | Path,
    *,
    target_count: int,
    audio_duration_sec: float,
    config: AlignmentConfig,
    prefer_balanced: bool = False,
    adjust_to_target: bool = True,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    if target_count <= 0:
        raise InputValidationError("Transcript must contain at least one line.")
    samples, sample_rate = load_audio_samples(path, sample_rate=16000)
    return _detect_speech_intervals_from_samples(
        samples,
        sample_rate,
        target_count=target_count,
        audio_duration_sec=audio_duration_sec,
        config=config,
        prefer_balanced=prefer_balanced,
        adjust_to_target=adjust_to_target,
    )


def detect_speech_intervals_in_window(
    path: str | Path,
    *,
    target_count: int,
    audio_duration_sec: float,
    config: AlignmentConfig,
    start_sec: float,
    end_sec: float,
    prefer_balanced: bool = False,
    adjust_to_target: bool = True,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    if target_count <= 0:
        raise InputValidationError("Transcript must contain at least one line.")
    samples, sample_rate = load_audio_samples(path, sample_rate=16000)
    start_sec = max(0.0, min(audio_duration_sec, start_sec))
    end_sec = max(start_sec, min(audio_duration_sec, end_sec))
    start_index = max(0, min(len(samples), int(start_sec * sample_rate)))
    end_index = max(start_index, min(len(samples), int(math.ceil(end_sec * sample_rate))))
    window = samples[start_index:end_index]
    intervals, debug = _detect_speech_intervals_from_samples(
        window,
        sample_rate,
        target_count=target_count,
        audio_duration_sec=len(window) / sample_rate,
        config=config,
        prefer_balanced=prefer_balanced,
        adjust_to_target=adjust_to_target,
    )
    debug["window_start_sec"] = round(start_sec, 3)
    debug["window_end_sec"] = round(end_sec, 3)
    if isinstance(debug.get("initial_intervals"), list):
        debug["initial_intervals"] = [
            [round(float(start) + start_sec, 3), round(float(end) + start_sec, 3)]
            for start, end in debug["initial_intervals"]  # type: ignore[index]
        ]
    return [(start + start_sec, end + start_sec) for start, end in intervals], debug


def _detect_speech_intervals_from_samples(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_count: int,
    audio_duration_sec: float,
    config: AlignmentConfig,
    prefer_balanced: bool = False,
    adjust_to_target: bool = True,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    frame_size = max(1, int(sample_rate * config.vad_frame_ms / 1000))
    usable = samples[: len(samples) // frame_size * frame_size]
    if len(usable) == 0:
        raise InputValidationError("Audio file contains no usable samples.")
    frames = usable.reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    frame_sec = frame_size / sample_rate

    thresholds = np.arange(0.004, 0.026, 0.0005)
    gaps_ms = range(config.min_silence_ms, config.max_silence_ms + 1, 25)
    min_active_frames = max(1, int(config.min_active_ms / config.vad_frame_ms))

    best: tuple[float, int, list[tuple[float, float]], list[tuple[float, float]], str, float] | None = None
    for threshold in thresholds:
        base_active = rms > float(threshold)
        for gap_ms in gaps_ms:
            max_gap_frames = max(1, int(gap_ms / config.vad_frame_ms))
            active = _postprocess_active(
                base_active,
                max_gap_frames=max_gap_frames,
                min_active_frames=min_active_frames,
            )
            intervals = _intervals_from_active(active, frame_sec)
            adjusted, adjustment = _adjust_to_target(intervals, target_count) if adjust_to_target else (list(intervals), "none")
            count_delta = abs(len(adjusted) - target_count)
            gap_preference = abs(gap_ms - 700) / 700
            threshold_preference = abs(float(threshold) - 0.013) / 0.013
            adjustment_penalty = 0.35 * abs(len(intervals) - target_count)
            balance_penalty = _duration_balance_penalty(adjusted) if prefer_balanced else 0.0
            score = count_delta * 1000 + adjustment_penalty + balance_penalty + gap_preference + threshold_preference
            if best is None or score < best[5]:
                best = (float(threshold), gap_ms, intervals, adjusted, adjustment, score)

    if best is None:
        raise PipelineError("Unable to analyze audio energy for alignment.")

    threshold, gap_ms, intervals, adjusted, adjustment, _ = best
    initial_count = len(intervals)

    if adjust_to_target and len(adjusted) != target_count:
        raise PipelineError(
            f"Alignment produced {len(adjusted)} speech regions for {target_count} transcript lines."
        )

    clamped = [(max(0.0, start), min(audio_duration_sec, end)) for start, end in adjusted]
    debug = {
        "engine": "energy_ordered",
        "threshold": threshold,
        "max_gap_ms": gap_ms,
        "frame_ms": config.vad_frame_ms,
        "min_active_ms": config.min_active_ms,
        "initial_segment_count": initial_count,
        "initial_intervals": [[round(start, 3), round(end, 3)] for start, end in intervals],
        "target_segment_count": target_count,
        "adjustment": adjustment,
        "prefer_balanced": prefer_balanced,
        "adjust_to_target": adjust_to_target,
        "rms_percentiles": {
            "p05": float(np.percentile(rms, 5)),
            "p50": float(np.percentile(rms, 50)),
            "p95": float(np.percentile(rms, 95)),
        },
    }
    return clamped, debug


def cut_audio_clip(input_path: str | Path, output_path: str | Path, start_sec: float, end_sec: float) -> str:
    duration = max(0.001, end_sec - start_sec)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    result = _run(command)
    if result.returncode != 0:
        raise PipelineError(f"ffmpeg failed while cutting clip: {result.stderr.strip()[:500]}")
    return result.stderr.strip()


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def analyze_clip_energy(path: str | Path) -> dict[str, float | int | bool]:
    samples, sample_rate = load_audio_samples(path, sample_rate=16000)
    if len(samples) == 0:
        return {
            "duration_sec": 0.0,
            "peak_dbfs": -120.0,
            "rms_dbfs": -120.0,
            "leading_silence_ms": 0,
            "trailing_silence_ms": 0,
            "start_energy_flag": False,
            "end_energy_flag": False,
        }
    duration = len(samples) / sample_rate
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak_dbfs = 20 * math.log10(max(peak, 1e-9))
    rms_dbfs = 20 * math.log10(max(rms, 1e-9))
    frame_size = max(1, int(sample_rate * 0.01))
    usable = samples[: len(samples) // frame_size * frame_size]
    frame_rms = np.sqrt(np.mean(usable.reshape(-1, frame_size) ** 2, axis=1)) if len(usable) else np.array([])
    silence_threshold = max(0.004, rms * 0.20)
    leading_frames = 0
    for value in frame_rms:
        if value <= silence_threshold:
            leading_frames += 1
        else:
            break
    trailing_frames = 0
    for value in frame_rms[::-1]:
        if value <= silence_threshold:
            trailing_frames += 1
        else:
            break
    edge_size = min(len(samples), int(sample_rate * 0.05))
    start_energy = float(np.sqrt(np.mean(samples[:edge_size] ** 2))) if edge_size else 0.0
    end_energy = float(np.sqrt(np.mean(samples[-edge_size:] ** 2))) if edge_size else 0.0
    edge_threshold = max(0.05, rms * 1.8)
    return {
        "duration_sec": duration,
        "peak_dbfs": round(peak_dbfs, 3),
        "rms_dbfs": round(rms_dbfs, 3),
        "leading_silence_ms": int(round(leading_frames * 10)),
        "trailing_silence_ms": int(round(trailing_frames * 10)),
        "start_energy_flag": start_energy > edge_threshold,
        "end_energy_flag": end_energy > edge_threshold,
    }
