import pytest

from wavesplit.alignment import _line_entries_from_word_timestamps, _refine_repeated_run
from wavesplit.config import AppConfig
from wavesplit.errors import AlignmentError
from wavesplit.models import LineAlignment
from wavesplit.text import build_line_manifest


def _repeated_alignments() -> list[LineAlignment]:
    return [
        LineAlignment("clip-000001", 1, "Take a Photo", "take a photo", 1.0, 1.7, 1.0, 1.7),
        LineAlignment("clip-000002", 2, "Take a Photo", "take a photo", 3.0, 3.7, 3.0, 3.7),
        LineAlignment("clip-000003", 3, "Take a Photo", "take a photo", 5.0, 5.7, 5.0, 5.7),
    ]


def test_repeated_vad_refinement_refuses_artificial_split(monkeypatch):
    def fake_detect(*args, **kwargs):
        return (
            [(1.0, 1.35), (1.35, 1.7), (3.0, 3.7)],
            {"initial_segment_count": 2, "adjustment": "split"},
        )

    monkeypatch.setattr("wavesplit.alignment.detect_speech_intervals_in_window", fake_detect)
    alignments = _repeated_alignments()

    refinements = _refine_repeated_run(
        alignments,
        0,
        3,
        audio_path="unused.wav",
        audio_duration_sec=8.0,
        config=AppConfig(),
    )

    assert refinements[0]["status"] == "skipped"
    assert refinements[0]["reason"] == "refuse_adjusted_repeated_run"
    assert alignments[0].raw_start_sec == 1.0
    assert all("repeated_text_vad_refined" not in item.flags for item in alignments)


def test_repeated_vad_refinement_reconciles_short_speech_count(monkeypatch):
    def fake_detect(*args, **kwargs):
        return (
            [(1.0, 1.7), (3.0, 3.7)],
            {
                "initial_segment_count": 2,
                "initial_intervals": [[1.0, 1.7], [3.0, 3.7]],
                "adjustment": "none",
            },
        )

    monkeypatch.setattr("wavesplit.alignment.detect_speech_intervals_in_window", fake_detect)
    alignments = _repeated_alignments()

    refinements = _refine_repeated_run(
        alignments,
        0,
        3,
        audio_path="unused.wav",
        audio_duration_sec=8.0,
        config=AppConfig(),
    )

    assert refinements[0]["status"] == "applied_missing_reconciled"
    assert refinements[0]["expected_segment_count"] == 3
    assert refinements[0]["detected_segment_count"] == 2
    assert alignments[0].raw_start_sec == 1.0
    assert alignments[1].raw_start_sec == 3.0
    assert alignments[2].start_sec is None
    assert "missing_audio_segment" in alignments[2].flags
    assert "unmatched_indices_vad_reconciled" in alignments[2].flags


def test_repeated_vad_refinement_can_mark_missing_middle(monkeypatch):
    def fake_detect(*args, **kwargs):
        return (
            [(1.05, 1.65), (5.05, 5.65)],
            {
                "initial_segment_count": 2,
                "initial_intervals": [[1.05, 1.65], [5.05, 5.65]],
                "adjustment": "none",
            },
        )

    monkeypatch.setattr("wavesplit.alignment.detect_speech_intervals_in_window", fake_detect)
    alignments = _repeated_alignments()

    refinements = _refine_repeated_run(
        alignments,
        0,
        3,
        audio_path="unused.wav",
        audio_duration_sec=8.0,
        config=AppConfig(),
    )

    assert refinements[0]["status"] == "applied_missing_reconciled"
    assert refinements[0]["assigned_line_indices"] == [1, 3]
    assert refinements[0]["missing_line_indices"] == [2]
    assert alignments[0].raw_start_sec == 1.05
    assert alignments[1].start_sec is None
    assert alignments[2].raw_start_sec == 5.05
    assert "missing_audio_segment" in alignments[1].flags


def test_repeated_vad_refinement_prefers_natural_exact_count(monkeypatch):
    calls: list[bool] = []

    def fake_detect(*args, **kwargs):
        adjust_to_target = kwargs["adjust_to_target"]
        calls.append(adjust_to_target)
        if not adjust_to_target:
            return (
                [(1.1, 1.8), (3.1, 3.8), (5.1, 5.8)],
                {"initial_segment_count": 3, "adjustment": "none", "adjust_to_target": False},
            )
        return (
            [(1.0, 1.35), (1.35, 1.7), (3.0, 3.7)],
            {"initial_segment_count": 2, "adjustment": "split", "adjust_to_target": True},
        )

    monkeypatch.setattr("wavesplit.alignment.detect_speech_intervals_in_window", fake_detect)
    alignments = _repeated_alignments()

    refinements = _refine_repeated_run(
        alignments,
        0,
        3,
        audio_path="unused.wav",
        audio_duration_sec=8.0,
        config=AppConfig(),
    )

    assert calls == [False]
    assert refinements[0]["status"] == "verified"
    assert [item.raw_start_sec for item in alignments] == [1.0, 3.0, 5.0]
    assert all("missing_audio_segment" not in item.flags for item in alignments)
    assert all("repeated_text_vad_refined" not in item.flags for item in alignments)


def test_ctc_line_entries_are_grouped_by_transcript_token_counts():
    lines = build_line_manifest(["Take a Photo", "Start Recording"])
    word_timestamps = [
        {"text": "Take", "start": 1.0, "end": 1.2, "score": -0.1},
        {"text": "a", "start": 1.2, "end": 1.3, "score": -0.1},
        {"text": "Photo", "start": 1.3, "end": 1.7, "score": -0.1},
        {"text": "Start", "start": 2.0, "end": 2.3, "score": -0.1},
        {"text": "Recording", "start": 2.3, "end": 2.9, "score": -0.1},
    ]

    entries = _line_entries_from_word_timestamps(word_timestamps, lines)

    assert entries == [
        {"index": 1, "start": 1.0, "end": 1.7, "text": "Take a Photo"},
        {"index": 2, "start": 2.0, "end": 2.9, "text": "Start Recording"},
    ]


def test_ctc_line_entries_reject_word_count_mismatch():
    lines = build_line_manifest(["Start Recording"])

    with pytest.raises(AlignmentError, match="1 word timestamps for 2 transcript tokens"):
        _line_entries_from_word_timestamps(
            [{"text": "Start", "start": 1.0, "end": 1.3}],
            lines,
        )
