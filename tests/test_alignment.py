from wavesplit.alignment import _refine_repeated_run
from wavesplit.config import AppConfig
from wavesplit.models import LineAlignment


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
    assert refinements[0]["reason"] == "refuse_artificial_split"
    assert alignments[0].raw_start_sec == 1.0
    assert all("repeated_text_vad_refined" not in item.flags for item in alignments)


def test_repeated_vad_refinement_marks_unmatched_tail_when_speech_count_is_short(monkeypatch):
    def fake_detect(*args, **kwargs):
        return (
            [(1.0, 1.7), (3.0, 3.35), (3.35, 3.7)],
            {
                "initial_segment_count": 2,
                "initial_intervals": [[1.0, 1.7], [3.0, 3.7]],
                "adjustment": "split",
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

    assert refinements[0]["status"] == "applied_missing_tail"
    assert refinements[0]["expected_segment_count"] == 3
    assert refinements[0]["detected_segment_count"] == 2
    assert alignments[0].raw_start_sec == 1.0
    assert alignments[1].raw_start_sec == 3.0
    assert alignments[2].start_sec is None
    assert "missing_audio_segment" in alignments[2].flags
    assert "unmatched_indices_assumed_tail" in alignments[2].flags
