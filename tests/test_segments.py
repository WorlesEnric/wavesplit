from wavesplit.config import AlignmentConfig
from wavesplit.models import LineAlignment
from wavesplit.segments import apply_padding_and_fix_overlaps


def test_padding_overlap_is_resolved_at_raw_gap_midpoint():
    alignments = [
        LineAlignment("clip-000001", 1, "one", "one", 1.0, 1.5, 1.0, 1.5),
        LineAlignment("clip-000002", 2, "two", "two", 1.55, 2.0, 1.55, 2.0),
    ]
    apply_padding_and_fix_overlaps(
        alignments,
        audio_duration_sec=3.0,
        config=AlignmentConfig(pre_padding_ms=100, post_padding_ms=200),
    )
    assert alignments[0].end_sec == alignments[1].start_sec
    assert "padding_overlap_adjusted" in alignments[0].flags
    assert "padding_overlap_adjusted" in alignments[1].flags


def test_silence_gap_boundary_is_shared_between_adjacent_clips():
    alignments = [
        LineAlignment("clip-000001", 1, "one", "one", 1.0, 1.5, 1.0, 1.5),
        LineAlignment("clip-000002", 2, "two", "two", 2.5, 3.0, 2.5, 3.0),
    ]
    apply_padding_and_fix_overlaps(
        alignments,
        audio_duration_sec=4.0,
        config=AlignmentConfig(pre_padding_ms=80, post_padding_ms=120),
        split_points=[{"time": 2.1, "gap_start": 1.5, "gap_end": 2.5, "method": "min_energy"}],
        include_edge_silence=True,
    )
    assert alignments[0].start_sec == 0.0
    assert alignments[0].end_sec == 2.1
    assert alignments[1].start_sec == 2.1
    assert alignments[1].end_sec == 4.0
    assert "silence_split_min_energy" in alignments[0].flags
    assert "silence_split_min_energy" in alignments[1].flags
    assert "leading_silence_attached" in alignments[0].flags
    assert "trailing_silence_attached" in alignments[1].flags
