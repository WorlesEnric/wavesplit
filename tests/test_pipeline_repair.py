from wavesplit.pipeline import _candidate_repair_bounds, _phrase_occurrences
from wavesplit.config import AppConfig


def test_phrase_occurrences_counts_non_overlapping_repeats():
    assert _phrase_occurrences("stop recording", "stop recording stop recording") == 2
    assert _phrase_occurrences("take a photo", "please take a photo then take a photo") == 2
    assert _phrase_occurrences("take a photo", "take a picture") == 0


def test_candidate_repair_bounds_clamps_between_duplicate_candidates():
    config = AppConfig()
    config.alignment.pre_padding_ms = 80
    config.alignment.post_padding_ms = 120

    start_sec, end_sec = _candidate_repair_bounds(
        intervals=[(10.0, 10.7), (12.0, 12.8)],
        selected_index=0,
        row_start=9.5,
        row_end=13.0,
        config=config,
    )

    assert start_sec == 9.92
    assert end_sec == 10.82
