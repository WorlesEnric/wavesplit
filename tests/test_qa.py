from wavesplit.config import QAConfig
from wavesplit.qa import normalize_for_qa, run_qa, score_clip


def test_alias_normalization_maps_known_variants():
    aliases = {"bleeker": ["bleaker", "bleeker."]}
    assert normalize_for_qa("Hey Bleaker.", aliases) == "hey bleeker"


def test_score_clip_passes_exact_match():
    result = score_clip(
        normalized_text="take a photo",
        asr_text="take a photo",
        duration_sec=1.2,
        metrics={
            "leading_silence_ms": 50,
            "trailing_silence_ms": 80,
            "start_energy_flag": False,
            "end_energy_flag": False,
        },
        config=QAConfig(),
        base_flags=[],
        same_text_median=1.2,
    )
    assert result["status"] == "pass"
    assert result["confidence"] == 100


def test_score_clip_fails_severe_text_mismatch():
    result = score_clip(
        normalized_text="take a photo",
        asr_text="open the door",
        duration_sec=1.2,
        metrics={
            "leading_silence_ms": 50,
            "trailing_silence_ms": 80,
            "start_energy_flag": False,
            "end_energy_flag": False,
        },
        config=QAConfig(),
        base_flags=[],
        same_text_median=1.2,
    )
    assert result["status"] == "fail"
    assert "text_mismatch_fail" in result["flags"]


def test_run_qa_preserves_missing_audio_rows(tmp_path):
    records = run_qa(
        [
            {
                "clip_id": "clip-000001",
                "line_index": 1,
                "original_text": "Take a Photo",
                "normalized_text": "take a photo",
                "output_file": None,
                "duplicate_index": 0,
                "start_sec": None,
                "end_sec": None,
                "duration_sec": None,
                "alignment_score_mean": None,
                "flags": ["missing_audio_segment", "repeated_text_count_mismatch"],
            }
        ],
        clips_dir=tmp_path,
        config=QAConfig(enabled=False),
    )

    assert records[0].status == "missing_audio"
    assert records[0].output_file is None
    assert records[0].start_sec is None
    assert "missing_audio_segment" in records[0].flags
