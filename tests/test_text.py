import pytest

from wavesplit.errors import InputValidationError
from wavesplit.text import build_line_manifest, normalize_text, read_transcript, tokenize_normalized


def test_normalize_text_removes_punctuation_and_collapses_space():
    assert normalize_text("  Take a Photo!  ") == "take a photo"
    assert tokenize_normalized("take a photo") == ["take", "a", "photo"]


def test_build_line_manifest_keeps_duplicate_order():
    records = build_line_manifest(["Hey Bleeker", "Hey Bleeker", "Bleeker"])
    assert records[0].token_start_index == 0
    assert records[0].token_end_index == 2
    assert records[1].token_start_index == 2
    assert records[1].token_end_index == 4
    assert records[2].token_start_index == 4
    assert records[2].token_end_index == 5


def test_read_transcript_rejects_empty_lines(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("hello\n\nworld\n", encoding="utf-8")
    with pytest.raises(InputValidationError):
        read_transcript(path)
