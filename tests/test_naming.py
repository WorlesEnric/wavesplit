from wavesplit.naming import assign_output_names, sanitize_filename_base
from wavesplit.text import build_line_manifest


def test_sanitize_filename_base_removes_illegal_characters():
    assert sanitize_filename_base(' hello / nissan: "go"? ') == "hello nissan go"


def test_assign_output_names_preserves_all_duplicates():
    lines = build_line_manifest(["hello nissan", "hello nissan", "hello nissan"])
    names = assign_output_names(lines)
    assert names[1] == ("hello nissan.wav", 0)
    assert names[2] == ("hello nissan-2.wav", 1)
    assert names[3] == ("hello nissan-3.wav", 2)
