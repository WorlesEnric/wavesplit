from __future__ import annotations

import re
from collections import defaultdict

from .models import LineRecord

ILLEGAL_FILENAME_RE = re.compile(r'[\/\\:\*\?"<>\|\x00]+')


def sanitize_filename_base(text: str, max_length: int = 120) -> str:
    base = text.strip()
    base = re.sub(r"\s+", " ", base)
    base = ILLEGAL_FILENAME_RE.sub(" ", base)
    base = re.sub(r"\s+", " ", base).strip(" .")
    if not base:
        base = "line"
    if len(base) > max_length:
        base = base[:max_length].rstrip(" .")
    return base or "line"


def assign_output_names(lines: list[LineRecord]) -> dict[int, tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    names: dict[int, tuple[str, int]] = {}
    for line in lines:
        base = sanitize_filename_base(line.original_text)
        if base == "line":
            base = f"line-{line.line_index}"
        duplicate_index = counts[base]
        counts[base] += 1
        suffix = "" if duplicate_index == 0 else f"-{duplicate_index + 1}"
        names[line.line_index] = (f"{base}{suffix}.wav", duplicate_index)
    return names
