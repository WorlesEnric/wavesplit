from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .errors import InputValidationError
from .models import LineRecord, dataclass_to_dict

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\u2018\u2019\u02bc]", "'", text)
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_normalized(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def read_transcript(path: str | Path) -> list[str]:
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = Path(path).read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines:
        raise InputValidationError("Transcript must contain at least one non-empty line.")

    cleaned: list[str] = []
    for index, line in enumerate(lines, start=1):
        value = line.rstrip("\r\n")
        if not value.strip():
            raise InputValidationError(f"Transcript contains an empty line at line {index}.")
        normalized = normalize_text(value)
        if not tokenize_normalized(normalized):
            raise InputValidationError(f"Transcript line {index} has no alignable English tokens.")
        cleaned.append(value.strip())
    return cleaned


def build_line_manifest(lines: list[str]) -> list[LineRecord]:
    records: list[LineRecord] = []
    token_cursor = 0
    for line_index, original_text in enumerate(lines, start=1):
        normalized = normalize_text(original_text)
        tokens = tokenize_normalized(normalized)
        start = token_cursor
        token_cursor += len(tokens)
        records.append(
            LineRecord(
                line_index=line_index,
                original_text=original_text,
                normalized_text=normalized,
                tokens=tokens,
                token_start_index=start,
                token_end_index=token_cursor,
            )
        )
    return records


def write_line_manifest(records: list[LineRecord], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([dataclass_to_dict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
