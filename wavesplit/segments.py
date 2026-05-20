from __future__ import annotations

from .config import AlignmentConfig
from .models import LineAlignment


def apply_padding_and_fix_overlaps(
    alignments: list[LineAlignment],
    *,
    audio_duration_sec: float,
    config: AlignmentConfig,
    split_points: list[dict[str, float | str]] | None = None,
    include_edge_silence: bool = False,
) -> list[LineAlignment]:
    pre = config.pre_padding_ms / 1000.0
    post = config.post_padding_ms / 1000.0
    for segment in alignments:
        segment.start_sec = max(0.0, segment.raw_start_sec - pre)
        segment.end_sec = min(audio_duration_sec, segment.raw_end_sec + post)
        if segment.end_sec <= segment.start_sec:
            midpoint = (segment.raw_start_sec + segment.raw_end_sec) / 2
            segment.start_sec = max(0.0, midpoint - 0.05)
            segment.end_sec = min(audio_duration_sec, midpoint + 0.05)
            segment.flags.append("boundary_invalid_repaired")

    if include_edge_silence and alignments:
        first = alignments[0]
        last = alignments[-1]
        if first.start_sec > 0.0:
            first.start_sec = 0.0
            first.flags.append("leading_silence_attached")
        if last.end_sec < audio_duration_sec:
            last.end_sec = audio_duration_sec
            last.flags.append("trailing_silence_attached")

    for index, (current, nxt) in enumerate(zip(alignments, alignments[1:])):
        raw_gap = nxt.raw_start_sec - current.raw_end_sec
        if split_points and index < len(split_points):
            boundary = float(split_points[index]["time"])
            method = str(split_points[index].get("method", "midpoint"))
            flag = "boundary_overlap" if raw_gap < 0 else f"silence_split_{method}"
            current.end_sec = max(current.start_sec, min(audio_duration_sec, boundary))
            nxt.start_sec = min(nxt.end_sec, max(0.0, boundary))
            current.flags.append(flag)
            nxt.flags.append(flag)
            continue
        if current.end_sec <= nxt.start_sec:
            continue
        if raw_gap >= 0:
            midpoint = (current.raw_end_sec + nxt.raw_start_sec) / 2
            flag = "padding_overlap_adjusted"
        else:
            midpoint = (current.raw_end_sec + nxt.raw_start_sec) / 2
            flag = "boundary_overlap"
        current.end_sec = max(current.start_sec, min(audio_duration_sec, midpoint))
        nxt.start_sec = min(nxt.end_sec, max(0.0, midpoint))
        current.flags.append(flag)
        nxt.flags.append(flag)
    return alignments
