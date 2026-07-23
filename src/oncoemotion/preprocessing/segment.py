"""Segment a free-text field into clause-level spans (multi-symptom support).

Splits on strong punctuation and a small set of Italian coordinating
conjunctions while preserving character offsets into the (display) text, so
downstream spans map back to the original field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Strong boundaries: sentence punctuation, newlines, semicolons, " e " / " ed ".
_BOUNDARY = re.compile(r"(?:[.;\n]+|,| e | ed | ma | però | oppure )", re.IGNORECASE)


@dataclass(frozen=True)
class Segment:
    text: str
    start: int
    end: int


def segment_text(display_text: str) -> list[Segment]:
    """Return non-empty segments with offsets into ``display_text``.

    Whole-text is always available as a fallback segment when nothing splits.
    """
    segments: list[Segment] = []
    pos = 0
    for m in _BOUNDARY.finditer(display_text):
        chunk = display_text[pos:m.start()]
        if chunk.strip():
            lead = len(chunk) - len(chunk.lstrip())
            seg_text = chunk.strip()
            start = pos + lead
            segments.append(Segment(seg_text, start, start + len(seg_text)))
        pos = m.end()
    tail = display_text[pos:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        seg_text = tail.strip()
        start = pos + lead
        segments.append(Segment(seg_text, start, start + len(seg_text)))
    if not segments and display_text.strip():
        seg = display_text.strip()
        start = display_text.find(seg)
        segments.append(Segment(seg, start, start + len(seg)))
    return segments
