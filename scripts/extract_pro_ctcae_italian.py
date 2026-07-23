#!/usr/bin/env python
"""Extract official Italian PRO-CTCAE labels from the NCI Italian item library PDF.

The NCI PRO-CTCAE Italian PDF has a regular structure:

    N. PRO-CTCAE(R) Symptom Term: <English canonical term>
    <ITALIAN LABEL (possibly multi-line)>
    a. Negli ultimi 7 giorni, ...

The item number N aligns with the canonical ordinal (PRO_00N). This script
parses all 80 items and writes an official-labels JSON to
``terminology/official/pro_ctcae_italian_labels.json``. That file is kept out of
version control (NCI Terms of Use) and merged into the built terminology by
``scripts/build_terminology.py``.

Requires: pypdf.  Usage:
    python scripts/extract_pro_ctcae_italian.py --pdf pro-ctcae_italian.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.terminology.canonical import CANONICAL_TERMS, canonical_id  # noqa: E402

HEADER = re.compile(r"^\s*(\d+)\.\s*PRO-CTCAE[®®]?\s*Symptom Term:\s*(.+?)\s*$")
QUESTION = re.compile(r"^\s*(?:[a-e]\.\s|Negli ultimi)")
NOISE = (
    "NCI-", "Item Library", "NATIONAL", "subject to", "Version date",
    "As individuals", "Quando un individuo", "For each question",
    "Per ciascuna", "===PAGE", "PRO-CTCAE",
)


def _read_pdf_text(pdf: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _clean_label(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if out.endswith("-"):          # de-hyphenate word split across lines
            out = out[:-1] + ln
        else:
            out = (out + " " + ln) if out else ln
    out = re.sub(r"\s+", " ", out).strip()
    out = out.replace("( ", "(").replace(" )", ")")
    return out


def _derived(label: str) -> list[str]:
    """Concise lowercase synonyms derived from an official label."""
    low = label.lower().strip()
    syns: list[str] = []
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", low)
    if m:
        primary = m.group(1).strip()
        paren = m.group(2).strip()
        if primary:
            syns.append(primary)
        # only keep a parenthetical if it is short (a real synonym, not a gloss)
        if paren and len(paren.split()) <= 3:
            syns.append(paren)
    else:
        syns.append(low)
    return [s for s in dict.fromkeys(syns) if s]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, default=_ROOT / "pro-ctcae_italian.pdf")
    ap.add_argument("--out", type=Path, default=_ROOT / "terminology/official/pro_ctcae_italian_labels.json")
    args = ap.parse_args()

    text = _read_pdf_text(args.pdf)
    lines = text.split("\n")
    items: dict[int, tuple[str, str]] = {}
    i = 0
    while i < len(lines):
        m = HEADER.match(lines[i])
        if not m:
            i += 1
            continue
        num = int(m.group(1))
        eng = m.group(2).strip()
        label_lines: list[str] = []
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if not s or any(s.startswith(n) or n in s for n in NOISE):
                j += 1
                continue
            if QUESTION.match(s) or HEADER.match(s):
                break
            label_lines.append(s)
            j += 1
        items[num] = (eng, _clean_label(label_lines))
        i = j

    missing = [n for n in range(1, 81) if n not in items]
    if missing:
        raise SystemExit(f"Failed to extract items: {missing}")

    canon = {ord_: eng for ord_, eng, _, _ in CANONICAL_TERMS}
    terms = {}
    mismatches = []
    for num in sorted(items):
        eng, label = items[num]
        cid = canonical_id(num)
        if eng.lower() != canon.get(num, "").lower():
            mismatches.append((num, eng, canon.get(num)))
        terms[cid] = {
            "canonical_english": canon.get(num, eng),
            "pdf_english": eng,
            "official_italian_label": label,
            "derived_synonyms": _derived(label),
        }

    doc = {
        "source": args.pdf.name,
        "source_version": "NCI PRO-CTCAE Italian Item Library Version 1.0",
        "version_date": "4/21/2022",
        "license_note": "NCI PRO-CTCAE Terms of Use apply; keep out of public VCS.",
        "term_count": len(terms),
        "terms": terms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(terms)} Italian labels -> {args.out}")
    if mismatches:
        print(f"NOTE: {len(mismatches)} English-name mismatches vs canonical (kept canonical):")
        for num, pdf_eng, canon_eng in mismatches:
            print(f"  {num}: pdf={pdf_eng!r} canon={canon_eng!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
