#!/usr/bin/env python
"""Extract the official CTCAE v6.0 dictionary from the NCI CTCAE v6.0 PDF.

The PDF linearizes each term block as:

    <SOC header>  <page>
    CTCAE v6.0
    <CTCAE Term>
    Grade 1 Grade 2 Grade 3 Grade 4 Grade 5
    <grade definitions (columns linearized -> stored raw)>
    Definition: <definition>
    Navigational Note: <note>

We reliably recover: CTCAE Term, SOC (System Organ Class), Definition, and the
raw grade block (columns are linearized by the text layer, so per-grade columns
are stored as raw text rather than falsely split). Grading remains a separate,
abstaining module.

CTCAE v6.0 is English (MedDRA). Italian matching for the fallback is provided
separately via a clearly-labelled Italian bridge (terminology/ctcae_italian_bridge.json)
and, from Phase 2, multilingual embeddings.

Requires: pypdf.  Usage:
    python scripts/extract_ctcae_v6.py --pdf CTCAEv6_Jan2026.pdf
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

GRADE_HDR = re.compile(r"^\s*Grade 1\s+Grade 2\s+Grade 3\s+Grade 4\s+Grade 5\s*$")
PAGE_MARK = re.compile(r"^===PAGE (\d+)===$")
SOC_TRAILING_NUM = re.compile(r"^(.*?)\s+\d+\s*$")


def _read_pdf_pages(pdf: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    return [(p.extract_text() or "") for p in reader.pages]


def parse(pages: list[str]) -> list[dict]:
    terms: list[dict] = []
    current_soc = ""
    for page in pages:
        lines = [ln.rstrip() for ln in page.split("\n")]
        # SOC header: on running pages it is "<SOC> <page>" BEFORE "CTCAE v6.0";
        # on the first page of a SOC the page number is alone before "CTCAE v6.0"
        # and the SOC title appears on the line AFTER it.
        for k, ln in enumerate(lines):
            if ln.strip() == "CTCAE v6.0" and k >= 1:
                before = lines[k - 1].strip()
                after = lines[k + 1].strip() if k + 1 < len(lines) else ""
                cand = ""
                if re.fullmatch(r"\d+", before):          # first page of a SOC
                    cand = after
                else:
                    m = SOC_TRAILING_NUM.match(before)     # running header
                    cand = (m.group(1) if m else before).strip()
                if cand and "CTCAE" not in cand and len(re.sub(r"[^A-Za-z]", "", cand)) >= 3:
                    current_soc = cand
                break
        # find terms on this page
        i = 0
        while i < len(lines):
            if GRADE_HDR.match(lines[i]):
                # term = nearest previous meaningful line (skip page header / SOC)
                term = ""
                j = i - 1
                while j >= 0:
                    s = lines[j].strip()
                    if (not s) or s == "CTCAE v6.0" or s == current_soc or re.search(r"\s\d+$", s):
                        j -= 1
                        continue
                    term = s
                    break
                # collect grade block + definition + nav note
                grade_lines = []
                definition = ""
                nav = ""
                k = i + 1
                while k < len(lines):
                    s = lines[k].strip()
                    if s.startswith("Definition:"):
                        definition = s[len("Definition:"):].strip()
                    elif s.startswith("Navigational Note:"):
                        nav = s[len("Navigational Note:"):].strip()
                        k += 1
                        break
                    elif GRADE_HDR.match(lines[k]) or s == "CTCAE v6.0":
                        break
                    elif not definition:
                        grade_lines.append(s)
                    k += 1
                if term:
                    terms.append({
                        "term": term,
                        "soc": current_soc,
                        "definition": definition,
                        "grade_block_raw": " ".join(g for g in grade_lines if g),
                        "navigational_note": nav,
                    })
                i = k
            else:
                i += 1
    return terms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, default=_ROOT / "CTCAEv6_Jan2026.pdf")
    ap.add_argument("--out", type=Path, default=_ROOT / "terminology/official/ctcae_v6.json")
    args = ap.parse_args()

    pages = _read_pdf_pages(args.pdf)
    raw_terms = parse(pages)

    # de-duplicate by (term, soc), assign ids
    seen = set()
    terms = []
    for t in raw_terms:
        key = (t["term"].lower(), t["soc"].lower())
        if key in seen:
            continue
        seen.add(key)
        terms.append(t)
    for idx, t in enumerate(terms, 1):
        t["ctcae_id"] = f"CTCAE_{idx:04d}"

    doc = {
        "schema_version": "ctcae_dictionary/1.0",
        "version": "6.0",
        "meddra_version": "28.0",
        "published": "2025-07-22",
        "source": args.pdf.name,
        "license_note": "NCI CTCAE Terms of Use apply; keep out of public VCS.",
        "language": "en",
        "term_count": len(terms),
        "terms": [
            {
                "ctcae_id": t["ctcae_id"],
                "term": t["term"],
                "soc": t["soc"],
                "italian_labels": [],
                "definition": t["definition"],
                "grade_block_raw": t["grade_block_raw"],
                "navigational_note": t["navigational_note"],
                "grade_definitions": {},
            }
            for t in terms
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(terms)} CTCAE v6.0 terms -> {args.out}")
    socs = sorted({t['soc'] for t in terms if t['soc']})
    print(f"{len(socs)} SOCs. Sample terms:")
    for t in terms[:12]:
        print(f"  [{t['soc'][:30]:30}] {t['term']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
