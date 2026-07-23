#!/usr/bin/env python
"""Run the baseline PRO-CTCAE/CTCAE mapper over a JSONL of records.

Reads records ({"record_id","text","language",...}) and writes one MapResponse
JSON per line. Logging is PII-safe: free text is never logged.

Usage:
    python scripts/run_mapping_baseline.py --input data/synthetic/clinical_controls.jsonl
    python scripts/run_mapping_baseline.py --input in.jsonl --output outputs/tables/mapping.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oncoemotion.factory import build_default_mapper  # noqa: E402
from oncoemotion.schemas import MapRequest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true", help="print a human summary")
    args = parser.parse_args()

    mapper = build_default_mapper()
    out_path = args.output
    out_f = out_path.open("w", encoding="utf-8") if out_path else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        req = MapRequest(**rec)
        resp = mapper.map(req)
        n += 1
        payload = resp.model_dump()
        if out_f:
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if args.pretty or not out_f:
            top = resp.pro_ctcae.predictions[0].term if resp.pro_ctcae.predictions else "-"
            ctc = resp.ctcae.predictions[0].term if resp.ctcae.predictions else "-"
            flag = " [URGENT]" if resp.safety.urgent_human_review else ""
            print(
                f"{req.record_id:>5} | PRO={resp.pro_ctcae.status:<28} top={top:<22} "
                f"CTCAE={resp.ctcae.status:<12} ({ctc}) abstain={resp.abstain}{flag}"
            )
    if out_f:
        out_f.close()
        print(f"\nWrote {n} responses -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
