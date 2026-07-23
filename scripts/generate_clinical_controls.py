#!/usr/bin/env python
"""Generate synthetic clinical-control records from the factorial template.

Phase 1 implements the progressive-content SCALES (mobility, nails, risk) from
`data/templates/clinical_controls_template.yaml`. The full factorial expansion
(>=30 concepts x >=8 formulations) is completed in Phase 3.

All output is synthetic and clearly non-personal (spec section 17).

Usage:
    python scripts/generate_clinical_controls.py --out data/synthetic/scales.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=_ROOT / "data/templates/clinical_controls_template.yaml")
    parser.add_argument("--out", type=Path, default=_ROOT / "data/synthetic/scales.jsonl")
    args = parser.parse_args()

    tpl = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    scales = tpl.get("scales", {})
    rows: list[dict] = []
    i = 0
    for scale_name, scale in scales.items():
        steps = scale.get("steps", [])
        for step_idx, text in enumerate(steps):
            i += 1
            rows.append(
                {
                    "record_id": f"scale_{scale_name}_{step_idx:02d}",
                    "text": text,
                    "language": "it",
                    "meta": {"scale": scale_name, "step": step_idx, "synthetic": True},
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} synthetic scale records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
