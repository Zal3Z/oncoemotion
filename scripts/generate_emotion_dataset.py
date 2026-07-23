#!/usr/bin/env python
"""[PHASE 2] Build the emotion-concept dataset (spec section 7).

Expands the Italian seeds into contrastive examples (positive = concept present,
negative = neutral / negation / metalanguage), tagged with variety and split
(extraction / validation / held-out test), independent of the clinical fields.

Usage:
    python scripts/generate_emotion_dataset.py --out data/synthetic/emotion_dataset.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.emotion_vectors.dataset import build_dataset, save_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_ROOT / "data/synthetic/emotion_dataset.jsonl")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ds = build_dataset(seed=args.seed)
    save_jsonl(ds, args.out)
    by_concept = Counter(e.concept for e in ds)
    by_split = Counter(e.split for e in ds)
    by_label = Counter(e.label for e in ds)
    print(f"Wrote {len(ds)} examples -> {args.out}")
    print(f"  concepts: {len(by_concept)}  splits: {dict(by_split)}  labels: {dict(by_label)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
