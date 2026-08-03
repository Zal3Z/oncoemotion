#!/usr/bin/env python
"""Re-map the model's stored answers with the current term matcher. No GPU.

The generated string is saved in every row, so improving the surface list does not
require re-running a single model: the same answers can be scored again offline.

Why this exists
---------------
The first full run discarded 22-59% of answers per model (median ~30%) as
unmappable. Inspecting them showed the matcher, not the models, was failing: the
answer "Disfagia" IS PRO_002 "difficolta a deglutire", "Tinnitus" IS PRO_045,
"Epistassi" IS PRO_078. The accuracy metric was largely measuring which words
happened to be in the surface list.

Three distinct things were being lumped together, and they are now separated:
  * a clinical synonym the list did not have  -> a matcher bug, now fixed;
  * a correct abstention ("nessun evento avverso correlato al trattamento") -> the
    model declining properly, which was scored as a failure;
  * a non-answer ("0000000000", "CTCAE_5.0_Grade_", "E10.0") -> a real model failure
    mode, worth counting on its own rather than hiding inside "unmapped".

Usage:
    python scripts/rescore_rows.py --glob "oncoemotion_results_7/outputs/role_emotion/*__rows.jsonl"
    python scripts/rescore_rows.py --glob "..." --write        # sovrascrive le righe
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.clinical.classify import ABSTAIN_MARKERS, build_term_matcher  # noqa: E402
from oncoemotion.terminology.pro_ctcae import load_pro_ctcae  # noqa: E402

# Codes, identifiers and digit runs: the model failed to answer at all. Counting
# these as "unmapped" alongside genuine clinical synonyms hid a real failure mode.
_NONANSWER = re.compile(r"^(?:[\d\s.\-_/]+|[A-Z]\d{2}(?:\.\d+)?|CTCAE[_\s].*|PRO[_\s]?\d+)$", re.I)


def _classify(term: str, matcher, floor: float):
    t = (term or "").strip()
    low = t.lower().strip('.,;:"\' ')
    if not t or _NONANSWER.match(t):
        return None, "non_answer", 0.0
    if low in ABSTAIN_MARKERS:
        return None, "abstained", 0.0
    cid, _en, sc = matcher(t)
    if cid is not None and sc >= floor:
        return cid, "mapped", sc
    return None, "unmapped", sc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--floor", type=float, default=0.72)
    ap.add_argument("--write", action="store_true",
                    help="overwrite the rows in place (default: report only)")
    args = ap.parse_args()

    matcher = build_term_matcher(load_pro_ctcae())
    paths = sorted(_glob.glob(args.glob))
    if not paths:
        print(f"nessun file per {args.glob}")
        return 1

    print(f"{'modello':26s} {'acc prima':>10s} {'acc dopo':>9s} {'delta':>7s} "
          f"{'agganciate':>11s} {'astenute':>9s} {'non-risposte':>13s}")
    tot_before, tot_after = [], []
    for p in paths:
        rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
        kinds = Counter()
        before = after = n = 0
        for r in rows:
            if r.get("gold_class") != "term" or r.get("arm", "intact") != "intact":
                continue
            n += 1
            before += int(r.get("model_top1_id") == r.get("gold_pro_id"))
            cid, kind, sc = _classify(r.get("model_generated"), matcher, args.floor)
            kinds[kind] += 1
            after += int(cid == r.get("gold_pro_id"))
            r["rescored_top1_id"] = cid
            r["rescored_kind"] = kind
            r["rescored_score"] = round(sc, 3)
        if not n:
            continue
        b, a = before / n, after / n
        tot_before.append(b); tot_after.append(a)
        print(f"{Path(p).name.split('__')[0]:26s} {b:10.3f} {a:9.3f} {a-b:+7.3f} "
              f"{kinds['mapped']/n:10.1%} {kinds['abstained']/n:8.1%} {kinds['non_answer']/n:12.1%}")
        if args.write:
            with Path(p).open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import statistics as st
    print(f"\naccuratezza mediana: {st.median(tot_before):.3f} -> {st.median(tot_after):.3f} "
          f"({st.median(tot_after) - st.median(tot_before):+.3f})")
    print("Le righe" + (" sono state riscritte." if args.write else
                        " NON sono state toccate: rilancia con --write per applicare."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
