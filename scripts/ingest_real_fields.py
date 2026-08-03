#!/usr/bin/env python
"""Ingest the real PRO-CTCAE free-text fields with the physician's associations.

Turns the clinician-annotated spreadsheet into the JSONL schema the rest of the
pipeline already reads, so no experiment code changes to run on real text.

What the source carries that the synthetic set cannot
-----------------------------------------------------
* a **grade** (0-5) assigned per entry, which is the variable the strongest result
  so far needs -- the model's fear axis tracks clinical severity (r ~ +0.67 across
  12 of 13 models) and until now that was only measurable on synthetic gradients;
* a **three-way** decision by a clinician: a PRO-CTCAE code, a CTCAE-v5 term with no
  PRO equivalent, or not codeable at all. That is the same three-class structure the
  synthetic categories were built to imitate, made by a physician on real text;
* real surface form -- median 3 words, 65% at three words or fewer, 22% exact
  duplicates.

Limits that must travel with the data
-------------------------------------
* **One annotator.** There is no second reading, so there is no inter-rater ceiling:
  we can measure agreement with this physician but not what agreement is achievable.
* **Duplicates.** 22% of the strings repeat. ``source_id`` groups them so they can be
  collapsed or given clustered standard errors; treating them as independent makes
  every interval too narrow.
* **The tail is the informative part.** Two thirds of entries are three words or
  fewer, usually a bare symptom name that any lexical system codes correctly. The
  >=7-word tail is ~10% of rows and is where a model can actually differ.

Usage:
    python scripts/ingest_real_fields.py --xlsx sinomi_campi_aperti.xlsx
    python scripts/ingest_real_fields.py --xlsx ... --min-words 7   # solo la coda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# The physician's item names are PRO-CTCAE concepts under different labels: British
# spellings, clinical synonyms, and abbreviations. Written out once rather than
# fuzzy-matched, because a wrong automatic match here silently corrupts the gold.
ITEM_ALIASES = {
    "pain": "PRO_048",                      # General pain
    "hand_foot_neuropathy": "PRO_039",      # Numbness & tingling (hands/feet)
    "sore_mouth": "PRO_003",                # Mouth/throat sores
    "arm_leg_swelling": "PRO_022",          # Swelling (arms or legs)
    "blurry_vision": "PRO_041",             # Blurred vision
    "itchy_skin": "PRO_028",                # Itching
    "appetite": "PRO_008",                  # Decreased appetite
    "epistaxis": "PRO_078",                 # Nosebleed
    "anxiety": "PRO_054",                   # Anxious
    "dysgeusia": "PRO_007",                 # Taste changes
    "depression": "PRO_055",                # Discouraged -- see AMBIGUOUS below
    "dyspnoea": "PRO_019",                  # Shortness of breath
    "swallowing": "PRO_002",                # Difficulty swallowing
    "dry_skin": "PRO_025",                  # Skin dryness
    "diarrhoea": "PRO_016",                 # Diarrhea (British spelling)
    "hot_flushes": "PRO_077",               # Hot flashes
    "flatulence": "PRO_012",                # Gas
    "excessive_sweating": "PRO_075",        # Increased sweating
    "palpitations": "PRO_023",              # Heart palpitations
    "sunlight_sensitivity": "PRO_034",      # Sensitivity to sunlight
    "bruise": "PRO_073",                    # Bruising
    "breast_tenderness": "PRO_072",         # Breast swelling and tenderness
    "voice_changes": "PRO_005",             # Voice quality changes
    "erectyl_disfunction": "PRO_066",       # Achieve and maintain erection
    "loss_of_nails": "PRO_031",             # Nail loss
    "ejaculation_problems": "PRO_067",      # Ejaculation
    "hoarse_voice": "PRO_006",              # Hoarseness
    "injection_site_problems": "PRO_079",   # Pain and swelling at injection site
    "libido_loss": "PRO_068",               # Decreased libido
    "urgent_urination": "PRO_062",          # Urinary urgency
    "irregular_menses": "PRO_057",          # Irregular periods/vaginal bleeding
}

# Judgement calls, recorded so a reviewer can disagree with the mapping rather than
# with a number that depends on it. Both are reported in the summary.
AMBIGUOUS = {
    "depression": "PRO_055 (Discouraged) vs PRO_056 (Sad): PRO-CTCAE has no "
                  "'depression' item; Discouraged is the closer construct.",
    "hand_foot_neuropathy": "PRO_039 (Numbness & tingling in hands/feet) vs PRO_030 "
                            "(Hand-foot syndrome, a skin reaction). Read as neuropathy.",
}

EXACT, NOD, INS = "EXACT_PRO_MATCH", "NO_DIRECT_PRO_MATCH", "INSUFFICIENT_CONTEXT"


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _source_id(text: str) -> str:
    """Stable key for identical strings, so the 22% duplicates can be collapsed."""
    return hashlib.sha1(_norm(text).encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=_ROOT / "data/real/clinical_real.jsonl")
    ap.add_argument("--min-words", type=int, default=0,
                    help="keep only entries with at least N words (7 = the informative tail)")
    ap.add_argument("--dedup", action="store_true",
                    help="keep one row per distinct text instead of all repeats")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_excel(args.xlsx)
    lib = json.loads((_ROOT / "terminology/pro_ctcae_terms.json").read_text(encoding="utf-8"))
    by_canon = {_norm(t["canonical_english"]): t for t in lib["terms"]}
    by_id = {t["canonical_id"]: t for t in lib["terms"]}

    recs, unmapped, seen = [], Counter(), set()
    for i, row in df.iterrows():
        text = row.get("campo_aperto")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if args.min_words and len(text.split()) < args.min_words:
            continue
        sid = _source_id(text)
        if args.dedup and sid in seen:
            continue
        seen.add(sid)

        src = str(row.get("fonte_associazione") or "").strip()
        item = row.get("item_associato")
        key = _norm(item) if isinstance(item, str) else ""
        pro_id = None
        if src == "PRO-CTCAE":
            pro_id = ITEM_ALIASES.get(key) or (by_canon[key]["canonical_id"] if key in by_canon else None)
            if pro_id is None:
                unmapped[item] += 1
                continue
            cls, cat, status = "term", EXACT, EXACT
        elif src == "CTCAE v5":
            # a real clinical term exists but has no PRO-CTCAE equivalent: coding a
            # PRO term here is a false positive, which is what the class tests
            cls, cat, status = "abstain", NOD, NOD
        else:
            cls, cat, status = "abstain", INS, INS

        grade = row.get("valore_associato")
        recs.append({
            "record_id": f"real_{i:05d}",
            "pair_id": sid,          # no neutral/emotional pairing on real text
            "framing": "real",
            "text": text,
            "language": "it",
            "category": cat,
            "gold_class": cls,
            "gold_pro_id": pro_id,
            "gold_pro_term": by_id[pro_id]["canonical_english"] if pro_id else None,
            "gold_pro_status": status,
            "gold_ctcae_term": str(item) if src == "CTCAE v5" and isinstance(item, str) else None,
            "urgent": False,
            "grade": int(grade) if pd.notna(grade) else None,
            "source_id": sid,        # groups the duplicates
            "assessment_id": None,   # not available: no patient/visit key in the source
            "n_words": len(text.split()),
            "annotation_source": src,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_term = sum(1 for r in recs if r["gold_class"] == "term")
    dup = 1 - len({r["source_id"] for r in recs}) / len(recs) if recs else 0
    g = Counter(r["grade"] for r in recs if r["grade"] is not None)
    print(f"{len(recs)} righe -> {args.out}")
    print(f"  term (codice PRO): {n_term}   astensione: {len(recs) - n_term}")
    print(f"  categorie: {dict(Counter(r['category'] for r in recs))}")
    print(f"  gradi: {dict(sorted(g.items()))}")
    print(f"  duplicati residui: {dup:.1%}   termini PRO distinti: "
          f"{len({r['gold_pro_id'] for r in recs if r['gold_pro_id']})}")
    w = sorted(r["n_words"] for r in recs)
    print(f"  parole: mediana {w[len(w)//2]}   >=7 parole: "
          f"{sum(1 for x in w if x >= 7) / len(w):.1%}")
    if unmapped:
        print(f"\n[!] {sum(unmapped.values())} righe scartate, item non mappato:")
        for k, v in unmapped.most_common(10):
            print(f"      {v:4d}  {k}")
    print("\nMappature discutibili (dichiararle, non nasconderle):")
    for k, why in AMBIGUOUS.items():
        print(f"  {k}: {why}")
    print("\nUn solo annotatore: nessun tetto di concordanza noto. La concordanza col")
    print("modello va letta contro quel limite, non contro il 100%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
