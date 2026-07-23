#!/usr/bin/env python
"""[PHASE 4b] Activation patching (spec section 12).

Cache a SOURCE run's activation at (layer, point E) and transfer it into a
RECIPIENT run — either the full activation or only its component along the emotion
direction — then measure the change in the recipient's decision. Controls: a
random direction and full-activation transfer.

Usage:
    python scripts/run_patching.py --concept afraid_alarmed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.config import ModelConfig  # noqa: E402
from oncoemotion.models.base import load_adapter  # noqa: E402
from oncoemotion.patching.runtime import PatchingRuntime  # noqa: E402
from oncoemotion.clinical.prompt import build_decision_prompt  # noqa: E402
from oncoemotion.emotion_vectors.vectors import random_vector  # noqa: E402

# source (high-affect) -> recipient (low-affect) pairs
PAIRS = [
    ("severe->mild", "ho un dolore lancinante e insopportabile che non passa",
     "ho un lieve fastidio ogni tanto"),
    ("severe->neutral", "non riesco quasi più a respirare e ho paura di soffocare",
     "ho la nausea"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--method", default="diff_of_means")
    ap.add_argument("--variant", default="resid", choices=["resid", "raw"])
    ap.add_argument("--concept", default="afraid_alarmed")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--vecs", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    ap.add_argument("--val-report", type=Path, default=_ROOT / "outputs/reports/vector_validation.json")
    ap.add_argument("--report", type=Path, default=_ROOT / "outputs/reports/patching_effects.json")
    args = ap.parse_args()

    V = np.load(args.vecs, allow_pickle=True)
    val = json.loads(args.val_report.read_text(encoding="utf-8"))
    layer = args.layer if args.layer >= 0 else val["concepts"][args.concept]["best_layer"]
    rk = f"{args.concept}|{args.method}|resid"
    key = rk if (args.variant != "raw" and rk in V) else f"{args.concept}|{args.method}"
    direction = V[key][layer]
    rand_dir = random_vector(direction.shape[0], seed=2024, norm=float(np.linalg.norm(direction)))

    cfg = ModelConfig(dtype=args.dtype, device_map=args.device)
    adapter = load_adapter(args.model, cfg)
    print(f"Loading {adapter.config.model_id} ...", flush=True)
    adapter.load()
    pr = PatchingRuntime(adapter)
    print(f"Patching at layer {layer} (concept={args.concept}, key={key})", flush=True)

    results = {"model_id": adapter.config.model_id, "layer": int(layer),
               "concept": args.concept, "vector_key": key, "pairs": {}}
    for tag, src, rec in PAIRS:
        sp, rp = build_decision_prompt(src), build_decision_prompt(rec)
        emo = pr.patch_and_summarize(sp, rp, layer, direction, mode="direction")
        rnd = pr.patch_and_summarize(sp, rp, layer, rand_dir, mode="direction")
        full = pr.patch_and_summarize(sp, rp, layer, direction, mode="full")
        results["pairs"][tag] = {
            "source": src, "recipient": rec,
            "emotion_direction": {k: round(v, 4) if isinstance(v, float) else v for k, v in emo.items()
                                  if k in ("delta_entropy", "delta_margin", "top1_changed")},
            "random_direction": {k: round(v, 4) if isinstance(v, float) else v for k, v in rnd.items()
                                 if k in ("delta_entropy", "delta_margin", "top1_changed")},
            "full_activation": {k: round(v, 4) if isinstance(v, float) else v for k, v in full.items()
                                if k in ("delta_entropy", "delta_margin", "top1_changed")},
        }
        print(f"  {tag}: emotion dEnt={emo['delta_entropy']:+.3f} (flip={emo['top1_changed']}) | "
              f"random dEnt={rnd['delta_entropy']:+.3f} | full dEnt={full['delta_entropy']:+.3f} (flip={full['top1_changed']})")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
