#!/usr/bin/env python
"""End-to-end smoke test of the whole chain, sized for a small local GPU.

The point is not the numbers -- with these limits they mean nothing. The point is
that every stage runs and every artefact appears with the fields the analyses
expect, so a change to prompts, seeds, layer selection or the ablation arms is
caught here instead of eight hours into a Colab session.

Sizing. Weights in fp16 need roughly 2 GB per billion parameters, plus room for
activations, so on an 8 GB card 1.5B is comfortable and 3B is the practical
ceiling. On CPU it works too, just slowly: pass --device cpu.

Usage:
    python scripts/smoke.py                              # Qwen2.5-1.5B on cuda
    python scripts/smoke.py --model Qwen/Qwen2.5-3B-Instruct
    python scripts/smoke.py --device cpu --stage data    # no model needed
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STAGES = ("data", "vectors", "validate", "role", "spectrum", "analysis")


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def _run(cmd: list[str], tag: str) -> bool:
    print(f"\n{'=' * 72}\n[{tag}] {' '.join(str(c) for c in cmd[1:])}\n{'=' * 72}", flush=True)
    t = time.time()
    rc = subprocess.call([str(c) for c in cmd], cwd=_ROOT)
    dt = time.time() - t
    print(f"[{tag}] {'ok' if rc == 0 else f'FALLITO (rc={rc})'} in {dt:.1f}s", flush=True)
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs/smoke")
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=STAGES,
                    help="run only these stages (default: all)")
    ap.add_argument("--pairs", type=int, default=6, help="item pairs for the role experiment")
    ap.add_argument("--baseline", type=int, default=8,
                    help="neutral baseline sentences per cell (full run uses all 53; "
                         "the baseline is re-measured per cell so it dominates a tiny run)")
    ap.add_argument("--stimuli", type=int, default=4, help="clinical stimuli for the spectrum")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    acts, vecs = out / "emotion_acts.npz", out / "emotion_vectors.npz"
    valrep = out / "vector_validation.json"
    common = ["--model", args.model, "--dtype", args.dtype, "--device", args.device]
    ok, skipped = True, []

    if "data" in args.stage:
        ok &= _run([PY, "scripts/generate_labeled_clinical.py", "--check-only"],
                   "item clinici: vincoli di disegno")
        ok &= _run([PY, "scripts/generate_emotion_dataset.py"], "dataset emozioni")
        if not ok:
            print("\n[stop] i vincoli di disegno non passano: correggere i seed prima di proseguire.")
            return 1

    if "vectors" in args.stage:
        ok &= _run([PY, "scripts/build_vectors.py", *common, "--methods", "diff_of_means",
                    "--acts-out", acts, "--vec-out", vecs], "vettori emotivi")
    if "validate" in args.stage and ok:
        ok &= _run([PY, "scripts/validate_vectors.py", "--acts", acts, "--vecs", vecs,
                    "--report", valrep, "--figure", out / "layer_sweep.png"],
                   "validazione (CV + layer condiviso + cancello lessicale)")

    role_dir = out / "role_emotion"
    if "role" in args.stage and ok:
        ok &= _run([PY, "scripts/run_role_emotion.py", *common,
                    "--limit", str(args.pairs),
                    "--arms", "intact", "emotion", "random",
                    "--ablation-limit", str(max(2, args.pairs // 2)),
                    "--baseline-limit", str(args.baseline),
                    "--vecs", vecs, "--val-report", valrep, "--out", role_dir],
                   "ruolo x emotivita (tre bracci)")
        rows = role_dir / f"{_slug(args.model)}__rows.jsonl"
        if rows.exists():
            ok &= _run([PY, "scripts/analyze_role_emotion.py", "--rows", rows],
                       "analisi per modello (sezione G = endpoint primario)")
        else:
            skipped.append(f"analisi per modello: {rows.name} non prodotto")

    spec_dir = out / "role_spectrum"
    if "spectrum" in args.stage and ok:
        ok &= _run([PY, "scripts/run_role_spectrum.py", *common,
                    "--limit", str(args.stimuli), "--null-draws", "200",
                    "--baseline-limit", str(args.baseline),
                    "--vecs", vecs, "--val-report", valrep, "--out", spec_dir],
                   "spettro di ruoli (con null casuale)")
        if any(spec_dir.glob("*__spectrum.json")):
            ok &= _run([PY, "scripts/reanalyze_direction.py", "--dir", spec_dir],
                       "C2 per rango")
        else:
            skipped.append("C2 per rango: nessuno spectrum.json prodotto")

    if "analysis" in args.stage and ok and any(role_dir.glob("*__rows.jsonl")):
        ok &= _run([PY, "scripts/analyze_results.py",
                    "--rows-glob", str(role_dir / "*__rows.jsonl"),
                    "--out", out / "primary_analysis.json"],
                   "analisi primaria aggregata")

    print(f"\n{'=' * 72}")
    if skipped:
        for s in skipped:
            print(f"[salta] {s}")
    if ok:
        print(f"smoke OK - artefatti in {out}")
        print("I numeri non vogliono dire niente con questi limiti: conta che ogni fase")
        print("abbia girato e che i campi attesi ci siano.")
    else:
        print("smoke FALLITO - vedi il primo passo con rc != 0 qui sopra.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
