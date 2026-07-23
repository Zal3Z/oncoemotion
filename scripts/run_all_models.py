#!/usr/bin/env python
"""[MULTI-MODEL] Run the full interpretability pipeline for several models.

For each model it runs, into a per-model output dir (outputs/models/<slug>/):
  build_vectors -> validate_vectors -> run_probing -> run_steering -> run_patching

Each step reuses the existing single-model scripts via subprocess (so the model
is reloaded per step — fine on an A100). Emotion vectors are rebuilt PER model:
directions live in each model's own representation space and are not transferable.

Default trio (China / Europe / USA), Colab A100 bf16:
  Qwen/Qwen3-8B · mistralai/Ministral-3-8B-Instruct-2512 · google/gemma-4-12B

Gated models (Mistral, Gemma): accept the license on their HF page and export
HF_TOKEN before running.

Usage:
    python scripts/run_all_models.py                      # default trio, bf16/auto
    python scripts/run_all_models.py --models Qwen/Qwen3-4B  # override
    python scripts/run_all_models.py --dtype float16 --device cuda  # single small GPU
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

DEFAULT_TRIO = [
    "Qwen/Qwen3-8B",                          # China — Alibaba, open
    "mistralai/Ministral-8B-Instruct-2410",   # Europe — Mistral (FR); plain-text decoder
    "google/gemma-4-12B",                     # USA — Google (unified config; text CausalLM)
]
# Note: Ministral-3-8B-Instruct-2512 is a MULTIMODAL (Mistral3) model that
# AutoModelForCausalLM can't load; Ministral-8B-Instruct-2410 is the clean text one.


def slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_TRIO)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    # All 4 methods by default — the vector build now runs on the GPU (torch),
    # so pca/logistic/lda are fast even at large hidden sizes.
    ap.add_argument("--methods", nargs="+", default=["diff_of_means", "pca", "logistic", "lda"])
    ap.add_argument("--outroot", type=Path, default=_ROOT / "outputs/models")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a model if its clinical_probing.json already exists")
    args = ap.parse_args()

    # shared, model-agnostic dataset
    run([PY, str(_ROOT / "scripts/generate_emotion_dataset.py")])

    summary = []
    for model_id in args.models:
        s = slug(model_id)
        d = args.outroot / s
        d.mkdir(parents=True, exist_ok=True)
        rep = d / "clinical_probing.json"
        if args.skip_existing and rep.exists():
            print(f"[skip] {model_id} (already done)")
            summary.append((model_id, "skipped"))
            continue

        acts, vecs = d / "emotion_acts.npz", d / "emotion_vectors.npz"
        val = d / "vector_validation.json"
        t0 = time.time()
        common = ["--model", model_id, "--dtype", args.dtype, "--device", args.device]
        steps = [
            [PY, str(_ROOT / "scripts/build_vectors.py"), *common,
             "--methods", *args.methods, "--acts-out", str(acts), "--vec-out", str(vecs)],
            [PY, str(_ROOT / "scripts/validate_vectors.py"),
             "--acts", str(acts), "--vecs", str(vecs), "--report", str(val),
             "--figure", str(d / "layer_sweep_auroc.png")],
            [PY, str(_ROOT / "scripts/run_probing.py"), *common,
             "--vecs", str(vecs), "--val-report", str(val),
             "--report", str(rep), "--figure", str(d / "clinical_gradients.png")],
            [PY, str(_ROOT / "scripts/run_steering.py"), *common,
             "--vecs", str(vecs), "--val-report", str(val),
             "--report", str(d / "steering_effects.json"), "--figure", str(d / "steering_effects.png")],
            [PY, str(_ROOT / "scripts/run_patching.py"), *common,
             "--vecs", str(vecs), "--val-report", str(val),
             "--report", str(d / "patching_effects.json")],
        ]
        ok = True
        for step in steps:
            if run(step) != 0:
                print(f"[FAIL] {model_id}: step {step[1].split('/')[-1]} failed", flush=True)
                ok = False
                break
        dt = time.time() - t0
        summary.append((model_id, f"{'ok' if ok else 'FAILED'} in {dt/60:.1f} min"))

    print("\n=== multi-model run summary ===")
    for m, st in summary:
        print(f"  {m:48} {st}")
    print(f"\nNow build the comparison:  {PY} scripts/compare_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
