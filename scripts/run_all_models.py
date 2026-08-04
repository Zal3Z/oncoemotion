#!/usr/bin/env python
"""[MULTI-MODEL] Run the full interpretability pipeline for several models.

For each model it runs, into a per-model output dir (outputs/models/<slug>/):
  build_vectors -> validate_vectors -> run_probing -> run_steering -> run_patching

Each step reuses the existing single-model scripts via subprocess (so the model
is reloaded per step — fine on an A100). Emotion vectors are rebuilt PER model:
directions live in each model's own representation space and are not transferable.

Default trio (China / Europe / USA), Colab A100 bf16:
  Qwen/Qwen3-8B · mistralai/Ministral-8B-Instruct-2410 · google/gemma-3-4b-it

Gated models (Mistral, Gemma): accept the license on their HF page and export
HF_TOKEN before running.

Usage:
    python scripts/run_all_models.py                      # default trio, bf16/auto
    python scripts/run_all_models.py --models Qwen/Qwen3-4B  # override
    python scripts/run_all_models.py --dtype float16 --device cuda  # single small GPU
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    "google/gemma-3-4b-it",                   # USA — Google, instruction-tuned
]
# Note: Ministral-3-8B-Instruct-2512 is a MULTIMODAL (Mistral3) model that
# AutoModelForCausalLM can't load; Ministral-8B-Instruct-2410 is the clean text one.
# Gemma-4 currently needs a newer multimodal Transformers stack and is therefore
# not a default model for the preregistered run.


def slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


VECTOR_INPUTS = [
    "configs/study_esmo_2026.yaml",
    "pyproject.toml",
    "src/oncoemotion/config.py",
    "src/oncoemotion/models/base.py",
    "src/oncoemotion/models/hf_decoder.py",
    "src/oncoemotion/activations/extract.py",
    "src/oncoemotion/emotion_vectors/build.py",
    "src/oncoemotion/emotion_vectors/dataset.py",
    "src/oncoemotion/emotion_vectors/seeds.py",
    "src/oncoemotion/emotion_vectors/vectors.py",
    "src/oncoemotion/clinical/baseline.py",
    "src/oncoemotion/clinical/prompt.py",
    "src/oncoemotion/probing/probe.py",
    "scripts/generate_emotion_dataset.py",
    "scripts/build_vectors.py",
    "scripts/validate_vectors.py",
]
STAGE_INPUTS = {
    "vectors": VECTOR_INPUTS,
    "probing": ["scripts/run_probing.py"],
    "steering": ["scripts/run_steering.py"],
    "patching": ["scripts/run_patching.py"],
}
ALL_STAGES = ["vectors", "probing", "steering", "patching"]


def _inputs_for(stages) -> list[str]:
    inputs = []
    for stage in stages:
        inputs.extend(STAGE_INPUTS[stage])
    return list(dict.fromkeys(inputs))


def pipeline_fingerprint(stages=ALL_STAGES) -> str:
    """Content hash of source/config files that feed the requested stages."""
    inputs = _inputs_for(stages)
    digest = hashlib.sha256()
    for rel in inputs:
        path = _ROOT / rel
        digest.update(rel.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _complete_manifest(path: Path, model_id: str, fingerprint: str, stages) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    completed = set(data.get("stages") or [])
    return bool(data.get("complete") and data.get("model_id") == model_id
                and data.get("pipeline_fingerprint") == fingerprint
                and set(stages).issubset(completed))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_TRIO)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    # All 4 methods by default — the vector build now runs on the GPU (torch),
    # so pca/logistic/lda are fast even at large hidden sizes.
    ap.add_argument("--methods", nargs="+", default=["diff_of_means", "pca", "logistic", "lda"])
    ap.add_argument("--outroot", type=Path, default=_ROOT / "outputs/models")
    ap.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES,
                    help="run only the requested stages; ESMO role study needs vectors")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a model if its clinical_probing.json already exists")
    args = ap.parse_args()

    # shared, model-agnostic dataset
    if run([PY, str(_ROOT / "scripts/generate_emotion_dataset.py")]) != 0:
        print("[FAIL] could not generate the shared emotion dataset", flush=True)
        return 1
    stages = list(dict.fromkeys(args.stages))
    if any(stage != "vectors" for stage in stages) and "vectors" not in stages:
        ap.error("probing/steering/patching require the vectors stage")
    fingerprint = pipeline_fingerprint(stages)

    summary = []
    for model_id in args.models:
        s = slug(model_id)
        d = args.outroot / s
        d.mkdir(parents=True, exist_ok=True)
        rep = d / "clinical_probing.json"
        manifest = d / "pipeline_manifest.json"
        if args.skip_existing and _complete_manifest(
                manifest, model_id, fingerprint, stages):
            print(f"[skip] {model_id} (complete manifest matches current pipeline)")
            summary.append((model_id, "skipped"))
            continue
        if args.skip_existing and rep.exists():
            print(f"[rerun] {model_id}: outputs exist but manifest is absent or stale")
        # A failed rerun must never leave an older "complete" marker behind.
        manifest.unlink(missing_ok=True)

        acts, vecs = d / "emotion_acts.npz", d / "emotion_vectors.npz"
        val = d / "vector_validation.json"
        t0 = time.time()
        common = ["--model", model_id, "--dtype", args.dtype, "--device", args.device]
        steps = []
        if "vectors" in stages:
            steps.extend([
                [PY, str(_ROOT / "scripts/build_vectors.py"), *common,
                 "--methods", *args.methods, "--acts-out", str(acts), "--vec-out", str(vecs)],
                [PY, str(_ROOT / "scripts/validate_vectors.py"),
                 "--acts", str(acts), "--vecs", str(vecs), "--report", str(val),
                 "--figure", str(d / "layer_sweep_auroc.png")],
            ])
        if "probing" in stages:
            steps.append([PY, str(_ROOT / "scripts/run_probing.py"), *common,
                          "--vecs", str(vecs), "--val-report", str(val),
                          "--report", str(rep), "--figure", str(d / "clinical_gradients.png")])
        if "steering" in stages:
            steps.append([PY, str(_ROOT / "scripts/run_steering.py"), *common,
                          "--vecs", str(vecs), "--val-report", str(val),
                          "--report", str(d / "steering_effects.json"),
                          "--figure", str(d / "steering_effects.png")])
        if "patching" in stages:
            steps.append([PY, str(_ROOT / "scripts/run_patching.py"), *common,
                          "--vecs", str(vecs), "--val-report", str(val),
                          "--report", str(d / "patching_effects.json")])
        ok = True
        for step in steps:
            if run(step) != 0:
                print(f"[FAIL] {model_id}: step {step[1].split('/')[-1]} failed", flush=True)
                ok = False
                break
        dt = time.time() - t0
        if ok:
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
            except Exception:
                git_commit = None
            manifest.write_text(json.dumps({
                "complete": True,
                "model_id": model_id,
                "git_commit": git_commit,
                "pipeline_fingerprint": fingerprint,
                "pipeline_inputs": _inputs_for(stages),
                "stages": stages,
                "dtype": args.dtype,
                "device": args.device,
                "methods": args.methods,
            }, indent=2), encoding="utf-8")
        summary.append((model_id, f"{'ok' if ok else 'FAILED'} in {dt/60:.1f} min"))

    print("\n=== multi-model run summary ===")
    for m, st in summary:
        print(f"  {m:48} {st}")
    print(f"\nNow build the comparison:  {PY} scripts/compare_models.py")
    return 1 if any("FAILED" in status for _, status in summary) else 0


if __name__ == "__main__":
    raise SystemExit(main())
