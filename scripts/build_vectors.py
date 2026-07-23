#!/usr/bin/env python
"""[PHASE 2] Extract activations and build emotion concept vectors (sections 7-8).

One-vs-rest construction: for each concept, positives are that concept's examples
and negatives are the neutral pool PLUS every OTHER concept's examples (so a
direction must capture the specific affect, not generic negative valence).

Emotion vectors are additionally saved in a RESIDUALIZED variant, orthogonalized
against the control (confounder) directions per layer (spec section 8) — this is
what Phase-3/4 use to reduce the negative-valence confound.

Usage (local T1000, FP16):
    python scripts/build_vectors.py --methods diff_of_means pca logistic lda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.config import ModelConfig  # noqa: E402
from oncoemotion.models.base import load_adapter  # noqa: E402
from oncoemotion.activations.extract import pooled_hidden_states  # noqa: E402
from oncoemotion.emotion_vectors.build import build_layer_vector  # noqa: E402
from oncoemotion.emotion_vectors.vectors import orthogonalize  # noqa: E402
from oncoemotion.emotion_vectors.dataset import build_dataset, load_jsonl, save_jsonl  # noqa: E402
from oncoemotion.emotion_vectors.seeds import EMOTION_SEEDS, CONTROL_SEEDS  # noqa: E402

EMOTIONS = list(EMOTION_SEEDS)
CONTROLS = list(CONTROL_SEEDS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--methods", nargs="+", default=["diff_of_means"])
    ap.add_argument("--dataset", type=Path, default=_ROOT / "data/synthetic/emotion_dataset.jsonl")
    ap.add_argument("--acts-out", type=Path, default=_ROOT / "outputs/activations/emotion_acts.npz")
    ap.add_argument("--vec-out", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    args = ap.parse_args()

    if args.dataset.exists():
        ds = load_jsonl(args.dataset)
    else:
        ds = build_dataset()
        save_jsonl(ds, args.dataset)
    texts = [e.text for e in ds]
    concepts_arr = np.array([e.concept for e in ds])
    splits = np.array([e.split for e in ds])
    print(f"Dataset: {len(ds)} examples ({int((concepts_arr!='neutral').sum())} concept + "
          f"{int((concepts_arr=='neutral').sum())} neutral), {len(set(EMOTIONS+CONTROLS))} concepts")

    cfg = ModelConfig(dtype=args.dtype, device_map=args.device)
    adapter = load_adapter(args.model, cfg)
    print(f"Loading {adapter.config.model_id} ({args.dtype}) on {args.device} ...", flush=True)
    t0 = time.time()
    adapter.load()
    print(f"  loaded in {time.time()-t0:.1f}s | layers={adapter.n_layers} hidden={adapter.hidden_size}")
    print("Extracting pooled hidden states ...", flush=True)
    t0 = time.time()
    acts = pooled_hidden_states(adapter, texts, pooling=args.pooling, progress=True)
    print(f"  activations {acts.shape} in {time.time()-t0:.1f}s")

    np.savez_compressed(args.acts_out, acts=acts.astype(np.float32), concepts=concepts_arr,
                        splits=splits, texts=np.array(texts, dtype=object),
                        model_id=adapter.config.model_id, pooling=args.pooling)
    args.acts_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saved activations -> {args.acts_out}")

    n_layers = acts.shape[1]
    ex = splits == "extraction"
    all_concepts = EMOTIONS + CONTROLS

    # 1) raw one-vs-rest vectors per concept/method/layer
    vec_store: dict[str, np.ndarray] = {}
    norm_store: dict[str, np.ndarray] = {}
    for concept in all_concepts:
        pos = ex & (concepts_arr == concept)
        neg = ex & (concepts_arr != concept)  # neutral + all other concepts
        if pos.sum() == 0 or neg.sum() == 0:
            print(f"  [skip] {concept}: missing pos/neg")
            continue
        y = np.concatenate([np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
        Xall = np.concatenate([acts[pos], acts[neg]], axis=0)  # [npos+nneg, L+1, H]
        for method in args.methods:
            lv = np.zeros((n_layers, acts.shape[2]), np.float32)
            nm = np.zeros(n_layers, np.float32)
            for l in range(n_layers):
                evv = build_layer_vector(Xall[:, l, :], y, method, concept, l)
                lv[l] = evv.vector.astype(np.float32)
                nm[l] = evv.original_norm
            vec_store[f"{concept}|{method}"] = lv
            norm_store[f"{concept}|{method}|norm"] = nm

    # 2) residualize EMOTION vectors against the CONTROL (confounder) directions
    #    (uses control diff_of_means vectors as the confounder basis, per layer)
    ctrl_key = "diff_of_means"
    n_resid = 0
    for concept in EMOTIONS:
        for method in args.methods:
            key = f"{concept}|{method}"
            if key not in vec_store:
                continue
            resid = np.zeros_like(vec_store[key])
            for l in range(n_layers):
                C = np.stack([vec_store[f"{c}|{ctrl_key}"][l] for c in CONTROLS
                              if f"{c}|{ctrl_key}" in vec_store])
                resid[l] = orthogonalize(vec_store[key][l], C).astype(np.float32)
            vec_store[f"{key}|resid"] = resid
            n_resid += 1

    np.savez_compressed(args.vec_out, **vec_store, **norm_store,
                        model_id=adapter.config.model_id, methods=np.array(args.methods, dtype=object),
                        emotions=np.array(EMOTIONS, dtype=object), controls=np.array(CONTROLS, dtype=object))
    args.vec_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saved {len([k for k in vec_store])} vector sets "
          f"({n_resid} residualized) -> {args.vec_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
