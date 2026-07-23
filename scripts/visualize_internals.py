#!/usr/bin/env python
"""[VIZ] Animated "x-ray" of the model's internals during the PRO-CTCAE decision.

For each of several clinical inputs, captures hidden states at EVERY token and
EVERY layer, projects them onto the emotion/control directions, z-scores against a
neutral baseline built from ALL neutral tokens (non-degenerate scale), and emits:

  * a combined JSON (all sentences) for the self-contained interactive player;
  * one static montage PNG per sentence;
  * optional GIFs (per sentence) with --gifs.

Layer 0 (embedding) is excluded from displays — there the projection is mere token
identity, not an emotion-like representation. These show emotion-LIKE internal
representations, not conscious emotions.

Usage:
    python scripts/visualize_internals.py            # default sentence set
    python scripts/visualize_internals.py --gifs     # also render GIFs
    python scripts/visualize_internals.py --texts-file my_sentences.txt  # "label | text" per line
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.config import ModelConfig  # noqa: E402
from oncoemotion.models.base import load_adapter  # noqa: E402
from oncoemotion.clinical.prompt import build_decision_prompt  # noqa: E402
from oncoemotion.emotion_vectors.seeds import NEUTRAL  # noqa: E402

SHOW = ["afraid_alarmed", "anxious_nervous", "sad", "calm"]
L0_SKIP = 1

# (label, sentence). Spans severity, formulation and affect — mostly WITHOUT
# explicit emotion words so the internal response to clinical content is visible.
DEFAULT_SENTENCES = [
    ("dolore severo", "ho un dolore lancinante e insopportabile che non passa"),
    ("dolore lieve", "ho un lieve fastidio ogni tanto"),
    ("respiro grave", "faccio molta fatica a respirare anche stando fermo"),
    ("prognosi negativa", "mi hanno detto che le cure non stanno più funzionando"),
    ("negazione/risolto", "non ho più dolore, ora va molto meglio"),
    ("ansia esplicita", "sono terrorizzato all'idea dei risultati degli esami"),
    ("controllo neutro", "la visita di controllo è fissata per giovedì mattina"),
]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:24]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--method", default="diff_of_means")
    ap.add_argument("--vecs", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    ap.add_argument("--val-report", type=Path, default=_ROOT / "outputs/reports/vector_validation.json")
    ap.add_argument("--outdir", type=Path, default=_ROOT / "outputs/figures")
    ap.add_argument("--texts-file", type=Path, default=None)
    ap.add_argument("--gifs", action="store_true", help="also render per-sentence GIFs")
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()

    if args.texts_file and args.texts_file.exists():
        sentences = []
        for ln in args.texts_file.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            lbl, _, txt = ln.partition("|")
            sentences.append((lbl.strip() or txt.strip(), (txt or lbl).strip()))
    else:
        sentences = DEFAULT_SENTENCES

    V = np.load(args.vecs, allow_pickle=True)
    val = json.loads(args.val_report.read_text(encoding="utf-8")) if args.val_report.exists() else {"concepts": {}}

    def key(c):
        rk = f"{c}|{args.method}|resid"
        return rk if rk in V else f"{c}|{args.method}"

    concepts = [c for c in SHOW if key(c) in V]
    vecs = {c: V[key(c)] for c in concepts}
    n_layers = next(iter(vecs.values())).shape[0]
    best_layer = {c: max(L0_SKIP, int(val["concepts"].get(c, {}).get("best_layer", 3 * n_layers // 4)))
                  for c in concepts}
    U = {c: np.stack([unit(vecs[c][l]) for l in range(n_layers)]) for c in concepts}

    adapter = load_adapter(args.model, ModelConfig())
    print(f"Loading {adapter.config.model_id} ...", flush=True)
    adapter.load()

    def project_all_tokens(prompt):
        cap = adapter.forward_capture(prompt)
        hs = np.stack([h[0].float().cpu().numpy() for h in cap["hidden_states"]], axis=0)  # [L+1,seq,H]
        ids = cap["input_ids"][0].tolist()
        toks = [adapter.tokenizer.decode([i]).replace("\n", "\\n") for i in ids]
        P = {c: np.einsum("lsh,lh->ls", hs, U[c]) for c in concepts}
        return P, toks

    print("Computing neutral baseline over all tokens ...", flush=True)
    acc = {c: [[] for _ in range(n_layers)] for c in concepts}
    for t in NEUTRAL[:12]:
        P, _ = project_all_tokens(build_decision_prompt(t))
        for c in concepts:
            for l in range(n_layers):
                acc[c][l].extend(P[c][l].tolist())
    bmean = {c: np.array([np.mean(acc[c][l]) for l in range(n_layers)]) for c in concepts}
    bstd = {c: np.array([np.std(acc[c][l]) + 1e-6 for l in range(n_layers)]) for c in concepts}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_Z = []
    payload_sentences = []
    for label, text in sentences:
        print(f"Capturing: {label!r} — {text!r}", flush=True)
        P, toks = project_all_tokens(build_decision_prompt(text))
        Z = {c: (P[c] - bmean[c][:, None]) / bstd[c][:, None] for c in concepts}
        all_Z.append(Z)
        payload_sentences.append({
            "label": label, "text": text, "tokens": toks,
            "traj": {c: [round(float(v), 3) for v in Z[c][best_layer[c]]] for c in concepts},
            "heat": {c: [[round(float(Z[c][l][t]), 2) for l in range(n_layers)] for t in range(len(toks))]
                     for c in concepts},
        })

    # shared colour limit across all sentences (layers >= 1)
    vlim = float(np.percentile(
        np.abs(np.concatenate([np.stack([Z[c] for c in concepts])[:, L0_SKIP:, :].ravel() for Z in all_Z])), 97))

    # per-sentence montage PNGs
    for (label, text), Z in zip(sentences, all_Z):
        stackZ = np.stack([Z[c] for c in concepts])
        T = stackZ.shape[2]
        snaps = sorted(set(list(range(0, T, max(1, (T - 1) // 5))) + [T - 1]))[:6]
        toks = payload_sentences[sentences.index((label, text))]["tokens"]
        fig, axes = plt.subplots(1, len(snaps), figsize=(3.0 * len(snaps), 3.0), sharey=True)
        for ax, i in zip(np.atleast_1d(axes), snaps):
            ax.imshow(stackZ[:, L0_SKIP:, i], aspect="auto", cmap="coolwarm", vmin=-vlim, vmax=vlim,
                      extent=[L0_SKIP, n_layers, len(concepts) - 0.5, -0.5])
            ax.set_yticks(range(len(concepts)))
            ax.set_yticklabels(concepts if i == snaps[0] else [], fontsize=7)
            tok = toks[i].strip() or "·"
            ax.set_title((f"E: '{tok}'" if i == T - 1 else f"tok {i}: '{tok}'")[:16], fontsize=8)
            ax.set_xlabel("layer", fontsize=7)
        fig.suptitle(f'[{label}] "{text}"', fontsize=9)
        fig.tight_layout()
        fig.savefig(args.outdir / f"internal_montage_{slug(label)}.png", dpi=120)
        plt.close(fig)
    print(f"Wrote {len(sentences)} montage PNGs")

    payload = {"model_id": adapter.config.model_id, "concepts": concepts,
               "best_layer": best_layer, "n_layers": n_layers, "vlim": round(vlim, 3),
               "sentences": payload_sentences}
    out_json = _ROOT / "outputs/reports/internal_trajectory_multi.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_json} ({out_json.stat().st_size/1024:.0f} KB, {len(sentences)} sentences)")

    # optional GIFs (first sentence only unless --gifs renders all)
    if args.gifs:
        from matplotlib.animation import FuncAnimation, PillowWriter
        for (label, text), Z in zip(sentences, all_Z):
            toks = payload_sentences[sentences.index((label, text))]["tokens"]
            T = len(toks); x = np.arange(T)
            fig, ax = plt.subplots(figsize=(11, 4))
            ys = np.concatenate([Z[c][best_layer[c]] for c in concepts])
            ax.set_ylim(np.percentile(ys, 1) - 1, np.percentile(ys, 99) + 1); ax.set_xlim(-.5, T - .5)
            ax.axhline(0, color="gray", lw=.6, ls=":")
            lines = {c: ax.plot([], [], "-o", ms=3, label=f"{c} L{best_layer[c]}")[0] for c in concepts}
            ax.legend(fontsize=8); ttl = ax.set_title("")

            def fr(i, Z=Z, lines=lines, toks=toks, T=T):
                for c in concepts:
                    lines[c].set_data(x[:i + 1], Z[c][best_layer[c]][:i + 1])
                ttl.set_text(f"[{label}] token {i+1}/{T}: '{toks[i].strip()}'" + (" ← E" if i == T - 1 else ""))
                return list(lines.values()) + [ttl]

            FuncAnimation(fig, fr, frames=T, blit=False).save(
                args.outdir / f"internal_traj_{slug(label)}.gif", writer=PillowWriter(fps=args.fps))
            plt.close(fig)
        print(f"Wrote {len(sentences)} trajectory GIFs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
