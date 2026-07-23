# oncoemotion

**PRO-CTCAE / CTCAE symptom mapping + emotion-concept mechanistic
interpretability for oncology symptom free-text (Italian).**

This project studies whether open-field oncology-symptom inputs induce
*emotion-like internal representations* in an open-weight language model during
its association of that text with PRO-CTCAE / CTCAE terminology, replicating and
extending the emotion-concepts + activation-steering paradigm in the clinical
domain.

> **Framing.** This repository makes **no claim** that the model possesses
> consciousness, sentience, or subjective experience. It studies *emotion-like
> internal representations*, *emotion concepts*, *functional states associated
> with an emotion*, and *causally relevant signals*. It is a **research and
> clinician-support tool**; it must not perform autonomous diagnosis nor replace
> clinical review. Inputs indicating potential immediate risk are routed to an
> organization-defined human workflow.

---

## Status: Phase 1 complete ✅

| Phase | Scope | State |
|---|---|---|
| **1** | Repo, terminology loader (80 PRO-CTCAE terms), baseline mapper, safety routing, API, tests | ✅ done, 82 tests green |
| **1b** | Official terminology integrated: PRO-CTCAE Italian labels (80) + CTCAE v6.0 (850 terms) + IT→CTCAE bridge | ✅ done |
| **2** | Activation extraction, emotion-concept dataset, vector construction (4 methods), layer sweep, validation | ✅ ran on the local T1000 (Qwen2.5-3B FP16) |
| **3** | Controlled clinical dataset, point-E emotion probing, confounder analysis, persistence | ✅ ran — see [phase3_report.md](outputs/reports/phase3_report.md) |
| **4** | Steering / ablation causal experiment at the decision point | ✅ ran (raw + residualized) |
| **4b** | Activation patching (source→recipient, emotion-direction only) | ✅ ran |
| **5** | Docker, Streamlit dashboard, final scientific report | ✅ [docs/REPORT.md](docs/REPORT.md), [Dockerfile](Dockerfile), [dashboard](dashboard/streamlit_app.py) |

**Final synthesis (residualized run):** emotion concepts are decodable and mutually
distinct (one-vs-rest held-out AUROC 0.78–1.00, deep layers); residualization
**removed the fear↔negative-valence confound (r 0.76 → 0.00)**; but once that
confound is removed the fear↔severity link becomes inconsistent and **causal
steering/patching do not exceed random-vector controls** (0/7 decision flips). So:
emotion-like representations exist and persist to the decision, but this experiment
does **not** establish a fear-specific *causal* driver of the coding decision. Full
write-up: [docs/REPORT.md](docs/REPORT.md).

### Phase 3/4 — how to run

```bash
python scripts/run_probing.py    # point-E emotion scores vs severity, confounders, persistence
python scripts/run_steering.py   # causal steering/ablation effect on the decision
python scripts/run_patching.py   # activation patching (source→recipient)
python scripts/visualize_internals.py   # animated token×layer "x-ray" (GIFs + montage + interactive HTML)
python scripts/run_all_models.py && python scripts/compare_models.py   # multi-model comparison
```

**Multi-model comparison (China / Europe / USA).** `run_all_models.py` runs the
whole pipeline per model into `outputs/models/<slug>/`; `compare_models.py`
aggregates them. Default trio on a Colab A100: `Qwen/Qwen3-8B` ·
`mistralai/Ministral-8B-Instruct-2410` · `google/gemma-4-12B` (the last two are
gated — accept the license + set `HF_TOKEN`). Ready-to-run notebook:
[notebooks/colab_multimodel.ipynb](notebooks/colab_multimodel.ipynb); full guide:
[docs/MULTIMODEL.md](docs/MULTIMODEL.md).

**Internal x-ray:** `scripts/visualize_internals.py` renders a token-by-token,
layer-by-layer view of the emotion-like directions as the model reads a symptom —
`outputs/figures/internal_token_trajectory.gif`, `internal_layer_heatmap.gif`,
`internal_montage.png`, and a self-contained interactive player
`internal_player.html`. It shows `afraid_alarmed` (L33) staying low during the
neutral instruction and rising as "dolore lancinante e insopportabile" is read,
peaking just before the decision point E (no explicit emotion words). A single
token's readout is exploratory, not a validated per-token measure.

**Phase 3 headline (illustrative, small dataset):** at the pre-decision point E,
the `afraid_alarmed` / `anxious_nervous` directions rise with symptom severity
(Pearson +0.7–0.9 across mobility/pain/breath/nausea; +4.2 SD at the severe step),
the signal **persists** through an inserted neutral sentence, and `safety_policy`
stays flat (0.0) — but `afraid_alarmed` is **strongly correlated with generic
negative valence (r≈0.76)**, so an emotion-specific claim is not yet warranted.
Details + caveats in [phase3_report.md](outputs/reports/phase3_report.md).

**Phase 4 headline:** steering the emotion direction at layer 31 causally shifts
the decision entropy dose-dependently, **but a random same-norm vector and the
opposite emotion produce comparable shifts** — so the effect is not
emotion-specific. The controls correctly prevent over-claiming; top-1 decision
flips are rare. See [phase4_report.md](outputs/reports/phase4_report.md).

### Phase 2 — how to run (local T1000, FP16)

```bash
.venv/Scripts/python.exe -m pip install -e ".[ml]"        # torch cu124 + transformers + sklearn
.venv/Scripts/python.exe scripts/generate_emotion_dataset.py
.venv/Scripts/python.exe scripts/build_vectors.py --methods diff_of_means pca logistic lda
.venv/Scripts/python.exe scripts/validate_vectors.py       # -> outputs/reports/vector_validation.json + figure
```

Builds emotion + control (confounder) direction vectors per layer from Italian,
non-clinical contrastive text (independent of the clinical fields), then measures
held-out AUROC per layer with bootstrap CIs and cross-concept collinearity.

> **Dataset-size caveat.** The shipped synthetic emotion dataset is small
> (~18 examples/concept) for a fast first pass — results are *illustrative*.
> Scale to the spec's 50–100/condition by extending
> `src/oncoemotion/emotion_vectors/seeds.py` (and set
> `extraction.min_examples_per_condition: 50` in `configs/experiment.yaml`)
> before drawing scientific conclusions.

Phase 1 runs with **zero ML dependencies** (no torch): the mapper is a
deterministic lexical+fuzzy engine so terminology mapping, safety routing, the
API and the full test suite work on the local machine and in CI immediately.

---

## Key decisions (this machine)

- **GPU:** NVIDIA **T1000 8GB** (Turing, compute 7.5). Turing has no efficient
  bf16 → local runs use **float16**. On Colab (Ampere+) switch to bfloat16.
- **Default model (local):** `Qwen/Qwen2.5-3B-Instruct` (strong Italian, fits
  8GB in FP16, standard `model.model.layers` structure for clean hooks).
  Configurable in [`configs/model.yaml`](configs/model.yaml) with presets for a
  1.5B CI model, a 7B Colab model, and an FP32 CPU control.
- **Official files not yet provided** (PRO-CTCAE Italian PDF, CTCAE v6.0):
  loaders + interfaces are built; a **clearly-labelled synthetic** CTCAE
  placeholder and synthetic Italian PRO seeds are used for dev/tests only.
  `official_italian_labels` stay empty until the real PDF is loaded — nothing is
  invented.

### Still needed from you
1. **Official Italian PRO-CTCAE PDF** → to populate `official_italian_labels`.
2. **Official CTCAE v6.0 file** (JSON in the documented schema) → to replace the
   synthetic placeholder. Load via `terminology/ctcae_loader.py --path ...`.
3. Optionally, a custom term list (defaults to the 80 official PRO-CTCAE terms).

---

## Install & run

```bash
# 1) create venv + core (no ML stack needed for Phase 1)
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,api]"   # Windows
# .venv/bin/python -m pip install -e ".[dev,api]"         # Linux/Colab

# 2) build the terminology file (80 PRO-CTCAE terms)
.venv/Scripts/python.exe scripts/build_terminology.py

# 3) run the baseline mapper on synthetic controls
.venv/Scripts/python.exe scripts/run_mapping_baseline.py \
    --input data/synthetic/clinical_controls.jsonl

# 4) tests
.venv/Scripts/python.exe -m pytest        # 73 passing

# 5) API (mapping endpoint only; steering disabled by default)
.venv/Scripts/python.exe -m uvicorn oncoemotion.api.app:create_app --factory
```

Phase 2+ ML stack (installed separately; CUDA-specific):

```bash
.venv/Scripts/python.exe -m pip install -e ".[ml,viz]"
# On the T1000 install a cu121 torch build; keep dtype=float16.
```

---

## Baseline mapper (Phase 1)

Modular pipeline (spec §5): normalize → segment (multi-symptom) →
assertion/temporality/experiencer (Italian rules) → candidate generation
(lexical + fuzzy; optional embeddings/reranker) → thresholded decision →
mandatory abstention → **separate** CTCAE fallback → **independent** safety
routing → reproducible diagnostics.

PRO status values: `EXACT_PRO_MATCH`, `PRO_MATCH_WITH_AMBIGUITY`,
`MULTIPLE_POSSIBLE_PRO_MATCHES`, `NO_DIRECT_PRO_MATCH`, `NEGATED_SYMPTOM`,
`INSUFFICIENT_CONTEXT`, `OUT_OF_SCOPE`; CTCAE: `MATCH|AMBIGUOUS|NO_MATCH|NOT_EVALUATED`.

Guarantees enforced by tests (spec §16): exactly 80 terms, unique ids, correct
attribute mapping, no invented terms, `"non ho nausea"` → negated,
`"...nausea... ora è passata"` → resolved, `"mi si ingialliscono le unghie"` →
Nail discoloration, `"ho messo lo smalto giallo"` → not a clinical event,
`"ansia"` → Anxious, `"febbre"` → NO_DIRECT_PRO_MATCH + CTCAE Fever,
`"suicidio"` → NO_DIRECT_PRO_MATCH + urgent human review,
`"non riesco più a camminare"` → not force-coded (human review), deterministic
across batch order, hooks removed after exceptions, reproducible random vectors,
schema always valid, no free text in logs.

> **Calibration honesty:** Phase 1 has no logits, so probabilities are labelled
> `uncalibrated_heuristic` in `analysis_meta`. A calibrated reranker (temperature
> / isotonic over LLM logits) plugs into `oncoemotion.mapping.calibration` in a
> later phase — probabilities are never obtained by just asking the model.

---

## Repository layout

```
configs/        model.yaml, experiment.yaml, terminology.yaml
terminology/    pro_ctcae_terms.json (80), loaders, synthetic CTCAE placeholder
src/oncoemotion/
  preprocessing/  normalize, segment, assertion/temporality/experiencer
  retrieval/      lexical + fuzzy (difflib fallback), optional embeddings
  mapping/        pipeline (baseline), calibration
  safety/         independent risk router
  models/         adapter interface + HF decoder adapter (lazy torch)   [P2]
  activations/    framework-agnostic hook manager (tested now)          [P2]
  emotion_vectors/ vector types + QR-stable orthogonalization (tested)  [P2]
  probing/ steering/ patching/ evaluation/ statistics/                  [P2-4]
  api/            FastAPI app
scripts/        build_terminology, run_mapping_baseline, + phase scaffolds
tests/          unit / integration / regression  (73 passing)
data/           templates + synthetic controls (no real patient data)
outputs/        tables/figures/activations/checkpoints/reports (gitignored)
notebooks/      analysis notebooks (added with their phases)
```

> Package uses a standard `src/oncoemotion/<subpkg>` layout; the subpackage names
> match the spec §14 tree (preprocessing, retrieval, mapping, models, activations,
> emotion_vectors, probing, steering, patching, evaluation, statistics, safety, api).

---

## Research questions → where they are addressed

- **RQ1–RQ4** (do clinical inputs activate emotion vectors; pre-decision; general-
  ization; separable from confounders): emotion-vector construction (§7), QR-stable
  confounder orthogonalization (`emotion_vectors.orthogonalize`, §8), measurement
  points A–G with primary point **E** pre-decision (§10), probing (§11) — Phases 2–3.
- **RQ5** (causal effects of adding/removing/transferring an emotion component):
  steering/ablation/patching (§12), vector ops `steer_add`/`ablate_projection`
  ready and tested — Phase 4.
- **RQ6** (persistence after the stimulus, during the decision): persistence
  analysis with an inserted neutral sequence (§11) — Phase 3.

---

## Privacy & governance (§17)

No real/identifiable patient data. Pseudonymous record IDs; PII-safe logging
(free text never logged); data / activations / results kept separate; the
`/run-steering` endpoint is research-only and **disabled by default** (never in
production) with an audit log of model/layer/vector/alpha. Grading is a separate
module that abstains when required information is absent (never auto-derived from
PRO attributes).

## License

MIT (code). Official terminology sources are **not** included and may be
license-restricted; load them locally (kept out of version control).
