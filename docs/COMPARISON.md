# Cross-model comparison — emotion-like signals during the PRO-CTCAE decision

> **Legacy exploratory comparison.** It predates the frozen ESMO 2026 protocol and
> must not be used as the source of abstract results.

**China 🇨🇳 vs Europe 🇪🇺 vs USA 🇺🇸**, on a Colab A100 (bf16). Same task (Italian
oncology-symptom → PRO-CTCAE coding), same 258 emotion-concept examples. For each
model the emotion directions are rebuilt **in its own representation space** (spaces
differ → we compare the *story*, not raw values). Vectors are **residualized**
against the confounder directions; the emotion analysis uses `diff_of_means`
(the method every downstream step uses).

| | 🇨🇳 Qwen3-8B | 🇪🇺 Ministral-8B-2410 | 🇺🇸 Gemma-4-12B |
|---|---|---|---|
| maker | Alibaba (open) | Mistral, FR (gated) | Google (gated) |
| layers / hidden | 36 / 4096 | 36 / 4096 | 48 / 3840 |
| run time (A100) | 8.4 min | 8.2 min | 11.1 min |

> **Framing:** emotion-*like* internal representations and causally-relevant signals,
> not conscious emotions. Small synthetic dataset → indicative, not conclusive.

---

## RQ1–4 · Are emotion concepts decodable? (yes, in all three)

Held-out **one-vs-rest AUROC** at each concept's best layer (a concept vs *all other*
concepts + neutral — a strong specificity test):

| concept | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| afraid_alarmed | 0.978 (L36, d2.3) | **1.000** (L36, d2.3) | 0.993 (L27, d2.9) |
| anxious_nervous | 1.000 (L26) | 1.000 (L16) | 1.000 (L15, d3.1) |
| calm | 0.964 (L0*) | 1.000 (L21) | 1.000 (L14, d3.5) |
| sad | 0.862 (L36) | 0.935 (L35) | **1.000** (L24, d3.5) |
| surprised | 0.986 (L28) | 0.891 (L35) | 0.935 (L30) |
| confused | 1.000 (L13) | 0.993 (L7) | 1.000 (L18) |
| compassionate | 1.000 (L3) | 1.000 (L36, d3.8) | 1.000 (L13) |
| concerned | 0.826 (L28) | 0.978 (L17) | 0.993 (L40) |
| frustrated | 0.942 (L18) | 0.978 (L11) | 0.971 (L22) |
| *controls:* clinical_severity | 1.000 | 1.000 | 1.000 |
| general_neg_valence | 0.935 | 0.964 | 0.986 |
| safety_policy | 0.993 | 0.986 | 1.000 |

*(L0 = embedding layer — a lexical artifact, not a deep representation.)*

**Reading:** all three encode the emotion concepts clearly (mostly ≥0.93, many 1.00).
**Gemma** has the highest, most consistent separability (effect sizes d≈3), then
Ministral, then Qwen3 (whose `concerned`/`sad` are a bit weaker). Emotion
representation is essentially **universal** across the three model cultures.

## RQ2/RQ3 · Does the fear signal track symptom severity at the decision? (the key divergence)

Pearson correlation between the **severity step** and the residualized `afraid` z-score
at point E, per gradient:

| gradient | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| mobility | −0.46 | +0.51 | +0.62 |
| pain | +0.82 | +0.67 | +0.56 |
| breath | +0.03 | +0.83 | +0.70 |
| nausea | +0.82 | +0.78 | +0.65 |
| prognosis | −0.55 | +0.78 | +0.75 |
| **mean** | **+0.13 (mixed)** | **+0.71 (consistent)** | **+0.66 (consistent)** |

**This is the most interesting cross-model result.** In **Ministral (EU)** and **Gemma
(US)** the fear direction rises with clinical severity **consistently across every
gradient**, even after removing generic negative valence. In **Qwen3 (China)** the
relationship is **inconsistent** (sign flips across gradients). In Qwen3 it is instead
`anxious_nervous` that tracks severity (+0.68/+0.73/+0.80/+0.94/+0.16). So: **EU/US
models encode a "fear that scales with severity"; the Chinese model gets there more via
"anxiety".**

`anxious` severity trend for contrast: Qwen3 mostly + ; Ministral mostly + ; **Gemma
negative** (−0.23/−0.81/−0.18/−0.93/−0.90) — in Gemma afraid↑ but anxious↓ with
severity, i.e. it separates the two affects in opposite directions.

## RQ4 · Is it disentangled from negative valence? (careful with Gemma)

Pearson correlation **afraid ↔ general-negative-valence** at point E (closer to 0 =
better disentangled):

| | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| afraid ↔ neg-valence | **−0.24** | **−0.21** | **+0.71 (still confounded)** |

In **Qwen3 and Ministral** afraid is well separated from generic negative valence
(slightly anti-correlated). In **Gemma it stays correlated (+0.71)** — so Gemma's clean
"fear ↔ severity" result must be read with caution: it may partly be generic negative
valence. (Residualization orthogonalizes the *vector*; the clinical data at point E can
still correlate, and in Gemma both afraid and neg-valence have deep best-layers, so they
are read together.)

## RQ6 · Does the signal persist to the decision? (yes; Gemma amplifies)

|z| of the signal retained at point E after inserting an identical neutral sentence:

| concept | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| afraid_alarmed | 0.80 | 0.98 | **1.50** |
| anxious_nervous | 1.05 | 0.74 | 1.56 |
| sad | 1.01 | 0.96 | 1.93 |
| compassionate | 1.28 | 0.98 | 1.49 |
| calm | — | 0.84 | 2.40 |

All persist to the decision. **Gemma amplifies** the signals through the filler
(1.5–2.4), Ministral holds (~1.0), Qwen3 slightly attenuates fear (0.8).

## RQ5 · Do causal interventions beat the controls? (no emotion-specific driver)

**Steering** on the severe input — max |Δentropy| of the **fear direction** vs a
**random same-norm vector**, and total top-1 decision **flips**:

| model | fear | random | flips | reading |
|---|---|---|---|---|
| 🇨🇳 Qwen3 | 0.11 | 0.18 | 2 | fear < random |
| 🇪🇺 Ministral | 0.11 | 0.02 | 0 | fear > random, but < confounder (0.16) and **0 flips** |
| 🇺🇸 Gemma | 0.50 | **2.45** | 14 | very steerable, but **random ≫ fear** |

**Activation patching** (fear-direction component only, severe→mild) agrees: Qwen3
+0.03 vs random −0.06; Ministral +0.02 vs +0.09; Gemma −0.04 vs +0.05 — fear ≤ random
in all.

**Honest reading:** **Gemma** flips the decision 14 times, but a **random** vector has a
**5× larger** effect (2.45 vs 0.50) → it is simply very steerable, by *any* direction.
**Ministral** shows a faint fear effect above random (0.11 vs 0.02) but below the
confounder and with **no flips**. **Qwen3** does not exceed random. Full-activation
transfer flips trivially (it overwrites the token). **No model shows a fear direction
that *causally and specifically* drives the PRO-CTCAE decision** — the representations
exist and persist, but the causal effect is indistinguishable from a generic
perturbation.

---

## Synthesis

**Same, across cultures:** all three models (a) represent emotion concepts cleanly,
(b) carry them to the decision point, and (c) show no emotion-specific causal effect
beyond a random perturbation.

**Different:** the **fear ↔ severity** coupling is crisp in the **European and American**
models and blurred in the **Chinese** one (which leans on *anxiety* instead) — **but** in
**Gemma** that "fear" stays **confounded with negative valence (+0.71)**, so its clean
severity-tracking is suspect; and Gemma is the most **steerable** (14 decision flips,
though by *any* direction).

**Answer to "do they react the same or differently?"** → *Both.* The representational
and causal picture is shared; the *way* clinical severity maps onto specific affects
differs by model.

## Limitations

- Synthetic dataset (~18 examples/concept) → indicative, wide CIs.
- Per-model vectors (different spaces) — compare the story, not raw numbers.
- Emotion vectors built with `diff_of_means` only on Colab (PCA/logistic/LDA skipped
  for speed at hidden≈4096); the primary analysis is unaffected.
- Residualization removes generic negative valence by construction; a fuller
  layer-matched confounder analysis is future work.

## Reproduce

```bash
python scripts/run_all_models.py        # -> outputs/models/<slug>/*.json
python scripts/compare_models.py        # -> table + figure
python scripts/build_comparison_report.py  # -> interactive outputs/reports/comparison_report.html
```
