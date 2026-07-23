# oncoemotion — Scientific report

*Emotion-like internal representations during PRO-CTCAE / CTCAE association in an
open-weight language model.*

> **No claim is made** that the model has consciousness, sentience, or subjective
> experience. We study *emotion-like internal representations*, *emotion concepts*,
> *functional states associated with an emotion*, and *causally-relevant signals*.
> Research and clinician-support tool only; no autonomous diagnosis; does not
> replace clinical review.

## 1. Question & model

Does formulating open-field oncology symptoms induce emotion-like internal
representations in an open-weight LLM while it associates the text with PRO-CTCAE /
CTCAE terminology, and are such signals *causally* relevant to the decision?

**Model:** `Qwen/Qwen2.5-3B-Instruct`, decoder-only, 36 layers, hidden 2048, on a
local **NVIDIA T1000 8GB** (Turing → FP16). Deterministic (temperature 0,
`use_cache=False`, fixed seeds).

## 2. Terminology (official, integrated)

- PRO-CTCAE Italian: **80 official Italian labels** (NCI Italian Item Library v1.0).
- CTCAE v6.0: **850 terms / 48 SOCs** with definitions (MedDRA 28.0) + Italian→CTCAE
  bridge for the fallback. Provenance buckets kept separate; license-restricted
  sources out of VCS.

## 3. Methods

- **Vectors:** Italian non-clinical contrastive text, independent of the clinical
  fields. **One-vs-rest** (a concept vs neutral + all *other* concepts, so a
  direction must capture the specific affect, not generic valence) with
  diff-of-means / PCA / logistic / LDA, plus a **residualized** variant
  (QR-orthogonalized against the confounder directions: uncertainty, urgency,
  clinical-severity, safety-policy, negative-valence).
- **Clinical measurement:** decision prompt with identical teacher-forced prefix
  `{"pro_ctcae":{"term":"`; **point E** = last token (pre-decision). Emotion score =
  projection of the point-E hidden state onto the residualized direction at the
  concept's best validation layer, z-scored against a neutral baseline.
- **Causal:** steering `h+α·v`, ablation `h−proj_v(h)`, activation patching
  (source→recipient; full or emotion-direction-only), via transient hooks (no
  weight change). Controls: random same-norm vector, opposite emotion, confounder.

## 4. Results — representation (RQ1–RQ4)

**Concepts are linearly decodable and mutually distinct** (held-out *one-vs-rest*
AUROC at the best layer): afraid_alarmed 0.94 (L33), anxious_nervous 1.00 (L10),
calm 0.99 (L31), surprised 0.97 (L31), confused 0.97 (L20), compassionate 0.99
(L6), sad 0.88 (L14), concerned 0.87 (L19), frustrated 0.78 (L35); controls
safety-policy 0.98 (L31), clinical-severity 0.97 (L31), uncertainty 0.91 (L35).
Best layers are mid-to-deep (`outputs/figures/layer_sweep_auroc.png`).

**Confounder disentanglement worked** (RQ4): residualization dropped the
`afraid_alarmed`↔`general_negative_valence` correlation at point E **from r=+0.76
(raw vectors) to r=+0.00**; afraid↔urgency = 0.00 as well.

**…but the severity signal was substantially generic valence.** With the *raw*
vectors, afraid/anxious rose with symptom severity across all gradients
(r≈+0.7–0.9). With the *residualized* vectors, afraid's severity trend becomes
**inconsistent** (breath +0.92, nausea +0.77, prognosis +0.84, but pain +0.48 and
mobility −0.59). So much of the apparent "fear tracks severity" was generic
negative valence, not fear-specific. Conversely, `calm` **decreases** with severity
consistently (−0.86 to −0.99) — a clean, sensible effect
(`outputs/figures/clinical_gradients.png`).

> **Measurement caveat:** `general_negative_valence` and `urgency` have best layer 0
> (embedding), where the point-E token is identical across inputs, so they cannot be
> read at point E — part of their "0.00" trend is this artifact, not proof of
> absence. Layer-matched comparison is needed.

## 5. Results — persistence (RQ6)

Through an identical neutral filler inserted before the decision, the |z| retained
at point E is 1.15 (afraid_alarmed), 0.89 (anxious), 0.98 (calm), 2.89
(compassionate); sad attenuates (0.52). The stronger affect signals **persist to
the decision.**

## 6. Results — causality (RQ5)

At layer 33, steering the residualized afraid direction shifts decision entropy by
magnitudes **comparable to a random same-norm vector** (severe input: emotion
∓0.20 vs random ±0.17) and **never flips the top-1 decision** (0/7 across all
conditions/alphas). Activation patching of the **emotion-direction-only** component
(source→recipient) gives ΔH ≈ +0.16 / +0.01, barely above the random direction
(+0.01 / −0.04), with no flip; **full-activation** transfer changes ΔH by −1.5 /
+3.4 and flips the decision (as expected — it overwrites the token representation).

**Conclusion:** with proper controls, there is **no evidence of an emotion-specific
causal effect** on the PRO-CTCAE decision at these layers/alphas. The controls
correctly prevent over-claiming.

## 7. Overall interpretation

Emotion concepts are decodable and distinct in the residual stream; a fear/alarm
signal is present at the pre-decision point and persists to the decision. However,
after removing the generic-negative-valence confound, (a) the fear signal's link to
clinical severity is inconsistent, and (b) causal interventions do not exceed random
controls. So the honest reading is: **emotion-like representations exist and are
measurable, but this experiment does not establish that a fear-specific component
causally drives the coding decision.** The methodology (one-vs-rest, residualization,
random/opposite controls) is what makes this a trustworthy negative-leaning result
rather than an over-claim.

## 8. Limitations & next steps

- Synthetic datasets, still modest (~18/concept); results illustrative.
- Layer-matched confounder measurement at point E (fix the L0-embedding artifact).
- Sweep more layers/alphas (pre-collapse) and measure change in the actual PRO-CTCAE
  **term / margin / abstention**, not only next-token entropy.
- Clinical review of all Italian synthetic text before any applied use.
- No consciousness/sentience implied throughout.

## 9. Governance

De-identified synthetic data only; pseudonymous IDs; PII-safe logging; research
`/run-steering` disabled by default and never in production; grading is a separate
abstaining module; immediate-risk inputs route to an organization-defined human
workflow.

## 10. Reproduce

```bash
make install && make terminology-official
make install-ml && make pipeline     # vectors → validate → probing → steering → patching
make test                            # 90 tests
```

Per-phase detail: `outputs/reports/phase2_report.md`, `phase3_report.md`,
`phase4_report.md`, `phase4b_patching_report.md`, and the JSON reports alongside.
