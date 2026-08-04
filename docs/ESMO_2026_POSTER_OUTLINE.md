# ESMO AI & Digital Oncology 2026 - poster outline

Working title: **Affective representations and role-conditioned instability in
LLM-based PRO-CTCAE coding**

The poster must use the conclusion selected by `build_esmo_abstract.py`. It must not
upgrade an association into mediation or describe representation directions as
feelings experienced by the model.

## Three-column layout

### 1. Clinical problem and paired design

- Why it matters: patient-reported symptom text carries affective appraisal, while
  system-role prompts are implementation choices often omitted from validation.
- Show the paired 2x2 design: neutral/marked patient wording x oncologist/
  token-matched identity-free prompt -> constrained top-1 PRO-CTCAE code.
- State the predeclared 69-pair `affective_reaction` subset and 43-pair
  `symptom_intensity` specificity control.
- State synthetic stress-test scope, clinician-review status and governance of the
  1,194-response calibration corpus.

### 2. Behavioral and affective results

- `primary_disagreement.svg`: per-model disagreement plus equal-model pooled CI.
- `accuracy_equivalence.svg`: paired accuracy CI against the +/-5-point equivalence
  region.
- `affective_role_interaction.svg`: emotional-minus-neutral change in the role effect
  for affective-reaction pairs beside the symptom-intensity control.
- Put the estimand, 95% CI, model and record denominators beside each figure.
- Do not use "no significant difference" as a synonym for equivalence.

### 3. Representation, mechanism and implications

- `affective_profile_read_point.svg`: grouped, validated affect projections at the
  patient-text read point. Call these representations, not model emotions.
- Use `mechanistic_gate.svg` only when generated. Its absence means the causal
  comparison was unavailable.
- If the gate fails, describe internal affect measures as explanatory and keep the
  patient-wording result separate from the mechanistic result.
- Report free-generation abstention separately from constrained coding.
- Report base/medicalized contrasts as secondary evidence without generalising from
  three model families.
- Close with: system prompts, item-level stability and affect-robustness should be
  reported in oncology LLM validation.

## Footer

Include authors/affiliations, funding, conflicts, legal entity, ethics determination,
checkpoint versions, protocol ID, repository/QR code if permitted, and a clear
"research evaluation - not autonomous clinical use" statement. Do not show patient
text or licensed terminology assets.

Generate figures after the definitive Colab analysis:

```bash
python scripts/build_esmo_poster_figures.py
```
