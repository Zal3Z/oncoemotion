# ESMO AI & Digital Oncology Congress 2026 — submission checklist

Use this checklist only after the definitive Colab export has completed without
placeholders. The frozen analysis contract is in `docs/ESMO_2026_PROTOCOL.md`.

## Scientific readiness

- [ ] All Tier 1 models completed the same 224 codeable records for both primary roles.
- [ ] An independent oncology clinician reviewed the synthetic scenarios and gold codes,
      or the abstract accurately states the actual review status.
- [ ] `esmo_primary_analysis.json` reports `data_quality.passed: true`.
- [ ] `data_quality.affective_subset.passed` is true and reports exactly 69 pairs
      per Tier 1 model.
- [ ] The primary disagreement estimate and hierarchical 95% CI are present.
- [ ] The affective-reaction role-by-framing estimate is reported beside the
      symptom-intensity specificity control, regardless of direction.
- [ ] Accuracy equivalence is claimed only if its full CI lies within ±5 percentage points.
- [ ] A causal affective statement is used only if `mechanistic_gate.gate_passes: true`.
- [ ] Only model-axis cells that passed both vector gates contribute to affect profiles;
      the read-point and decision-point results are labelled separately.
- [ ] Base/medicalized results are described as three matched-family secondary contrasts.
- [ ] Generative abstention outcomes are not inferred from constrained classification.
- [ ] Sensitivity analyses and model-level estimates agree with the stated interpretation.

## Abstract form

- [ ] Submit by **Tuesday 1 September 2026, 21:00 CEST** through the ESMO portal.
- [ ] Preferred category: **Methodology, regulations and policy**.
- [ ] Preferred presentation: **Poster**; ePoster is an acceptable fallback.
- [ ] Title describes the topic without stating results or conclusions.
- [ ] Body uses Background, Methods, Results and Conclusions.
- [ ] Title + body + any table stay within 2,000 characters excluding spaces.
- [ ] No graph or illustration is included; at most one short table is used if indispensable.
- [ ] AI use for analysis and writing is disclosed in Methods under human oversight.
- [ ] No patient, hospital or author-identifying information appears in the abstract text.

## Governance and authorship

- [ ] Document the lawful basis, ethics determination and permitted secondary use for the
      1,194-response clinical corpus used to calibrate aggregate surface properties.
- [ ] Confirm whether the synthetic experiment itself requires formal ethics review or a
      documented exemption/non-human-subject determination at the submitting institution.
- [ ] Obtain approval from every listed author and designate the presenting author.
- [ ] Enter the legal entity responsible for the study.
- [ ] Declare funding and all author conflicts of interest.
- [ ] Verify originality/encore status and compliance with the ESMO publication policy.
- [ ] Retain the protocol, hashes, model versions, prompts and final export as an audit bundle.
- [ ] If accepted as an onsite poster, register an attending presenter; otherwise follow the
      ePoster instructions and deadlines.

Official references: [ESMO congress abstract page](https://www.esmo.org/meeting-calendar/esmo-ai-and-digital-oncology-congress-2026/abstracts)
and [2026 Abstract Regulations](https://dam.esmo.org/image/upload/v1777899617/ESMO-AI-Digital-Oncology-Congress-2026-Abstract-Regulations_prollz.pdf).
