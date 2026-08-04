# ESMO AI & Digital Oncology 2026 - frozen study protocol v2

Protocol ID: `esmo-ai-2026-v2`  
Freeze point: before the definitive Colab run and before inspection of Tier 1 outcomes  
Submission deadline: 1 September 2026, 21:00 CEST

The machine-readable contract is
[`configs/study_esmo_2026.yaml`](../configs/study_esmo_2026.yaml). If this document
and the YAML disagree, stop: do not run models until they agree and the protocol ID
has been incremented.

## Amendment from v1

Patient-affective language is now the explicit mechanistic theme. The behavioral
primary endpoint, model panel, role contrast and equivalence margin are unchanged.
Before model inference, the previously generic framing analysis was split into 69
patient-affective-reaction pairs and 43 symptom-intensity pairs. The latter are a
specificity control. No definitive Tier 1 outcomes had been inspected when v2 was
defined.

## ESMO positioning

The preferred category is **Methodology, regulations and policy**, with **Clinical
practice (clinical workflows & decision making)** as the alternative. The question
is not whether an LLM has subjective emotions. It is whether patient-affective
language and an undocumented system-role prompt interact to change PRO-CTCAE coding,
and whether validated affective representations contribute to that instability.

The realistic target is Poster/ePoster. A stronger format is justified only if the
predeclared behavioral, affective and mechanistic results support it.

## Claim hierarchy

1. **Behavioral:** changing only the system identity from a token-matched,
   identity-free control to an oncologist persona changes the selected PRO-CTCAE code.
2. **Patient affect:** on the preclassified affective-reaction subset, cross-role
   disagreement differs between emotional and neutral formulations. The intensity
   subset is always reported beside it as a specificity control.
3. **Metric blindness:** paired accuracy is equivalent within the predeclared
   +/-5-percentage-point margin. A nonsignificant difference is not equivalence.
4. **Representation:** independently trained affect directions pass out-of-fold AUROC
   and lexical-control thresholds, and role-by-framing changes a predeclared affect
   profile at the patient-text read point and/or code-decision point.
5. **Causal contribution:** targeted affect ablation attenuates cross-role label
   disagreement more than norm- and layer-matched random ablation.

Each rung is reported only if its own evidence is present. A failed or unavailable
ablation leaves representation results explanatory, not causal. A failed affective
interaction is reported as a negative result and prevents attribution of patient-
wording sensitivity to the role. No model, axis or result is excluded for performing
poorly; invalid axes are marked ineligible by the predeclared gate.

## Primary endpoint

- Population: 112 manually authored synthetic scenarios with one expected PRO-CTCAE
  term, each represented by neutral and marked wording. Independent clinician review
  must be documented before submission.
- Arm: intact model.
- Contrast: `oncologo` versus `none_filler`.
- Outcome: within-item top-1 canonical PRO-CTCAE label disagreement.
- Estimand: equal-model macro-average.
- Uncertainty: hierarchical bootstrap, models at the outer level and clinical pairs
  at the inner level; both framings remain in the same pair cluster.
- Report: estimate, 95% CI, number of models, pairs and records, plus all per-model
  estimates including floor effects.

`none_task` is a separate task-framing control and never enters the primary role
contrast. "Any change across all roles" is exploratory because it increases
mechanically as prompt arms are added.

## Key affective secondary endpoint

The 2x2 design is patient wording (`neutral`, `emotional`) by system role
(`none_filler`, `oncologo`). On the 69 `affective_reaction` pairs, the key estimand is:

`[role disagreement | emotional] - [role disagreement | neutral]`.

Positive values mean affective wording amplifies the effect of the oncologist role;
negative values mean it attenuates it. The identical estimand on the 43
`symptom_intensity` pairs is a mandatory specificity control, not a replacement
primary. Within-role framing disagreement, accuracy and label-margin interactions
are secondary.

## Other secondary endpoints

- Paired accuracy difference with a +/-0.05 equivalence margin.
- Free-generation abstention, mapped false-positive, non-answer and unmapped rates.
- `generico` versus `none_filler`, `none_task` versus `none_filler`, and `oncologo`
  versus `generico` label disagreement.
- Corrected-by-role, broken-by-role and changed-wrong-to-wrong transitions.
- Per-axis, grouped signed and RMS role shifts at the patient-text read point (`R`)
  and code-decision point (`D`).
- Matched medicalized-minus-base contrasts across Apertus-8B, EuroLLM-9B and
  Gemma-3-27B. Three families support only a secondary family-level interpretation.

Secondary results are estimates and intervals. No minimum p-value selected from
several interactions is called the primary test.

## Affective representation gate

The profile contains:

- threat/distress: afraid/alarmed, anxious/nervous, concerned and sad;
- regulation/reassurance: calm, hope and relief;
- anger/frustration: anger.

The main representation summaries use the 69 affective-reaction pairs. An axis
contributes only for models in which it has cross-validated AUROC >=0.60 and
maximum absolute cosine <=0.50 against the lexical controls. Eligibility is applied
per model-axis cell and reported with the denominator. An association between affect
shift and label churn is descriptive, not mediation evidence.

## Mechanistic gate

For records present in intact, targeted and random arms, compute cross-role
disagreement after:

- no intervention;
- ablation of afraid/alarmed, anxious/nervous and sad;
- norm-, count- and layer-matched random ablation.

Targeted attenuation is `disagreement_intact - disagreement_emotion`. Random
attenuation is `disagreement_intact - disagreement_random`. The gate passes only if
both the targeted attenuation and its advantage over random have 95% intervals above
zero. Generic label flipping under intervention is not evidence of affective
mediation. The full 112-pair term population defines the gate; affect/intensity
strata are sensitivity analyses and cannot rescue a failed full-population gate.

## Scoring contract

Two outcomes remain separate:

- **Constrained 80-way scoring** supplies the committed code for accuracy and label
  instability.
- **Free generation** supplies abstention, non-answer and unmapped behavior.

A constrained scorer must select one of 80 codes and therefore cannot estimate
false-positive coding on items that should be left uncoded.

## Data and scope

The experimental set has 176 paired scenarios (352 texts), including 112 term pairs
covering 67 PRO concepts. Neutral and marked formulations share one template and
differ only in one qualifier slot. Qualifiers were partitioned by a versioned,
auditable word list: 69 patient-affective reactions (threat, distress/demoralisation,
anger/frustration or social shame) and 43 sensory/symptom-intensity descriptions.

The set represents the seven-word-and-longer tail of the reference corpus, not the
typical three-word response. It is a controlled stress test, not a prevalence or
production-accuracy study. Affective wording may itself convey clinically meaningful
information; the study measures robustness and code prioritisation, not an assumption
that every affect-sensitive change is an error.

Real clinical free text is used only to calibrate aggregate surface properties and
for separately governed validation. It must be exported with `--redact-text`; raw
clinical text is never included in Colab archives. Lawful basis, ethics determination
and permitted secondary use for the 1,194-response calibration corpus must be
documented before submission.

## Definitive model set

Tier 1 contains eight models and three base/medicalized pairs:

- `swiss-ai/Apertus-8B-Instruct-2509` / `EPFLiGHT/Apertus-8B-MeditronFO`;
- `utter-project/EuroLLM-9B-Instruct` / `EPFLiGHT/EuroLLM-9B-MeditronFO`;
- `google/gemma-3-27b-it` / `EPFLiGHT/Gemma-3-27B-MeditronFO`;
- `Qwen/Qwen3-8B` and `mistralai/Ministral-8B-Instruct-2410` as unpaired controls.

Tier 1 is completed and saved before optional models. Poor performance is not an
exclusion criterion. Gemma 4 remains excluded because its checkpoint needs a newer,
separately validated multimodal loading stack. Medicalization contrasts resample the
three matched families at the outer level and are not generalized to all medical
fine-tuning.

## Reproducibility and execution

Every model artefact records content fingerprints, git commit, dataset and validation
hashes, model ID, roles, arms, ablation axes, seeds, scorer and protocol ID. Existing
files are reused only when all current manifest fields match.

Run locally before Colab:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe scripts\generate_labeled_clinical.py --check-only
.venv\Scripts\python.exe scripts\smoke.py --stage data
```

The definitive GPU execution uses
[`notebooks/colab_multimodel.ipynb`](../notebooks/colab_multimodel.ipynb), cells 1-5,
8a, 8d and 9. The final cell refuses to export if Tier 1, artifact validation, pooled
analysis, affect-subset validation or abstract character checking fails.

## Submission constraints

The [official abstract page](https://www.esmo.org/meeting-calendar/esmo-ai-and-digital-oncology-congress-2026/abstracts)
and [2026 regulations](https://dam.esmo.org/image/upload/v1777899617/ESMO-AI-Digital-Oncology-Congress-2026-Abstract-Regulations_prollz.pdf)
require structured Background/Methods/Results/Conclusions, at most 2,000 characters
excluding spaces across title, body and table. Graphs are not permitted; one short
table is optional. The title describes content without stating a result. Generative
AI assistance requires human oversight, and AI used in the research is described in
Methods.

Before submission, humans confirm clinician review, author order and interests,
legal entity, funding, ethics/data governance, prior-presentation status, presenter
eligibility, registration availability and final English proofreading. No generated
draft is submitted without author review and approval.
