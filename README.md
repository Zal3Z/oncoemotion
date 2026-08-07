# oncoemotion

Italian PRO-CTCAE/CTCAE symptom mapping and mechanistic evaluation of
role-conditioned affective representations in open-weight language models.

The repository has two deliberately separated components:

1. a deterministic clinical-support mapper with abstention and independent safety
   routing;
2. a research pipeline that measures how system roles change internal affective
   representations and model-generated PRO-CTCAE codes.

It does **not** claim consciousness, sentience or subjective emotion. It is not an
autonomous diagnostic system and does not replace clinical review.

## Current study

The active target is an abstract/poster for ESMO AI & Digital Oncology 2026. The
pre-run protocol is [`docs/ESMO_2026_PROTOCOL.md`](docs/ESMO_2026_PROTOCOL.md), the
machine-readable analysis contract is
[`configs/study_esmo_2026.yaml`](configs/study_esmo_2026.yaml), and the execution
guide is [`docs/ESECUZIONE.md`](docs/ESECUZIONE.md). The final human checks are in
[`docs/ESMO_2026_SUBMISSION_CHECKLIST.md`](docs/ESMO_2026_SUBMISSION_CHECKLIST.md),
and the result-driven layout is in
[`docs/ESMO_2026_POSTER_OUTLINE.md`](docs/ESMO_2026_POSTER_OUTLINE.md).

Primary clinical question: holding the item fixed, how often does an oncologist
persona change the selected PRO-CTCAE code relative to a token-matched,
identity-free control? The key affective secondary asks whether patient-affective
wording amplifies or attenuates that role effect. The 69 affective-reaction pairs are
analysed separately from 43 symptom-intensity controls. A causal affect claim is made
only if targeted ablation attenuates cross-role disagreement more than matched random
ablation.

All previously generated HTML reports and `oncoemotion_results*` directories predate
the frozen protocol. They are exploratory and must not be cited as definitive ESMO
results.

## Local verification

Python 3.10+ is required.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,api]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe scripts\generate_labeled_clinical.py --check-only
```

The heavy model stack is separate:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[ml,viz]"
```

The local 8 GB GPU is for smoke testing. The complete definitive execution uses the
single entry point
[`notebooks/oncoemotion_colab.ipynb`](notebooks/oncoemotion_colab.ipynb).

The clinician-validated real-field protocol has a separate runner and analysis
contract so its observational estimands cannot be mixed with the paired synthetic
experiment, although the single Colab notebook orchestrates both:

```powershell
.venv\Scripts\python.exe scripts\run_real_study.py `
  --xlsx sinomi_campi_aperti.xlsx --cohort primary
```

See [`docs/REAL_FIELDS_PROTOCOL.md`](docs/REAL_FIELDS_PROTOCOL.md). The same single
Colab notebook runs this protocol after the controlled study and before the
reasoning/explicit-abstention extension.

The rerun uses an eight-model panel with the matched Apertus 8B pair in plain
BF16. The reasoning extension uses the three matched
base/medicalized pairs, MedGemma 27B, Italian-capable Apollo2-7B and Qwen3.6-27B.
Qwen3.6 receives a separate native
thinking arm; see
[`docs/MODEL_PANEL_REASONING.md`](docs/MODEL_PANEL_REASONING.md).

## Deterministic mapper

The clinical path is:

```text
normalize -> segment -> assertion/temporality/experiencer
          -> lexical/fuzzy PRO retrieval -> threshold/abstention
          -> separate CTCAE fallback -> independent safety routing
```

Run the mapper and API:

```powershell
.venv\Scripts\python.exe scripts\build_terminology.py
.venv\Scripts\python.exe scripts\run_mapping_baseline.py --input data\synthetic\clinical_controls.jsonl
.venv\Scripts\python.exe -m uvicorn oncoemotion.api.app:create_app --factory
```

The `/map` endpoint never performs steering. Research activation and intervention
code is executed offline and is not part of the production mapping path.

## ESMO research pipeline

The definitive path is:

```text
emotion/control seeds
  -> per-model activations and directions
  -> cross-validated shared-layer + lexical gate
  -> preclassified affective-reaction / symptom-intensity clinical pairs
  -> token-matched role prompts in a paired 2x2 design
  -> constrained 80-way code + separate free-generation abstention
  -> intact / targeted-ablation / matched-random arms
  -> hierarchical paired analysis + matched base/medicalized contrasts
  -> regulation-checked abstract draft
```

Core commands after GPU artefacts exist:

```powershell
.venv\Scripts\python.exe scripts\analyze_results.py
.venv\Scripts\python.exe scripts\build_esmo_abstract.py
.venv\Scripts\python.exe scripts\build_esmo_poster_figures.py
```

Every definitive artefact records protocol ID, content hashes and the git commit.
Resume logic uses manifests rather than the mere existence of an output file.

## Data and terminology

- The experimental clinical dataset is synthetic and manually authored; clinician
  review must be documented before submission if it has not already occurred.
- Real clinical free text is excluded from version control and must be redacted in
  exported result rows.
- Official PRO-CTCAE/CTCAE sources may be licence-restricted and remain outside Git.
- The deterministic mapper is a reference, not the source of the authored gold.

The paired stress-test set represents the longer tail of free-text responses; it is
not intended to estimate population prevalence or production accuracy.

## Repository layout

```text
configs/                 model, terminology and frozen ESMO study contract
data/                    synthetic generators/records; real data ignored
docs/                    protocols, runbook and audit guides
notebooks/               definitive Colab runner and exploratory notebooks
scripts/                 data, vector, experiment, analysis and reporting CLIs
src/oncoemotion/         mapper, safety, model hooks, vectors and interventions
tests/                   unit, regression and API tests
terminology/             generated libraries; restricted official files ignored
```

## Governance

Do not commit patient text, model weights, raw activations or licence-restricted
terminology. For human/clinical data, institutional governance, lawful basis, ethics
requirements, de-identification and author accountability must be resolved outside
the software before publication or deployment.

Code licence: MIT. Terminology and source clinical data may have separate terms.
