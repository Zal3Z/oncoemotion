# Command-line pipeline

## Clinical mapper

- `build_terminology.py` - build the 80-term PRO library.
- `run_mapping_baseline.py` - deterministic PRO/CTCAE mapper.
- `ingest_real_fields.py` - governed, fail-closed local ingestion through the explicit
  crosswalk; output remains ignored.

## ESMO 2026 study

1. `generate_emotion_dataset.py`
2. `build_vectors.py`
3. `validate_vectors.py` - cross-validation, shared layer, lexical and protocol gate.
4. `generate_labeled_clinical.py` - paired clinical stress-test items with a frozen
   affective-reaction versus symptom-intensity partition.
5. `run_role_emotion.py` - four prompt conditions and three intervention arms.
6. `analyze_role_emotion.py` - descriptive per-model audit.
7. `analyze_results.py` - preregistered pooled endpoint, affective 2x2 interaction,
   matched base/medicalized contrasts and causal gate.
8. `build_esmo_abstract.py` - structured draft and ESMO character validation.
9. `build_esmo_poster_figures.py` - result-driven PNG/SVG poster panels.

## Clinician-validated real-field study

1. `ingest_real_fields.py` - preserves every assessment row and creates a redaction key.
2. `run_real_study.py` - builds expanded independent affect vectors, runs the frozen
   primary or extended cohort and measures two token-matched roles. On constrained
   Colab disks, `--ephemeral-model-cache-root` keeps one model cache at a time and
   removes it only after that model's vector and real-field stages complete.
3. `analyze_real_fields.py` - gold-code accuracy, non-PRO abstention, within-item
   affect/value slopes, affect/error association and clustered sensitivity analyses.
4. `package_real_results.py` - privacy gate and result archive; raw text is rejected.

The authoritative parameters are in `configs/study_esmo_2026_real.yaml`; operational
details and estimands are in `docs/REAL_FIELDS_PROTOCOL.md`.

## Reasoning and explicit-abstention extension

1. `run_reasoning_classification.py` - one-model joint choice among validated
   PRO-CTCAE items, validated CTCAE items and `NON_CLASSIFICABILE`, in direct and
   standardized two-pass deliberative modes.
2. `run_reasoning_study.py` - resumable cohort orchestrator with isolated model
   caches.
3. `analyze_reasoning_classification.py` - strict item/taxonomy metrics, paired
   direct-deliberative effects and base/medicalized interactions.
4. `package_reasoning_results.py` - rejects both raw clinical text and generated
   deliberation text before export.
5. `export_item_audit.py` - creates local JSON/JSONL audit payloads joining the
   private validated source to the redacted v1 model outputs.

Parameters are frozen in `configs/study_esmo_2026_reasoning.yaml`; rationale and
interpretation are in `docs/REASONING_ABSTENTION_PROTOCOL.md`.

## Complete Colab orchestration

`run_complete_colab_study.py` is the single end-to-end orchestrator used by
`notebooks/oncoemotion_colab.ipynb`. For each model it runs the controlled study,
the real-field v2 study and the direct/deliberative extension before deleting that
model's isolated weight cache. Aggregate analyses, privacy-checked packages and
both real-field and reasoning item-audit payloads are created only after the requested
cohort is complete.

The controlled and real-field reruns use eight models, all in plain BF16 under
`configs/runtime_blackwell_96gb.yaml` (the swiss pair is Apertus 8B). The reasoning extension
runs the three base/medicalized pairs, MedGemma 27B,
Apollo2-7B and Qwen3.6-27B; only Qwen3.6 gets the additional
`native_reasoning` mode. BioMistral-7B is the optional secondary model.

The authoritative parameters live in `configs/study_esmo_2026.yaml`. Existing
artefacts are reusable only when their manifest/fingerprint matches the current
pipeline.

`run_all_models.py --stages vectors` builds only the artefacts needed by the role
study; historical probing, steering and patching can be selected explicitly but are
not part of the definitive poster run.

## Local checks

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe scripts\generate_labeled_clinical.py --check-only
.venv\Scripts\python.exe scripts\smoke.py --stage data
```

Run commands from the repository root. GPU scripts require the `ml` optional
dependencies; the deterministic mapper and tests do not.
