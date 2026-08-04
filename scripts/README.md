# Command-line pipeline

## Clinical mapper

- `build_terminology.py` - build the 80-term PRO library.
- `run_mapping_baseline.py` - deterministic PRO/CTCAE mapper.
- `ingest_real_fields.py` - governed local ingestion; output remains ignored.

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
