# Real-field validation protocol

This protocol is operational and analytic documentation. It does not replace a
submission abstract or poster.

## Question

Do affect-associated directions learned independently from synthetic Italian text
activate in clinician-validated oncology free-text fields, and is that activation
associated with PRO-CTCAE coding error or with role-conditioned code instability?

Projection onto an affect direction is a representational measurement. It does not
mean that a model experiences an emotion.

## Data roles

- `data/synthetic/emotion_dataset.jsonl` builds and validates model-specific affect
  directions independently of the clinical fields.
- `sinomi_campi_aperti.xlsx` supplies real text, the source association and its value.
- `data/synthetic/clinical_labeled.jsonl` remains a separate controlled
  neutral/emotional stress test. It is not the gold source for real-field accuracy.

The current source yields 1,275 non-empty assessment rows: 968 PRO-CTCAE, 245 CTCAE
v5 without a direct PRO target and 62 non-associable. It covers 63 PRO identifiers,
996 exact normalized source strings and 127 records of at least seven words.

Identical text does not imply an interchangeable duplicate. Some identical strings
carry different item and/or value associations. Every source row is retained as its
own `record_id`; the non-reversible `source_id` is only a model-computation cache key
and bootstrap cluster.

## Affect directions

The eight predeclared axes are fear/alarm, anxiety, concern, sadness, anger, calm,
hope and relief. Fear/alarm plus concern form the primary threat/concern composite.
The expanded dataset has at least 111 examples for every declared axis, with explicit
and implicit paraphrases. Structured paraphrase families are never split across
extraction, validation, test or cross-validation folds.

Clinical severity, general negative valence, urgency and uncertainty are nuisance
controls. Emotion-word mentions and explicitly negated emotions form the lexical
gate. An axis is interpreted only when it has at least 60 out-of-fold positives,
cross-validated AUROC >= 0.60 and maximum absolute cosine <= 0.50 with the lexical
controls.

## Cohorts

The primary cohort remains the frozen eight-model set used by protocol v2. The real
protocol adds a secondary replication cohort with MedGemma 27B and the Gemma/MedGemma
4B pair. Secondary models never rescue or redefine a failed primary result.

Only open-weight models expose the hidden states needed for projection. Closed API
models may later be compared for behavioral accuracy, but not pooled into the
mechanistic representation analysis.

## Outcomes

For PRO rows, committed constrained top-1 code is compared with `gold_pro_id`. For
CTCAE-v5 and non-associable rows, the constrained scorer is not used to infer
abstention because it is forced to select one of 80 codes; free generation supplies
abstention and false-positive coding outcomes. The real runner uses the adaptive
policy: constrained scoring only for PRO rows and free generation only for non-PRO
rows, avoiding an otherwise redundant second scoring pass on every record.

The primary affect estimand is the absolute error-probability change per within-model
standard deviation of the fear/concern score. It is centered within source item and
associated value, then adjusted for text length, clinical-severity projection and
general-negative-valence projection.

Secondary outcomes include:

- within-item affect slope per associated value;
- oncologist-minus-length-matched-control affect shift on the same text;
- oncologist/control top-1 disagreement;
- macro recall across the 63 observed PRO identifiers;
- generative abstention on non-PRO rows;
- sensitivity restricted to texts of at least seven words;
- sensitivity excluding source-text clusters with conflicting assessment labels.

Inference weights models equally and resamples `source_id` clusters within model.
Real-text analyses are observational. Causal wording claims remain confined to the
separate controlled neutral/emotional dataset.

## Execution

Local check:

```powershell
.venv\Scripts\python.exe scripts\ingest_real_fields.py `
  --xlsx sinomi_campi_aperti.xlsx --check-only --expected-records 1275
.venv\Scripts\python.exe -m pytest
```

Primary Colab/H100 run:

```bash
python scripts/run_real_study.py \
  --xlsx /content/private/sinomi_campi_aperti.xlsx \
  --dtype bfloat16 --device auto --cohort primary \
  --ephemeral-model-cache-root /content/oncoemotion_hf_model_cache \
  --min-free-disk-gb 100
```

The low-disk mode processes one model end to end, then deletes only that model's
isolated local Hub/Xet cache. It prevents several 8B/27B weight caches from
accumulating on the Colab VM. Valid vectors and redacted real-field artifacts are
retained and still resume through their content fingerprints.

Secondary replication:

```bash
python scripts/run_real_study.py \
  --xlsx /content/private/sinomi_campi_aperti.xlsx \
  --dtype bfloat16 --device auto --cohort all \
  --ephemeral-model-cache-root /content/oncoemotion_hf_model_cache \
  --min-free-disk-gb 100
```

The secondary run requires gated access to both Google MedGemma repositories on
the same Hugging Face account used for the Colab token. A `403` is an account-access
failure, not a study-data or ingestion failure.

Package only redacted outputs:

```bash
python scripts/package_real_results.py
```

The archive excludes the workbook, ingested real JSONL, raw activations, vectors,
weights and license-restricted official terminology. Free-generated strings are
also removed from redacted row artifacts; only their mapped code, response class
and numeric scores remain.
