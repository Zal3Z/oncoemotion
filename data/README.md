# Data boundaries

`data/synthetic/` contains generated or manually authored synthetic records used by
tests and the controlled role/framing experiment. It contains no patient records.

`data/real/` is ignored by Git and may exist only in an authorised local workspace.
Its presence does not imply permission to upload it to Colab or distribute it. The
ingestion schema stores free text; exported model rows must therefore use
`run_role_emotion.py --redact-text`. `package_real_results.py` refuses to package a
row file unless every text field has been replaced by its non-reversible `source_id`.

The spreadsheet supplied for local development is source clinical material and must
be handled under the applicable institutional governance. Do not add additional raw
clinical files to version control. Before any publication, confirm de-identification,
lawful basis/ethics requirements, annotator provenance and data-sharing restrictions.

The real-field converter uses `terminology/real_field_association_map.json` and fails
closed on an unknown PRO label. Every source row remains a distinct assessment;
identical strings are clustered by `source_id` rather than silently deduplicated,
because identical text can carry a different item or associated value.

## Synthetic record schema

```json
{"record_id":"r001","pair_id":"p001","text":"campo libero sintetico",
 "framing":"neutral","manipulation_type":"affective_reaction",
 "affect_family":"threat","gold_class":"term","gold_pro_id":"PRO_001"}
```

The paired ESMO generator is `scripts/generate_labeled_clinical.py`. It writes 176
neutral/emotional pairs and refuses to write when its length, punctuation, surface
balance or sample-size constraints fail. For codeable terms it also freezes an
auditable semantic partition: 69 `affective_reaction` pairs and 43
`symptom_intensity` pairs. The latter are the specificity control and must not be
silently pooled into an emotion-specific claim.

```powershell
.venv\Scripts\python.exe scripts\generate_labeled_clinical.py --check-only
```

The deterministic mapper cross-check is a reference only; authored gold labels are
not silently replaced by mapper output.

## Emotion dataset

The independently generated emotion dataset includes authored seeds plus structured
paraphrase expansion for the eight ESMO-relevant axes and their nuisance controls.
Each expansion family is assigned wholly to one split; family-aware cross-validation
prevents template siblings from appearing in both train and held-out folds.
