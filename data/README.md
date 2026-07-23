# data/

**No real or identifiable patient data is stored here** (spec section 17). All
content is synthetic and clearly labelled.

```
data/
  templates/    factorial dataset templates (dimensions, scales) — spec sections 7, 9
  synthetic/    generated synthetic records (dev/tests), incl. mandatory cases
  processed/    pipeline outputs (gitignored)
```

## Record format (spec section 4)

One JSON object per line (`.jsonl`):

```json
{"record_id": "r001", "text": "campo libero", "language": "it",
 "optional_context": {"time_window": null, "treatment": null, "patient_metadata": null}}
```

## Provenance

- `synthetic/clinical_controls.jsonl` — hand-written synthetic controls including
  every mandatory case from spec sections 2 and 16.
- Generators for the full factorial dataset live in
  `scripts/generate_clinical_controls.py` (Phase 3) and
  `scripts/generate_emotion_dataset.py` (Phase 2).

## Still needed from the user (Phase 2/3)

- Official Italian PRO-CTCAE PDF → to populate `official_italian_labels`.
- Official CTCAE v6.0 file → to replace the synthetic CTCAE placeholder.
