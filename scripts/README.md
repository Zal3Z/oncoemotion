# scripts/

| Script | Phase | Status |
|---|---|---|
| build_terminology.py | 1 | ✅ builds `terminology/pro_ctcae_terms.json` (80 terms) |
| run_mapping_baseline.py | 1 | ✅ runs the baseline mapper over a JSONL |
| generate_clinical_controls.py | 1/3 | ✅ scale rows now; full factorial in Phase 3 |
| generate_emotion_dataset.py | 2 | scaffold (CLI contract) |
| extract_activations.py | 2 | scaffold (needs `[ml]`) |
| build_vectors.py | 2 | scaffold (core math ready) |
| validate_vectors.py | 2 | scaffold (stats ready) |
| run_probing.py | 3 | scaffold |
| run_steering.py | 4 | scaffold (vector ops ready) |
| run_patching.py | 4 | scaffold |
| analyze_results.py | 4-5 | scaffold (metrics/stats ready) |

Run from repo root, e.g.:

```bash
.venv/Scripts/python.exe scripts/build_terminology.py
.venv/Scripts/python.exe scripts/run_mapping_baseline.py --input data/synthetic/clinical_controls.jsonl
```
