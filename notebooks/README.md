# notebooks/

Analysis notebooks are added with their phases (they require the ML stack and
generated activations):

| Notebook | Phase | Purpose |
|---|---|---|
| 01_dataset_audit.ipynb | 1-3 | audit synthetic datasets, provenance, class balance |
| 02_layer_sweep.ipynb | 2 | emotion-vector quality per layer (held-out probe AUROC) |
| 03_emotion_geometry.ipynb | 2 | vector geometry, collinearity vs confounders |
| 04_clinical_probing.ipynb | 3 | emotion scores on clinical inputs, persistence curves |
| 05_causal_interventions.ipynb | 4 | steering/ablation/patching effects, controls |
| colab_multimodel.ipynb | ESMO v2 | frozen controlled neutral/emotional study |
| colab_real_fields.ipynb | real validation | private Excel ingestion, isolated per-model caches, guarded secondary run, redacted export |

Until then, the same analyses are runnable as scripts under `scripts/`.
