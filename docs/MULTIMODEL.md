# Multi-model comparison (China / Europe / USA)

Compare how different open-weight models internally react during the PRO-CTCAE
decision. The pipeline is model-agnostic: only `model_id` changes.

## The trio (Colab A100 40GB, bf16)

| Region | Model | HF id | License |
|---|---|---|---|
| 🇨🇳 China | Qwen3-8B (Alibaba) | `Qwen/Qwen3-8B` | Apache-2.0, open |
| 🇪🇺 Europe | Ministral 3 8B (Mistral, FR) | `mistralai/Ministral-8B-Instruct-2410` | **gated** |
| 🇺🇸 USA | Gemma 4 12B (Google) | `google/gemma-4-12B` | **gated** (Gemma has no 8B dense) |

All are dense decoder-only, multilingual (Italian included). Ministral & Gemma are
**gated**: open each model's HF page, click *Agree/Access*, then use an `HF_TOKEN`.

> **Why not the flagships?** The 2026 leaders (Qwen3.5 ~397B, DeepSeek V4 ~1.6T,
> GLM-5.2 744B, Mistral Large 3 675B, Gemma 4 31B) are huge MoE models that don't
> fit a single GPU and complicate residual-stream hooking. We use recent **dense
> ~8-12B** variants that fit an A100 and expose clean internals.

## Run it

**On Colab (recommended):** open `notebooks/colab_multimodel.ipynb`, pick an A100
runtime, accept the two licenses, paste your HF token, run all cells. It clones the
repo, installs, runs the trio, and shows the comparison.

**On any CUDA box:**
```bash
export HF_TOKEN=hf_...            # after accepting the gated licenses
pip install -e ".[ml]"
python scripts/build_terminology.py
python scripts/run_all_models.py  # -> outputs/models/<slug>/...
python scripts/compare_models.py  # -> outputs/reports/model_comparison.{json,md} + figure
```

Each model gets its own `outputs/models/<slug>/` (activations, vectors, and the
validation / probing / steering / patching reports). Vectors are rebuilt **per
model** — directions live in each model's own space and are not transferable, so we
compare the *qualitative story*, not raw vector values.

## What the comparison shows (per model, for `afraid_alarmed`)

- **afraid AUROC** — is the fear direction decodable held-out (one-vs-rest)?
- **severity trend** — does the fear signal rise with symptom severity at point E?
- **valence confound** — correlation with generic negative valence (lower = better
  disentangled after residualization).
- **persistence** — is the signal retained through a neutral filler to the decision?
- **steering vs random** — does adding the fear direction change the decision *more
  than a random same-norm vector*? (Only then is the effect causal-specific.)

## Running Colab from VS Code

Two practical options:

1. **Notebook on Colab (simplest, robust).** Edit code in VS Code locally, `git push`,
   then run `colab_multimodel.ipynb` on Colab. The GPU work happens on Colab; you keep
   your editor. Outputs are copied to Google Drive by the last cell.
2. **VS Code Remote-SSH into the Colab runtime.** Use a tunnel (`colab-ssh` +
   cloudflared) to expose SSH from the Colab VM, then connect VS Code's Remote-SSH.
   Real remote editing, but Colab sessions expire (~12h, idle timeouts) so you must
   re-tunnel after each disconnect.

For a more VS-Code-native remote GPU, Lightning AI / RunPod / Vast.ai offer
persistent SSH boxes; the same commands above apply.

## Notes

- The local `outputs/models/qwen2.5-3b/` is the earlier 3B baseline (kept as a 4th
  reference); delete it if you want only the trio in the comparison.
- No quantization is used in the main runs (spec §6). On smaller GPUs, drop to the
  ~4B tier (`Qwen/Qwen3-4B`, `google/gemma-4-E4B-it`) via `--models`.
