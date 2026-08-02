# oncoemotion — Makefile
# On Windows (Git Bash) the venv python lives under .venv/Scripts; on Linux/Colab
# override:  make test PY=.venv/bin/python   (or PY=python)
PY ?= .venv/Scripts/python.exe

.PHONY: help venv install install-ml terminology test test-v api mapping lint clean \
        terminology-official vectors validate probing steering patching pipeline \
        smoke dashboard docker-build docker-up

help:
	@echo "Targets: venv install install-ml terminology test api mapping lint clean"

venv:
	python -m venv .venv

install:
	$(PY) -m pip install -e ".[dev,api]"

install-ml:
	$(PY) -m pip install -e ".[ml,viz]"

terminology:
	$(PY) scripts/build_terminology.py

test:
	$(PY) -m pytest

test-v:
	$(PY) -m pytest -v

api:
	$(PY) -m uvicorn oncoemotion.api.app:create_app --factory --reload

mapping:
	$(PY) scripts/run_mapping_baseline.py --input data/synthetic/clinical_controls.jsonl

terminology-official:
	$(PY) scripts/extract_pro_ctcae_italian.py
	$(PY) scripts/extract_ctcae_v6.py
	$(PY) scripts/build_terminology.py

# --- Phase 2-4 ML pipeline (needs .[ml]) ---
vectors:
	$(PY) scripts/generate_emotion_dataset.py
	$(PY) scripts/build_vectors.py --methods diff_of_means pca logistic lda

validate:
	$(PY) scripts/validate_vectors.py

probing:
	$(PY) scripts/run_probing.py

steering:
	$(PY) scripts/run_steering.py

patching:
	$(PY) scripts/run_patching.py

pipeline: vectors validate probing steering patching

viz:
	$(PY) scripts/visualize_internals.py

# --- local smoke test: validates the whole chain before spending Colab time ---
# Sized for an 8 GB card. Qwen2.5-1.5B in fp16 is ~3 GB of weights and leaves room
# for activations; 3B is the practical ceiling on 8 GB and will be tight. The point
# is not the numbers, it is that every stage runs and every artefact appears with
# the fields the analyses expect. Run this after any change to prompts, seeds,
# layer selection or the ablation arms.
SMOKE_MODEL ?= Qwen/Qwen2.5-1.5B-Instruct
SMOKE_DIR   ?= outputs/smoke

smoke:
	$(PY) scripts/generate_labeled_clinical.py --check-only
	$(PY) scripts/generate_emotion_dataset.py
	$(PY) scripts/build_vectors.py --model $(SMOKE_MODEL) --dtype float16 --device cuda \
	    --methods diff_of_means \
	    --acts-out $(SMOKE_DIR)/emotion_acts.npz --vec-out $(SMOKE_DIR)/emotion_vectors.npz
	$(PY) scripts/validate_vectors.py --acts $(SMOKE_DIR)/emotion_acts.npz \
	    --vecs $(SMOKE_DIR)/emotion_vectors.npz \
	    --report $(SMOKE_DIR)/vector_validation.json --figure $(SMOKE_DIR)/layer_sweep.png
	$(PY) scripts/run_role_emotion.py --model $(SMOKE_MODEL) --dtype float16 --device cuda \
	    --limit 6 --arms intact emotion random --ablation-limit 4 \
	    --vecs $(SMOKE_DIR)/emotion_vectors.npz --val-report $(SMOKE_DIR)/vector_validation.json \
	    --out $(SMOKE_DIR)/role_emotion
	for f in $(SMOKE_DIR)/role_emotion/*__rows.jsonl; do $(PY) scripts/analyze_role_emotion.py --rows "$$f"; done
	$(PY) scripts/run_role_spectrum.py --model $(SMOKE_MODEL) --dtype float16 --device cuda \
	    --limit 4 --null-draws 200 \
	    --vecs $(SMOKE_DIR)/emotion_vectors.npz --val-report $(SMOKE_DIR)/vector_validation.json \
	    --out $(SMOKE_DIR)/role_spectrum
	$(PY) scripts/reanalyze_direction.py --dir $(SMOKE_DIR)/role_spectrum
	$(PY) scripts/analyze_results.py --rows-glob "$(SMOKE_DIR)/role_emotion/*__rows.jsonl" \
	    --out $(SMOKE_DIR)/primary_analysis.json
	@echo ""
	@echo "smoke OK - ogni fase ha prodotto i suoi artefatti in $(SMOKE_DIR)"

# --- multi-model comparison (needs a big GPU + HF_TOKEN for gated models) ---
models:
	$(PY) scripts/run_all_models.py --dtype bfloat16 --device auto --skip-existing

compare:
	$(PY) scripts/compare_models.py

# --- dashboard / docker ---
dashboard:
	$(PY) -m streamlit run dashboard/streamlit_app.py

docker-build:
	docker build -t oncoemotion-api:latest .

docker-up:
	docker compose up --build

lint:
	$(PY) -m ruff check src tests scripts || true

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info src/*.egg-info
