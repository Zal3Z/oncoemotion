# oncoemotion — Makefile
# On Windows (Git Bash) the venv python lives under .venv/Scripts; on Linux/Colab
# override:  make test PY=.venv/bin/python   (or PY=python)
PY ?= .venv/Scripts/python.exe

.PHONY: help venv install install-ml terminology test test-v api mapping lint clean \
        terminology-official vectors validate probing steering patching pipeline \
        dashboard docker-build docker-up

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
