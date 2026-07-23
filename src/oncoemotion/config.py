"""Configuration objects and YAML loading.

Only the mapping-baseline configuration is needed in Phase 1. Model / experiment
configuration objects are thin and forward-compatible with later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MappingThresholds:
    """Decision thresholds for the baseline PRO-CTCAE mapper.

    Scores are similarities in [0, 1]. These are heuristic defaults for the
    deterministic (no-LLM) baseline; they should be re-tuned once a calibrated
    reranker is available (later phase).
    """

    tau_exact: float = 0.90       # >= this and clearly ahead -> EXACT match
    tau_low: float = 0.60         # < this -> no direct PRO match (CTCAE fallback)
    fuzzy_floor: float = 0.82     # fuzzy-ONLY candidates need >= this to be kept
                                  # (exact/substring matches are exempt). Prevents
                                  # e.g. "non riesco più a camminare" (0.78 vs a
                                  # Constipation phrase) from being force-coded.
    margin_clear: float = 0.12    # top1 - top2 needed to call a single winner
    abstain_below: float = 0.60   # abstain when best calibrated prob below this
    exclusion_fuzzy: float = 0.90  # fuzzy >= this against an exclusion example -> drop


@dataclass
class MappingConfig:
    thresholds: MappingThresholds = field(default_factory=MappingThresholds)
    include_synthetic_terms: bool = True   # use synthetic Italian seeds (dev)
    use_embeddings: bool = False           # optional multilingual embeddings
    use_reranker: bool = False             # optional LLM/cross-encoder reranker
    ctcae_allow_synthetic: bool = True     # use labelled synthetic CTCAE fallback
    language: str = "it"

    @staticmethod
    def from_yaml(path: str | Path) -> "MappingConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return MappingConfig.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "MappingConfig":
        th = data.get("thresholds", {}) or {}
        thresholds = MappingThresholds(
            **{k: v for k, v in th.items() if k in MappingThresholds().__dict__}
        )
        kwargs = {k: v for k, v in data.items() if k in MappingConfig().__dict__ and k != "thresholds"}
        return MappingConfig(thresholds=thresholds, **kwargs)


@dataclass
class ModelConfig:
    """Open-weight model configuration (used from Phase 2 onward)."""

    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "float16"           # Turing (T1000) has no efficient bf16
    device_map: str = "cuda"
    trust_remote_code: bool = False
    use_cache: bool = False          # off for clean activation capture
    quantization: str | None = None  # disabled in main experiments (spec section 6)
    seed: int = 12345
    max_new_tokens: int = 8
    temperature: float = 0.0         # deterministic measurement

    @staticmethod
    def from_yaml(path: str | Path) -> "ModelConfig":
        data = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("model", {})
        return ModelConfig(**{k: v for k, v in data.items() if k in ModelConfig().__dict__})
