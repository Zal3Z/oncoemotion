"""Model adapter interface + registry (spec section 6).

Concrete adapters (e.g. :mod:`oncoemotion.models.hf_decoder`) implement this ABC.
Nothing here imports torch; the registry maps model-id patterns to adapter
classes so ``load_adapter("Qwen/Qwen2.5-3B-Instruct", cfg)`` returns the right one.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Type

from oncoemotion.config import ModelConfig


@dataclass
class AdapterCapabilities:
    hidden_states: bool = True
    residual_stream: bool = True
    mlp_outputs: bool = False
    attention_outputs: bool = False
    logits: bool = True


class ModelAdapter(ABC):
    """Uniform surface over a decoder-only LM for interpretability work."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None

    # --- shape / capability introspection --- #
    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...

    # --- module accessors (hook sites) --- #
    @abstractmethod
    def get_block(self, layer: int) -> Any:
        """Transformer block at ``layer`` (residual-stream write site)."""

    def get_mlp(self, layer: int) -> Any:
        raise NotImplementedError("MLP output hook not available for this family.")

    def get_attention(self, layer: int) -> Any:
        raise NotImplementedError("Attention output hook not available for this family.")

    # --- inference / capture --- #
    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> Any: ...

    @abstractmethod
    def forward_capture(self, text_or_ids: Any, **kwargs) -> dict:
        """Run a forward pass with output_hidden_states=True.

        Returns a dict with at least: ``hidden_states`` (tuple length n_layers+1),
        ``logits``, ``input_ids``, ``attention_mask``.
        """

    @abstractmethod
    def generate(self, text: str, **kwargs) -> str: ...


# --------------------------- registry --------------------------- #
_REGISTRY: list[tuple[re.Pattern, Type[ModelAdapter]]] = []


def register_adapter(pattern: str):
    """Class decorator: register an adapter for model-ids matching ``pattern``."""

    def deco(cls: Type[ModelAdapter]) -> Type[ModelAdapter]:
        _REGISTRY.append((re.compile(pattern, re.IGNORECASE), cls))
        return cls

    return deco


def load_adapter(model_id: str | None = None, config: ModelConfig | None = None) -> ModelAdapter:
    cfg = config or ModelConfig()
    mid = model_id or cfg.model_id
    if model_id:
        cfg.model_id = model_id
    # Import concrete adapters lazily to keep torch out of the import path.
    from oncoemotion.models import hf_decoder  # noqa: F401  (side-effect: registration)

    for pattern, cls in _REGISTRY:
        if pattern.search(mid):
            return cls(cfg)
    # Fallback: generic HF decoder adapter.
    return hf_decoder.HFDecoderAdapter(cfg)
