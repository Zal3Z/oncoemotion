"""Open-weight model adapters (spec section 6).

Adapters expose a uniform surface over decoder-only HF models: hidden states,
transformer blocks (residual stream), MLP / attention outputs (when available)
and logits. Family-specific adapters avoid assuming every model exposes
``model.model.layers``.

torch / transformers are imported lazily inside the concrete adapters so this
package imports cleanly without the ML stack (Phase 1 baseline needs neither).
"""

from oncoemotion.models.base import ModelAdapter, AdapterCapabilities, register_adapter, load_adapter

__all__ = ["ModelAdapter", "AdapterCapabilities", "register_adapter", "load_adapter"]
