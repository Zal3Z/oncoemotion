"""HuggingFace decoder-only adapter for the Qwen2.5 / Llama-3.x families.

torch + transformers are imported lazily inside methods so importing this module
never requires the ML stack. Loading a real model happens only when
:meth:`HFDecoderAdapter.load` is called (Phase 2+).

On the local NVIDIA T1000 (Turing, compute 7.5) bf16 is not efficiently
supported; ``ModelConfig.dtype`` defaults to ``float16``. On Ampere+ (Colab) set
``dtype='bfloat16'`` in ``configs/model.yaml``.
"""

from __future__ import annotations

from typing import Any

from oncoemotion.config import ModelConfig
from oncoemotion.models.base import AdapterCapabilities, ModelAdapter, register_adapter


@register_adapter(r"(qwen2|qwen3|llama-3|llama-4|meta-llama|mistral|ministral|gemma)")
class HFDecoderAdapter(ModelAdapter):
    """Adapter for decoder-only models exposing a stack of transformer blocks.

    Layer discovery is robust across families (Qwen2/3, Llama, Mistral/Ministral,
    Gemma, GPT-NeoX/GPT-2), so it does not assume ``model.model.layers``.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config or ModelConfig())
        self._loaded = False

    # ------------------------------------------------------------------ #
    def load(self) -> "HFDecoderAdapter":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=self.config.trust_remote_code
        )

        dev = (self.config.device_map or "cpu")
        kw = dict(trust_remote_code=self.config.trust_remote_code)
        if dev == "auto":
            kw["device_map"] = "auto"
        # dtype arg name changed across transformers versions.
        try:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_id, dtype=dtype, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_id, torch_dtype=dtype, **kw)

        if dev != "auto":
            target = "cuda" if dev.startswith("cuda") and torch.cuda.is_available() else "cpu"
            model = model.to(target)

        model.eval()
        model.config.use_cache = self.config.use_cache
        torch.manual_seed(self.config.seed)
        self.model = model
        self._loaded = True
        return self

    @property
    def device(self):
        self._require()
        return next(self.model.parameters()).device

    def _require(self) -> None:
        if not self._loaded:
            raise RuntimeError("Adapter not loaded. Call .load() first (requires torch+transformers).")

    # --- introspection --- #
    @property
    def n_layers(self) -> int:
        self._require()
        return len(self._layers())

    @property
    def hidden_size(self) -> int:
        self._require()
        cfg = self.model.config
        hs = getattr(cfg, "hidden_size", None)
        if hs is None:                       # unified/multimodal configs nest it
            for sub in ("text_config", "language_model_config", "llm_config"):
                s = getattr(cfg, sub, None)
                if s is not None and getattr(s, "hidden_size", None):
                    hs = s.hidden_size
                    break
        if hs is None:                       # last resort: infer from a decoder weight
            for p in self._layers()[0].parameters():
                hs = p.shape[-1]
                break
        return int(hs)

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            hidden_states=True, residual_stream=True,
            mlp_outputs=True, attention_outputs=True, logits=True,
        )

    def _layers(self):
        """Locate the ModuleList of decoder blocks across model families."""
        import torch.nn as nn

        m = self.model
        candidates = (
            ("model", "layers"),          # Qwen2/3, Llama, Mistral/Ministral, Gemma
            ("model", "language_model", "layers"),        # unified/multimodal (Gemma4, ...)
            ("language_model", "model", "layers"),
            ("model", "model", "language_model", "layers"),
            ("language_model", "layers"),
            ("model", "model", "layers"),  # some wrapped variants
            ("layers",),
            ("transformer", "h"),          # GPT-2 style
            ("gpt_neox", "layers"),        # GPT-NeoX style
        )
        for path in candidates:
            obj, ok = m, True
            for attr in path:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, nn.ModuleList) and len(obj) > 0:
                return obj
        # last resort: the first non-empty ModuleList of decoder-like blocks
        for mod in m.modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0 and hasattr(mod[0], "forward"):
                return mod
        raise AttributeError(f"Could not locate decoder layers on {type(m).__name__}")

    # --- hook sites --- #
    def get_block(self, layer: int) -> Any:
        self._require()
        return self._layers()[layer]

    def get_mlp(self, layer: int) -> Any:
        self._require()
        return self._layers()[layer].mlp

    def get_attention(self, layer: int) -> Any:
        self._require()
        return self._layers()[layer].self_attn

    # --- inference --- #
    def tokenize(self, text: str, add_generation_prompt: bool = False, **kwargs) -> Any:
        self._require()
        dev = self.device
        if add_generation_prompt and hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, return_tensors="pt",
            )
            return ids.to(dev)
        enc = self.tokenizer(text, return_tensors="pt", **kwargs)
        return {k: v.to(dev) for k, v in enc.items()}

    def forward_capture(self, text_or_ids: Any, **kwargs) -> dict:
        import torch

        self._require()
        if isinstance(text_or_ids, str):
            enc = self.tokenize(text_or_ids)
            input_ids = enc["input_ids"] if isinstance(enc, dict) else enc
            attention_mask = enc.get("attention_mask") if isinstance(enc, dict) else None
        else:
            input_ids = text_or_ids
            attention_mask = kwargs.get("attention_mask")
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=self.config.use_cache,
            )
        return {
            "hidden_states": out.hidden_states,  # tuple len n_layers+1
            "logits": out.logits,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def generate(self, text: str, **kwargs) -> str:
        import torch

        self._require()
        ids = self.tokenize(text, add_generation_prompt=True)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                do_sample=False,  # temperature=0 deterministic measurement
                temperature=None,
                top_p=None,
            )
        gen = out[0][ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)
