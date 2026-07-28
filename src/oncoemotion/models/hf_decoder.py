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

        dev = (self.config.device_map or "cpu")

        # Try several AutoModel classes so multimodal decoders (Gemma-3 / MedGemma,
        # Mistral-3, ...) load for TEXT-ONLY interpretability: AutoModelForCausalLM
        # first, then the image-text-to-text / vision-seq wrappers, then AutoModel.
        # Layer discovery (_layers) navigates the nested language_model, and
        # forward_capture runs text-only with output_hidden_states.
        import transformers as _tf
        loaders = [AutoModelForCausalLM]
        for _name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"):
            _cls = getattr(_tf, _name, None)
            if _cls is not None:
                loaders.append(_cls)

        def _acquire(trc: bool):
            """Load tokenizer + model with a given trust_remote_code setting."""
            tok = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=trc)
            kw = dict(trust_remote_code=trc)
            if dev == "auto":
                kw["device_map"] = "auto"
            mdl, last = None, None
            for loader in loaders:
                try:
                    try:  # dtype arg name changed across transformers versions
                        mdl = loader.from_pretrained(self.config.model_id, dtype=dtype, **kw)
                    except TypeError:
                        mdl = loader.from_pretrained(self.config.model_id, torch_dtype=dtype, **kw)
                    break
                except Exception as e:  # arch not supported by this class -> next
                    last = e
            if mdl is None:
                raise last
            return tok, mdl

        try:
            self.tokenizer, model = _acquire(self.config.trust_remote_code)
        except Exception as e:
            # some models (e.g. Apertus) require custom code -> auto-enable it and retry
            if not self.config.trust_remote_code and "trust_remote_code" in str(e).lower():
                self.tokenizer, model = _acquire(True)
            else:
                raise

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

    def build_prompt_ids(self, user: str, system: str | None = None,
                         assistant_prefix: str = "") -> Any:
        """Chat-template ids for ``[system?, user]`` + an assistant-turn prefix.

        Applies the model's chat template so a SYSTEM role actually reaches the
        model, then appends ``assistant_prefix`` (no special tokens) as the start
        of the assistant turn. The LAST token is the last token of
        ``assistant_prefix`` — measurement point E. Falls back to merging the
        system text into the user turn for templates without a system slot
        (e.g. some Gemma/Mistral templates).
        """
        import torch

        self._require()
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            try:
                base = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               return_tensors="pt")
            except Exception:
                merged = (system + "\n\n" + user) if system else user
                base = tok.apply_chat_template(
                    [{"role": "user", "content": merged}],
                    add_generation_prompt=True, return_tensors="pt")
        else:  # no chat template: plain concatenation
            text = (system + "\n" if system else "") + user + "\n"
            base = tok(text, return_tensors="pt").input_ids
        # apply_chat_template may return a tensor or a BatchEncoding depending on
        # the transformers version — normalize to a plain [1, T] id tensor.
        if not torch.is_tensor(base):
            base = base["input_ids"]
        if base.dim() == 1:
            base = base.unsqueeze(0)
        if assistant_prefix:
            pref = tok(assistant_prefix, add_special_tokens=False,
                       return_tensors="pt").input_ids
            base = torch.cat([base, pref.to(base.device)], dim=1)
        return base.to(self.device)

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
