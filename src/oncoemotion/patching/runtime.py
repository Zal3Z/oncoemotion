"""Activation patching runtime (spec section 12).

Cache a SOURCE run's activation at (layer, token); run a RECIPIENT and transfer
either the full activation OR only its component along the emotion direction into
the recipient's hidden state at that position; measure the change in the
recipient's decision. Transient forward hooks (removed on exit); no weight change.

Direction-only transfer (the spec's "solo la componente nella direzione emotiva"):

    h' = h - (h·d̂) d̂ + (s·d̂) d̂

i.e. replace the recipient's component along the unit emotion direction ``d̂``
with the source's component, leaving the orthogonal complement untouched.
"""

from __future__ import annotations

import numpy as np

from oncoemotion.activations.hooks import safe_hooks


class PatchingRuntime:
    def __init__(self, adapter):
        self.adapter = adapter

    def capture(self, prompt: str, layer: int, token_index: int = -1) -> np.ndarray:
        """Cache the hidden state at (layer, token) for a source prompt. Returns [H]."""
        cap = self.adapter.forward_capture(prompt)
        return cap["hidden_states"][layer][0, token_index].float().cpu().numpy()

    def _hidden(self, output):
        if isinstance(output, tuple):
            return output[0], lambda hs: (hs,) + tuple(output[1:])
        return output, lambda hs: hs

    def run_patched(self, prompt: str, layer: int, source_value: np.ndarray,
                    mode: str = "direction", direction: np.ndarray | None = None,
                    token_index: int = -1):
        """Run ``prompt`` with the (layer, token) hidden patched; return point-E logits."""
        import torch

        _nl = getattr(self.adapter, "n_layers", None)
        blk_i = min(int(layer), _nl - 1) if isinstance(_nl, int) and _nl > 0 else int(layer)
        block = self.adapter.get_block(blk_i)
        dev = self.adapter.device
        dtype = next(self.adapter.model.parameters()).dtype
        s = torch.tensor(np.asarray(source_value), dtype=dtype, device=dev)
        d = None
        if mode == "direction":
            if direction is None:
                raise ValueError("direction required for mode='direction'")
            d = torch.tensor(np.asarray(direction), dtype=dtype, device=dev)
            d = d / d.norm().clamp_min(1e-8)

        def hook(module, inputs, output):
            hs, rewrap = self._hidden(output)
            tok = hs[:, token_index, :]
            if mode == "full":
                new = s.to(hs.dtype)
            else:  # direction-only transfer
                recip_coeff = (tok * d).sum(-1, keepdim=True)
                src_coeff = float((s * d).sum())
                new = tok - recip_coeff * d + src_coeff * d
            hs = hs.clone()
            hs[:, token_index, :] = new
            return rewrap(hs)

        with safe_hooks() as mgr:
            mgr.register(block, hook)
            cap = self.adapter.forward_capture(prompt)
            return cap["logits"][0, -1].float()

    def patch_and_summarize(self, source_prompt: str, recipient_prompt: str, layer: int,
                            direction: np.ndarray, mode: str = "direction") -> dict:
        """Transfer source→recipient; return baseline vs patched decision summary."""
        import torch

        def summ(logits):
            p = torch.softmax(logits, dim=-1)
            tv, ti = torch.topk(p, 2)
            ent = float(-(p * p.clamp_min(1e-12).log()).sum())
            return {"entropy": ent, "margin": float(tv[0] - tv[1]),
                    "top1_prob": float(tv[0]), "top1_id": int(ti[0])}

        src_val = self.capture(source_prompt, layer)
        base = summ(self.adapter.forward_capture(recipient_prompt)["logits"][0, -1].float())
        patched = summ(self.run_patched(recipient_prompt, layer, src_val, mode=mode, direction=direction))
        return {
            "baseline": base, "patched": patched,
            "delta_entropy": patched["entropy"] - base["entropy"],
            "delta_margin": patched["margin"] - base["margin"],
            "top1_changed": base["top1_id"] != patched["top1_id"],
        }
