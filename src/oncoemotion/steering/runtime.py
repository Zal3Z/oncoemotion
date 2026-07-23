"""Activation steering / ablation runtime via transient forward hooks (section 12).

Applies interventions to a loaded model WITHOUT modifying weights: a forward hook
adds ``alpha * unit(v)`` to (or removes the v-component from) a block's residual
output, and is removed on context exit even if an exception occurs (guaranteed by
:mod:`oncoemotion.activations.hooks`). Alpha is scaled to the residual-stream norm
so a given alpha has comparable effect across layers (spec section 12).

Nothing here mutates the model parameters; a steered run and a baseline run of the
same model are identical once the context exits.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from oncoemotion.activations.hooks import safe_hooks


def _unit_torch(vec, torch, dtype, device):
    v = torch.tensor(np.asarray(vec), dtype=dtype, device=device)
    n = v.norm()
    return v / n if n > 0 else v


class SteeringRuntime:
    def __init__(self, adapter):
        self.adapter = adapter

    def _block_hidden(self, output):
        """Decoder blocks may return a tuple; the hidden state is element 0."""
        if isinstance(output, tuple):
            return output[0], lambda hs: (hs,) + tuple(output[1:])
        return output, lambda hs: hs

    @contextmanager
    def intervene(self, layer: int, vector, alpha: float, mode: str = "add",
                  norm_scale: bool = True):
        """Context manager applying an intervention at ``layer``.

        mode='add'    -> h' = h + alpha_eff * unit(v)
        mode='ablate' -> h' = h - (h . unit(v)) unit(v)   (alpha ignored)
        """
        import torch

        # best_layer is an index into the hidden_states tuple (0..n_layers);
        # the hookable transformer blocks are 0..n_layers-1, so clamp.
        _nl = getattr(self.adapter, "n_layers", None)
        if isinstance(_nl, int) and _nl > 0:
            layer = min(int(layer), _nl - 1)
        block = self.adapter.get_block(layer)
        dev = self.adapter.device
        dtype = next(self.adapter.model.parameters()).dtype
        u = _unit_torch(vector, torch, dtype, dev)

        def hook(module, inputs, output):
            hs, rewrap = self._block_hidden(output)
            if mode == "ablate":
                coeff = (hs * u).sum(dim=-1, keepdim=True)  # [...,1]
                hs = hs - coeff * u
            else:
                a = float(alpha)
                if norm_scale:
                    # scale to the mean residual norm at this position
                    rn = hs.norm(dim=-1, keepdim=True).mean()
                    a = a * float(rn)
                hs = hs + a * u
            return rewrap(hs)

        with safe_hooks() as mgr:
            mgr.register(block, hook)
            yield

    # --- convenience: measure the effect on the point-E decision ---
    def decision_logits(self, prompt: str):
        import torch

        cap = self.adapter.forward_capture(prompt)
        return cap["logits"][0, -1].float()

    def steer_and_summarize(self, prompt: str, layer: int, vector, alpha: float,
                            mode: str = "add") -> dict:
        """Return next-token entropy/margin at point E, baseline vs intervened."""
        import torch

        def summ(logits):
            p = torch.softmax(logits, dim=-1)
            tv, ti = torch.topk(p, 2)
            ent = float(-(p * p.clamp_min(1e-12).log()).sum())
            return {"entropy": ent, "margin": float(tv[0] - tv[1]),
                    "top1_prob": float(tv[0]), "top1_id": int(ti[0])}

        base = summ(self.decision_logits(prompt))
        with self.intervene(layer, vector, alpha, mode=mode):
            steered = summ(self.decision_logits(prompt))
        return {
            "baseline": base, "steered": steered,
            "delta_entropy": steered["entropy"] - base["entropy"],
            "delta_margin": steered["margin"] - base["margin"],
            "top1_changed": base["top1_id"] != steered["top1_id"],
        }
