"""Patching runtime: direction-only transfer, hook removal, no weight change."""

from __future__ import annotations

import numpy as np
import pytest


def _fake_adapter(torch, seq=3, H=4):
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = nn.Identity()
            self._p = nn.Parameter(torch.zeros(1))  # gives device/dtype

        def forward(self, x):
            return self.blk(x)

    model = M().eval()

    class FakeAdapter:
        def __init__(self, m):
            self.model = m
            self._seq = seq
            self._H = H

        @property
        def device(self):
            return next(self.model.parameters()).device

        def get_block(self, layer):
            return self.model.blk

        def forward_capture(self, prompt):
            # deterministic hidden states: block output = ones, logits from it
            seq = self._seq
            hs = torch.arange(seq * self._H, dtype=torch.float32).reshape(1, seq, self._H)
            out = self.model.blk(hs)  # triggers hook
            # real adapter logits are [batch, seq, vocab]; mirror that shape here
            return {"hidden_states": (hs, out), "logits": out, "input_ids": None,
                    "attention_mask": None}

    return FakeAdapter(model), model


def test_direction_only_transfer_replaces_component():
    torch = pytest.importorskip("torch")
    from oncoemotion.patching.runtime import PatchingRuntime

    adapter, model = _fake_adapter(torch)
    pr = PatchingRuntime(adapter)
    d = np.array([1.0, 0.0, 0.0, 0.0])
    # source value with a known component (5) along d
    src = np.array([5.0, 9.0, 9.0, 9.0])
    logits = pr.run_patched("recip", layer=1, source_value=src, mode="direction",
                            direction=d, token_index=-1)
    # last-token hidden was [8,9,10,11]; component along d replaced 8 -> 5
    assert abs(float(logits[0]) - 5.0) < 1e-4
    assert abs(float(logits[1]) - 9.0) < 1e-4  # orthogonal untouched


def test_patch_hook_removed_and_no_weight_change():
    torch = pytest.importorskip("torch")
    from oncoemotion.patching.runtime import PatchingRuntime

    adapter, model = _fake_adapter(torch)
    pr = PatchingRuntime(adapter)
    p_before = model._p.detach().clone()
    d = np.array([1.0, 0.0, 0.0, 0.0])
    pr.run_patched("recip", layer=1, source_value=np.zeros(4), mode="direction", direction=d)
    # a subsequent plain forward is unaffected (hook removed)
    out = adapter.forward_capture("x")["logits"]
    assert abs(float(out[0, -1, 0]) - 8.0) < 1e-4   # last-token first comp back to original
    assert torch.allclose(model._p, p_before)
