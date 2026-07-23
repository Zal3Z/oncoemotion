"""Steering runtime: forward-hook interventions, no weight change (spec section 16)."""

from __future__ import annotations

import numpy as np
import pytest


def _fake_adapter(torch):
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = nn.Linear(4, 4)

        def forward(self, x):
            return self.blk(x)

    model = M().eval()

    class FakeAdapter:
        def __init__(self, m):
            self.model = m

        @property
        def device(self):
            return next(self.model.parameters()).device

        def get_block(self, layer):
            return self.model.blk

    return FakeAdapter(model), model


def test_steer_add_shifts_output_and_restores():
    torch = pytest.importorskip("torch")
    from oncoemotion.steering.runtime import SteeringRuntime

    adapter, model = _fake_adapter(torch)
    rt = SteeringRuntime(adapter)
    x = torch.zeros(1, 4)
    v = np.array([1.0, 0.0, 0.0, 0.0])

    base = model.blk(x).detach().clone()
    w_before = model.blk.weight.detach().clone()

    with rt.intervene(layer=0, vector=v, alpha=2.0, mode="add", norm_scale=False):
        steered = model.blk(x).detach().clone()

    after = model.blk(x).detach().clone()
    # steered output shifted by +2 * unit(v) on the first component
    assert torch.allclose(steered - base, torch.tensor([[2.0, 0.0, 0.0, 0.0]]), atol=1e-5)
    # hook removed -> back to baseline
    assert torch.allclose(after, base, atol=1e-6)
    # weights never changed
    assert torch.allclose(model.blk.weight, w_before)


def test_ablate_removes_component():
    torch = pytest.importorskip("torch")
    from oncoemotion.steering.runtime import SteeringRuntime

    adapter, model = _fake_adapter(torch)
    # make the block output a known vector by setting identity weights + bias
    with torch.no_grad():
        model.blk.weight.copy_(torch.eye(4))
        model.blk.bias.copy_(torch.tensor([3.0, 4.0, 0.0, 0.0]))
    rt = SteeringRuntime(adapter)
    x = torch.zeros(1, 4)
    v = np.array([1.0, 0.0, 0.0, 0.0])
    with rt.intervene(layer=0, vector=v, alpha=0.0, mode="ablate"):
        out = model.blk(x).detach()
    # component along v removed -> first coord ~ 0
    assert abs(float(out[0, 0])) < 1e-5
    assert abs(float(out[0, 1]) - 4.0) < 1e-5
