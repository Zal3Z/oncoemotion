"""Hook lifecycle: removed even after exceptions (spec section 16).

Uses a torch-free fake module exposing register_forward_hook -> handle.remove().
"""

from __future__ import annotations

import pytest

from oncoemotion.activations.hooks import HookManager, safe_hooks


class FakeHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class FakeModule:
    def __init__(self):
        self.handles = []

    def register_forward_hook(self, fn):
        h = FakeHandle()
        self.handles.append(h)
        return h


def test_hooks_removed_on_normal_exit():
    mod = FakeModule()
    with safe_hooks() as mgr:
        mgr.register(mod, lambda *a: None)
        mgr.register(mod, lambda *a: None)
        assert mgr.active_count == 2
    assert all(h.removed for h in mod.handles)


def test_hooks_removed_after_exception():
    mod = FakeModule()
    with pytest.raises(RuntimeError):
        with safe_hooks() as mgr:
            mgr.register(mod, lambda *a: None)
            raise RuntimeError("boom during forward")
    assert all(h.removed for h in mod.handles)
    assert len(mod.handles) == 1


def test_hookmanager_context_removes():
    mod = FakeModule()
    with HookManager() as mgr:
        mgr.register(mod, mgr.make_capture_hook("blk"))
        assert mgr.active_count == 1
    assert mod.handles[0].removed
    assert mgr.active_count == 0


def test_capture_hook_stores_output():
    mgr = HookManager()
    hook = mgr.make_capture_hook("layer0")
    hook(None, None, "activation-tensor")
    assert mgr.captured["layer0"] == "activation-tensor"
