"""Activation capture and intervention plumbing (spec sections 6, 10-12).

The hook manager here is deliberately framework-agnostic so it can be unit
tested without torch: it only needs objects exposing ``register_forward_hook``
returning a handle with ``.remove()``. The real model adapter passes torch
modules; tests pass fakes.
"""

from oncoemotion.activations.hooks import HookManager, safe_hooks

__all__ = ["HookManager", "safe_hooks"]
