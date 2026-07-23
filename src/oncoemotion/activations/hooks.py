"""Framework-agnostic forward-hook manager with guaranteed removal.

Guarantees (spec section 16):
  * hooks are removed even if an exception is raised during the forward pass;
  * no permanent modification is made to the module (only forward hooks, which
    are transient and removed on exit).

Works with any object exposing ``register_forward_hook(fn) -> handle`` where
``handle`` has a ``.remove()`` method (torch ``nn.Module`` satisfies this). Unit
tests use a lightweight fake so the guarantees are verified without torch.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator


class HookManager:
    """Register forward hooks and remove them all reliably."""

    def __init__(self) -> None:
        self._handles: list[Any] = []
        self.captured: dict[str, Any] = {}

    def register(self, module: Any, hook: Callable) -> "HookManager":
        handle = module.register_forward_hook(hook)
        self._handles.append(handle)
        return self

    def make_capture_hook(self, name: str) -> Callable:
        """Return a hook that stores the module output under ``name``."""

        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.captured[name] = output

        return _hook

    def remove_all(self) -> None:
        errors: list[Exception] = []
        for h in self._handles:
            try:
                h.remove()
            except Exception as e:  # collect, keep removing the rest
                errors.append(e)
        self._handles.clear()
        if errors:
            raise RuntimeError(f"Errors while removing {len(errors)} hook(s): {errors!r}")

    @property
    def active_count(self) -> int:
        return len(self._handles)

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.remove_all()
        return False  # never suppress exceptions


@contextmanager
def safe_hooks() -> Iterator[HookManager]:
    """Context manager yielding a :class:`HookManager`; always removes hooks."""
    mgr = HookManager()
    try:
        yield mgr
    finally:
        mgr.remove_all()
