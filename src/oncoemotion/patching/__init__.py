"""Activation patching (spec section 12).

Cache a source run's activation at (layer, token); transfer either the full
activation or only its component along the emotion direction into a recipient
run; measure the change in the recipient's decision. See :mod:`.runtime`.
"""

from oncoemotion.patching.runtime import PatchingRuntime

__all__ = ["PatchingRuntime"]

PHASE = 4
