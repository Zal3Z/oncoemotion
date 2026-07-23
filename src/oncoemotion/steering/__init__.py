"""Activation steering & ablation (spec section 12).

PHASE 4. Steering  h' = h + alpha * v ; ablation  h' = h - proj_v(h).
Alpha grid [-0.10,-0.05,-0.02,0,0.02,0.05,0.10] adapted to the residual-stream
norm (never large enough to cause generative collapse). Interventions applied
to: all post-stimulus tokens / pre-decision tokens only / single layers / layer
windows / residual stream / MLP output / attention output.

Guarantee (spec section 16): steering must never permanently modify weights —
implemented as transient forward hooks via
:mod:`oncoemotion.activations.hooks` and removed after each run.
"""

from oncoemotion.steering.spec import SteeringSpec, steer_add, ablate_projection

__all__ = ["SteeringSpec", "steer_add", "ablate_projection"]

PHASE = 4
