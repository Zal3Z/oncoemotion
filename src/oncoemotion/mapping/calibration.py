"""Probability calibration.

The specification requires that probabilities be *calibrated* and NOT obtained by
simply asking the model for a number — they must come from logits / a reranker /
a separately calibrated model.

Phase 1 ships a deterministic retriever with no logits, so probabilities here are
honestly labelled as **uncalibrated heuristic scores** (monotone in the retrieval
similarity). The :class:`Calibrator` interface (temperature / isotonic fit) is the
seam where a fitted calibrator plugs in once an LLM reranker produces logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class Calibrator(Protocol):
    is_calibrated: bool
    method: str

    def transform(self, score: float) -> float: ...


@dataclass
class HeuristicCalibrator:
    """Monotone squashing of a retrieval similarity into a pseudo-probability.

    Explicitly *uncalibrated*. Kept monotone so ranking is preserved; the value
    should be treated as a confidence proxy, not a calibrated probability.
    """

    temperature: float = 1.0
    is_calibrated: bool = False
    method: str = "uncalibrated_heuristic"

    def transform(self, score: float) -> float:
        s = max(0.0, min(1.0, float(score)))
        # Gentle logistic centred at 0.5 to spread mid-range scores.
        z = (s - 0.5) / max(self.temperature, 1e-6)
        p = 1.0 / (1.0 + math.exp(-6.0 * z))
        # Blend with raw score so exact matches stay near 1.0.
        return round(0.5 * p + 0.5 * s, 4)


@dataclass
class TemperatureCalibrator:
    """Placeholder for a *fitted* temperature-scaling calibrator over logits.

    Fit ``temperature`` on a validation set (later phase). Marked calibrated only
    after :meth:`fit` is called with real data.
    """

    temperature: float = 1.0
    is_calibrated: bool = False
    method: str = "temperature_scaling"

    def transform(self, logit: float) -> float:
        return 1.0 / (1.0 + math.exp(-logit / max(self.temperature, 1e-6)))
