"""Evaluation metrics (spec section 13).

Ships pure-python mapping metrics used from Phase 1 onward (accuracy, top-k
recall, macro-F1, abstention rate, selective accuracy). Calibration diagrams,
AUROC, coverage-risk curves and causal-effect metrics arrive with their phases.
"""

from oncoemotion.evaluation.metrics import (
    top1_accuracy,
    topk_recall,
    macro_f1,
    abstention_rate,
    selective_accuracy,
)

__all__ = [
    "top1_accuracy",
    "topk_recall",
    "macro_f1",
    "abstention_rate",
    "selective_accuracy",
]
