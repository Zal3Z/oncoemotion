"""Pure-python mapping metrics (no heavy deps)."""

from __future__ import annotations

from collections import defaultdict


def top1_accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def topk_recall(gold: list[str], pred_topk: list[list[str]], k: int | None = None) -> float:
    if not gold:
        return 0.0
    hit = 0
    for g, preds in zip(gold, pred_topk):
        cand = preds[:k] if k else preds
        hit += int(g in cand)
    return hit / len(gold)


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = set(gold) | set(pred)
    if not labels:
        return 0.0
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    f1s = []
    for lab in labels:
        prec = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        rec = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def abstention_rate(abstained: list[bool]) -> float:
    if not abstained:
        return 0.0
    return sum(bool(a) for a in abstained) / len(abstained)


def selective_accuracy(gold: list[str], pred: list[str], abstained: list[bool]) -> float:
    """Accuracy on the answered (non-abstained) subset."""
    answered = [(g, p) for g, p, a in zip(gold, pred, abstained) if not a]
    if not answered:
        return 0.0
    return sum(g == p for g, p in answered) / len(answered)
