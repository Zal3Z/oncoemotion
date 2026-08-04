"""Build the emotion/control concept dataset from seeds (spec section 7).

Each example carries: concept, text, label (always 1 for a concept example; the
shared neutral pool is concept="neutral", label 0), condition, variety, family,
source and split.

Vector building uses a ONE-VS-REST scheme: the negatives for concept C are the
neutral pool PLUS every OTHER concept's examples. This forces each direction to
capture the *specific* affect rather than generic negative valence (the Phase-3
confound), because the negative set itself contains other negative-affect
concepts. Splits are deterministic given a seed and stratified per concept. All
members of one structured paraphrase family stay in one split, so near-identical
template siblings cannot leak between extraction and held-out evaluation.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from oncoemotion.emotion_vectors.augment import augmented_seed_bank
from oncoemotion.emotion_vectors.seeds import CONTROL_SEEDS, EMOTION_SEEDS, NEUTRAL


@dataclass
class EmotionExample:
    concept: str
    text: str
    label: int                # 1 = concept example, 0 = neutral pool
    condition: str = "comprehension"
    variety: str = "explicit"
    split: str = "extraction"
    family: str | None = None
    source: str = "authored_seed"


def _assign_splits(
    items: list[EmotionExample],
    rng: random.Random,
    fractions=(0.6, 0.2, 0.2),
) -> None:
    groups: dict[str, list[EmotionExample]] = defaultdict(list)
    for index, item in enumerate(items):
        family = item.family or f"singleton:{index}"
        groups[family].append(item)
    families = sorted(groups)
    rng.shuffle(families)
    n_groups = len(families)
    n_ex = int(round(n_groups * fractions[0]))
    n_val = int(round(n_groups * fractions[1]))
    for rank, family in enumerate(families):
        split = (
            "extraction" if rank < n_ex else
            "validation" if rank < n_ex + n_val else
            "test"
        )
        for item in groups[family]:
            item.split = split


def build_dataset(seed: int = 12345) -> list[EmotionExample]:
    rng = random.Random(seed)
    examples: list[EmotionExample] = []
    augmented = augmented_seed_bank()
    for concept, seeds in {**EMOTION_SEEDS, **CONTROL_SEEDS}.items():
        pos = [
            EmotionExample(
                concept,
                text,
                1,
                variety=tag,
                family=f"base:{concept}:{index:03d}",
            )
            for index, (text, tag) in enumerate(seeds)
        ]
        pos.extend(
            EmotionExample(
                concept,
                text,
                1,
                variety=tag,
                family=family,
                source="structured_augmentation",
            )
            for text, tag, family in augmented.get(concept, [])
        )
        _assign_splits(pos, rng)
        examples.extend(pos)
    neutrals = [
        EmotionExample(
            "neutral",
            text,
            0,
            variety="neutral",
            family=f"base:neutral:{index:03d}",
        )
        for index, text in enumerate(NEUTRAL)
    ]
    _assign_splits(neutrals, rng)
    examples.extend(neutrals)
    return examples


def save_jsonl(examples: list[EmotionExample], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[EmotionExample]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(EmotionExample(**json.loads(line)))
    return out
