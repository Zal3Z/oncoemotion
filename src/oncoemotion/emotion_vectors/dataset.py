"""Build the emotion/control concept dataset from seeds (spec section 7).

Each example carries: concept, text, label (always 1 for a concept example; the
shared neutral pool is concept="neutral", label 0), condition, variety, split.

Vector building uses a ONE-VS-REST scheme: the negatives for concept C are the
neutral pool PLUS every OTHER concept's examples. This forces each direction to
capture the *specific* affect rather than generic negative valence (the Phase-3
confound), because the negative set itself contains other negative-affect
concepts. Splits are deterministic given a seed and stratified per concept.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from oncoemotion.emotion_vectors.seeds import CONTROL_SEEDS, EMOTION_SEEDS, NEUTRAL


@dataclass
class EmotionExample:
    concept: str
    text: str
    label: int                # 1 = concept example, 0 = neutral pool
    condition: str = "comprehension"
    variety: str = "explicit"
    split: str = "extraction"


def _assign_splits(items: list[EmotionExample], rng: random.Random,
                   fractions=(0.6, 0.2, 0.2)) -> None:
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n = len(items)
    n_ex = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    for rank, i in enumerate(idx):
        if rank < n_ex:
            items[i].split = "extraction"
        elif rank < n_ex + n_val:
            items[i].split = "validation"
        else:
            items[i].split = "test"


def build_dataset(seed: int = 12345) -> list[EmotionExample]:
    rng = random.Random(seed)
    examples: list[EmotionExample] = []
    for concept, seeds in {**EMOTION_SEEDS, **CONTROL_SEEDS}.items():
        pos = [EmotionExample(concept, text, 1, variety=tag) for text, tag in seeds]
        _assign_splits(pos, rng)
        examples.extend(pos)
    neutrals = [EmotionExample("neutral", text, 0, variety="neutral") for text in NEUTRAL]
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
