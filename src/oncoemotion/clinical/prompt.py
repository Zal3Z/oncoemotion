"""Decision-prompt construction for the clinical measurement task (spec section 10).

The measurement uses an IDENTICAL output prefix via teacher forcing so that the
pre-decision point (E) is the same token position for every input:

    {"pro_ctcae":{"term":"

Point E is the LAST token of the prompt (the opening quote before the term is
generated) — activations there are captured immediately before the model commits
to a PRO-CTCAE term. An optional neutral filler is inserted between the patient
free-text and the decision prefix to test persistence (spec section 11).
"""

from __future__ import annotations

DECISION_INSTRUCTION = (
    "Sei un sistema di codifica clinica PRO-CTCAE. "
    "Leggi il testo del paziente e indica il termine PRO-CTCAE più appropriato."
)

# Identical teacher-forced output prefix (spec section 10).
TEACHER_PREFIX = '{"pro_ctcae":{"term":"'

# Standard neutral sentence for the persistence test (identical for all inputs).
NEUTRAL_FILLER = (
    "Questa è una procedura di codifica standard, ordinaria e di routine."
)


def build_decision_prompt(free_text: str, neutral_filler: str | None = None) -> str:
    """Return the full prompt whose LAST token is measurement point E."""
    parts = [
        DECISION_INSTRUCTION,
        f'Testo del paziente: "{free_text}"',
    ]
    if neutral_filler:
        parts.append(neutral_filler)
    parts.append("Rispondi in formato JSON.")
    # No trailing newline/space after the prefix: the last char is the opening
    # quote, so the next generated token is the first token of the term.
    return "\n".join(parts) + "\n" + TEACHER_PREFIX
