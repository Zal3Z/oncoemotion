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

# Role-neutral coding task (constant across role conditions). The persona lives in
# the SYSTEM message (see ROLE_PERSONAS) so that only the role varies between
# conditions and the task framing is identical.
TASK_INSTRUCTION = (
    "Leggi il testo del paziente e indica il termine PRO-CTCAE più appropriato."
)

# System personas for the role experiment. ``None`` = no system message (baseline:
# the coding task alone, no persona). Keys are the role identifiers used in scripts.
ROLE_PERSONAS = {
    # --- medici ---
    "oncologo": (
        "Sei un oncologo esperto, un medico specializzato nella diagnosi e "
        "nella cura dei pazienti oncologici."
    ),
    "infermiere": (
        "Sei un infermiere esperto, abituato ad assistere ogni giorno pazienti "
        "oncologici."
    ),
    # --- tecnici / distaccati ---
    "ingegnere": (
        "Sei un ingegnere, abituato ad analizzare problemi tecnici in modo "
        "preciso e distaccato."
    ),
    "avvocato": (
        "Sei un avvocato, abituato a valutare i fatti con lucidità e distacco "
        "professionale."
    ),
    "contabile": (
        "Sei un contabile, abituato a registrare dati con precisione e senza "
        "coinvolgimento emotivo."
    ),
    # --- emotivi / profani ---
    "paziente_ansioso": (
        "Sei una persona molto in ansia per la propria salute, spaventata dai "
        "sintomi che avverti."
    ),
    "bambino": (
        "Sei un bambino di otto anni, senza alcuna conoscenza medica."
    ),
    "poeta": (
        "Sei un poeta, una persona molto sensibile ed emotiva, che vive ogni "
        "cosa in modo intenso."
    ),
    # --- controlli ---
    "generico": (
        "Sei un assistente generico, senza alcuna competenza medica specifica."
    ),
    "empatico": (
        "Sei una persona molto empatica e sensibile, profondamente partecipe "
        "delle sofferenze delle persone."
    ),
    "none": None,
}

# Identical teacher-forced output prefix (spec section 10).
TEACHER_PREFIX = '{"pro_ctcae":{"term":"'

# Standard neutral sentence for the persistence test (identical for all inputs).
NEUTRAL_FILLER = (
    "Questa è una procedura di codifica standard, ordinaria e di routine."
)


def build_decision_prompt(free_text: str, neutral_filler: str | None = None) -> str:
    """Return the full raw prompt whose LAST token is measurement point E.

    Legacy single-string path (no chat template / no system role) used by
    ``run_probing.py``. For the role experiment use :func:`build_decision_messages`.
    """
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


def build_decision_messages(
    free_text: str, role: str | None = None, neutral_filler: str | None = None
) -> tuple[str | None, str]:
    """Return ``(system_text, user_text)`` for the role-conditioned decision.

    The persona (``ROLE_PERSONAS[role]``) becomes the SYSTEM message; the constant
    coding task + patient text + JSON instruction become the USER message. The
    teacher-forced prefix (``TEACHER_PREFIX``) is added by the adapter as the start
    of the assistant turn, so point E remains the identical final token across
    roles. ``role=None`` or ``"none"`` yields no system message.
    """
    system = ROLE_PERSONAS.get(role) if role else None
    parts = [
        TASK_INSTRUCTION,
        f'Testo del paziente: "{free_text}"',
    ]
    if neutral_filler:
        parts.append(neutral_filler)
    parts.append("Rispondi in formato JSON.")
    return system, "\n".join(parts)
