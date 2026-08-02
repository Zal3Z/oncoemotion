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


# --- token-matched role spans -------------------------------------------------
# The personas were written freehand and come out between 18 and 29 tokens, and the
# "none" control had no system block at all -- zero tokens. Everything after the
# system block therefore sat at a different absolute position in every condition,
# and in a transformer position is not neutral: the C2 result was confounded with it
# by construction. Worse, "none" was not a control at all, it was a structurally
# different prompt.
#
# Padding is computed against the real tokenizer at run time, never hardcoded: the
# same string is a different number of tokens in Qwen, Gemma and EuroLLM, so a fixed
# pad would equalize one model and skew the rest. The clauses are procedural, carry
# no identity, no affect and no clinical content, and go after the persona so that
# the end of the system block -- and hence every later position -- lands identically.
# Graded in length so a greedy fill lands on the target with real sentences instead
# of a repeated stub.
ROLE_PAD_CLAUSES = [
    "La sessione segue la procedura ordinaria prevista dal servizio.",
    "Il formato della risposta e' quello consueto.",
    "Le voci restano in ordine standard.",
    "La registrazione avviene come sempre.",
    "I passaggi sono quelli abituali.",
    "Il riferimento resta invariato.",
    "L'ordine non cambia.",
    "Come di consueto.",
    "Come sempre.",
    "Si procede.",
]

# Semantically empty stand-in for "no role": present in structure, absent in
# identity, so it can be padded to the same length as every persona.
NO_ROLE_STUB = "Questa e' una sessione di codifica."


def build_padded_personas(tokenizer, target: int | None = None,
                          personas: dict | None = None) -> tuple[dict, dict]:
    """Return ``(padded_personas, token_counts)`` all of identical token length.

    ``target`` defaults to the longest persona, rounded up to fit whole padding
    clauses. Raises if any span cannot be brought to the target, so a silent
    mismatch is impossible.
    """
    src = dict(personas or ROLE_PERSONAS)
    src = {k: (v if v else NO_ROLE_STUB) for k, v in src.items()}

    def n_tok(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False).input_ids)

    base = {k: n_tok(v) for k, v in src.items()}
    # (cost, clause) sorted longest first, so the greedy fill closes most of the gap
    # with one real sentence and trims with the short ones
    clauses = sorted(((n_tok(" " + c), c) for c in ROLE_PAD_CLAUSES), reverse=True)
    goal = target or max(base.values())

    out, counts = {}, {}
    for k, text in src.items():
        cur = base[k]
        used = set()
        progress = True
        while cur < goal and progress:
            progress = False
            for i, (cost, clause) in enumerate(clauses):
                if i not in used and cur + cost <= goal:
                    text += " " + clause
                    cur += cost
                    used.add(i)
                    progress = True
                    break
        out[k], counts[k] = text, cur

    spread = max(counts.values()) - min(counts.values())
    if spread > 2:
        raise ValueError(
            f"role spans still differ by {spread} tokens after padding "
            f"({counts}); add shorter entries to ROLE_PAD_CLAUSES")
    return out, counts


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
    free_text: str, role: str | None = None, neutral_filler: str | None = None,
    personas: dict | None = None
) -> tuple[str | None, str]:
    """Return ``(system_text, user_text)`` for the role-conditioned decision.

    The persona becomes the SYSTEM message; the constant coding task + patient text
    + JSON instruction become the USER message. The teacher-forced prefix
    (``TEACHER_PREFIX``) is added by the adapter as the start of the assistant turn,
    so point E remains the identical final token across roles.

    Pass ``personas`` from :func:`build_padded_personas` to get token-matched spans,
    including a real length-matched control for ``role="none"``. Without it the
    unpadded ``ROLE_PERSONAS`` are used and everything after the system block sits
    at a role-dependent position.
    """
    table = personas if personas is not None else ROLE_PERSONAS
    system = table.get(role) if role else None
    parts = [
        TASK_INSTRUCTION,
        f'Testo del paziente: "{free_text}"',
    ]
    if neutral_filler:
        parts.append(neutral_filler)
    parts.append("Rispondi in formato JSON.")
    return system, "\n".join(parts)
