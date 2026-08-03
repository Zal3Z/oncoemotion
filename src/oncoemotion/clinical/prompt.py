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

# Controls, defined before the persona table that uses them.
NO_ROLE_STUB = ""
TASK_FRAME_STUB = "Questa e' una sessione di codifica."

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
    # --- controlli: nessuna identita' ---
    "none_filler": "",          # solo riempitivo: il controllo vero
    "none_task": TASK_FRAME_STUB,   # nomina il compito, non un'identita'
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
# Padding must be inert. The first version said things like "la sessione segue la
# procedura ordinaria" and the no-role stub said "questa e' una sessione di
# codifica" -- both point at the task. That turned the control into a task
# instruction, so "role vs no role" was really "identity framing vs task framing",
# and the control won. These are discourse markers: grammatical, natural in a system
# message, and about neither the task nor any identity.
ROLE_PAD_CLAUSES = [
    "Va bene cosi', senza altre indicazioni particolari da aggiungere.",
    "D'accordo, nulla di piu' da precisare.",
    "Certamente, niente altro da aggiungere.",
    "Va bene, tutto chiaro.",
    "D'accordo, si intende.",
    "Certamente, senz'altro.",
    "Va bene cosi'.",
    "D'accordo.",
    "Certamente.",
    "Bene.",
]

# Three distinct controls instead of one, because the first version silently mixed
# them. They decompose what a system message can contribute:
#   none_empty  -- no system block at all: what you get by writing no system prompt,
#                  but structurally a different prompt, so not a clean contrast;
#   none_filler -- padding only: length-matched, no identity, no task. THE control;
#   none_task   -- names the task without naming an identity. This is what the old
#                  "none" accidentally was, and it is worth keeping as its own arm:
#                  identity-vs-task is a real question, it just is not the one the
#                  label "no role" claims to answer.


def build_padded_personas(tokenizer, target: int | None = None,
                          personas: dict | None = None) -> tuple[dict, dict]:
    """Return ``(padded_personas, token_counts)`` all of identical token length.

    ``target`` defaults to the longest persona, rounded up to fit whole padding
    clauses. Raises if any span cannot be brought to the target, so a silent
    mismatch is impossible.
    """
    src = dict(personas or ROLE_PERSONAS)
    # "none" stays a genuinely empty block; the padded controls are separate
    # entries so that "no role" and "task named, no role" never collapse again.
    # A None persona means literally no system block, and stays None: that is a
    # structurally different prompt, kept as a separate arm rather than quietly
    # padded into something that only looks like a control.
    keep_none = {k for k, v in src.items() if v is None}
    src = {k: (v or "") for k, v in src.items()}

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
        out[k], counts[k] = text.strip(), cur

    for k in keep_none:
        out[k], counts[k] = None, 0
    padded = {k: v for k, v in counts.items() if k not in keep_none}
    spread = max(padded.values()) - min(padded.values())
    if spread > 2:
        raise ValueError(
            f"role spans still differ by {spread} tokens after padding "
            f"({padded}); add shorter entries to ROLE_PAD_CLAUSES")
    return out, counts


def build_decision_ids(adapter, free_text: str, role: str = "none",
                       neutral_filler: str | None = None, personas: dict | None = None):
    """Token ids for the decision prompt, whose LAST token is measurement point E.

    The single construction path. Before this, the probing / steering / patching
    scripts used :func:`build_decision_prompt` -- a raw string with no chat template
    and no system role -- while the two role experiments used
    :func:`build_decision_messages` with a templated system persona. A3 and A4 were
    therefore measured on a different object than C2 and C5, and the master report
    placed them side by side as though they were the same experiment.

    ``role`` defaults to the no-role control, which is now a real padded system
    block rather than an absent one.
    """
    system, user = build_decision_messages(free_text, role=role,
                                           neutral_filler=neutral_filler,
                                           personas=personas)
    return adapter.build_prompt_ids(user, system, assistant_prefix=TEACHER_PREFIX)


def read_point_index(adapter, free_text: str, ids, role: str = "none",
                     neutral_filler: str | None = None,
                     personas: dict | None = None) -> int | None:
    """Token index of the LAST token of the patient text -- the read point, R.

    The cheap half of the R/D split. The thesis distinguishes the state the patient's
    text leaves the reader in (R) from the state it decides in (D, the existing point
    E), but separating them properly means moving the coding instruction after the
    patient text, which restructures the prompt and breaks comparability with every
    published number. This locates R inside the prompt as it already stands: not a
    clean pre-instruction read point, but enough to answer whether the two positions
    behave differently at all -- and therefore whether the full rewrite is worth
    paying for later.

    Returns None when the position cannot be located exactly, which is the honest
    outcome for a tokenizer whose template output does not round-trip. Callers should
    treat None as "no R for this item" rather than as an error.
    """
    system, user = build_decision_messages(free_text, role=role,
                                           neutral_filler=neutral_filler,
                                           personas=personas)
    tok = adapter.tokenizer
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    try:
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None
    full = rendered + TEACHER_PREFIX

    marker = f'"{free_text}"'
    pos = full.rfind(marker)
    if pos < 0:
        return None
    char_end = pos + len(marker)

    try:
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        return None            # slow tokenizer: no offsets, no exact index
    offsets = enc["offset_mapping"]

    # the ids we computed here must be the ids the model actually sees, otherwise the
    # index points into a different sequence
    have = ids[0].tolist() if hasattr(ids, "shape") else list(ids)
    got = list(enc["input_ids"])
    if got != have:
        if len(have) >= len(got) and have[-len(got):] == got:
            got = have          # template added leading specials: shift the offsets
            offsets = [(-1, -1)] * (len(have) - len(offsets)) + list(offsets)
        else:
            return None

    for i, (a, b) in enumerate(offsets):
        if a < 0:
            continue
        if b >= char_end:
            return i
    return None


def build_decision_prompt(free_text: str, neutral_filler: str | None = None) -> str:
    """Return the full raw prompt whose LAST token is measurement point E.

    DEPRECATED for measurement. No chat template, no system role, so a model that
    was instruction-tuned on a chat format sees something it was never trained on,
    and the result is not comparable with the role experiments. Every measurement
    script now goes through :func:`build_decision_ids`; what is left on this path is
    the x-ray viewer and the dashboard, which display a single prompt rather than
    produce numbers that enter the analysis.
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
