"""Assertion, temporality and experiencer detection (Italian, rule-based).

This is a transparent, deterministic baseline. It intentionally uses curated
negation patterns that attach to symptom nouns (``non ho``, ``senza`` ...)
rather than bare ``non``, because in Italian ``non vedo bene`` / ``non riesco a``
express a *present* symptom, not its negation. Known limitation, documented and
covered by regression tests for the mandatory cases.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# Window (characters) around a mention to search for cues.
WINDOW = 60


def _fold(text: str) -> str:
    d = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in d if unicodedata.category(ch) != "Mn")


# Negation patterns that attach to a following symptom noun.
_NEGATION = [_fold(x) for x in (
    "non ho ", "non ha ", "non ho avuto", "non ho piu", "non abbiamo ",
    "senza ", "nessun ", "nessuna ", "nessuno ", "niente ", "assenza di ",
    "non presenta", "non accuso", "non accusa", "non lamento", "non lamenta",
    "non avverto", "non avverte", "non provo", "non prova", "non c'e ",
    "non ci sono", "mai avuto", "negativo per",
)]

# Resolution cues -> temporality "resolved".
_RESOLVED = [_fold(x) for x in (
    "e passat", "ora e passat", "adesso e passat", "non piu", "risolt",
    "guarit", "scompars", "ora sto bene", "adesso sto bene", "e finit",
    "e cessat", "e sparit", "sono guarit",
)]

# Past cues -> temporality "past".
_PAST = [_fold(x) for x in (
    "il mese scorso", "la settimana scorsa", "l'anno scorso", "ieri",
    "l'altro ieri", "tempo fa", "in passato", "avevo ", "ho avuto",
    "ha avuto", "giorni fa", "settimane fa", "mesi fa", "scorso", "scorsa",
)]

# Hypothetical cues -> assertion "hypothetical".
_HYPO = [_fold(x) for x in (
    "se ", "qualora", "nel caso", "se dovessi", "in caso di", "in futuro",
)]

# Uncertainty cues -> assertion "uncertain".
_UNCERTAIN = [_fold(x) for x in (
    "forse", "magari", "credo", "penso", "mi sembra", "non sono sicur",
    "dubito", "probabilmente", "potrebbe", "sospetto",
)]

# Experiencer = other (someone other than the patient).
_OTHER = [_fold(x) for x in (
    "mia madre", "mio padre", "mia moglie", "mio marito", "mio figlio",
    "mia figlia", "mio fratello", "mia sorella", "un amico", "un'amica",
    "il vicino", "la vicina", "un parente", "mia nonna", "mio nonno",
    "il paziente accanto", "mia zia", "mio zio",
)]


@dataclass
class AssertionResult:
    assertion: str = "present"          # present|negated|uncertain|hypothetical
    temporality: str = "current"        # current|past|resolved|unknown
    experiencer: str = "patient"        # patient|other|unknown
    cues: list[str] = field(default_factory=list)


def _any(patterns: list[str], *windows: str) -> str | None:
    for w in windows:
        for p in patterns:
            if p and p in w:
                return p
    return None


def detect_assertion_temporality(
    display_text: str, span_start: int, span_end: int
) -> AssertionResult:
    """Detect assertion/temporality/experiencer for a mention span."""
    lower = display_text.lower()
    before = _fold(lower[max(0, span_start - WINDOW):span_start])
    after = _fold(lower[span_end:span_end + WINDOW])
    full = _fold(lower)

    cues: list[str] = []

    # --- assertion ---
    assertion = "present"
    neg = _any(_NEGATION, before)
    hypo = _any(_HYPO, before)
    unc = _any(_UNCERTAIN, before, after)
    if neg:
        assertion = "negated"
        cues.append(f"neg:{neg.strip()}")
    elif hypo:
        assertion = "hypothetical"
        cues.append(f"hypo:{hypo.strip()}")
    elif unc:
        assertion = "uncertain"
        cues.append(f"unc:{unc.strip()}")

    # --- temporality (resolved > past > current) ---
    temporality = "current"
    res = _any(_RESOLVED, after, before)
    past = _any(_PAST, before, after)
    if res:
        temporality = "resolved"
        cues.append(f"res:{res.strip()}")
    elif past:
        temporality = "past"
        cues.append(f"past:{past.strip()}")

    # --- experiencer ---
    experiencer = "patient"
    other = _any(_OTHER, before, full)
    if other:
        experiencer = "other"
        cues.append(f"other:{other.strip()}")

    return AssertionResult(assertion, temporality, experiencer, cues)
