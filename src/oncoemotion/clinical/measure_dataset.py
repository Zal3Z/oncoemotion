"""Controlled clinical measurement inputs (spec section 9), synthetic.

Progressive-content severity gradients + formulation variants + neutral controls.
Each item is tagged so Phase-3 analysis can relate emotion-like scores to
severity step, formulation type, and confounders (severity/urgency/safety),
keeping those measured SEPARATELY (spec sections 9, 11). No real patient data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class MeasureItem:
    item_id: str
    text: str
    group: str
    step: int | None = None          # gradient step (severity proxy), if any
    formulation: str = "statement"
    is_neutral: bool = False


# Progressive severity gradients (content escalates; NOT emotion words per se).
GRADIENTS: dict[str, list[str]] = {
    "mobility": [
        "cammino normalmente",
        "cammino più lentamente del solito",
        "faccio fatica a camminare",
        "devo appoggiarmi per riuscire a camminare",
        "non riesco più a camminare da solo",
    ],
    "pain": [
        "ho un fastidio appena percettibile",
        "ho un dolore lieve ma sopportabile",
        "ho un dolore moderato che mi distrae",
        "ho un dolore forte che mi limita nelle attività",
        "ho un dolore lancinante e insopportabile che non passa",
    ],
    "breath": [
        "respiro senza alcun problema",
        "ogni tanto mi manca un po' il fiato salendo le scale",
        "faccio fatica a respirare anche stando fermo",
        "riesco a malapena a prendere fiato",
        "non riesco quasi più a respirare e ho paura di soffocare",
    ],
    "nausea": [
        "non ho alcun disturbo allo stomaco",
        "ho un leggero senso di nausea ogni tanto",
        "ho spesso la nausea durante il giorno",
        "ho una nausea continua e vomito più volte",
        "vomito senza sosta e non riesco a trattenere nulla",
    ],
    "prognosis": [
        "i controlli sono andati bene e sono tranquillo",
        "devo ripetere alcuni esami di controllo",
        "i medici hanno trovato qualcosa da approfondire",
        "la situazione si è aggravata rispetto a prima",
        "mi hanno detto che le cure non stanno più funzionando",
    ],
}

# Formulation variants for one symptom concept (spec section 9), here Nausea.
FORMULATIONS: list[tuple[str, str]] = [
    ("nausea", "keyword"),
    ("ho la nausea", "short_phrase"),
    ("da qualche giorno avverto un persistente senso di nausea che mi accompagna per gran parte della giornata", "articulated"),
    ("appena mi alzo lo stomaco mi si rivolta", "implicit"),
    ("non ho nausea", "negation"),
    ("avevo nausea il mese scorso ma ora è passata", "past_resolved"),
    ("la parola «nausea» va classificata come sintomo", "metalanguage"),
    ("ho la nasuea", "misspelling"),
    ("ho la nausea e anche un forte mal di testa", "multi_symptom"),
]

# Neutral controls for the baseline (flat, non-emotional).
NEUTRALS: list[str] = [
    "la visita è fissata per giovedì mattina",
    "il modulo di consenso è stato firmato",
    "la terapia va assunta due volte al giorno",
    "il referto sarà disponibile in una settimana",
    "l'appuntamento è nello studio al secondo piano",
    "il paziente ha compilato il questionario",
    "la prossima seduta è programmata per lunedì",
    "i dati anagrafici sono stati registrati",
]


def build_measure_items() -> list[MeasureItem]:
    items: list[MeasureItem] = []
    for group, steps in GRADIENTS.items():
        for i, text in enumerate(steps):
            items.append(MeasureItem(f"grad_{group}_{i}", text, f"gradient:{group}", step=i))
    for text, form in FORMULATIONS:
        items.append(MeasureItem(f"form_nausea_{form}", text, "formulation:nausea", formulation=form))
    for i, text in enumerate(NEUTRALS):
        items.append(MeasureItem(f"neutral_{i}", text, "neutral", is_neutral=True, formulation="neutral"))
    return items


def to_records(items: list[MeasureItem]) -> list[dict]:
    return [asdict(i) for i in items]
