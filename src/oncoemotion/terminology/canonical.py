"""Single source of truth for the 80 canonical PRO-CTCAE symptom terms.

The English strings and their applicable attributes are the *canonical
identifiers* taken verbatim from the PRO-CTCAE item library as reproduced in
the project specification (section 3). They are treated as ground truth and are
covered by regression tests (exact count of 80, unique ids, attribute mapping).

IMPORTANT PROVENANCE RULE
-------------------------
Everything in :data:`SYNTHETIC_SEEDS` (Italian synonyms, patient phrases,
negation / exclusion examples) is **synthetic developer placeholder data**,
clearly labelled as such. It exists only so the baseline mapper and the
regression tests are meaningful *before* the official Italian PRO-CTCAE PDF is
available. None of it should be mistaken for validated clinical terminology.
Official Italian labels must be loaded from the official document and stored in
``official_italian_labels`` (empty until then).

Attribute letter legend (spec section 3):
    F = Frequency, S = Severity, I = Interference, P = Presence/Absence, A = Amount
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Attribute code -> human readable attribute name.
# --------------------------------------------------------------------------- #
ATTRIBUTE_NAMES: dict[str, str] = {
    "F": "frequency",
    "S": "severity",
    "I": "interference",
    "P": "presence",
    "A": "amount",
}

# --------------------------------------------------------------------------- #
# The 80 canonical terms: (ordinal, canonical_english, category, attr_letters).
# Order and content follow the specification exactly.
# --------------------------------------------------------------------------- #
CANONICAL_TERMS: list[tuple[int, str, str, str]] = [
    # ORAL
    (1, "Dry mouth", "Oral", "S"),
    (2, "Difficulty swallowing", "Oral", "S"),
    (3, "Mouth/throat sores", "Oral", "SI"),
    (4, "Cracking at the corners of the mouth (cheilosis/cheilitis)", "Oral", "S"),
    (5, "Voice quality changes", "Oral", "P"),
    (6, "Hoarseness", "Oral", "S"),
    # GASTROINTESTINAL
    (7, "Taste changes", "Gastrointestinal", "S"),
    (8, "Decreased appetite", "Gastrointestinal", "SI"),
    (9, "Nausea", "Gastrointestinal", "FS"),
    (10, "Vomiting", "Gastrointestinal", "FS"),
    (11, "Heartburn", "Gastrointestinal", "FS"),
    (12, "Gas", "Gastrointestinal", "P"),
    (13, "Bloating", "Gastrointestinal", "FS"),
    (14, "Hiccups", "Gastrointestinal", "FS"),
    (15, "Constipation", "Gastrointestinal", "S"),
    (16, "Diarrhea", "Gastrointestinal", "F"),
    (17, "Abdominal pain", "Gastrointestinal", "FSI"),
    (18, "Fecal incontinence", "Gastrointestinal", "FI"),
    # RESPIRATORY
    (19, "Shortness of breath", "Respiratory", "SI"),
    (20, "Cough", "Respiratory", "SI"),
    (21, "Wheezing", "Respiratory", "S"),
    # CARDIO/CIRCULATORY
    (22, "Swelling", "Cardiac/Circulatory", "FSI"),
    (23, "Heart palpitations", "Cardiac/Circulatory", "FS"),
    # CUTANEOUS
    (24, "Rash", "Cutaneous", "P"),
    (25, "Skin dryness", "Cutaneous", "S"),
    (26, "Acne", "Cutaneous", "S"),
    (27, "Hair loss", "Cutaneous", "A"),
    (28, "Itching", "Cutaneous", "S"),
    (29, "Hives", "Cutaneous", "P"),
    (30, "Hand-foot syndrome", "Cutaneous", "S"),
    (31, "Nail loss", "Cutaneous", "P"),
    (32, "Nail ridging", "Cutaneous", "P"),
    (33, "Nail discoloration", "Cutaneous", "P"),
    (34, "Sensitivity to sunlight", "Cutaneous", "P"),
    (35, "Bed/pressure sores", "Cutaneous", "P"),
    (36, "Radiation skin reaction", "Cutaneous", "S"),
    (37, "Skin darkening", "Cutaneous", "P"),
    (38, "Stretch marks", "Cutaneous", "P"),
    # NEUROLOGICAL
    (39, "Numbness & tingling", "Neurological", "SI"),
    (40, "Dizziness", "Neurological", "SI"),
    # VISUAL/PERCEPTUAL
    (41, "Blurred vision", "Visual/Perceptual", "SI"),
    (42, "Flashing lights", "Visual/Perceptual", "P"),
    (43, "Visual floaters", "Visual/Perceptual", "P"),
    (44, "Watery eyes", "Visual/Perceptual", "SI"),
    (45, "Ringing in ears", "Visual/Perceptual", "S"),
    # ATTENTION/MEMORY
    (46, "Concentration", "Attention/Memory", "SI"),
    (47, "Memory", "Attention/Memory", "SI"),
    # PAIN
    (48, "General pain", "Pain", "FSI"),
    (49, "Headache", "Pain", "FSI"),
    (50, "Muscle pain", "Pain", "FSI"),
    (51, "Joint pain", "Pain", "FSI"),
    # SLEEP/WAKE
    (52, "Insomnia", "Sleep/Wake", "SI"),
    (53, "Fatigue", "Sleep/Wake", "SI"),
    # MOOD
    (54, "Anxious", "Mood", "FSI"),
    (55, "Discouraged", "Mood", "FSI"),
    (56, "Sad", "Mood", "FSI"),
    # GENITOURINARY
    (57, "Irregular periods/vaginal bleeding", "Genitourinary", "P"),
    (58, "Missed expected menstrual period", "Genitourinary", "P"),
    (59, "Vaginal discharge", "Genitourinary", "A"),
    (60, "Vaginal dryness", "Genitourinary", "S"),
    (61, "Painful urination", "Genitourinary", "S"),
    (62, "Urinary urgency", "Genitourinary", "FI"),
    (63, "Urinary frequency", "Genitourinary", "FI"),
    (64, "Change in usual urine color", "Genitourinary", "P"),
    (65, "Urinary incontinence", "Genitourinary", "FI"),
    # SEXUAL
    (66, "Achieve and maintain erection", "Sexual", "S"),
    (67, "Ejaculation", "Sexual", "F"),
    (68, "Decreased libido", "Sexual", "S"),
    (69, "Delayed orgasm", "Sexual", "P"),
    (70, "Unable to have orgasm", "Sexual", "P"),
    (71, "Pain w/sexual intercourse", "Sexual", "S"),
    # MISCELLANEOUS
    (72, "Breast swelling and tenderness", "Miscellaneous", "S"),
    (73, "Bruising", "Miscellaneous", "P"),
    (74, "Chills", "Miscellaneous", "FS"),
    (75, "Increased sweating", "Miscellaneous", "FS"),
    (76, "Decreased sweating", "Miscellaneous", "P"),
    (77, "Hot flashes", "Miscellaneous", "FS"),
    (78, "Nosebleed", "Miscellaneous", "FS"),
    (79, "Pain and swelling at injection site", "Miscellaneous", "P"),
    (80, "Body odor", "Miscellaneous", "S"),
]

EXPECTED_TERM_COUNT = 80


def canonical_id(ordinal: int) -> str:
    """Return the zero-padded canonical id, e.g. ``33 -> 'PRO_033'``."""
    return f"PRO_{ordinal:03d}"


def attribute_names(letters: str) -> list[str]:
    """Map an attribute-letter string (e.g. ``'FS'``) to attribute names."""
    return [ATTRIBUTE_NAMES[ch] for ch in letters]


# --------------------------------------------------------------------------- #
# SYNTHETIC developer seeds (Italian). CLEARLY LABELLED PLACEHOLDER DATA.
# Keyed by canonical id. Only a subset of terms is seeded — enough to make the
# baseline mapper and the mandatory regression cases meaningful. These strings
# are NOT official terminology and must be replaced by reviewed data.
#
# Field separation (spec section 3 "keep separated"):
#   synonyms            -> here used ONLY for synthetic dev synonyms
#   common_patient_phrases -> synthetic dev free-text patient phrasings
#   negation_examples   -> synthetic examples that must resolve to NEGATED
#   exclusion_examples  -> synthetic phrasings that must NOT map to the term
# Official labels + clinically reviewed synonyms are intentionally left empty.
# --------------------------------------------------------------------------- #
SYNTHETIC_SEEDS: dict[str, dict[str, list[str]]] = {
    "PRO_009": {  # Nausea
        "synonyms": ["nausea", "senso di nausea", "nauseato", "nauseata"],
        "common_patient_phrases": ["ho la nausea", "mi viene da vomitare", "sento la nausea"],
        "negation_examples": ["non ho nausea", "nessuna nausea", "senza nausea"],
        "exclusion_examples": [],
    },
    "PRO_010": {  # Vomiting
        "synonyms": ["vomito", "vomitare"],
        "common_patient_phrases": ["ho vomitato", "continuo a vomitare"],
        "negation_examples": ["non ho vomitato"],
        "exclusion_examples": [],
    },
    "PRO_015": {  # Constipation
        "synonyms": ["stitichezza", "stipsi", "stitico", "stitica"],
        "common_patient_phrases": ["non riesco ad andare in bagno"],
        "negation_examples": [],
        "exclusion_examples": [],
    },
    "PRO_016": {  # Diarrhea
        "synonyms": ["diarrea"],
        "common_patient_phrases": ["ho la diarrea", "scariche liquide"],
        "negation_examples": [],
        "exclusion_examples": [],
    },
    "PRO_019": {  # Shortness of breath
        "synonyms": ["fiato corto", "affanno", "dispnea", "mancanza di fiato"],
        "common_patient_phrases": ["fatico a respirare", "mi manca il respiro"],
        "negation_examples": [],
        "exclusion_examples": [],
    },
    "PRO_020": {  # Cough
        "synonyms": ["tosse"],
        "common_patient_phrases": ["ho la tosse", "tossisco molto"],
        "negation_examples": [],
        "exclusion_examples": [],
    },
    "PRO_027": {  # Hair loss
        "synonyms": ["perdita di capelli", "caduta dei capelli", "alopecia"],
        "common_patient_phrases": ["mi cadono i capelli", "sto perdendo i capelli"],
        "negation_examples": [],
        "exclusion_examples": [],
    },
    "PRO_028": {  # Itching
        "synonyms": ["prurito"],
        "common_patient_phrases": ["ho prurito", "mi prude la pelle"],
        "negation_examples": [],
        "exclusion_examples": [],
    },
    "PRO_033": {  # Nail discoloration
        "synonyms": [
            "unghie gialle",
            "unghie ingiallite",
            "scolorimento delle unghie",
            "unghie scolorite",
            "cambiamento di colore delle unghie",
        ],
        "common_patient_phrases": [
            "mi si ingialliscono le unghie",
            "le unghie stanno diventando gialle",
            "le unghie cambiano colore",
            "le unghie sono completamente gialle",
        ],
        "negation_examples": [],
        # Cosmetic causes that must NOT be coded as a clinical symptom event:
        "exclusion_examples": [
            "ho messo lo smalto giallo",
            "ho messo uno smalto giallo",
            "ho applicato lo smalto",
            "smalto giallo",
        ],
    },
    "PRO_049": {  # Headache
        "synonyms": ["mal di testa", "cefalea", "emicrania"],
        "common_patient_phrases": ["mi fa male la testa"],
        "negation_examples": [],
        "exclusion_examples": [],
    },
    "PRO_053": {  # Fatigue
        "synonyms": ["stanchezza", "affaticamento", "spossatezza", "fatica", "astenia"],
        "common_patient_phrases": ["mi sento stanco", "mi sento stanca", "sono sempre stanco"],
        "negation_examples": [],
        "exclusion_examples": [],
    },
    "PRO_054": {  # Anxious
        "synonyms": ["ansia", "ansioso", "ansiosa", "angoscia", "nervosismo"],
        "common_patient_phrases": ["mi sento in ansia", "provo ansia", "sono in ansia"],
        "negation_examples": ["non ho ansia"],
        "exclusion_examples": [],
    },
    "PRO_056": {  # Sad
        "synonyms": ["tristezza", "triste", "abbattuto", "abbattuta"],
        "common_patient_phrases": ["mi sento triste", "sono giù di morale"],
        "negation_examples": [],
        "exclusion_examples": [],
    },
}
