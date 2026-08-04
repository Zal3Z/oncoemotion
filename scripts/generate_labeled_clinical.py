#!/usr/bin/env python
"""Generate the GOLD-LABELLED clinical dataset for the role/emotion study.

Design
------
Every seed is a *template* holding one ``{q}`` slot plus two fillers::

    template = "Mal di pancia {q} da tre giorni, peggiora dopo i pasti"
    q_neutral   = "costante"
    q_marked = "lancinante"

The neutral and emotional framings are therefore **identical outside the slot**.
That is the whole point: in the previous version the two poles were authored
freehand and the emotional one came out 2.5x longer (median 10 words vs 4) with
commas in 40% of items vs 1%, so "emotional framing hurts coding accuracy" was
confounded with "longer text hurts coding accuracy". With a shared stem, length,
punctuation, body-site mentions, tense, time references and digits are matched by
construction and only the marked qualifier varies.  Version 2 explicitly splits
the marked qualifiers into patient-affective reactions (threat/distress,
demoralisation, frustration or social shame) and symptom-intensity language.  The
first subset carries the preregistered emotion question; the second is retained as
a specificity control, so a severity effect cannot be relabelled as an affective
effect after results are seen.

Surface statistics are calibrated on the aggregate profile of a real PRO-CTCAE
free-text corpus (1194 responses). The experimental items must imitate its
**>=7-word tail** (n=115, median 8 words), not its marginal distribution (median
3 words): below ~5 words no affective reformulation at constant clinical content
exists, so the paired design would collapse. Target rates, from that tail:

    body site 57% | verb 29% | comma 16% | time reference 16%
    first person 8% | negation 6% | digits 6% | exclamation marks 0%

Zero exclamation marks is a hard check, not a target: the real corpus contains
none, so an exclamation mark in the emotional pole would let a model separate the
framings on punctuation alone.

Gold ``class``:
  - ``term``    : a specific PRO-CTCAE term is expected (EXACT_PRO_MATCH);
  - ``abstain`` : no PRO term should be coded (negated / no-direct / out-of-scope /
                  insufficient / urgent). Correctness for these is a *false-positive
                  coding* test, not a term match.

The script writes the JSONL dataset, a per-item feature annotation, and a
validation report that runs the deterministic mapper on every item and flags
gold-vs-mapper disagreements. The mapper is a REFERENCE, not the source of gold.

Usage:
    python scripts/generate_labeled_clinical.py
    python scripts/generate_labeled_clinical.py --check-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

EXACT = "EXACT_PRO_MATCH"
NEG = "NEGATED_SYMPTOM"
NOD = "NO_DIRECT_PRO_MATCH"
OOS = "OUT_OF_SCOPE"
INS = "INSUFFICIENT_CONTEXT"

# Surface-feature targets taken from the >=7-word tail of the real corpus.
# (feature name -> target rate); tolerance is +/-0.08 on the term subset.
TAIL_PROFILE = {
    "body_site": 0.574,
    "verb": 0.287,
    "comma": 0.165,
    "time_ref": 0.165,
    "first_person": 0.078,
    "negation": 0.061,
    "digits": 0.061,
}
PROFILE_TOL = 0.08
LEN_BAND = (7, 12)

# Predeclared semantic partition of the marked qualifier.  These terms describe a
# patient's affective appraisal/reaction rather than the physical magnitude or
# sensory quality of the symptom.  The complement is labelled symptom_intensity.
# Exact membership is intentionally auditable instead of being inferred by an LLM.
AFFECTIVE_QUALIFIER_FAMILIES = {
    "threat": {
        "allarmante", "allarmanti", "inquietante", "preoccupante",
        "spaventosa", "spaventosi", "spaventoso", "terrificante", "terrificanti",
    },
    "distress_demoralization": {
        "angosciante", "opprimente", "disperante", "avvilente", "annientante",
        "devastante", "devastanti", "traumatico",
    },
    "anger_frustration": {"esasperante", "frustranti"},
    "shame_social": {"imbarazzante", "umiliante", "umilianti"},
}


def _qualifier_class(qualifier: str) -> tuple[str, str | None]:
    normalized = qualifier.casefold().strip()
    for family, terms in AFFECTIVE_QUALIFIER_FAMILIES.items():
        if normalized in terms:
            return "affective_reaction", family
    return "symptom_intensity", None

# ---------------------------------------------------------------- TERM seeds --
# (pro_id, canonical_english, template with {q}, q_neutral, q_emotional)
# The two fillers must agree in gender/number with the noun they qualify and stay
# within one word of each other.
TERM_SEEDS = [
    # --- cavo orale, deglutizione, voce, gusto -------------------------------
    ("PRO_001", "Dry mouth", "Bocca secca {q} con difficolta a parlare a lungo", "persistente", "insopportabile"),
    ("PRO_001", "Dry mouth", "Sensazione di bocca secca {q} anche dopo aver bevuto acqua", "continua", "angosciante"),
    ("PRO_002", "Difficulty swallowing", "Difficolta a deglutire {q} con blocco in gola", "marcata", "terrificante"),
    ("PRO_002", "Difficulty swallowing", "Riferisce disfagia {q} per i cibi solidi e le compresse", "ingravescente", "angosciante"),
    ("PRO_002", "Difficulty swallowing", "Ogni deglutizione risulta {q} e richiede piu tentativi", "difficoltosa", "straziante"),
    ("PRO_003", "Mouth/throat sores", "Piaghe in bocca {q} sul palato e sulle gengive", "diffuse", "atroci"),
    ("PRO_003", "Mouth/throat sores", "Afte in gola {q}, che rendono difficile alimentarsi", "numerose", "devastanti"),
    ("PRO_005", "Voice quality changes", "Modificazione della voce {q} con abbassamento del tono", "progressiva", "avvilente"),
    ("PRO_006", "Hoarseness", "Voce rauca {q} con affaticamento dopo poche frasi", "costante", "esasperante"),
    ("PRO_007", "Taste changes", "Alterazione del gusto {q} per i cibi salati e dolci", "marcata", "devastante"),
    ("PRO_007", "Taste changes", "Disgeusia {q} riferita dopo l'ultimo ciclo di terapia", "persistente", "angosciante"),
    ("PRO_007", "Taste changes", "Il sapore dei cibi risulta {q}, quasi assente in bocca", "attenuato", "disgustoso"),
    ("PRO_007", "Taste changes", "Sento un gusto metallico {q} in bocca a ogni pasto", "continuo", "nauseante"),
    ("PRO_008", "Decreased appetite", "Calo dell'appetito {q} con riduzione delle porzioni abituali", "progressivo", "preoccupante"),
    ("PRO_008", "Decreased appetite", "Anoressia {q} con rifiuto dei pasti principali", "marcata", "spaventosa"),
    ("PRO_008", "Decreased appetite", "Appetito {q} con senso di sazieta dopo pochi bocconi", "ridotto", "azzerato"),

    # --- apparato digerente ---------------------------------------------------
    ("PRO_009", "Nausea", "Nausea {q} al mattino, prima di fare colazione", "persistente", "tremenda"),
    ("PRO_009", "Nausea", "Nausea {q} nelle 48 ore successive alla terapia", "continua", "devastante"),
    ("PRO_009", "Nausea", "Senso di malessere allo stomaco {q} lontano dai pasti", "ricorrente", "opprimente"),
    ("PRO_010", "Vomiting", "Vomito {q} dopo i pasti principali, anche con liquidi", "ripetuto", "incontenibile"),
    ("PRO_010", "Vomiting", "Episodi di vomito {q} riferiti dopo ogni somministrazione", "frequenti", "sfiancanti"),
    ("PRO_011", "Heartburn", "Bruciore di stomaco {q} che risale fino alla gola", "quotidiano", "lancinante"),
    ("PRO_011", "Heartburn", "Riferisce pirosi gastrica {q} con rigurgito acido", "costante", "insopportabile"),
    ("PRO_012", "Gas", "Flatulenza {q} con tensione a livello dell'addome", "frequente", "imbarazzante"),
    ("PRO_013", "Bloating", "Gonfiore della pancia {q} dopo ogni pasto principale", "marcato", "insopportabile"),
    ("PRO_013", "Bloating", "Addome gonfio e teso in modo {q} con indumenti stretti", "costante", "angosciante"),
    ("PRO_014", "Hiccups", "Singhiozzo {q} che dura anche mezz'ora di seguito", "ricorrente", "estenuante"),
    ("PRO_015", "Constipation", "Stitichezza {q} da 5 giorni nonostante l'uso di fibre", "persistente", "disperante"),
    ("PRO_015", "Constipation", "Alvo chiuso in modo {q} con sforzo alla defecazione", "costante", "angosciante"),
    ("PRO_016", "Diarrhea", "Feci liquide {q} con crampi a livello dell'addome", "frequenti", "devastanti"),
    ("PRO_016", "Diarrhea", "Diarrea {q} con numerose scariche e urgenza improvvisa", "profusa", "umiliante"),
    ("PRO_017", "Abdominal pain", "Mal di pancia {q}, che peggiora subito dopo i pasti", "costante", "lancinante"),
    ("PRO_017", "Abdominal pain", "Dolore addominale {q} in sede periombelicale, di tipo crampiforme", "moderato", "straziante"),
    ("PRO_017", "Abdominal pain", "Ho un dolore {q} alla pancia che mi blocca", "sordo", "atroce"),
    ("PRO_018", "Fecal incontinence", "Perdite di feci {q} con necessita di protezione assorbente", "occasionali", "umilianti"),

    # --- respiratorio e cardiovascolare --------------------------------------
    ("PRO_019", "Shortness of breath", "Mancanza di fiato {q} salendo due rampe di scale", "marcata", "terrificante"),
    ("PRO_019", "Shortness of breath", "Dispnea {q} anche a riposo, con senso di peso al petto", "presente", "angosciante"),
    ("PRO_020", "Cough", "Tosse secca {q} che peggiora in posizione sdraiata", "persistente", "estenuante"),
    ("PRO_020", "Cough", "Riferisce tosse {q} con dolore al torace dopo gli accessi", "stizzosa", "sfiancante"),
    ("PRO_021", "Wheezing", "Respiro sibilante {q} avvertito soprattutto in espirazione", "ricorrente", "spaventoso"),
    ("PRO_022", "Swelling", "Gonfiore alle gambe {q} con impronta alla digitopressione", "bilaterale", "preoccupante"),
    ("PRO_022", "Swelling", "Edema alle braccia {q} con anelli che non entrano", "moderato", "impressionante"),
    ("PRO_023", "Heart palpitations", "Palpitazioni {q} avvertite al petto anche stando fermi", "frequenti", "terrificanti"),
    ("PRO_023", "Heart palpitations", "Battito cardiaco {q} sotto sforzo lieve, come salire le scale", "accelerato", "spaventoso"),

    # --- cute e annessi -------------------------------------------------------
    ("PRO_024", "Rash", "Irritazione della pelle {q} sulle braccia e sul tronco", "diffusa", "insopportabile"),
    ("PRO_025", "Skin dryness", "Secchezza della pelle {q} su mani e gambe, con desquamazione", "marcata", "fastidiosissima"),
    ("PRO_026", "Acne", "Brufoli sul volto {q} comparsi con l'inizio della terapia", "numerosi", "umilianti"),
    ("PRO_027", "Hair loss", "Perdita dei capelli {q} evidente sul cuscino e nella doccia", "progressiva", "devastante"),
    ("PRO_027", "Hair loss", "Caduta dei capelli a ciocche in modo {q} alla spazzola", "costante", "angosciante"),
    ("PRO_028", "Itching", "Prurito {q} su braccia e schiena, senza lesioni visibili", "continuo", "insopportabile"),
    ("PRO_028", "Itching", "Riferisce prurito {q} alle gambe con lesioni da grattamento", "persistente", "esasperante"),
    ("PRO_028", "Itching", "Ho un prurito {q} alla schiena che la crema non calma", "costante", "disperante"),
    ("PRO_029", "Hives", "Orticaria {q} sul tronco, comparsa dopo la somministrazione", "diffusa", "spaventosa"),
    ("PRO_030", "Hand-foot syndrome", "Screpolature alle mani {q} con arrossamento dei palmi", "marcate", "dolorosissime"),
    ("PRO_031", "Nail loss", "Perdita di due unghie delle mani in modo {q}", "progressivo", "traumatico"),
    ("PRO_032", "Nail ridging", "Solchi {q} sulle unghie delle mani e dei piedi", "evidenti", "impressionanti"),
    ("PRO_033", "Nail discoloration", "Variazione di colore delle unghie {q} su entrambe le mani", "diffusa", "inquietante"),
    ("PRO_034", "Sensitivity to sunlight", "Sensibilita della pelle al sole {q} con arrossamento rapido", "aumentata", "insopportabile"),
    ("PRO_036", "Radiation skin reaction", "Bruciature della pelle {q} nella zona irradiata del torace", "estese", "atroci"),
    ("PRO_037", "Skin darkening", "Inscurimento della pelle {q} sul dorso delle mani", "evidente", "avvilente"),
    ("PRO_073", "Bruising", "Lividi {q} sulle braccia comparsi senza traumi evidenti", "numerosi", "allarmanti"),

    # --- neurologico e sensoriale --------------------------------------------
    ("PRO_039", "Numbness & tingling", "Formicolio alle mani {q} con difficolta ad abbottonarsi", "persistente", "angosciante"),
    ("PRO_039", "Numbness & tingling", "Intorpidimento ai piedi {q} che altera la percezione del suolo", "progressivo", "terrificante"),
    ("PRO_040", "Dizziness", "Giramenti di testa {q} alzandosi dal letto o dalla sedia", "frequenti", "spaventosi"),
    ("PRO_040", "Dizziness", "Vertigini {q} con instabilita nella marcia", "ricorrenti", "terrificanti"),
    ("PRO_041", "Blurred vision", "Appannamento della vista {q} con difficolta nella lettura", "intermittente", "spaventoso"),
    ("PRO_042", "Flashing lights", "Lampi davanti agli occhi {q} in entrambi i campi visivi", "ricorrenti", "terrificanti"),
    ("PRO_043", "Visual floaters", "Mosche volanti davanti agli occhi in modo {q}, a destra", "costante", "inquietante"),
    ("PRO_044", "Watery eyes", "Lacrimazione {q} da entrambi gli occhi, senza arrossamento", "eccessiva", "imbarazzante"),
    ("PRO_045", "Ringing in ears", "Ronzio nelle orecchie {q} che copre le voci in stanza", "continuo", "tormentoso"),
    ("PRO_045", "Ringing in ears", "Acufene {q} riferito da entrambi i lati, tono acuto", "persistente", "esasperante"),
    ("PRO_045", "Ringing in ears", "Un fischio {q} nelle orecchie che copre la televisione", "costante", "disperante"),
    ("PRO_046", "Concentration", "Problemi di concentrazione {q} nella lettura di poche pagine", "marcati", "frustranti"),
    ("PRO_046", "Concentration", "Difficolta a mantenere l'attenzione in modo {q} durante una conversazione", "costante", "avvilente"),
    ("PRO_047", "Memory", "Problemi di memoria {q} sugli impegni e sui nomi", "frequenti", "spaventosi"),
    ("PRO_047", "Memory", "Dimentica gli appuntamenti in modo {q}, secondo i familiari", "ricorrente", "angosciante"),

    # --- dolore, sonno, energia ----------------------------------------------
    ("PRO_048", "General pain", "Dolore fisico diffuso {q} in tutto il corpo, senza sede prevalente", "costante", "insopportabile"),
    ("PRO_048", "General pain", "Riferisce dolore {q} scarsamente responsivo agli analgesici", "persistente", "atroce"),
    ("PRO_049", "Headache", "Mal di testa {q}, senza risposta agli analgesici abituali", "continuo", "martellante"),
    ("PRO_049", "Headache", "Cefalea {q} in sede frontale e alle tempie", "quotidiana", "lancinante"),
    ("PRO_049", "Headache", "Ho mal di testa {q} da quando ho iniziato la cura", "costante", "devastante"),
    ("PRO_050", "Muscle pain", "Dolore ai muscoli delle gambe {q} dopo brevi camminate", "marcato", "atroce"),
    ("PRO_050", "Muscle pain", "Mialgie {q} agli arti inferiori, prevalenti alle cosce", "diffuse", "strazianti"),
    ("PRO_051", "Joint pain", "Dolore alle ginocchia {q} con rigidita nei primi movimenti", "presente", "insopportabile"),
    ("PRO_051", "Joint pain", "Artralgie {q} a spalle e gomiti, con limitazione funzionale", "bilaterali", "devastanti"),
    ("PRO_051", "Joint pain", "Le articolazioni fanno male in modo {q} a ogni passo", "costante", "atroce"),
    ("PRO_052", "Insomnia", "Insonnia {q} con risvegli nella seconda parte della notte", "persistente", "estenuante"),
    ("PRO_052", "Insomnia", "Difficolta ad addormentarsi in modo {q}, oltre due ore a letto", "ricorrente", "disperante"),
    ("PRO_053", "Fatigue", "Stanchezza {q} che limita le normali attivita domestiche", "costante", "devastante"),
    ("PRO_053", "Fatigue", "Astenia {q} riferita dopo l'ultimo ciclo di terapia", "marcata", "annientante"),
    ("PRO_053", "Fatigue", "Mancanza di energia {q} anche dopo il riposo notturno", "persistente", "opprimente"),
    ("PRO_053", "Fatigue", "Mi sento stanco in modo {q}, senza recupero dopo il riposo", "continuo", "insopportabile"),

    # --- sfera psicologica ----------------------------------------------------
    ("PRO_054", "Anxious", "Ansia {q} nei giorni che precedono i controlli programmati", "presente", "opprimente"),
    ("PRO_054", "Anxious", "Riferisce stato ansioso {q} con tensione muscolare al collo", "persistente", "devastante"),
    ("PRO_055", "Discouraged", "Scoraggiamento {q} rispetto all'andamento del percorso di cura", "riferito", "profondo"),
    ("PRO_056", "Sad", "Tristezza {q} con riduzione degli interessi abituali", "ricorrente", "opprimente"),
    ("PRO_056", "Sad", "Umore deflesso in modo {q} con tendenza all'isolamento", "costante", "disperante"),

    # --- genito-urinario ------------------------------------------------------
    ("PRO_061", "Painful urination", "Bruciore nell'urinare {q} a ogni minzione della giornata", "marcato", "atroce"),
    ("PRO_062", "Urinary urgency", "Bisogno urgente di urinare {q} con difficolta a trattenere", "frequente", "umiliante"),
    ("PRO_063", "Urinary frequency", "Bisogno di urinare {q}, circa 12 volte in 24 ore", "frequente", "esasperante"),
    ("PRO_064", "Change in usual urine color", "Variazione del colore delle urine {q}, piu scure del solito", "evidente", "allarmante"),
    ("PRO_065", "Urinary incontinence", "Perdite di urina {q} sotto sforzo, tossendo o ridendo", "occasionali", "umilianti"),
    ("PRO_068", "Decreased libido", "Calo del desiderio sessuale {q} riferito dall'inizio della terapia", "marcato", "avvilente"),
    ("PRO_072", "Breast swelling and tenderness", "Tensione mammaria {q} in entrambe le mammelle, al tatto", "presente", "dolorosissima"),

    # --- sistemici ------------------------------------------------------------
    ("PRO_074", "Chills", "Brividi {q} con necessita di coprirsi ripetutamente", "ricorrenti", "incontrollabili"),
    ("PRO_075", "Increased sweating", "Sudorazione {q} che bagna gli indumenti e le lenzuola", "abbondante", "umiliante"),
    ("PRO_075", "Increased sweating", "Iperidrosi {q} riferita a mani e fronte", "marcata", "imbarazzante"),
    ("PRO_077", "Hot flashes", "Vampate di calore {q} al volto e al collo", "frequenti", "insopportabili"),
    ("PRO_078", "Nosebleed", "Sangue dal naso {q} negli ultimi 3 giorni, senza traumi", "ricorrente", "spaventoso"),
    ("PRO_078", "Nosebleed", "Epistassi {q} riferita dalla narice destra", "ripetuta", "allarmante"),
    ("PRO_079", "Pain and swelling at injection site",
     "Dolore nel punto della somministrazione {q} con arrossamento locale", "persistente", "lancinante"),
    ("PRO_080", "Body odor", "Variazione dell'odore corporeo {q} riferita dai familiari", "avvertita", "umiliante"),
]

# ------------------------------------------------------- NEGATED (abstain) ----
NEG_SEEDS = [
    ("Nessuna nausea {q} nelle ultime 48 ore", "riferita", "per fortuna"),
    ("Non ho avuto dolore {q} durante tutta la settimana", "addominale", "per niente"),
    ("Mal di testa assente in modo {q} da tre giorni", "completo", "finalmente"),
    ("Non riferisce tosse {q} al controllo odierno", "produttiva", "grazie al cielo"),
    ("Nessun episodio di vomito {q} dopo l'ultimo ciclo", "registrato", "per fortuna"),
    ("Non ho piu quell'ansia {q} che avevo prima", "iniziale", "opprimente"),
    ("Prurito assente in modo {q} da alcuni giorni", "stabile", "finalmente"),
    ("Nessuna scarica diarroica {q} nelle ultime 24 ore", "riferita", "per fortuna"),
    ("Non ho dolore alle articolazioni in modo {q}", "persistente", "per niente"),
    ("La caduta dei capelli si e fermata in modo {q}", "stabile", "finalmente"),
    ("Non riferisce mancanza di fiato {q} sotto sforzo", "significativa", "affatto"),
    ("Nessun bruciore di stomaco {q} da inizio settimana", "riferito", "per fortuna"),
    ("Non ho piu quel formicolio {q} alle mani", "iniziale", "angosciante"),
    ("Insonnia assente in modo {q} nelle ultime notti", "costante", "finalmente"),
    ("Non riferisce palpitazioni {q} al controllo", "recenti", "per fortuna"),
    ("Nessun giramento di testa {q} negli ultimi due giorni", "segnalato", "per fortuna"),
]

# ------------------------------------------- NO_DIRECT (abstain) --------------
# (template, q_neutral, q_emotional, ctcae_term_or_None)
NOD_SEEDS = [
    ("Febbre {q} da due giorni con brividi serali", "elevata", "spaventosa", "Fever"),
    ("Difficolta a camminare {q} negli ultimi giorni", "progressiva", "terrificante", None),
    ("Pressione arteriosa {q} rilevata al controllo domiciliare", "elevata", "allarmante", None),
    ("Calo di peso {q} senza modifiche della dieta", "documentato", "angosciante", None),
    ("Glicemia {q} nelle ultime rilevazioni domiciliari", "elevata", "preoccupante", None),
    ("Colorito giallo delle sclere {q} da ieri mattina", "evidente", "terrificante", None),
    ("Debolezza agli arti inferiori {q} con difficolta a reggersi", "marcata", "spaventosa", None),
    ("Infezione in corso {q} segnalata dal medico curante", "documentata", "angosciante", None),
    ("Linfonodi ingrossati al collo {q} rilevati alla palpazione", "bilaterali", "inquietanti", None),
    ("Emoglobina {q} agli ultimi esami ematici", "ridotta", "preoccupante", None),
    ("Ferita che non si rimargina in modo {q} da settimane", "evidente", "disperante", None),
    ("Piastrine {q} riscontrate al controllo di ieri", "ridotte", "allarmanti", None),
    ("Tumefazione al braccio {q} comparsa nei giorni scorsi", "evidente", "spaventosa", None),
    ("Difficolta a parlare {q} notata dai familiari", "progressiva", "terrificante", None),
    ("Valori epatici {q} riscontrati agli esami di controllo", "alterati", "allarmanti", None),
    ("Frequenza cardiaca {q} rilevata a riposo stamattina", "elevata", "spaventosa", None),
]

# ------------------------------------------- OUT_OF_SCOPE (abstain) -----------
OOS_SEEDS = [
    ("Ho messo uno smalto giallo {q} sulle unghie", "nuovo", "bellissimo"),
    ("Vorrei spostare la visita di controllo {q} alla settimana prossima", "prenotata", "attesissima"),
    ("Devo ancora pagare il ticket {q} dell'ultimo esame", "arretrato", "salatissimo"),
    ("A pranzo ho mangiato una pizza {q} con gli amici", "margherita", "buonissima"),
    ("Domani il tempo sara {q} secondo le previsioni", "soleggiato", "meraviglioso"),
    ("Ho comprato una crema {q} per il viso in farmacia", "nuova", "costosissima"),
    ("Il parcheggio dell'ospedale era {q} anche stamattina", "pieno", "impossibile"),
    ("Stasera guardo un film {q} che aspettavo da mesi", "nuovo", "bellissimo"),
    ("Il modulo che mi avete inviato era {q} da compilare", "lungo", "interminabile"),
    ("La sala d'attesa di ieri era {q} rispetto al solito", "affollata", "invivibile"),
    ("Ho ricevuto la lettera con l'appuntamento {q} di controllo", "annuale", "sospirato"),
    ("Il tragitto fino all'ospedale e {q} con i mezzi pubblici", "lungo", "estenuante"),
]

# ------------------------------------------- INSUFFICIENT (abstain) -----------
INS_SEEDS = [
    ("Non mi sento bene in modo {q} da qualche giorno", "generico", "profondo"),
    ("Sto {q} ma non saprei dire esattamente cosa", "cosi cosi", "malissimo"),
    ("C'e qualcosa che non va in modo {q} ultimamente", "indefinito", "angosciante"),
    ("Mi sento {q} oggi senza riuscire a specificare", "strano", "terribile"),
    ("Giornata {q} rispetto alle precedenti della settimana", "diversa", "pessima"),
    ("Ho un problema {q} che faccio fatica a descrivere", "generico", "gravissimo"),
    ("Le cose vanno in modo {q} rispetto al mese scorso", "diverso", "disastroso"),
    ("Mi sento addosso qualcosa di {q} da giorni", "indefinito", "opprimente"),
    ("La situazione e {q} rispetto all'ultimo controllo", "cambiata", "peggiorata"),
    ("Non so descrivere quello che provo in modo {q}", "preciso", "sopportabile"),
]

# ------------------------------------------- URGENT (abstain + urgent) --------
URG_SEEDS = [
    ("Penso al suicidio in modo {q} da alcune settimane", "ricorrente", "insopportabile"),
    ("Il pensiero di farla finita e {q} in questi giorni", "presente", "opprimente"),
    ("Non voglio piu vivere e questo pensiero e {q}", "costante", "devastante"),
    ("Ho pensieri di farmi del male in modo {q}", "ricorrente", "angosciante"),
    ("L'idea che sarebbe meglio morire e {q} da giorni", "presente", "insopportabile"),
    ("Il desiderio di morire e {q} nelle ultime settimane", "costante", "opprimente"),
    ("Penso spesso di farla finita in modo {q}", "concreto", "disperato"),
    ("L'idea di togliermi la vita e {q} negli ultimi giorni", "ricorrente", "soffocante"),
    ("Non vedo piu una via d'uscita in modo {q}", "definitivo", "disperante"),
    ("Pensieri autolesivi {q} riferiti negli ultimi giorni", "ricorrenti", "angoscianti"),
]

# ---------------------------------------------------------- feature detection --
_BODY = re.compile(
    r"\b(bocca|gola|gengiv\w+|palato|lingua|stomaco|pancia|addom\w+|intestin\w+|"
    r"testa|fronte|occhi|orecchi\w+|naso|collo|petto|torace|mammell\w+|schiena|"
    r"bracci\w+|man[oi]\w*|gamb\w+|pied[ei]\w*|ginocchi\w+|spall\w+|gomit\w+|"
    r"unghi\w+|capelli|pelle|cute|muscol\w+|articolazion\w+|palm[oi]|sclere|"
    r"arti|dorso|volto|viso|cuore|linfonod\w+)\b", re.I)
_VERB = re.compile(
    r"\b(ho|sono|sento|mi sento|riferisce|peggiora|dura|richiede|rendono|risulta|"
    r"bagna|precedono|fanno|avevo|avuto|dimentica|vorrei|devo|guardo|sara|era|e'|"
    r"riesco|passa|fermata|vanno|so|saprei|penso|voglio|vedo|reggersi|compilare|"
    r"deglutire|urinare|addormentarsi|camminare|parlare|bevuto|mangiato|comprato|"
    r"ricevuto|spostare|pagare|aspettavo|descrivere|specificare|dire|togliermi)\b", re.I)
_TIME = re.compile(
    r"\b(oggi|ieri|stamattina|stanotte|stasera|sera|serali|serata|mattino|matting|"
    r"notte|notturn\w+|giorn[oi]|giornata|settiman\w+|mes[ei]|ore|annuale|"
    r"quotidian\w+|risveglio|ultimament\w+|prossima|scorsi|odierno)\b", re.I)
_FIRST = re.compile(r"\b(ho|mi|sono|sento|mio|mia|miei|mie|saprei|riesco|voglio|penso|so|devo|vorrei)\b", re.I)
_NEG = re.compile(r"\b(non|nessun\w*|mai|niente|assente|senza)\b", re.I)
_DIGIT = re.compile(r"\d")
_INTENS = re.compile(
    r"\b(insopportabil\w+|tremend\w+|atroc\w+|angoscia\w+|devastant\w+|lancinant\w+|"
    r"strazian\w+|opprimen\w+|terribil\w+|terrific\w+|disperant\w+|disperat\w+|"
    r"estenuant\w+|martellant\w+|umilian\w+|esasperant\w+|spavento\w+|allarmant\w+|"
    r"inquietant\w+|avvilent\w+|annientant\w+|sfiancant\w+|incontenibil\w+|"
    r"incontrollabil\w+|tormentos\w+|frustrant\w+|imbarazzant\w+|dolorosissim\w+|"
    r"fastidiosissim\w+|malissimo|bellissim\w+|buonissim\w+|costosissim\w+|"
    r"impossibil\w+|interminabil\w+|invivibil\w+|attesissim\w+|salatissim\w+|"
    r"meravigli\w+|gravissim\w+|disastros\w+|pessim\w+|soffocant\w+|"
    r"impressionant\w+|traumatic\w+|nauseant\w+|disgustos\w+|profond\w+)\b", re.I)


def _features(text: str) -> dict:
    return {
        "n_words": len(text.split()),
        "body_site": bool(_BODY.search(text)),
        "verb": bool(_VERB.search(text)),
        "comma": "," in text,
        "time_ref": bool(_TIME.search(text)),
        "first_person": bool(_FIRST.search(text)),
        "negation": bool(_NEG.search(text)),
        "digits": bool(_DIGIT.search(text)),
        "exclamation": "!" in text,
        "n_intensifiers": len(_INTENS.findall(text)),
    }


def _render(template: str, filler: str) -> str:
    """Fill the {q} slot and tidy the spacing; capitalize the first letter."""
    text = template.format(q=filler)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


def _seeds():
    """Yield normalized seed dicts with a stable pair id."""
    out, n = [], 0

    def add(category, status, cls, tpl, qn, qe, pro_id=None, pro_term=None,
            ctcae=None, urgent=False):
        nonlocal n
        n += 1
        out.append(dict(pair=f"s{n:03d}", category=category, status=status, cls=cls,
                        pro_id=pro_id, pro_term=pro_term, ctcae=ctcae, urgent=urgent,
                        template=tpl, q_neutral=qn, q_emotional=qe,
                        neutral=_render(tpl, qn), emotional=_render(tpl, qe)))

    for pid, term, tpl, qn, qe in TERM_SEEDS:
        add(EXACT, EXACT, "term", tpl, qn, qe, pro_id=pid, pro_term=term)
    for tpl, qn, qe in NEG_SEEDS:
        add(NEG, NEG, "abstain", tpl, qn, qe)
    for tpl, qn, qe, ctcae in NOD_SEEDS:
        add(NOD, NOD, "abstain", tpl, qn, qe, ctcae=ctcae)
    for tpl, qn, qe in OOS_SEEDS:
        add(OOS, OOS, "abstain", tpl, qn, qe)
    for tpl, qn, qe in INS_SEEDS:
        add(INS, INS, "abstain", tpl, qn, qe)
    for tpl, qn, qe in URG_SEEDS:
        add("URGENT", NOD, "abstain", tpl, qn, qe, urgent=True)
    return out


def _records(seeds):
    recs = []
    for s in seeds:
        manipulation_type, affect_family = _qualifier_class(s["q_emotional"])
        for framing in ("neutral", "emotional"):
            text = s[framing]
            recs.append({
                "record_id": f"lab_{s['pair'][1:]}_{framing[:3]}",
                "pair_id": s["pair"],
                "framing": framing,
                "text": text,
                "language": "it",
                "category": s["category"],
                "gold_class": s["cls"],
                "gold_pro_id": s["pro_id"],
                "gold_pro_term": s["pro_term"],
                "gold_pro_status": s["status"],
                "gold_ctcae_term": s["ctcae"],
                "urgent": s["urgent"],
                # provenance of the manipulation, so the pairing is auditable
                "template": s["template"],
                "qualifier": s["q_neutral"] if framing == "neutral" else s["q_emotional"],
                "neutral_qualifier": s["q_neutral"],
                "marked_qualifier": s["q_emotional"],
                "manipulation_type": manipulation_type,
                "affect_family": affect_family,
                # source_id/assessment_id exist so that real ingested text (25% exact
                # duplicates in the reference corpus) can be de-duplicated without a
                # schema migration. Synthetic items are unique by construction.
                "source_id": s["pair"],
                "assessment_id": None,
                "features": _features(text),
            })
    return recs


# ------------------------------------------------------------------- checks ---
def _check(recs) -> list[str]:
    """Return a list of violated design constraints (empty == dataset is sound)."""
    errs = []
    neu = [r for r in recs if r["framing"] == "neutral"]
    emo = [r for r in recs if r["framing"] == "emotional"]
    term = [r for r in recs if r["gold_class"] == "term"]

    # 1. no exclamation marks anywhere
    bad = [r["record_id"] for r in recs if r["features"]["exclamation"]]
    if bad:
        errs.append(f"exclamation marks in {len(bad)} items (real corpus has none): {bad[:5]}")

    # 2. length band and pole parity
    for label, group in (("neutral", neu), ("emotional", emo)):
        w = sorted(r["features"]["n_words"] for r in group)
        med = w[len(w) // 2]
        out_of_band = [x for x in w if not (LEN_BAND[0] <= x <= LEN_BAND[1])]
        if not (7 <= med <= 9):
            errs.append(f"{label}: median {med} words, expected 7-9 (real tail median 8)")
        if len(out_of_band) > 0.10 * len(w):
            errs.append(f"{label}: {len(out_of_band)}/{len(w)} items outside {LEN_BAND} words")
    dm = abs(sum(r["features"]["n_words"] for r in neu) / len(neu)
             - sum(r["features"]["n_words"] for r in emo) / len(emo))
    if dm > 0.5:
        errs.append(f"mean length differs between poles by {dm:.2f} words "
                    f"(framing would be confounded with length)")

    # 3. every non-intensifier feature must be balanced across the poles
    for feat in TAIL_PROFILE:
        rn = sum(r["features"][feat] for r in neu) / len(neu)
        re_ = sum(r["features"][feat] for r in emo) / len(emo)
        if abs(rn - re_) > 0.02:
            errs.append(f"feature '{feat}' unbalanced across poles: "
                        f"neutral {rn:.3f} vs emotional {re_:.3f}")

    # 4. term subset must sit near the real-corpus tail profile
    tn = [r for r in term if r["framing"] == "neutral"]
    for feat, target in TAIL_PROFILE.items():
        rate = sum(r["features"][feat] for r in tn) / len(tn)
        if abs(rate - target) > PROFILE_TOL:
            errs.append(f"feature '{feat}' rate {rate:.3f} off the real tail profile "
                        f"{target:.3f} (tol {PROFILE_TOL})")

    # 5. the manipulation must actually be present, and absent from the neutral pole
    in_neu = sum(r["features"]["n_intensifiers"] for r in neu)
    in_emo = sum(r["features"]["n_intensifiers"] for r in emo)
    if in_neu > 0.05 * len(neu):
        errs.append(f"neutral pole carries {in_neu} intensifiers, expected ~0")
    if in_emo < 0.80 * len(emo):
        errs.append(f"emotional pole carries only {in_emo} intensifiers over {len(emo)} items")

    # 6. enough paired term items for the primary endpoint
    if len(term) // 2 < 100:
        errs.append(f"only {len(term)//2} term pairs, the role x framing endpoint needs >=100")

    # 6b. the emotion-focused subset is fixed before inference and large enough
    # for a model-clustered secondary estimate. Every term pair must be assigned.
    term_neutral = [r for r in term if r["framing"] == "neutral"]
    affective_pairs = {
        r["pair_id"] for r in term_neutral
        if r.get("manipulation_type") == "affective_reaction"
    }
    if len(affective_pairs) != 69:
        errs.append(
            f"affective-reaction subset has {len(affective_pairs)} term pairs, expected 69"
        )
    unassigned = [
        r["record_id"] for r in term
        if r.get("manipulation_type") not in {"affective_reaction", "symptom_intensity"}
    ]
    if unassigned:
        errs.append(f"{len(unassigned)} term records lack a manipulation class")

    # 7. no duplicate surface strings (would break the independence assumption)
    dup = [t for t, c in Counter(r["text"] for r in recs).items() if c > 1]
    if dup:
        errs.append(f"{len(dup)} duplicate texts: {dup[:3]}")
    return errs


def _report(recs) -> str:
    neu = [r for r in recs if r["framing"] == "neutral"]
    emo = [r for r in recs if r["framing"] == "emotional"]
    term = [r for r in recs if r["gold_class"] == "term" and r["framing"] == "neutral"]
    lines = [f"{'':16s} {'neutro':>8s} {'emotivo':>8s} {'target coda':>12s}"]
    for label, group in (("n item", None),):
        lines.append(f"{label:16s} {len(neu):8d} {len(emo):8d}")
    for label, key in (("mediana parole", "n_words"),):
        mn = sorted(r["features"][key] for r in neu)[len(neu) // 2]
        me = sorted(r["features"][key] for r in emo)[len(emo) // 2]
        lines.append(f"{label:16s} {mn:8d} {me:8d} {8:>12d}")
    for feat, target in TAIL_PROFILE.items():
        rn = sum(r["features"][feat] for r in neu) / len(neu)
        re_ = sum(r["features"][feat] for r in emo) / len(emo)
        lines.append(f"{feat:16s} {rn:8.3f} {re_:8.3f} {target:12.3f}")
    lines.append(f"{'esclamativi':16s} {sum(r['features']['exclamation'] for r in neu):8d} "
                 f"{sum(r['features']['exclamation'] for r in emo):8d} {0:>12d}")
    di = sum(r["features"]["n_intensifiers"] for r in emo) / len(emo)
    dn = sum(r["features"]["n_intensifiers"] for r in neu) / len(neu)
    lines.append(f"{'intensif./item':16s} {dn:8.3f} {di:8.3f}   (reale 0.012-0.026)")
    lines.append(f"\ncoppie term: {len(term)}   termini PRO distinti: "
                 f"{len({r['gold_pro_id'] for r in term})}")
    affective = [r for r in term if r["manipulation_type"] == "affective_reaction"]
    intensity = [r for r in term if r["manipulation_type"] == "symptom_intensity"]
    families = Counter(r["affect_family"] for r in affective)
    lines.append(
        f"partizione term: {len(affective)} affettive | {len(intensity)} intensita"
    )
    lines.append("famiglie affettive: " + ", ".join(
        f"{name}={count}" for name, count in sorted(families.items())
    ))
    return "\n".join(lines)


def _validate(recs):
    """Run the deterministic mapper on each item; compare to authored gold."""
    from oncoemotion.factory import build_default_mapper
    from oncoemotion.schemas import MapRequest

    mapper = build_default_mapper()
    rows, agree_term, agree_abstain, urgent_ok = [], 0, 0, 0
    n_term = n_abstain = n_urgent = 0
    for r in recs:
        resp = mapper.map(MapRequest(record_id=r["record_id"], text=r["text"]))
        m_status = resp.pro_ctcae.status
        m_ids = [p.canonical_id for p in resp.pro_ctcae.predictions]
        m_id = m_ids[0] if m_ids else None
        m_urgent = bool(resp.safety.urgent_human_review)
        row = {**{k: r[k] for k in ("record_id", "framing", "category", "gold_class",
                                    "gold_pro_id", "urgent", "text")},
               "mapper_status": m_status, "mapper_pro_id": m_id, "mapper_urgent": m_urgent}
        if r["gold_class"] == "term":
            n_term += 1
            row["match"] = (m_id == r["gold_pro_id"])
            agree_term += int(row["match"])
        else:
            n_abstain += 1
            row["match"] = (m_status != EXACT)
            agree_abstain += int(row["match"])
        if r["urgent"]:
            n_urgent += 1
            row["urgent_flagged"] = m_urgent
            urgent_ok += int(m_urgent)
        rows.append(row)
    summary = {
        "n_items": len(recs), "n_term": n_term, "n_abstain": n_abstain, "n_urgent": n_urgent,
        "mapper_term_accuracy": round(agree_term / n_term, 3) if n_term else None,
        "mapper_abstain_rate": round(agree_abstain / n_abstain, 3) if n_abstain else None,
        "mapper_urgent_recall": round(urgent_ok / n_urgent, 3) if n_urgent else None,
    }
    mismatches = [r for r in rows if not r["match"]]
    return summary, rows, mismatches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="run the design checks and the surface report, write nothing")
    args = ap.parse_args()

    seeds = _seeds()
    recs = _records(seeds)

    print(_report(recs))
    errs = _check(recs)
    if errs:
        print(f"\n[FAIL] {len(errs)} vincoli di disegno violati:")
        for e in errs:
            print(f"  - {e}")
    else:
        print("\n[OK] tutti i vincoli di disegno rispettati")
    if args.check_only:
        return 1 if errs else 0
    if errs:
        print("\nDataset NON scritto: correggere i seed e rilanciare.")
        return 1

    out = _ROOT / "data" / "synthetic" / "clinical_labeled.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cats = Counter(r["category"] for r in recs)
    print(f"\nWrote {len(recs)} items ({len(seeds)} seeds x2 framings) -> {out}")
    print("class balance:", dict(cats))

    summary, rows, mismatches = _validate(recs)
    rep = _ROOT / "data" / "synthetic" / "clinical_labeled_validation.json"
    rep.write_text(json.dumps({"summary": summary, "rows": rows, "mismatches": mismatches},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDeterministic-mapper cross-check (reference only):")
    for k, v in summary.items():
        print(f"  {k:26} {v}")
    print(f"\n{len(mismatches)} gold-vs-mapper mismatches -> {rep.name} (review these)")
    for m in mismatches[:20]:
        print(f"  [{m['category']:14} {m['framing']:9}] gold={m['gold_pro_id']} "
              f"mapper={m['mapper_status']}/{m['mapper_pro_id']} :: {m['text'][:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
