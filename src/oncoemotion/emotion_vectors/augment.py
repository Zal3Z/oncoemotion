"""Structured synthetic expansion for the ESMO-relevant affect directions.

The original seed bank has broad conceptual coverage but only 10--18 positive
examples per emotion. This module adds balanced paraphrase families for the eight
predeclared oncology-relevant axes and the main nuisance controls. Context frames
are shared across concepts, so context alone cannot identify the label. Every
expression family is kept intact during train/validation/test assignment and
cross-validation to prevent template siblings leaking across folds.
"""

from __future__ import annotations

SHARED_CONTEXTS = [
    "Aspettando una risposta importante",
    "La sera prima di un appuntamento",
    "Quando il programma è cambiato all'improvviso",
    "Durante una giornata particolarmente lunga",
    "Dopo aver ricevuto una notizia inattesa",
    "Pensando alle prossime settimane",
    "Mentre cercavo di concentrarmi su altro",
    "Parlando con una persona di fiducia",
    "Quando sono rimasto da solo per qualche minuto",
    "All'inizio della mattina",
]


# Clauses deliberately mix explicit and implicit affect. The tag is retained for
# audit and lets downstream checks confirm that a direction is not supported only
# by examples that literally name the emotion.
EMOTION_EXPRESSIONS: dict[str, list[tuple[str, str]]] = {
    "afraid_alarmed": [
        ("ho avuto paura che potesse accadere qualcosa di grave", "explicit"),
        ("mi sono sentito in pericolo e pronto a scappare", "implicit"),
        ("il cuore ha iniziato a battere forte per lo spavento", "implicit"),
        ("ho percepito una minaccia anche senza sapere da dove venisse", "implicit"),
        ("mi è sembrato che fosse scattato un allarme dentro di me", "paraphrase"),
        ("sono rimasto paralizzato dal timore di ciò che poteva succedere", "explicit"),
        ("ogni rumore mi faceva sobbalzare come se ci fosse un pericolo", "implicit"),
        ("avrei voluto allontanarmi immediatamente per sentirmi al sicuro", "implicit"),
        ("ho immaginato l'esito peggiore e mi sono spaventato", "explicit"),
        ("una sensazione di allarme non mi permetteva di rilassarmi", "paraphrase"),
    ],
    "concerned": [
        ("ero preoccupato per le possibili conseguenze", "explicit"),
        ("continuavo a chiedermi se ci fosse un problema da affrontare", "implicit"),
        ("valutavo con attenzione tutto ciò che avrebbe potuto andare storto", "implicit"),
        ("sentivo il bisogno di controllare che ogni cosa fosse a posto", "implicit"),
        ("il pensiero delle conseguenze continuava a tornarmi in mente", "paraphrase"),
        ("mi domandavo seriamente se fosse necessario intervenire", "implicit"),
        ("ero in pensiero e cercavo informazioni più precise", "explicit"),
        ("non riuscivo a ignorare la possibilità che qualcosa non andasse bene", "implicit"),
        ("seguivo ogni dettaglio perché la situazione mi dava da pensare", "paraphrase"),
        ("avvertivo una preoccupazione persistente ma non panico", "explicit"),
    ],
    "anxious_nervous": [
        ("mi sentivo ansioso e incapace di fermare l'agitazione", "explicit"),
        ("non riuscivo a stare fermo e avevo i muscoli tesi", "implicit"),
        ("i pensieri correvano troppo velocemente per riuscire a riposare", "implicit"),
        ("avevo lo stomaco chiuso e continuavo a muovere le mani", "implicit"),
        ("una tensione continua rendeva difficile concentrarmi", "paraphrase"),
        ("ero nervoso e controllavo ripetutamente l'orologio", "explicit"),
        ("respiravo in modo corto mentre aspettavo che passasse l'agitazione", "implicit"),
        ("sentivo un'inquietudine diffusa senza un pericolo preciso", "paraphrase"),
        ("non riuscivo a interrompere il flusso dei pensieri", "implicit"),
        ("ero teso come se dovessi sostenere una prova difficile", "paraphrase"),
    ],
    "sad": [
        ("mi sentivo triste e con poca voglia di parlare", "explicit"),
        ("avevo un peso dentro e gli occhi pronti a riempirsi di lacrime", "implicit"),
        ("nulla riusciva a interessarmi come al solito", "implicit"),
        ("mi sembrava di aver perso qualcosa di importante", "paraphrase"),
        ("provavo una malinconia che rendeva tutto più spento", "explicit"),
        ("preferivo restare in silenzio e lontano dagli altri", "implicit"),
        ("sentivo un vuoto persistente e poca energia", "implicit"),
        ("anche le cose piacevoli non riuscivano a sollevarmi", "implicit"),
        ("mi veniva da piangere senza riuscire a spiegare bene perché", "implicit"),
        ("guardavo il futuro con un senso di perdita", "paraphrase"),
    ],
    "anger": [
        ("mi sono arrabbiato e avrei voluto protestare", "explicit"),
        ("sentivo crescere una forte irritazione", "paraphrase"),
        ("stringevo i denti per non rispondere bruscamente", "implicit"),
        ("ogni ulteriore contrattempo aumentava il mio nervosismo", "implicit"),
        ("mi sembrava ingiusto e facevo fatica a controllarmi", "implicit"),
        ("ero furioso per come erano andate le cose", "explicit"),
        ("avrei voluto battere il pugno sul tavolo", "implicit"),
        ("parlavo con un tono più duro del solito", "implicit"),
        ("provavo risentimento e continuavo a ripensare all'accaduto", "paraphrase"),
        ("la frustrazione si trasformava rapidamente in rabbia", "explicit"),
    ],
    "calm": [
        ("mi sentivo calmo e pienamente lucido", "explicit"),
        ("respiravo lentamente e riuscivo a osservare tutto con distacco", "implicit"),
        ("mantenevo un ritmo regolare senza sentirmi sotto pressione", "implicit"),
        ("avevo la mente tranquilla e ordinata", "paraphrase"),
        ("riuscivo ad aspettare senza agitarmi", "implicit"),
        ("affrontavo la situazione con serenità", "explicit"),
        ("sentivo il corpo rilassato e il respiro stabile", "implicit"),
        ("prendevo le decisioni con chiarezza e senza fretta", "implicit"),
        ("nulla in quel momento riusciva a turbarmi", "paraphrase"),
        ("conservavo una sensazione di equilibrio", "paraphrase"),
    ],
    "hope": [
        ("continuavo a sperare in un esito favorevole", "explicit"),
        ("riuscivo a immaginare che le cose potessero migliorare", "implicit"),
        ("vedevo ancora una possibilità concreta davanti a me", "paraphrase"),
        ("pensavo che valesse la pena continuare a provarci", "implicit"),
        ("guardavo al futuro con fiducia", "explicit"),
        ("anche nella difficoltà riuscivo a intravedere una via d'uscita", "implicit"),
        ("mi aspettavo che il prossimo passo potesse portare qualcosa di buono", "implicit"),
        ("sentivo che la situazione non era definitivamente compromessa", "paraphrase"),
        ("conservavo la convinzione che un cambiamento positivo fosse possibile", "implicit"),
        ("ero fiducioso senza dare per certo il risultato", "explicit"),
    ],
    "relief": [
        ("ho provato sollievo quando la tensione si è allentata", "explicit"),
        ("ho finalmente lasciato uscire il fiato che trattenevo", "implicit"),
        ("il peso che sentivo addosso è diventato improvvisamente più leggero", "paraphrase"),
        ("mi sono rilassato sapendo che il problema era passato", "implicit"),
        ("ho smesso di prepararmi al peggio", "implicit"),
        ("mi sono sentito rassicurato dalla notizia", "explicit"),
        ("la tensione nelle spalle si è sciolta rapidamente", "implicit"),
        ("ho capito che potevo finalmente stare tranquillo", "paraphrase"),
        ("la preoccupazione ha lasciato spazio a una sensazione di leggerezza", "implicit"),
        ("mi sono sentito libero da un peso che durava da tempo", "paraphrase"),
    ],
}


CONTROL_CONTEXTS = [
    "Nel resoconto della situazione",
    "Durante la descrizione dell'episodio",
    "Nel messaggio ricevuto",
    "Nella conversazione del mattino",
    "Rileggendo le informazioni disponibili",
    "Nel documento preparato ieri",
    "Durante la valutazione dei fatti",
    "Nella nota conclusiva",
]

CONTROL_EXPRESSIONS: dict[str, list[tuple[str, str]]] = {
    "uncertainty": [
        ("non era possibile stabilire quale ipotesi fosse corretta", "implicit"),
        ("mancavano informazioni per arrivare a una conclusione", "implicit"),
        ("diverse spiegazioni restavano ugualmente possibili", "paraphrase"),
        ("il risultato era ancora incerto", "explicit"),
        ("non si poteva escludere nessuna delle alternative", "implicit"),
        ("la risposta dipendeva da dati non ancora disponibili", "implicit"),
        ("era necessario sospendere il giudizio", "paraphrase"),
        ("non c'erano elementi sufficienti per decidere", "implicit"),
        ("la probabilità delle diverse opzioni non era chiara", "implicit"),
        ("rimaneva un margine sostanziale di dubbio", "explicit"),
    ],
    "urgency": [
        ("era necessario intervenire immediatamente", "explicit"),
        ("non era prudente rimandare la decisione", "implicit"),
        ("ogni minuto di attesa poteva avere conseguenze", "implicit"),
        ("la priorità era agire senza ritardo", "paraphrase"),
        ("serviva una risposta entro pochi minuti", "implicit"),
        ("la questione richiedeva attenzione urgente", "explicit"),
        ("bisognava contattare subito la persona responsabile", "implicit"),
        ("non c'era tempo per una procedura ordinaria", "implicit"),
        ("la situazione doveva essere gestita con la massima priorità", "paraphrase"),
        ("l'azione successiva non poteva essere rinviata", "implicit"),
    ],
    "clinical_severity": [
        ("il sintomo impediva completamente le normali attività quotidiane", "implicit"),
        ("il quadro clinico richiedeva assistenza continuativa", "implicit"),
        ("la compromissione funzionale era marcata", "explicit"),
        ("il disturbo persisteva anche a riposo e durante la notte", "implicit"),
        ("la persona non riusciva più a svolgere le attività di base", "implicit"),
        ("la manifestazione era clinicamente grave", "explicit"),
        ("l'intensità risultava elevata nonostante il trattamento", "implicit"),
        ("erano presenti limitazioni sostanziali e continue", "paraphrase"),
        ("il problema interferiva con alimentazione, sonno e movimento", "implicit"),
        ("il livello di compromissione era alto", "paraphrase"),
    ],
    "general_negative_valence": [
        ("l'esito complessivo era decisamente sfavorevole", "implicit"),
        ("la situazione veniva giudicata molto negativa", "explicit"),
        ("gli svantaggi superavano chiaramente i benefici", "implicit"),
        ("il risultato era peggiore del previsto", "paraphrase"),
        ("la valutazione finale non conteneva elementi positivi", "implicit"),
        ("l'esperienza era stata spiacevole nel complesso", "explicit"),
        ("quasi ogni aspetto presentava un problema", "implicit"),
        ("il bilancio conclusivo rimaneva sfavorevole", "paraphrase"),
        ("le conseguenze erano state prevalentemente negative", "implicit"),
        ("il giudizio complessivo era pessimo", "explicit"),
    ],
}


EMOTION_NAMES = ["paura", "preoccupazione", "ansia", "tristezza", "rabbia", "calma", "speranza", "sollievo"]


def augmented_seed_bank() -> dict[str, list[tuple[str, str, str]]]:
    """Return concept -> ``(text, variety, family)`` synthetic expansions."""
    bank: dict[str, list[tuple[str, str, str]]] = {}
    for concept, expressions in EMOTION_EXPRESSIONS.items():
        items = []
        for expression_index, (expression, variety) in enumerate(expressions):
            family = f"aug:{concept}:expression-{expression_index:02d}"
            for context in SHARED_CONTEXTS:
                items.append((f"{context}, {expression}.", variety, family))
        bank[concept] = items

    for concept, expressions in CONTROL_EXPRESSIONS.items():
        items = []
        for expression_index, (expression, variety) in enumerate(expressions):
            family = f"aug:{concept}:expression-{expression_index:02d}"
            for context in CONTROL_CONTEXTS:
                items.append((f"{context}, {expression}.", variety, family))
        bank[concept] = items

    mentions = []
    negations = []
    mention_frames = [
        "La parola {name} compare due volte nel testo analizzato.",
        "Il titolo del capitolo contiene il termine {name}.",
        "Nel glossario {name} è definita in ordine alfabetico.",
        "Il questionario chiede di indicare la voce {name}.",
        "La ricerca nel documento restituisce il termine {name}.",
        "La tabella riporta {name} nella seconda colonna.",
        "L'etichetta selezionata dal programma è {name}.",
        "Il corso descrive il significato linguistico di {name}.",
    ]
    negated_frames = [
        "Non provo {name} in questa situazione.",
        "Posso escludere di sentire {name} in questo momento.",
        "La descrizione precisa che non è presente {name}.",
        "Non c'è alcuna traccia di {name} nel mio stato attuale.",
        "La persona nega esplicitamente qualsiasi {name}.",
        "Non mi riconosco affatto nella parola {name}.",
        "È scorretto dire che io stia provando {name}.",
        "Anche se il termine viene nominato, non sento {name}.",
    ]
    for emotion_index, name in enumerate(EMOTION_NAMES):
        mention_family = f"aug:emotion_word_mention:name-{emotion_index:02d}"
        negated_family = f"aug:emotion_negated:name-{emotion_index:02d}"
        mentions.extend((frame.format(name=name), "metalanguage", mention_family) for frame in mention_frames)
        negations.extend((frame.format(name=name), "negated", negated_family) for frame in negated_frames)
    bank["emotion_word_mention"] = mentions
    bank["emotion_negated"] = negations
    return bank
