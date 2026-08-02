"""Shared neutral baseline for point-E z-scores.

One list, used by every experiment that z-scores a projection. Two things were
wrong before: the list held 8 sentences, and the two role experiments used
different references.

Eight is not enough to estimate a standard deviation. Dividing by a sigma fitted on
8 points produced z-scores up to 27, and since each concept gets its own sigma, an
ordering of emotions by z was partly an ordering by 1/sigma-hat -- the C5 heatmap
was ranking noise in the reference set as much as signal in the items.

The second problem was comparability: ``run_role_emotion`` refitted the mean and sd
per (role, ablation) cell while ``run_role_spectrum`` used a single fixed reference,
so the z-scores of experiments B and C were never on the same scale even though the
report places them side by side.

The sentences are administrative and clinical-adjacent but carry no symptom and no
affect, so they sit where a "nothing is happening" reading should sit.
"""

from __future__ import annotations

# Minimum sentences required to fit a per-concept mean and sd. Below this the
# scale of the z-scores is not identified.
MIN_BASELINE_N = 40

NEUTRAL_BASELINE: list[str] = [
    # --- amministrativo / modulistica ---
    "Il modulo è stato compilato correttamente.",
    "La procedura di registrazione è terminata.",
    "Il documento è stato archiviato negli atti.",
    "L'appuntamento è confermato per la data prevista.",
    "I dati anagrafici risultano aggiornati.",
    "La pratica è stata protocollata questa mattina.",
    "Il questionario contiene dieci domande in totale.",
    "La sala d'attesa è al primo piano dell'edificio.",
    "La tessera sanitaria è stata registrata allo sportello.",
    "Il consenso informato è stato firmato e acquisito.",
    "La richiesta è stata inoltrata all'ufficio competente.",
    "Il codice identificativo è composto da otto cifre.",
    "La copia del referto è disponibile in formato digitale.",
    "L'orario di apertura è dalle otto alle sedici.",
    "Il numero di protocollo compare in alto a destra.",
    "La documentazione è stata trasmessa via posta certificata.",
    "Il modulo va restituito compilato in ogni sua parte.",
    "L'elenco degli esami è riportato nella seconda pagina.",
    "La prenotazione risulta registrata a sistema.",
    "Il fascicolo è stato aggiornato con i dati recenti.",
    # --- percorso di cura, senza sintomo né affetto ---
    "La visita di controllo è programmata come da calendario.",
    "Il prelievo si effettua al piano terra.",
    "La terapia è somministrata secondo lo schema previsto.",
    "Il ciclo successivo è fissato per la settimana indicata.",
    "L'esame richiede il digiuno nelle ore precedenti.",
    "La medicazione va sostituita secondo le indicazioni.",
    "Il dosaggio è riportato sulla confezione del farmaco.",
    "La cartella clinica contiene i referti degli ultimi mesi.",
    "Il reparto si trova nel padiglione adiacente.",
    "Il trasporto è organizzato dal servizio dedicato.",
    "La consegna dei referti avviene allo sportello dedicato.",
    "Le istruzioni per la preparazione sono allegate al modulo.",
    "Il controllo dei parametri è stato eseguito come da protocollo.",
    "La scheda di terapia è aggiornata alla data odierna.",
    "L'infermiere di riferimento è indicato nel foglio informativo.",
    # --- quotidiano non clinico, tono piatto ---
    "L'autobus passa ogni venti minuti circa.",
    "Il negozio all'angolo apre alle nove.",
    "La riunione si terrà nella sala al secondo piano.",
    "Il pacco è stato consegnato ieri pomeriggio.",
    "La bolletta scade alla fine del mese.",
    "Il treno parte dal binario indicato sul tabellone.",
    "La biblioteca resta aperta anche il sabato.",
    "Il parcheggio si trova sul lato est dell'edificio.",
    "La ricevuta è stata rilasciata allo sportello.",
    "Il modulo si scarica dal sito istituzionale.",
    "L'ascensore serve tutti i piani dell'edificio.",
    "La segreteria risponde nelle ore mattutine.",
    "Il calendario delle attività è affisso in bacheca.",
    "La chiave si ritira alla reception.",
    "Il corso dura complessivamente venti ore.",
    "L'indirizzo è riportato in fondo alla pagina.",
    "La riunione è durata poco meno di un'ora.",
    "Il testo è disponibile anche in versione stampata.",
]

assert len(NEUTRAL_BASELINE) >= MIN_BASELINE_N, (
    f"neutral baseline has {len(NEUTRAL_BASELINE)} sentences, at least "
    f"{MIN_BASELINE_N} are needed to fit a per-concept sd")
