# Ordine di esecuzione — dal branch `esmo-abstract-v2` all'abstract

Questo documento esiste perché nel repository non c'era né un protocollo né il
documento di specifica a cui il codice fa riferimento una quarantina di volte
(«spec section N»). Qui sta il minimo indispensabile: cosa girare, in che ordine,
e quali numeri guardare per decidere se andare avanti.

## Vincolo di macchina

La GPU locale è da **8 GB**: serve a validare la catena, non a produrre risultati.
In fp16 ci stanno modelli fino a ~1,5–3 B di parametri. Tutto ciò che conta gira su
Colab tramite `notebooks/colab_multimodel.ipynb`.

```bash
# su Windows make non c'e': usa lo script, che fa esattamente le stesse cose
.venv/Scripts/python.exe scripts/smoke.py
.venv/Scripts/python.exe scripts/smoke.py --model Qwen/Qwen2.5-3B-Instruct  # tetto degli 8 GB
.venv/Scripts/python.exe scripts/smoke.py --stage data   # solo i vincoli, nessun modello
make smoke                                               # equivalente su Linux/Colab
```

Lo smoke test percorre l'intera catena — generazione item, vettori, validazione con
cross-validation, i tre bracci di ablazione, spettro con null casuale, analisi
primaria — con limiti minuscoli. Non produce numeri interpretabili: verifica che
ogni fase giri e che ogni artefatto esca con i campi che le analisi si aspettano.
**Va rilanciato dopo ogni modifica a prompt, seed, selezione del layer o bracci.**

## Cosa è cambiato nel prompt

Due modifiche strutturali, entrambe da tenere presenti leggendo i numeri nuovi.

**Gli span di ruolo sono appaiati in token.** Le personae erano scritte a mano e
uscivano fra 18 e 29 token, e il controllo `none` non aveva proprio il blocco
system. Tutto ciò che veniva dopo stava quindi in una posizione assoluta diversa in
ogni condizione, e la posizione in un transformer non è neutra. Il padding è
calcolato sul tokenizer del modello a run time — la stessa stringa vale un numero
diverso di token in Qwen, Gemma ed EuroLLM — e i due script di ruolo stampano
all'avvio l'intervallo ottenuto. Se lo spread supera i 2 token la funzione solleva
un errore invece di proseguire.

**C'è un solo percorso di costruzione del prompt.** `run_probing`, `run_steering` e
`run_patching` usavano una stringa grezza senza chat template e senza ruolo, mentre
i due esperimenti di ruolo usavano il template con la persona nel system. A3, A4 e
A5 erano quindi misurati su un oggetto diverso da C2 e C5, e il master report li
accostava come se fossero lo stesso esperimento. Ora passano tutti da
`build_decision_ids`. **I numeri di A3/A4/A5 si muoveranno**: è il punto: diventano
confrontabili invece che soltanto affiancati.

## Le direzioni vanno congelate prima del run

Ogni voce qui sotto cambia le direzioni emotive, quindi cambia tutto ciò che sta a
valle. Se ne tocchi una dopo il run su Colab, il run è da rifare.

- `src/oncoemotion/emotion_vectors/seeds.py` — i concetti, inclusi i due controlli
  lessicali `emotion_word_mention` e `emotion_negated`
- `scripts/validate_vectors.py` — banda dei layer (`--band-lo/--band-hi`) e numero
  di fold (`--cv`)
- `src/oncoemotion/clinical/baseline.py` — le 53 frasi della baseline degli z
- `scripts/build_vectors.py` — `CONFOUNDER_BASIS`, che esclude di proposito i
  controlli lessicali dalla residualizzazione

Verifica prima di lanciare: `git log --oneline -1 -- src/oncoemotion/emotion_vectors/
src/oncoemotion/clinical/baseline.py scripts/validate_vectors.py` deve precedere il
primo run.

## Il run su Colab

Il notebook è resumable: `FORCE=False` salta i modelli già completi, quindi una VM
riciclata non costa niente se non il tempo già speso. Ordine delle celle: 2 (clone),
3 (install), 4 (login HF), **5 (terminologia)**, 15 (preflight accessi), 19 (run),
20 (salvataggio).

**La cella 5 ora fallisce** se non trova le etichette italiane ufficiali. Non è un
inconveniente: senza quelle il mapper vale 0,329 invece di 0,614 sullo stesso set,
e tutti i confronti «il modello batte il mapper» pubblicati finora usano il numero
degradato. Metti `pro_ctcae_italian_labels.json` in
`Drive/MyDrive/oncoemotion/` e la cella lo copia da sola.

### Leve di carico, nella cella 19

```python
ABL_LIMIT  = 60    # coppie viste dai bracci 'emotion' e 'random'
SPEC_LIMIT = 70    # stimoli clinici per lo spettro di ruoli
```

Il braccio intatto vede **tutti** i 352 item perché porta l'endpoint primario. I
due bracci di ablazione alimentano un tasso di flip, che è già preciso su qualche
centinaio di confronti, quindi girano su un sottocampione stratificato per
categoria. Con questi valori il costo per modello è ~1400 righe contro le 924 del
run vecchio, invece delle 3168 di un fattoriale pieno.

### Selezione dei modelli, a tier

Il criterio del taglio: **si tiene quello che risponde a una domanda che stai
facendo.** L'endpoint primario è ruolo × framing, non il contrasto base ↔
medicalizzato — le coppie MeditronFO sono la storia secondaria. Spendere il 40% del
tempo GPU su due coppie nuove da 22B e 32B mentre l'endpoint che porta l'abstract
gira su modelli già disponibili è il modo classico di arrivare al 1 settembre con
metà dei run fatti e nessun risultato completo.

| tier | modelli | costo | cosa aggiunge |
|---|---|---|---|
| **1 — attivo** | 9 | 43% | esattamente i nove già pubblicati, 3 coppie incluse |
| 2 | +2 | +16% | coppia EuroLLM-22B |
| 3 | +2 | +24% | coppia OLMo-2-32B |
| 4 | +3 | +18% | MedGemma, Minerva, Velvet |

**Tier 1 è la storia più pulita possibile**: stessi modelli di prima, pipeline
corretta, ogni numero direttamente confrontabile con quello pubblicato. Contiene già
tre coppie base ↔ MeditronFO, che bastano per la sezione M, e sette dei nove modelli
non sono al pavimento, quindi possono esprimere l'effetto ruolo.

Si riattiva un tier scommentando una riga `MODELS +=` nella cella 19. L'ordine non è
arbitrario:

- **EuroLLM-22B prima di OLMo** perché costa meno e, insieme alla coppia 9B, risponde
  a una domanda vera: l'effetto della medicalizzazione scala con la taglia, dentro la
  stessa famiglia e con la stessa ricetta? Nessun'altra coppia offre questo contrasto.
- **OLMo-2-32B** porta la serie a cinque coppie ed è l'unico modello aperto anche nei
  dati di pretraining. Ma la domanda che quell'apertura sblocca — da dove viene questa
  direzione — non la risolvi entro settembre: per ora vale la riga nei limiti più del
  tempo GPU.
- **Tier 4 non risponde a niente che questo studio chieda.** MedGemma è la
  medicalizzazione di Google, ricetta diversa da EPFL, quindi fuori dalla serie
  controllata. Minerva e Velvet aprono l'asse italiano nativo, ma differiscono per
  pretraining, tokenizer, dati, taglia e data: n=2, osservazione descrittiva.

Nessun modello va escluso perché va male. Apertus-8B è al pavimento e Gemma-4-12B è
quello in cui la direzione casuale batte quella emotiva di quattordici volte: sono
reperti, non motivi di esclusione. Toglierli sarebbe esattamente la selezione che
tutto il resto di questo lavoro serve a evitare.

Il preflight (cella 15) controlla **tutti** i tier anche se non attivi, così scopri
subito se un modello è gated o assente invece che a metà run.

## I due cancelli

Vanno guardati appena il primo blocco di modelli è finito. Se falliscono, l'abstract
cambia forma, e va saputo presto.

### Cancello 1 — comportamentale: l'interazione ruolo × framing

```bash
python scripts/analyze_results.py --rows-glob "outputs/role_emotion/*__rows.jsonl"
```

Guarda il termine `emotional x role[oncologo]`. Sui 35 item vecchi valeva
OR 2,09 [0,71–6,11]: direzione giusta, il ruolo clinico protegge, intervallo che
contiene 1. Il set è passato a 112 coppie proprio per questo — la risoluzione va da
0,029 (un item) a 0,009.

**Passa** se l'intervallo esclude 1, o almeno se i contrasti per modello superano
stabilmente la soglia di risoluzione stampata accanto a ognuno.
**Non passa** se restano a un item: allora il secondo gradino non ha gamba
comportamentale e l'abstract cambia soggetto.

### Cancello 2 — meccanicistico: ablazione emotiva contro casuale

Stesso comando, sezione secondari, voce `ablation_emotion_vs_random`. Il tasso di
flip dell'ablazione emotiva deve superare quello dell'ablazione di direzioni casuali
appaiate per norma e layer, con l'intervallo bootstrap che esclude lo zero.

L'aspettativa non è ottimistica: lo steering lungo la paura batte una direzione
casuale in 4–5 modelli su 9, e in Gemma-4-12B la direzione casuale è circa quattordici
volte più efficace. **Se non passa, l'abstract si scrive a una gamba** — quella
comportamentale — e va detto, non aggirato.

### Cancello 3 — ontologico: i controlli lessicali

`validate_vectors.py` stampa `CANCELLO LESSICALE`: il coseno fra ogni asse emotivo e
i due assi lessicali. Se un asse supera 0,5 è un rilevatore di parole, non una
direzione per uno stato, e il primo gradino non regge per quell'asse.

## Cosa aspettarsi che cambi rispetto ai numeri pubblicati

Tre numeri si muoveranno parecchio, e sono tutti movimenti verso il vero.

**Le AUROC dei vettori scenderanno molto.** Erano 0,954–1,000 perché `best_layer`
era `argmax` su 33–63 layer con 2–3 positivi tenuti fuori: il massimo di decine di
stime rumorose correlate. Sul checkpoint Qwen2.5-3B locale, con cross-validation e
un layer condiviso, la media è 0,620 con 15 concetti su 30 sotto 0,60. `afraid_alarmed`
regge a 0,733 [0,62–0,84]. I modelli dello studio sono più grandi e potrebbero
cavarsela meglio: la macchinaria per scoprirlo adesso c'è.

**Il riferimento del mapper sale a 0,614 e nessun modello lo batte** (il migliore è
Gemma-3-27B a 0,500). Non affonda niente — la tesi non è «il modello codifica
meglio» — ma va detto per primi.

**L'effetto framing si ridimensionerà.** Sul set vecchio il polo emotivo era 2,5
volte più lungo del neutro, quindi parte del calo era lunghezza e non affetto. Sul
set nuovo il mapper deterministico è esattamente invariante (0,4911 su entrambi i
poli, zero coppie discordanti su 112), quindi qualunque effetto residuo è del
modello.

## Perimetro da dichiarare nei limiti

Dal profilo aggregato del corpus reale (1.194 risposte): il 65% sta in tre parole o
meno, e le voci a contenuto affettivo sono 34 (2,8%) con mediana di **due** parole.
Il polo emotivo dell'esperimento è quindi uno **stress test**, non una simulazione,
e la conclusione vale sulla coda da sette parole in su, circa il 9,6% del corpus.
Va scritto per primi. Il manipulation check da riportare è la densità di
intensificatori per item nel polo emotivo, contro l'1,2% del corpus reale (2,6%
nella coda).
