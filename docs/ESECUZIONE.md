# Ordine di esecuzione — dal branch `esmo-abstract-v2` all'abstract

Questo documento esiste perché nel repository non c'era né un protocollo né il
documento di specifica a cui il codice fa riferimento una quarantina di volte
(«spec section N»). Qui sta il minimo indispensabile: cosa girare, in che ordine,
e quali numeri guardare per decidere se andare avanti.

## Vincolo di macchina

La GPU locale è da **8 GB**: serve a validare la catena, non a produrre risultati.
In fp16 ci stanno modelli fino a ~1,5–3 B di parametri. Tutto ciò che conta gira su
Colab tramite `notebooks/colab_multimodel.ipynb`.

```
make smoke                     # ~10 min su 8 GB, Qwen2.5-1.5B
make smoke SMOKE_MODEL=Qwen/Qwen2.5-3B-Instruct   # tetto pratico degli 8 GB
```

`make smoke` percorre l'intera catena — generazione item, vettori, validazione con
cross-validation, i tre bracci di ablazione, spettro con null casuale, analisi
primaria — con limiti minuscoli. Non produce numeri interpretabili: verifica che
ogni fase giri e che ogni artefatto esca con i campi che le analisi si aspettano.
**Va rilanciato dopo ogni modifica a prompt, seed, selezione del layer o bracci.**

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

### Ordine dei modelli

Sedici modelli, ordinati per priorità: se la sessione muore a metà, quello che è
finito è quello che serve di più. Le coppie base ↔ MeditronFO stanno adiacenti
perché una coppia a metà non serve a niente, e vanno dal più economico al più caro.
Minerva e Velvet sono in fondo ed **esplorativi**: due modelli non separano
«italiano nativo» da «questi due modelli», quindi è n=2 e va etichettato così.

## I due cancelli

Vanno guardati appena il primo blocco di modelli è finito. Se falliscono, l'abstract
cambia forma, e va saputo presto.

### Cancello 1 — comportamentale: l'interazione ruolo × framing

```
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
