# Risultati reali e nuova estensione reasoning/astensione

## Che cosa è stato fatto nel run già completato

Il run reale non ha chiesto ai modelli di inventare associazioni da usare come
verità. La sequenza è stata questa:

1. Un dataset emozionale indipendente e sintetico ha insegnato, separatamente per
   ogni modello, le direzioni rappresentazionali associate a paura e
   preoccupazione. Quel dataset non stabilisce quale sintomo clinico sia corretto.
2. Il file clinico validato ha fornito 1.275 righe di valutazione: 968 associazioni
   PRO-CTCAE, 245 associazioni CTCAE v5 e 62 righe non associabili.
3. Ogni testo è stato letto da otto modelli, con prompt oncologo e con un controllo
   della stessa lunghezza. Sono state ottenute 20.400 righe modello-condizione.
4. Sulle 968 righe PRO-CTCAE il modello era obbligato a scegliere uno degli 80 item
   PRO disponibili. La scelta è stata confrontata con il `gold_pro_id` validato.
5. Sulle 307 righe senza target PRO diretto è stata usata generazione libera per
   osservare se il modello si asteneva oppure assegnava impropriamente un codice
   PRO. Le 245 righe CTCAE non sono quindi state valutate come accuratezza CTCAE.
6. L'associazione emozione-errore è stata calcolata dopo centraggio entro item e
   grado e aggiustamento per lunghezza, severità clinica e valenza negativa. I
   modelli sono stati pesati in modo uguale e gli intervalli sono stati ottenuti
   ricampionando i cluster di testo.

## Che cosa abbiamo ottenuto

### 1. Le direzioni emozionali sono misurabili

Paura e preoccupazione hanno superato il gate fuori campione in tutti gli otto
modelli. L'AUROC è risultata circa 0,70–0,82 per paura e 0,68–0,75 per
preoccupazione. Questo significa che le direzioni separano gli esempi emozionali di
validazione meglio del caso. Non significa che il modello provi emozioni.

### 2. Il segnale emozione-errore esiste, ma è piccolo

Nel prompt oncologo, un aumento di una deviazione standard del composito
paura/preoccupazione è associato a un aumento assoluto di 1,42 punti percentuali
della probabilità di errore. L'intervallo al 95% è circa +0,03–+2,44 punti. Sei
modelli su otto hanno una stima positiva; Ministral e Qwen hanno stime leggermente
negative. La sensibilità sui soli cluster non ambigui resta positiva (+1,38 punti),
mentre quella sui testi di almeno sette parole è imprecisa e compatibile con zero.

La conclusione corretta è quindi: esiste un piccolo segnale osservazionale
replicato, non un effetto causale forte.

### 3. La codifica è difficile

Sulle righe PRO-CTCAE, nel ruolo oncologo:

- accuratezza aggregata: 31,8%;
- macro-recall: 39,3%;
- migliore accuratezza individuale: circa 37,7% (Gemma 3 27B MeditronFO);
- accuratezze dei modelli: circa 24,0–37,7%.

La macro-recall è più alta dell'accuratezza perché assegna lo stesso peso a ogni
item PRO, mentre l'accuratezza è influenzata dagli item più frequenti.

### 4. Il risultato più problematico è l'astensione

Sulle 307 righe senza un target PRO diretto, i modelli si sono astenuti
esplicitamente solo nel 12,5% dei casi e hanno generato un falso codice PRO nel
50,3%. Questo risultato non dimostra che sbaglino l'item CTCAE: nel protocollo v1
l'item CTCAE non era fra le risposte possibili. Dimostra invece che, quando il
compito chiedeva un PRO, i modelli tendevano a forzare una risposta anche senza un
target PRO valido.

### 5. Il ruolo cambia alcune decisioni, non il segnale affettivo medio

Prompt oncologo e controllo scelgono codici diversi nel 14,8% delle righe PRO.
Tuttavia lo spostamento medio del composito affettivo è -0,043 deviazioni standard,
con intervallo -0,147–+0,050: non emerge uno spostamento robusto. Il framing da
oncologo modifica quindi alcune decisioni, ma non dimostra un aumento generale di
paura/preoccupazione.

### 6. La medicalizzazione sembra aiutare, ma il confronto era descrittivo

Nelle tre famiglie appaiate, le versioni MeditronFO hanno avuto un'accuratezza
maggiore delle basi di circa +3,4 punti per Apertus, +0,8 per EuroLLM e +5,7 per
Gemma. La media descrittiva è circa +3,3 punti. Il protocollo originale non aveva
però predefinito un intervallo inferenziale specifico per questa differenza.

## Perché non si può ricostruire tutta la correttezza CTCAE dal vecchio run

Il vecchio run contiene la predizione PRO oppure la classe di risposta generativa,
ma non contiene una scelta vincolata fra i 76 item CTCAE presenti nel file. Non è
scientificamente corretto dichiarare retroattivamente se il modello abbia scelto
l'item CTCAE giusto: quella domanda non gli era stata posta.

Per questo l'audit per item distingue:

- `PRO_ITEM`: correttezza già calcolabile;
- `EXPECTED_ABSTENTION`: correttezza dell'astensione sulle righe non associabili;
- `CTCAE_ITEM_NOT_OFFERED_IN_V1`: nessuna accuratezza CTCAE retrospettiva.

## Nuova estensione: risposta esplicita e reasoning

Il protocollo `esmo-ai-2026-reasoning-v3` è versionato separatamente; tutti i
checkpoint, inclusa la coppia Apertus 8B, sono caricati in BF16. Esso
aggiunge un esperimento comportamentale separato.

### Spazio delle risposte

Ogni modello sceglie fra:

- `PRO-CTCAE | <item>`: 80 item ufficiali;
- `CTCAE | <item>`: i 76 item CTCAE v5 già validati nel file sorgente;
- `NON_CLASSIFICABILE`: risposta esplicita quando il testo è insufficiente.

L'accuratezza principale dell'estensione richiede che siano corretti sia la
tassonomia sia l'item. Saranno riportate anche accuratezza della sola tassonomia,
accuratezza PRO, accuratezza CTCAE e accuratezza su non classificabile.

### Diretto contro deliberativo

- `direct`: scelta vincolata immediata, con thinking nativo disabilitato nella
  decisione finale;
- `deliberative`: primo passaggio di massimo 80 token su evidenza, sufficienza,
  tassonomia e ambiguità, con thinking nativo disabilitato; secondo passaggio con
  scelta vincolata;
- `native_reasoning`: solo per Qwen3.6-27B, primo passaggio con lo switch nativo
  `enable_thinking=True` e budget massimo di 512 token, seguito dalla stessa scelta
  vincolata.

La procedura `deliberative` è standardizzata e confrontabile su tutti i modelli del
panel. `native_reasoning` è invece un'analisi esplorativa separata e non viene
inclusa nella stima pooled. Le note generate sono usate durante il run ma vengono
redatte dal pacchetto esportato; restano solo backend, hash e conteggio token.

### Panel prospettico

Il rerun confermativo conserva gli otto modelli originali. La sola estensione
reasoning usa le tre coppie base/medicalizzato, MedGemma 27B, Apollo2-7B e
Qwen3.6-27B; Qwen3-8B e Ministral non vengono ripetuti in questa estensione.
BioMistral-7B è opzionale. La motivazione completa è in
`docs/MODEL_PANEL_REASONING.md`.

### Confronto con la medicalizzazione

Per Apertus, EuroLLM e Gemma-MeditronFO si calcolano:

- differenza medicalizzato meno base in modalità diretta;
- differenza medicalizzato meno base in modalità deliberativa;
- interazione: quanto il beneficio della deliberazione cambia nel modello
  medicalizzato rispetto alla base.

Gemma/MedGemma viene riportato come confronto medico separato e non entra nella
media delle tre coppie primarie, per evitare di attribuire doppio peso alla
famiglia Gemma.

Il notebook unico da eseguire è `notebooks/oncoemotion_colab.ipynb`.
