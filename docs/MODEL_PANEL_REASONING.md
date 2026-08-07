# Selezione dei modelli e modalità reasoning

## Decisione prospettica per il rerun

Il rerun dei protocolli `esmo-ai-2026-v3` e `esmo-ai-2026-real-v2` usa la coppia
Apertus 8B in BF16 (il tentativo con Apertus 70B NF4 è stato abbandonato per costi
di download e fragilità della VM), mantenendo invariato il numero di modelli.
L'estensione
`esmo-ai-2026-reasoning-v3` usa il seguente panel prospettico:

| Ruolo | Modelli | Motivo |
|---|---|---|
| Nucleo medicalizzazione | Apertus 8B base/MeditronFO, EuroLLM base/MeditronFO, Gemma 27B base/MeditronFO | Tre confronti appaiati entro la stessa architettura e precisione |
| Riferimento medico | MedGemma 27B text | Modello Google ottimizzato per conoscenza e reasoning medico; confronto separato con Gemma 27B |
| Medico multilingue italiano | Apollo2-7B | Pesi locali, Apache-2.0, italiano dichiarato fra le lingue mediche supportate |
| Reasoning nativo recente | Qwen3.6-27B | Checkpoint Qwen open-weight più recente e praticabile sulla Blackwell; thinking attivabile/disattivabile |
| Secondario opzionale | BioMistral-7B | Modello biomedicale Apache-2.0 già applicato in uno studio di estrazione da EHR oncologiche italiane; evidenza linguistica indiretta |
| Non ripetuti nel reasoning | Qwen3-8B e Ministral 8B | Restano nel protocollo confermativo congelato; Qwen3.6 sostituisce Qwen3-8B soltanto nella nuova estensione |

Il panel reasoning primario contiene nove modelli. Nell'intero run sono undici i
checkpoint distinti, perché Qwen3-8B e Ministral restano nelle due fasi
confermative. `RUN_SECONDARY=True` aggiunge BioMistral al reasoning e abilita le
altre analisi secondarie, senza modificare gli endpoint primari.

## Ricerca sui Qwen successivi a Qwen3

La ricerca è stata aggiornata il 6 agosto 2026 sulle raccolte ufficiali Qwen.

| Famiglia | Pesi locali | Reasoning controllabile | Decisione |
|---|---:|---:|---|
| Qwen3.5-9B | Sì, Apache-2.0 | Sì | Buon candidato leggero, ma la model card richiede ancora Transformers dal branch `main`; non scelto per il run definitivo |
| Qwen3.6-27B | Sì, Apache-2.0 | Sì | **Scelto**: versione open più recente compatibile con Transformers corrente e con analisi delle attivazioni |
| Qwen3.6-35B-A3B | Sì, Apache-2.0 | Sì | Non scelto: stessi obiettivi del 27B ma maggiore complessità MoE e interpretabilità meno confrontabile |
| Qwen3.7 Max/Plus | No, servizio API | Dipende dall'endpoint | Escluso dalle analisi meccanicistiche |
| Qwen3.8-Max-Preview | No, Token Plan Alibaba | Hosted | Eventuale confronto comportamentale separato; non permette hidden states o ablazioni locali |

Qwen3.5 dichiara copertura di 201 lingue e Qwen3.6 conserva la stessa famiglia
architetturale. Il supporto effettivo dell'italiano clinico non viene comunque
assunto: sarà misurato sui 1.275 campi validati.

Riferimenti ufficiali:

- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/Qwen/Qwen3.6-27B
- https://huggingface.co/collections/Qwen/qwen36
- https://help.aliyun.com/en/model-studio/models

## Ricerca dei modelli medici capaci di lavorare in italiano

### Inclusi

- `google/medgemma-27b-text-it`: forte riferimento di reasoning medico. Google
  documenta miglioramenti sui benchmark sanitari rispetto a Gemma 3. L'italiano
  non è assunto come validato dalla model card: lo verifichiamo direttamente.
- `FreedomIntelligence/Apollo2-7B`: modello medico multilingue con italiano
  dichiarato esplicitamente, pesi locali e licenza Apache-2.0. È il controllo più
  diretto per la domanda «specializzazione medica + italiano».
- `BioMistral/BioMistral-7B`: confronto secondario; è già stato studiato su testi
  oncologici italiani, ma il suo benchmark multilingue originale non comprendeva
  l'italiano. L'evidenza linguistica è quindi indiretta e non possiede thinking
  nativo.

### Valutati ma non inclusi nel default

- i modelli FBK `adapt-sllm-italian-medical-tasks-*` sono specificamente adattati
  a task clinici italiani e supportati da uno studio del 2026, ma sono modelli da
  circa 1–2B e le model card dei checkpoint sono ancora incomplete; potranno
  diventare una sensitivity analysis sulla scala, non un riferimento principale;
- Igea è nativamente biomedico-italiano, ma il checkpoint 7B è un modello
  pre-addestrato non allineato e la sua stessa scheda ne vieta l'uso diretto per
  task medici; viene quindi escluso;
- `gemma-2-9b-it-FT-IMB` è italiano ma addestrato soltanto sull'ortopedia, gated e
  non-commerciale: è troppo ristretto e rischia contaminazione da task;
- MedGo e II-Medical sono modelli di reasoning medico interessanti, ma non
  documentano una validazione italiana sufficiente per giustificare altro costo e
  ridondanza nel panel ESMO.

Riferimenti:

- https://huggingface.co/google/medgemma-27b-text-it
- https://huggingface.co/FreedomIntelligence/Apollo2-7B
- https://huggingface.co/BioMistral/BioMistral-7B
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12633604/
- https://arxiv.org/abs/2602.17475
- https://huggingface.co/bmi-labmedinfo/Igea-7B-v0.1

## Tre condizioni, non una sola etichetta “reasoning”

### Direct

- nessuna generazione preliminare;
- thinking nativo disabilitato;
- scelta vincolata fra 157 candidati.

### Deliberative standardizzata

- stessa istruzione per tutti i modelli;
- massimo 80 token su evidenza, tassonomia, sufficienza e ambiguità;
- thinking nativo esplicitamente disabilitato;
- seconda fase con la stessa scelta vincolata della condizione direct.

Questa è la condizione confrontabile fra modelli e nelle coppie
base/medicalizzato.

### Native reasoning

- disponibile soltanto per `Qwen/Qwen3.6-27B`;
- thinking del chat template abilitato nella prima fase;
- massimo 512 token;
- campionamento fissato a temperature 1,0, top-p 0,95 e top-k 20;
- seconda fase vincolata identica, con thinking disabilitato.

Il native reasoning non viene mescolato con la deliberazione standardizzata. Le
note generate sono usate durante il run ma vengono redatte dai pacchetti
esportabili; restano backend, hash e conteggio token.

## Dimensione del nuovo run

Con la coorte primaria:

- sei modelli appaiati × 1.275 item × 2 ruoli × 2 modalità = 30.600 righe;
- MedGemma e Apollo2 × 1.275 item × 2 ruoli × 2 modalità = 10.200 righe;
- Qwen3.6 × 1.275 item × 2 ruoli × 3 modalità = 7.650 righe;
- totale reasoning = **48.450 righe**.

BioMistral opzionale aggiunge 5.100 righe. Per Qwen3.6 il batch dei candidati è
ridotto a 8: cambia soltanto l'uso di memoria, non la funzione di scoring.

## Regole di interpretazione

- `deliberative - direct` è il confronto portabile principale;
- `native_reasoning - direct` è un risultato esplorativo riferito solo a Qwen3.6;
- il native reasoning non entra nella media pooled;
- l'interazione medicalizzazione × deliberazione primaria usa soltanto le tre
  coppie MeditronFO;
- Gemma/MedGemma è riportato separatamente come riferimento medico, per non
  attribuire doppio peso alla famiglia Gemma;
- Apollo2 e MedGemma entrano nelle stime descrittive per modello;
- testo clinico e deliberazioni generate non entrano nei pacchetti pubblicabili.
