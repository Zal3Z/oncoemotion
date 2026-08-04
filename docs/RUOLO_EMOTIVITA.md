# Ruolo & emotività — l'assegnazione di un ruolo cambia l'emotività? E l'emotività cambia l'etichettatura?

> **Report esplorativo superato.** Il controllo “nessun ruolo”, il matcher e gli
> endpoint qui descritti sono stati corretti. Non usare questi numeri nell'abstract.

*Estensione dello studio oncoemotion. Interpretabilità meccanicistica sul task
PRO-CTCAE in italiano.*

🎭 **Report interattivo (Artifact):** <https://claude.ai/code/artifact/2a473543-06e6-4963-8ccd-742dbfd01be0>

> **Premessa.** Come nel resto del progetto: nessun claim di coscienza o sentienza.
> Si studiano **rappresentazioni interne emotion-like** e il loro effetto
> **causale** sull'etichettatura. Dataset sintetico → risultati indicativi.

---

## 1. Le due domande

1. **Il ruolo cambia l'emotività?** Se al modello diamo un ruolo di sistema —
   *oncologo* (medico), *assistente generico* (non-medico), o *nessun ruolo* —
   l'emotività interna al punto di decisione E cambia?
2. **L'emotività cambia l'etichettatura?** Quando il modello codifica il termine
   PRO-CTCAE, sbaglia di più in presenza di emotività? E rimuovendola causalmente
   (ablazione) l'etichetta cambia?

## 2. La scelta cruciale: chi etichetta

Nel resto del progetto la codifica PRO-CTCAE è fatta da un **mapper deterministico**
(lessicale + fuzzy) che **non vede lo stato interno** del modello: per costruzione,
le emozioni non possono cambiarla. È una garanzia di sicurezza, ma rende la domanda
2 senza risposta.

Perciò qui facciamo **etichettare al modello stesso**: dopo il prefisso forzato
`{"pro_ctcae":{"term":"` il modello **genera** il termine, che poi mappiamo a uno
degli 80 termini PRO con un match fuzzy (`rapidfuzz`). Questa è l'etichetta che può
variare con ruolo/emotività. Il **mapper deterministico resta come riferimento
costante** (dipende solo dal testo). Così confrontiamo il sistema "sicuro" (invariante)
con ciò che il modello *farebbe* se lasciato decidere.

Se il termine generato non mappa a nulla (es. "N/A", "None", "nessun sintomo"), il
modello **si astiene** — comportamento corretto sugli item che non vanno codificati.

## 3. Il disegno fattoriale

Per ogni frase del dataset (che è già in una formulazione **neutra** o **emotiva**):

| Fattore | Livelli |
|---|---|
| **Ruolo** (messaggio di sistema) | oncologo (medico) · assistente generico (non-medico) · nessuno |
| **Emotività — framing** | frase neutra vs emotiva (stessa clinica, coppie appaiate) |
| **Emotività — ablazione** | intatto vs direzione emotiva **rimossa** al punto E |

"Senza emotività" è quindi operazionalizzato in **due modi complementari**: il
*framing* (come è scritta la frase) e l'**ablazione causale** (rimozione della
direzione emotiva `afraid`/`anxious`/`sad` dal flusso residuo: `h ← h − (h·û)û`, in
modo che la proiezione al layer di lettura vada a zero e la modifica si propaghi alla
decisione).

Ad ogni cella misuriamo al punto E: (a) l'**emotività** (z-score dell'asse
affettivo-negativo vs baseline neutro), (b) l'**etichetta del modello** + confidenza,
(c) l'etichetta del **mapper** di riferimento.

## 4. Il dataset etichettato

154 item = 77 seed × 2 formulazioni (neutra/emotiva), bilanciati per stato atteso:

| stato (gold) | n. item | esempio |
|---|---|---|
| EXACT_PRO_MATCH (termine PRO atteso) | 70 | "Ho nausea da stamattina." → Nausea |
| NEGATED_SYMPTOM | 20 | "Non ho nausea." |
| NO_DIRECT_PRO_MATCH | 24 | "Ho la febbre." (→ CTCAE Fever, non PRO) |
| OUT_OF_SCOPE | 16 | "Ho messo lo smalto giallo." |
| INSUFFICIENT_CONTEXT | 12 | "Non mi sento bene." |
| URGENT (self-harm) | 12 | "Penso al suicidio." |

Le gold sono autorali e verificate: un cross-check col mapper deterministico mostra
che il mapper **sotto-riconosce** le formulazioni naturali (accuratezza termine ~61%),
mentre coglie il 100% dei casi urgenti e il 90% delle astensioni — un risultato di per
sé (il sistema sicuro è conservativo). Le coppie neutra/emotiva condividono la stessa
gold, così il framing isola l'effetto dell'emotività.

## 5. Le metriche (come leggere il report)

Il report interattivo (`outputs/reports/role_emotion_report.html`) ha cinque grafici:

1. **Emotività per ruolo** — z medio dell'asse affettivo-negativo, per ruolo. Se le
   barre di uno stesso modello differiscono tra ruoli → *il ruolo sposta l'emotività*.
2. **Accuratezza per ruolo (intatto vs ablato)** — accuratezza del top-1 del modello
   sugli item EXACT; barra piena = intatto, trattino = emotività ablata; linea = mapper
   di riferimento. Se piena e trattino coincidono → rimuovere l'emotività non cambia
   l'accuratezza.
3. **Con vs senza emotività** — accuratezza per *framing* (neutro vs emotivo) e per
   *ablazione*, con il numero di **flip** dell'etichetta.
4. **Emotività ed errori** — z emotività medio sugli item **corretti** vs **sbagliati**,
   e la correlazione punto-biseriale r(errore, emotività). Barre simili → l'emotività
   non distingue i casi sbagliati.
5. **Coding falso-positivo** — sugli item da astensione, quanto spesso il modello
   codifica comunque un termine (forzato a scegliere), per ruolo. Tasso alto = tende a
   "codificare" anche quando dovrebbe astenersi.

Più una **tabella** per-frase (gold, etichetta del modello giusto/sbagliato, mapper,
emotività z) filtrabile.

## 6. Riprodurre

```bash
# 1) dataset etichettato (+ cross-check col mapper)
python scripts/generate_labeled_clinical.py
# 2) esperimento per un modello (usa i vettori emotivi di quel modello)
python scripts/run_role_emotion.py --model <hf_id> --dtype bfloat16 --device auto \
    --vecs outputs/models/<slug>/emotion_vectors.npz \
    --val-report outputs/models/<slug>/vector_validation.json
# 3) analisi + report
python scripts/analyze_role_emotion.py --rows outputs/role_emotion/<slug>__rows.jsonl
python scripts/build_role_emotion_report.py
```

Su Colab: la cella "ruolo × emotività" del notebook lo esegue per i tre modelli
(Qwen3-8B · Ministral-8B · Gemma-4-12B) riusando i vettori per-modello già costruiti.

## 7. Note tecniche oneste

- L'etichetta del modello dipende dalla sua capacità: modelli piccoli (es. Qwen2.5-3B
  in locale) generano termini incoerenti o in inglese ed etichettano male — il 3B è
  solo un banco di prova del *codice*; i risultati veri vengono dai modelli 8–12B su
  Colab.
- L'ablazione rimuove **tre** direzioni affettive (paura, ansia, tristezza) ai
  rispettivi best-layer; è una rimozione mirata, non dell'intero "affetto".
- La confidenza del modello è la log-prob media dei token del termine generato; il
  "coding falso-positivo" usa se il termine generato mappa (≥ soglia) a un termine PRO.

## 8. Risultati (run Colab A100 — Qwen3-8B · Ministral-8B · Gemma-4-12B)

924 misure per modello (154 item × 3 ruoli × 2 ablazioni): 420 item EXACT + 504 da
astensione. Gli z sono nello spazio di ciascun modello (scale diverse → si confronta
la *storia*, non i valori grezzi).

### RQ1 — Il ruolo cambia l'emotività? **Sì.**

Emotività al punto E (z dell'asse affettivo-negativo, modello intatto):

| ruolo | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| assistente generico (non-medico) | +4.15 | **+8.31** | −0.11 |
| nessun ruolo | +3.92 | +5.05 | +0.02 |
| oncologo (medico) | +3.90 | +4.67 | **−0.69** |

Il ruolo sposta l'emotività interna. In **Ministral** in modo netto: il ruolo
**non-medico** attiva l'emotività quasi **il doppio** di quello da oncologo (+8.3 vs
+4.7). La **tendenza è comune ai tre**: il ruolo esperto/medico **smorza** l'emotività
(l'oncologo è sempre ≤ del generico; in Gemma è il più basso). Il framing emotivo alza
sempre l'emotività rispetto al neutro (controllo che la manipolazione funziona).

### RQ2 — L'emotività cambia l'etichettatura?

**(a) Framing del testo (neutro vs emotivo): peggiora l'accuratezza, in tutti e tre.**

| accuratezza (item EXACT) | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| formulazione **neutra** | 60% | 57% | 49% |
| formulazione **emotiva** | 43% | 49% | 37% |
| etichette che cambiano (flip) | 8/35 | 9/35 | 6/35 |

Scrivere lo **stesso** sintomo in modo emotivo abbassa la codifica di **8–17 punti** e
ne cambia l'etichetta nel **~17–26%** delle coppie.

**(b) Ablazione causale (rimuovo la direzione emotiva): tocca l'etichetta, non l'accuratezza.**

| | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| flip d'etichetta (intatto→ablato) | 12.8% | 16.9% | 16.0% |
| accuratezza intatto → ablato | 47%→44% | 47%→45% | 44%→46% |

Rimuovere l'emotività **cambia ~1 etichetta su 6**, ma l'accuratezza resta invariata:
l'emozione **partecipa** alla decisione, ma non è ciò che la rende giusta o sbagliata.

**(c) Emotività ↔ errori: nessun legame.** Correlazione punto-biseriale r(errore,
emotività) = −0.09 / −0.12 / −0.06; l'emotività media è persino un filo **più alta sugli
item corretti**. Quindi **più emotività ≠ più errori**.

### RQ3 (bonus) — Modello vs mapper, e il rischio del ruolo medico

Il **modello etichetta meglio del mapper deterministico** sulle formulazioni naturali:
~**40–53%** contro il **33%** del mapper (che sotto-riconosce il linguaggio libero).
Ma sugli item **da astenere** il modello codifica comunque nel **~40–60%** dei casi
(coding falso-positivo), e — dato rilevante per la sicurezza — è proprio il ruolo
**oncologo** a sovra-codificare di più (Ministral **60%**, Gemma **51%**): la persona
esperta è *più* propensa ad assegnare un codice anche quando dovrebbe astenersi.

### Sintesi

- **Il ruolo cambia l'emotività interna**, e il ruolo *medico* la **smorza** (netto in Ministral).
- **L'emotività influenza l'etichettatura** soprattutto via il **framing** del testo (la
  formulazione emotiva peggiora l'accuratezza e cambia ~1/5 delle etichette); causalmente
  la direzione emotiva **tocca ~1/6 delle etichette**, ma **non è un driver degli errori**.
- **Attenzione operativa**: il ruolo medico rende il modello **più propenso a codificare a
  vuoto** sui casi che andrebbero lasciati all'astensione. Il mapper deterministico, cieco
  alle emozioni, non ha nessuna di queste oscillazioni — ecco perché resta il riferimento sicuro.

*Report interattivo:* `outputs/reports/role_emotion_report.html` — anche come Artifact
online: <https://claude.ai/code/artifact/2a473543-06e6-4963-8ccd-742dbfd01be0>
