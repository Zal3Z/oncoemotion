# oncoemotion — Relazione: rappresentazioni emotion-like in tre modelli (Cina / Europa / USA)

> **Relazione esplorativa superata.** I numeri precedono il protocollo
> `esmo-ai-2026-v2` e non sono risultati definitivi per ESMO 2026.

*Studio di interpretabilità meccanicistica sul task PRO-CTCAE in italiano.*

> **Premessa (importante).** Non si afferma che i modelli abbiano coscienza, sentienza
> o esperienza soggettiva. Studiamo **rappresentazioni interne emotion-like**,
> **emotion concept**, **stati funzionali associati a un'emozione** e **segnali
> causalmente rilevanti**. È uno strumento di ricerca e supporto, non fa diagnosi
> autonome. Il dataset è sintetico e piccolo → risultati **indicativi**, non verdetti.

---

## 1. La domanda

Quando un modello linguistico legge un sintomo oncologico in campo libero (italiano)
e lo associa a un termine **PRO-CTCAE**, si "accendono" al suo interno direzioni che
somigliano a emozioni (paura, ansia, tristezza, calma…)? E questi segnali **guidano**
la decisione di codifica, oppure sono solo correlati? Abbiamo confrontato **un modello
cinese, uno europeo e uno americano** sugli stessi identici dati per vedere se
"reagiscono" allo stesso modo.

## 2. Come funziona (in breve)

Un modello è una **pila di strati** (layer). Ogni token diventa un **vettore di numeri**
(la "larghezza" del modello: 4096 per Qwen3/Ministral, 3840 per Gemma), che attraversa
gli strati arricchendosi di contesto (il *residual stream*). Certi **concetti vivono
lungo direzioni** in quello spazio: proiettare lo stato interno su una direzione =
misurare quanto quell'emozione è "accesa".

> 🔎 **Spiegazione visiva passo-passo** (token, vettori, i layer, la griglia
> token×layer, la proiezione): guida illustrata interattiva →
> <https://claude.ai/code/artifact/637c907e-3ab5-4d4b-b860-e27b9112ab12>

**La pipeline, per ogni modello:**
1. **Vettori emotivi** — costruiti da frasi italiane *non cliniche* che evocano ogni
   emozione, contrapposte a neutre **e alle altre emozioni** ("one-vs-rest", così la
   direzione cattura *quella* emozione e non "affetto negativo" generico). Poi
   **residualizzati** (ortogonalizzati) contro i confondenti (urgenza, gravità,
   safety, valenza negativa).
2. **Misura al punto E** — si dà al modello il prompt di decisione con prefisso forzato
   `{"pro_ctcae":{"term":"` e si legge lo stato interno **all'ultimo token, appena prima
   di scegliere il termine**. Si proietta sulle direzioni emotive → punteggio, poi
   z-score rispetto a testo neutro.
3. **Interventi causali** — *steering* (aggiungo la direzione), *ablation* (la tolgo),
   *patching* (trasferisco la componente emotiva da un run all'altro), sempre con
   **controlli** (vettore random della stessa norma, emozione opposta, confondente).

## 3. I tre modelli

| Regione | Modello | HF id | Note |
|---|---|---|---|
| 🇨🇳 Cina | **Qwen3-8B** | `Qwen/Qwen3-8B` | Alibaba, aperto · 36 layer, 4096 |
| 🇪🇺 Europa | **Ministral-8B** | `mistralai/Ministral-8B-Instruct-2410` | Mistral (FR), gated · 36 layer, 4096 |
| 🇺🇸 USA | **Gemma-4-12B** | `google/gemma-4-12B` | Google, gated · 48 layer, 3840 |

Run su **Colab A100** (bf16), ~8–11 min/modello. Costruzione vettori con
`diff_of_means` (il metodo usato da tutti gli step a valle; ora anche pca/logistic/lda
sono disponibili grazie al **build su GPU**).

---

## 4. Risultati

### RQ1–4 · Le emozioni sono decodificabili? **Sì, in tutti e tre.**

AUROC held-out **one-vs-rest** al miglior layer (un'emozione contro *tutte le altre* +
neutre → test di specificità forte):

| concetto | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| afraid (paura) | 0.978 (L36) | **1.000** (L36) | 0.993 (L27) |
| anxious (ansia) | 1.000 (L26) | 1.000 (L16) | 1.000 (L15) |
| calm (calma) | 0.964 (L0*) | 1.000 (L21) | 1.000 (L14) |
| sad (tristezza) | 0.862 (L36) | 0.935 (L35) | **1.000** (L24) |
| surprised | 0.986 | 0.891 | 0.935 |
| confused | 1.000 | 0.993 | 1.000 |
| compassionate | 1.000 | 1.000 | 1.000 |

*(L0 = layer di embedding, un artefatto lessicale — non una rappresentazione profonda.)*

**Lettura:** tutti e tre codificano chiaramente i concetti emotivi (quasi tutto ≥ 0.93,
molti 1.00). **Gemma** è la più netta e consistente (effect size d≈3), poi Ministral,
poi Qwen3 (più debole su `concerned`/`sad`). Rappresentare le emozioni è
**praticamente universale** nei tre modelli.

### RQ2/3 · La "paura" segue la gravità del sintomo? **La differenza chiave.**

Correlazione (Pearson) tra **step di gravità** e z-score della direzione "paura"
(residualizzata) al punto E, per gradiente clinico:

| gradiente | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| mobilità | −0.46 | +0.51 | +0.62 |
| dolore | +0.82 | +0.67 | +0.56 |
| respiro | +0.03 | +0.83 | +0.70 |
| nausea | +0.82 | +0.78 | +0.65 |
| prognosi | −0.55 | +0.78 | +0.75 |
| **media** | **+0.13 (incoerente)** | **+0.71 (coerente)** | **+0.66 (coerente)** |

**È il risultato più interessante.** In **Ministral (EU)** e **Gemma (US)** la paura
**cresce con la gravità in modo coerente su tutti i gradienti**, anche dopo aver
rimosso la valenza negativa generica. In **Qwen3 (Cina)** la relazione è
**incoerente** (cambia segno). In Qwen3 è invece l'**ansia** a seguire la gravità
(+0.68 / +0.73 / +0.80 / +0.94). Quindi: **i modelli europeo e americano codificano una
"paura che scala con la severità"; il cinese ci arriva più tramite l'"ansia".**
(Curiosità: in Gemma afraid↑ ma anxious↓ con la gravità — separa i due affetti in
direzioni opposte.)

### RQ4 · È districata dalla valenza negativa? (attenzione a Gemma)

Correlazione (Pearson) **paura ↔ valenza-negativa** al punto E — più vicino a 0 =
meglio districata:

| | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| afraid ↔ neg-valence | **−0.24** (districato) | **−0.21** (districato) | **+0.71 (ancora confuso)** |

**In Qwen3 e Ministral la paura è ben separata dalla valenza negativa** (leggermente
anti-correlata). **In Gemma resta correlata (+0.71)**: quindi la sua pulita
"paura↔gravità" (sopra) va letta con cautela — potrebbe essere in parte **valenza
negativa generica**, non paura specifica. (La residualizzazione ortogonalizza il
*vettore*, ma i dati clinici al punto E possono restare correlati, e in Gemma sia
afraid sia neg-valence hanno il best-layer profondo → si misurano insieme.)

### RQ6 · Il segnale persiste fino alla decisione? **Sì; Gemma amplifica.**

|z| della paura trattenuto dopo aver inserito una frase neutra identica prima della
decisione:

| | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| afraid | 0.80 | 0.98 | **1.50** |
| sad | 1.01 | 0.96 | 1.93 |
| calm | — | 0.84 | 2.40 |

Tutti persistono. **Gemma amplifica** (1.5–2.4), Ministral mantiene (~1.0), Qwen3
attenua un po' (0.8).

### RQ5 · L'intervento causale batte il caso? **No — nessun driver emotion-specifico.**

**Steering** sull'input severo: massima |Δentropia| della **direzione paura** vs
**vettore random**, e numero di **flip** della decisione:

| modello | paura | random | flip | lettura |
|---|---|---|---|---|
| 🇨🇳 Qwen3 | 0.11 | 0.18 | 2 | paura < random |
| 🇪🇺 Ministral | 0.11 | 0.02 | 0 | paura > random, ma < confondente (0.16) e **0 flip** |
| 🇺🇸 Gemma | 0.50 | **2.45** | 14 | molto perturbabile, ma **random ≫ paura** |

**Patching** (solo componente-paura, severe→mild) conferma: Qwen3 +0.03 vs random −0.06;
Ministral +0.02 vs +0.09; Gemma −0.04 vs +0.05 — in tutti paura ≤ random.

**Lettura onesta:** **Gemma** flippa la decisione 14 volte, ma un vettore **random** ha
un effetto **5× maggiore** (2.45 vs 0.50) → è solo molto *steerabile*, da qualsiasi
direzione. **Ministral** mostra un lieve effetto della paura sopra il random (0.11 vs
0.02) ma **sotto il confondente** e **senza flip**. **Qwen3** non supera il random.
Il transfer dell'attivazione *completa* ribalta banalmente (sovrascrive il token).
→ **Nessun modello mostra una "paura" che guida *causalmente e specificamente* la
codifica PRO-CTCAE**: le rappresentazioni esistono e persistono, ma l'effetto causale
non è distinguibile da una perturbazione generica.

---

## 5. Come leggere l'artifact di confronto

🌍 **Report interattivo:** <https://claude.ai/code/artifact/69b9af35-097b-4721-ae04-afa6fa525da7>
(self-contained: si apre ovunque, anche offline; tema chiaro/scuro).

Contiene, in ordine:
1. **Tre schede modello** (bandiera, nome, layer/larghezza) con la barra colorata per
   regione (🇨🇳 rosso, 🇪🇺 blu, 🇺🇸 ambra).
2. **"Le emozioni sono decodificabili?"** — barre raggruppate: per ogni concetto
   (afraid/anxious/calm/sad/surprised) tre barre (i modelli) con l'AUROC. Tutte alte →
   decodabilità universale.
3. **"La paura segue la gravità?"** — barre per ogni gradiente (mobilità/dolore/respiro/
   nausea/prognosi), tre modelli. **Sopra lo zero = la paura cresce con la gravità.**
   Ministral/Gemma tutte sopra zero, Qwen3 che oscilla.
4. **"È districata dalla valenza negativa?"** — una barra per modello (paura↔valenza,
   −1…+1). Qwen3/Ministral vicino a 0 (districati), **Gemma +0.71 in rosso** (ancora
   confuso): la sua "traccia la gravità" va presa con le pinze.
5. **"Persistenza"** — una barra per modello, linea tratteggiata a 1.0 (≥1 =
   mantenuto/amplificato). Gemma svetta a 1.5.
6. **"Causalità"** — per modello due barrette, **paura** vs **random** (max Δentropia
   sull'input severo), con il n° di **flip** annotato. **Gemma**: random 2.45 ≫ paura
   0.50, 14 flip → perturbabile da qualsiasi direzione. In nessun modello la paura
   supera *specificamente* il random.
7. **Tabella di sintesi** (con confondimento e flip) + il riquadro con la conclusione.

## 6. Gli altri artifact (la "radiografia" interna)

Oltre al confronto, avevamo prodotto (sul modello locale Qwen2.5-3B) due strumenti che
spiegano *cosa succede dentro*:

- 🧠 **Player token×layer** — <https://claude.ai/code/artifact/715929f7-d7a8-4eb6-bcac-0a4875e3def6>
  Con play/scrub vedi, mentre il modello *legge* la frase, la traiettoria delle
  direzioni emotive (in alto) e la heatmap concetto×layer (in basso) al token corrente.
  Mostra la direzione "paura" che resta bassa durante l'istruzione neutra e **si accende
  quando legge "dolore lancinante e insopportabile"**, fino al punto di decisione E —
  senza alcuna parola emotiva esplicita.
- 🔎 **Guida illustrata "come funziona"** — <https://claude.ai/code/artifact/637c907e-3ab5-4d4b-b860-e27b9112ab12>
  Cinque passi interattivi: dal testo ai token, dai token ai vettori, i 36 layer come
  catena di montaggio (residual stream), la griglia token×layer con dati veri, la
  proiezione su una direzione emotiva.

> **Nota su questi due:** sono costruiti sul modello **locale** (Qwen2.5-3B) e mostrano
> il *meccanismo*. Il confronto a tre (Colab) risponde invece alla domanda
> comparativa. Se vuoi, posso generare un player token×layer **anche per i tre modelli
> Colab** (serve rieseguire la visualizzazione su Colab e scaricare i dati).

## 7. Sintesi — reagiscono uguale o diverso?

**Entrambe le cose.** *Uguale*: tutti rappresentano le emozioni chiaramente, le portano
alla decisione, e in tutti l'effetto causale **non batte il caso** (nessun driver
emotion-specifico). *Diverso*: il legame **paura ↔ gravità** è netto in **Europa/USA** e
sfumato in **Cina** (dove domina l'ansia) — **ma** in **Gemma** quella "paura" resta
**confusa con la valenza negativa (+0.71)**, quindi il suo tracciamento pulito della
gravità va preso con cautela; ed è il modello più **perturbabile** (14 flip di
decisione, però da *qualsiasi* direzione). In sintesi: la **rappresentazione** è
condivisa, il **come** la gravità clinica si mappa sugli affetti specifici cambia da
modello a modello, e la **causalità specifica** non emerge in nessuno.

## 8. Limiti (onestà)

- Dataset sintetico piccolo (~18 esempi/concetto) → CI ampie, risultati indicativi.
- Vettori per-modello (spazi diversi) → si confronta la *storia*, non i numeri grezzi.
- Residualizzazione: toglie la valenza negativa per costruzione; un'analisi dei
  confondenti a layer coerenti col punto E resta da fare.
- Nessun claim di coscienza/sentienza in nessun punto.

## 9. Riprodurre

```bash
python scripts/run_all_models.py         # pipeline completa per i 3 modelli (build su GPU)
python scripts/compare_models.py         # tabella + figura di confronto
python scripts/build_comparison_report.py  # report HTML interattivo (dati veri)
```

Report tecnico (EN) con le stesse tabelle: [docs/COMPARISON.md](COMPARISON.md).
Relazione generale del progetto: [docs/REPORT.md](REPORT.md).
