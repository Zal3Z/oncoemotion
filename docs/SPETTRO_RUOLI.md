# Spettro dei ruoli — perché il ruolo cambia l'emotività (25 emozioni)

> **Analisi esplorativa.** Lo spettro a 11 personae non è un endpoint del protocollo
> ESMO 2026 e non deve guidare la selezione post-hoc degli assi affettivi.

*Estensione oncoemotion. Interpretabilità meccanicistica sul task PRO-CTCAE.*

🎭 **Report interattivo (Artifact):** <https://claude.ai/code/artifact/0a49b392-4006-43b0-bc18-8ae9c0b48f22>

> Rappresentazioni emotion-*like*, non emozioni coscienti. Dataset sintetico →
> indicazioni, non verdetti. Scale diverse tra modelli → si confronta la *storia*.

## La domanda

Il ruolo (oncologo, ingegnere, bambino…) cambia l'emotività interna, e **perché**?
11 personas in 4 gruppi — **medici** (oncologo, infermiere), **tecnici/distaccati**
(ingegnere, avvocato, contabile), **emotivi/profani** (paziente ansioso, bambino,
poeta), **controlli** — misurate su 70 sintomi al punto di decisione, con la
tavolozza completa di **25 emozioni**.

## Il risultato chiave: 3 emozioni nascondevano il segnale

Con il solo composito **paura + ansia + tristezza** (le 3 emozioni di prima) l'effetto
del ruolo è **debole e non specifico**: la differenza di stato "profano − medico"
è **quasi ortogonale all'asse della paura** (coseno ≈ 0):

| direzione sull'asse paura | 🇨🇳 Qwen3 | 🇪🇺 Ministral | 🇺🇸 Gemma |
|---|---|---|---|
| profani − medici (coseno) | −0.09 | +0.02 | −0.02 |

Cioè: **il ruolo NON muove lo stato "lungo la paura"**. L'intuizione "il medico ha meno
paura" è, letteralmente, sbagliata: la paura cambia poco.

**È guardando tutte le 25 emozioni che il segnale emerge** — e riguarda emozioni
diverse dalla paura, spesso specifiche del modello.

## Quali emozioni separano davvero i ruoli (profani vs medici)

| modello | i profani accendono PIÙ dei medici | i medici accendono PIÙ dei profani |
|---|---|---|
| 🇨🇳 Qwen3 | **rabbia** (Δ+2.6), ansia, noia, delusione | sorpresa, preoccupazione, **entusiasmo**, speranza |
| 🇪🇺 Ministral | tristezza, rabbia, vergogna (Δ piccoli) | sorpresa, speranza, curiosità |
| 🇺🇸 Gemma | orgoglio, amore, sollievo, ansia | **calma** (Δ−3.1!), noia, speranza, solitudine |

Letture:
- **La firma del ruolo professionale è la calma, non la mancanza di paura.** In
  **Gemma** i medici hanno una calma nettamente più alta dei profani (+4.95 vs +1.85):
  il medico non "ha meno paura", **è più sereno**. È l'emozione che il composito a 3
  ignorava del tutto.
- In **Qwen3** i profani (paziente, bambino, poeta) mostrano più **rabbia** e
  **delusione**; i medici più **speranza/entusiasmo** — un affetto più "orientato alla
  cura/soluzione".
- L'entità e le emozioni coinvolte **variano per modello**: non esiste un'unica
  "emozione del ruolo", ma un tema ricorrente — i ruoli professionali spostano l'affetto
  verso **calma/speranza** e via da **rabbia/ansia**.

## Familiarità medica o distacco professionale?

L'ingegnere/avvocato/contabile (tecnici, non-medici) **non replicano in modo pulito** i
medici: in Gemma la calma alta è soprattutto dei medici, ma anche il contabile è tra i
meno emotivi; in Qwen3 i tecnici stanno nel mezzo. Il quadro suggerisce una **composure
professionale** più che una conoscenza specificamente medica, ma il segnale è
model-specifico e non netto: serve un dataset più ampio per concludere.

## Persona vs reazione

L'emotività non è solo reazione al sintomo: alcune personas partono già "cariche". In
**Ministral** il *paziente ansioso* ha un'emotività-di-base (su testo neutro) altissima
(+5.5), molto sopra le altre → il ruolo imposta un **mood di partenza**, non solo un
diverso modo di reagire. La decomposizione persona/reazione (nel report) mostra entrambe
le componenti.

## In una riga

Con 3 emozioni sembrava che il ruolo non cambiasse quasi nulla (e non "la paura"); con
**25 emozioni** si vede che il ruolo professionale sposta l'affetto verso **calma e
speranza** e lontano da **rabbia e ansia** — con emozioni-chiave diverse da modello a
modello. La tavolozza ampia era necessaria per vederlo.

## Riprodurre

```bash
python scripts/run_role_spectrum.py --model <hf_id> --dtype bfloat16 --device auto \
    --vecs outputs/models/<slug>/emotion_vectors.npz \
    --val-report outputs/models/<slug>/vector_validation.json
python scripts/build_role_spectrum_report.py
```

Su Colab: cella 8c (dopo aver rigenerato i vettori a 25 emozioni con la cella 6).
