# Guida alla tabella di audit per item

## Payload già preparato

`scripts/export_item_audit.py` ricongiunge localmente il dataset clinico privato ai
risultati redatti. Produce sotto `outputs/tables/item_audit/`:

- `item_audit_long.jsonl`: 20.400 righe, una per valutazione × modello × ruolo;
- `item_audit_wide.jsonl`: 1.275 righe, con gli otto modelli affiancati per il
  ruolo oncologo;
- `item_audit_summary.json`: conteggi, hash e controllo della ricongiunzione.

Il payload contiene testo clinico e deve restare locale. Ogni riga dei risultati è
verificata contro `source_id`, riga Excel, fonte, item validato, grado, classe gold
e gold PRO. Il file locale rigenerato ha un hash binario diverso da quello Colab,
ma tutti questi campi coincidono per tutte le righe e per tutti i modelli.

## Come leggere il run v1

### Riga PRO-CTCAE

Campi principali:

- `gold_pro_id`: item corretto validato;
- `model_top1_id`: item scelto dal modello;
- `gold_rank`: posizione del gold nella classifica completa del modello;
- `label_margin`: distanza fra prima e seconda scelta;
- `auditable_correct_v1`: vero se `model_top1_id = gold_pro_id`.

### Riga CTCAE v5

Il run v1 non offriva gli item CTCAE. `generative_kind` descrive soltanto se il
modello, interrogato su un termine PRO, si è astenuto, ha prodotto un termine
mappabile, una risposta non mappabile o una non-risposta. La cella di correttezza
CTCAE deve restare vuota: non va trasformata in errore né in successo.

### Riga non associabile

La risposta attesa è l'astensione. Nel payload `auditable_correct_v1` è vero solo
quando `generative_kind = abstained`.

## Struttura prevista del workbook

Quando il renderer XLSX è disponibile, il workbook deve contenere:

1. `LEGGIMI`: significato delle tre popolazioni e legenda;
2. `CONFRONTO_ITEM`: 1.275 righe, otto modelli affiancati, filtri su fonte, item e
   grado;
3. `LONG_FORM`: 20.400 righe adatte a pivot e analisi;
4. `RIEPILOGO`: accuratezza PRO, astensione sulle righe non associabili e conteggi
   non valutabili CTCAE;
5. `REVISIONE_MANUALE`: colonne modificabili per sistema atteso, item atteso,
   correttezza manuale e nota.

La correttezza calcolata non deve sovrascrivere una revisione manuale. La formula
finale deve usare l'override solo quando compilato e conservare separatamente la
metrica automatica originale.

## Dopo il nuovo run reasoning

I nuovi output includeranno direttamente:

- `gold_choice_id` e `model_choice_id`;
- `gold_system` e `model_choice_system`;
- correttezza complessiva, PRO, CTCAE e non classificabile;
- top 5 con punteggi;
- modalità `direct` o `deliberative`;
- backend `none`, `standardized_prompted` o, per Qwen, `native_chat_template_thinking`;
- hash e numero di token della deliberazione redatta.

Per Qwen3.6 è presente anche la modalità `native_reasoning`. Deve essere confrontata
con `direct` all'interno di Qwen3.6 e non mediata insieme alla deliberazione
standardizzata degli altri modelli.

In quel caso la correttezza stretta è verificabile per tutte le 1.275 righe, perché
PRO, CTCAE e non classificabile sono davvero presenti nello spazio delle risposte.
