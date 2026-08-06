# Esecuzione definitiva - ESMO AI 2026

Il protocollo scientifico v3, centrato sulla sensibilità al linguaggio affettivo, è in
[`ESMO_2026_PROTOCOL.md`](ESMO_2026_PROTOCOL.md); i parametri autoritativi sono in
[`configs/study_esmo_2026.yaml`](../configs/study_esmo_2026.yaml). I vecchi report e
gli archivi `oncoemotion_results*` sono esplorativi e non alimentano l'abstract.

Il run reale validato e la successiva estensione con risposta esplicita
`NON_CLASSIFICABILE` sono documentati separatamente in
[`REASONING_ABSTENTION_PROTOCOL.md`](REASONING_ABSTENTION_PROTOCOL.md). Tutti i
protocolli vengono ora eseguiti da `notebooks/oncoemotion_colab.ipynb`; l'estensione
direct/deliberative resta un'analisi separata e non sostituisce il protocollo
reale v2.

## 1. Prima del run

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe scripts\generate_labeled_clinical.py --check-only
.venv\Scripts\python.exe scripts\smoke.py --stage data
git status --short
```

Congelare e committare prima di Colab:

- seed e controlli lessicali;
- dataset clinico e gold;
- personae e filler;
- layer band, cross-validation e soglie del protocollo;
- scorer, ruoli, bracci, semi e lista modelli.
- partizione semantica dei qualificatori (`affective_reaction` vs
  `symptom_intensity`) e famiglie affettive.

Il file ufficiale `terminology/official/pro_ctcae_italian_labels.json` deve essere
disponibile nella sessione Colab. La build deve fallire, non degradare, se manca.

## 2. Notebook Colab

Aprire `notebooks/oncoemotion_colab.ipynb` e usare **Runtime → Esegui tutto**. Le
celle eseguono in ordine:

1. configurazione del nuovo `RUN_ID` e controllo GPU;
2. clone del branch congelato;
3. installazione e login Hugging Face;
4. mount degli input privati e output persistenti su Drive;
5. preflight degli accessi;
6. studio controllato, campi reali v1 ed estensione reasoning/astensione;
7. analisi, figure, audit per item e pacchetti redatti.

Il run usa otto modelli Tier 1 nelle fasi controllata e reale. La coppia Apertus
70B base/medicalizzata è caricata in NF4 appaiato secondo
`configs/runtime_blackwell_96gb.yaml`; gli altri modelli usano BF16. L'estensione
reasoning viene eseguita su tre coppie base/medicalizzato, MedGemma 27B,
Apollo2-7B e Qwen3.6-27B; Qwen3-8B e Ministral restano nei rerun confermativi ma
non vengono ripetuti nell'estensione. `RUN_SECONDARY=True` aggiunge BioMistral-7B
e abilita le altre analisi secondarie. Per il poster non vengono
rieseguiti esperimenti storici fuori protocollo: le loro funzioni utili sono già
incorporate negli script dello studio controllato.

Ogni modello resta nella cache locale fino al completamento di tutte e tre le fasi;
solo allora i pesi vengono rimossi. I risultati vivono invece in
`MyDrive/oncoemotion/runs/<RUN_ID>/outputs`, quindi una disconnessione non li elimina.

Il run è riprendibile tramite due manifesti:

- `outputs/models/<slug>/pipeline_manifest.json` per vettori e validazione;
- `outputs/role_emotion/<slug>__meta.json` per protocollo, scorer e righe di ruolo.

Un file esistente senza fingerprint corrente viene rigenerato.

## 3. Cancelli automatici

### Asse interpretabile

`validate_vectors.py` applica per ogni asse del profilo preregistrato:

- AUROC cross-validata >= 0,60;
- massimo coseno assoluto con controlli lessicali <= 0,50.

Se il nucleo usato dall'ablazione non passa, `run_role_emotion.py` esegue comunque il
braccio intatto ma salta i bracci causali. L'opzione
`--force-unvalidated-ablation` è riservata allo smoke test e non va usata su Colab.

### Primario comportamentale

`analyze_results.py` misura il disaccordo top-1 `oncologo` vs `none_filler` sugli
item con termine. `none_task` non entra nel primario.

### Interazione affettiva chiave

Sulle 69 coppie `affective_reaction`, `analyze_results.py` confronta il disaccordo
fra ruoli nella formulazione emotiva con quello nella formulazione neutra. Le 43
coppie `symptom_intensity` sono sempre riportate come controllo di specificità. Se
la partizione non contiene esattamente gli item preregistrati, l'analisi si ferma.

Le proiezioni affettive sono riassunte sia all'ultimo token del testo paziente (`R`)
sia al punto di decisione (`D`), usando solo assi eleggibili nel singolo modello.
Un'associazione fra proiezione e cambio di codice è descrittiva, non causale.

### Accuratezza invisibile

La differenza appaiata di accuratezza supporta la claim soltanto se tutto il suo
intervallo al 95% cade in [-0,05, +0,05]. “Non significativo” non basta.

### Meccanismo

L'ablazione emotiva deve ridurre il disaccordo fra ruoli più dell'ablazione casuale,
con intervallo al 95% sopra zero. Se non passa, l'abstract resta comportamentale e
non attribuisce causalmente l'instabilità all'affetto.

## 4. Artefatti da conservare

La fase finale crea i pacchetti redatti sotto `outputs/packages/` e conserva:

- configurazione congelata;
- validazione vettori e manifesti;
- righe, metadati e analisi per modello;
- analisi pooled `esmo_primary_analysis.json`;
- draft `esmo_abstract_draft.md` con conteggio caratteri.
- figure affettive, controllo di specificità e contrasti base/medicalizzato.

Sotto `outputs/tables/item_audit/` vengono inoltre prodotte tabelle locali con il
testo clinico e le scelte di ogni modello, comprese le condizioni diretta e
deliberativa. Queste tabelle e il workbook di revisione non devono entrare negli ZIP
pubblicabili.

Non includere pesi, attivazioni grezze, terminologia soggetta a licenza o testo
clinico non redatto.

## 5. Decisione sul poster

- Interazione affettiva presente + cancello causale superato: poster affettivo,
  comportamentale e meccanicistico.
- Interazione affettiva presente + cancello causale fallito: poster sulla sensibilità
  affettiva, con rappresentazioni interne dichiarate esplicative e non causali.
- Interazione affettiva assente: riportare il risultato negativo e non attribuire
  all'emozione l'eventuale instabilità comportamentale del ruolo.
- Nessun primario riproducibile o equivalenza non supportata: non usare il titolo
  “accuracy cannot see”; riformulare o non inviare.
