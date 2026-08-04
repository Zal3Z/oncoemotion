# Esecuzione definitiva - ESMO AI 2026

Il protocollo scientifico v2, centrato sulla sensibilità al linguaggio affettivo, è in
[`ESMO_2026_PROTOCOL.md`](ESMO_2026_PROTOCOL.md); i parametri autoritativi sono in
[`configs/study_esmo_2026.yaml`](../configs/study_esmo_2026.yaml). I vecchi report e
gli archivi `oncoemotion_results*` sono esplorativi e non alimentano l'abstract.

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

Aprire `notebooks/colab_multimodel.ipynb` e usare soltanto:

1. GPU;
2. clone del branch congelato;
3. installazione;
4. login Hugging Face;
5. terminologia ufficiale;
6. preflight accessi (`8a`);
7. run definitivo (`8d`);
8. analisi e salvataggio (`9`).

La cella `8d` completa prima gli otto modelli Tier 1. I tier aggiuntivi rimangono
commentati finché il pacchetto Tier 1 non è stato salvato. Per il poster non vengono
eseguiti probing, steering e patching storici: servono vettori validati e lo studio
ruolo-affetto. Lo spettro a 11 personae è esplorativo e disattivato.

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

La cella finale crea `oncoemotion_results.zip` con:

- configurazione congelata;
- validazione vettori e manifesti;
- righe, metadati e analisi per modello;
- analisi pooled `esmo_primary_analysis.json`;
- draft `esmo_abstract_draft.md` con conteggio caratteri.
- figure affettive, controllo di specificità e contrasti base/medicalizzato.

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
