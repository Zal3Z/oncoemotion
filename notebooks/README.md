# Notebook operativo

Il repository contiene un solo notebook eseguibile:

- `oncoemotion_colab.ipynb` — run end-to-end su Colab dello studio controllato,
  della validazione sui campi reali e dell'estensione direct/deliberative con
  `NON_CLASSIFICABILE`.

Le analisi prima distribuite fra notebook separati sono state consolidate negli
script richiamati da questo notebook. In questo modo **Runtime → Esegui tutto** usa
un solo punto di ingresso, mantiene gli output su Google Drive e conserva una cache
dei pesi per il tempo necessario a completare tutte le fasi di ciascun modello.

Input privati attesi sotto `MyDrive/oncoemotion/`:

- `sinomi_campi_aperti.xlsx`;
- `pro_ctcae_italian_labels.json`.

Per iniziare un rerun indipendente basta cambiare `RUN_ID` nella prima cella. Il
workbook e le tabelle di audit contengono testo clinico e devono restare nella
cartella privata del run.

Il rerun usa otto modelli, inclusa la coppia Apertus 8B in BF16 come tutti gli
altri checkpoint. Il panel reasoning contiene tre
coppie base/medicalizzato, MedGemma 27B, Apollo2-7B con supporto italiano esplicito
e Qwen3.6-27B; Qwen riceve anche un braccio `native_reasoning` separato. Vedere
`docs/MODEL_PANEL_REASONING.md`.
