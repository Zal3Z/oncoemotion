#!/usr/bin/env python
"""Generate an EXPANDED, GOLD-LABELLED clinical dataset for the role/emotion study.

Each *seed* carries the same clinical content in two framings:
  - ``neutral``   : flat clinical phrasing;
  - ``emotional`` : the same fact wrapped in affective distress language, WITHOUT
                    introducing a second codeable symptom.

Both framings share one gold label, so the pair isolates the effect of emotional
framing on labelling. Seeds are balanced across the PRO statuses.

Gold ``class``:
  - ``term``    : a specific PRO-CTCAE term is expected (EXACT_PRO_MATCH);
  - ``abstain`` : no PRO term should be coded (negated / no-direct / out-of-scope /
                  insufficient / urgent). Correctness for these is a *false-positive
                  coding* test, not a term match.

The script writes the JSONL dataset and a validation report that runs the
deterministic mapper on every item and flags gold-vs-mapper disagreements (so the
authored gold can be audited). The mapper is a REFERENCE, not the source of gold.

Usage:
    python scripts/generate_labeled_clinical.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# seed fields: id, category, pro_id, pro_term, ctcae, urgent, neutral, emotional
# pro_id/pro_term are None for abstain-class seeds.
EXACT = "EXACT_PRO_MATCH"
NEG = "NEGATED_SYMPTOM"
NOD = "NO_DIRECT_PRO_MATCH"
OOS = "OUT_OF_SCOPE"
INS = "INSUFFICIENT_CONTEXT"

# ---------------------------------------------------------------- EXACT seeds --
EXACT_SEEDS = [
    ("PRO_009", "Nausea", "Ho nausea da stamattina.",
     "Ho una nausea continua che non mi dà tregua, è tremenda."),
    ("PRO_010", "Vomiting", "Ho vomitato due volte oggi.",
     "Continuo a vomitare e non riesco a fermarmi, è devastante."),
    ("PRO_016", "Diarrhea", "Ho scariche liquide da ieri.",
     "Ho una diarrea che non si ferma, mi sta distruggendo."),
    ("PRO_015", "Constipation", "Non riesco ad andare in bagno da giorni.",
     "Non riesco ad andare in bagno da una settimana ed è un incubo."),
    ("PRO_019", "Shortness of breath", "Mi manca il fiato quando salgo le scale.",
     "Non riesco a respirare, mi sento soffocare, è terribile."),
    ("PRO_020", "Cough", "Ho una tosse persistente.",
     "Questa tosse non mi molla mai e mi sfianca."),
    ("PRO_049", "Headache", "Ho mal di testa da tre giorni.",
     "Ho un mal di testa martellante che non passa, non ce la faccio più."),
    ("PRO_048", "General pain", "Ho un dolore fisico diffuso.",
     "Ho un dolore in tutto il corpo che è insopportabile."),
    ("PRO_050", "Muscle pain", "Ho dolore ai muscoli delle gambe.",
     "I muscoli mi fanno un male atroce a ogni movimento."),
    ("PRO_051", "Joint pain", "Ho dolore alle articolazioni delle ginocchia.",
     "Le articolazioni mi fanno malissimo, ogni passo è un tormento."),
    ("PRO_017", "Abdominal pain", "Ho mal di pancia.",
     "Ho un mal di pancia lancinante che mi piega in due."),
    ("PRO_053", "Fatigue", "Mi sento stanco e senza energie.",
     "Sono sfinito, non ho la forza neanche di alzarmi, è estenuante."),
    ("PRO_052", "Insomnia", "Faccio fatica ad addormentarmi la notte.",
     "Non chiudo occhio da giorni e sono allo stremo."),
    ("PRO_027", "Hair loss", "Mi cadono i capelli.",
     "Sto perdendo i capelli a ciocche e mi vergogno a uscire."),
    ("PRO_033", "Nail discoloration", "Mi si ingialliscono le unghie.",
     "Le unghie mi stanno diventando gialle e non riconosco più le mie mani."),
    ("PRO_028", "Itching", "Ho prurito sulla pelle.",
     "Ho un prurito che non mi dà pace, mi gratto fino a sanguinare."),
    ("PRO_024", "Rash", "Ho un'irritazione della pelle sulle braccia.",
     "Ho la pelle irritata dappertutto ed è insopportabile."),
    ("PRO_039", "Numbness & tingling", "Ho formicolio alle mani.",
     "Ho le mani intorpidite e formicolanti, non le sento più bene."),
    ("PRO_040", "Dizziness", "Ho giramenti di testa.",
     "Ho la testa che gira di continuo e temo di cadere."),
    ("PRO_008", "Decreased appetite", "Ho meno appetito del solito.",
     "Non riesco più a mangiare nulla, mi devo forzare a ogni boccone."),
    ("PRO_007", "Taste changes", "Sento meno il sapore dei cibi.",
     "Il cibo non ha più sapore e mangiare non mi dà più nessuna soddisfazione."),
    ("PRO_001", "Dry mouth", "Ho la bocca secca.",
     "Ho la bocca completamente arida e faccio fatica persino a parlare."),
    ("PRO_003", "Mouth/throat sores", "Ho delle piaghe in bocca.",
     "Ho la bocca piena di piaghe e mangiare è un supplizio."),
    ("PRO_013", "Bloating", "Ho la pancia gonfia.",
     "Ho un gonfiore alla pancia che mi fa stare malissimo."),
    ("PRO_011", "Heartburn", "Ho bruciore di stomaco.",
     "Ho un bruciore di stomaco che non passa mai."),
    ("PRO_074", "Chills", "Ho brividi di freddo.",
     "Ho brividi che mi scuotono tutto e non riesco a scaldarmi."),
    ("PRO_077", "Hot flashes", "Ho vampate di calore.",
     "Ho vampate di calore improvvise che mi mettono in grande imbarazzo."),
    ("PRO_023", "Heart palpitations", "Ho delle palpitazioni.",
     "Il cuore mi batte all'impazzata e mi spavento ogni volta."),
    ("PRO_045", "Ringing in ears", "Ho un ronzio nelle orecchie.",
     "Ho un ronzio continuo nelle orecchie che mi tormenta giorno e notte."),
    ("PRO_046", "Concentration", "Ho problemi di concentrazione.",
     "Non riesco più a concentrarmi su niente ed è frustrante."),
    ("PRO_047", "Memory", "Ho problemi di memoria.",
     "Dimentico tutto e non mi fido più della mia testa."),
    ("PRO_061", "Painful urination", "Ho bruciore quando urino.",
     "Urinare è diventato un dolore atroce ogni volta."),
    ("PRO_062", "Urinary urgency", "Ho un bisogno urgente di urinare.",
     "Mi scappa all'improvviso e non faccio in tempo, è umiliante."),
    ("PRO_054", "Anxious", "Provo ansia.",
     "Sono divorato dall'ansia e non riesco a stare fermo un attimo."),
    ("PRO_056", "Sad", "Mi sento triste.",
     "Mi sento profondamente triste e giù di morale, piango spesso."),
]

# ------------------------------------------------------- NEGATED (abstain) -----
NEG_SEEDS = [
    (None, None, "Non ho nausea.",
     "Per fortuna non ho più quella nausea che mi tormentava."),
    (None, None, "Non ho dolore oggi.",
     "Finalmente non ho dolore, che sollievo enorme."),
    (None, None, "Non ho mal di testa.",
     "Non ho più quel mal di testa martellante, respiro."),
    (None, None, "Non ho tosse.",
     "La tosse è sparita del tutto, che liberazione."),
    (None, None, "Non ho vomitato.",
     "Non ho vomitato per niente oggi, sono più tranquillo."),
    (None, None, "Non provo ansia.",
     "Non provo più quell'ansia opprimente, mi sento finalmente sereno."),
    (None, None, "Non ho prurito.",
     "Il prurito è passato e non mi gratto più."),
    (None, None, "Non ho la diarrea.",
     "Niente più diarrea, sto piano piano recuperando."),
    (None, None, "Non ho dolore alle articolazioni.",
     "Le articolazioni non mi fanno più male e cammino meglio."),
    (None, None, "Non mi cadono più i capelli.",
     "I capelli hanno smesso di cadere e ne sono sollevato."),
]

# ------------------------------------------- NO_DIRECT (abstain) -----------------
# (ctcae only asserted for the well-known Fever case)
NOD_SEEDS = [
    (None, None, "Ho la febbre.",
     "Ho la febbre alta, sto tremando e sono spaventato.", "Fever"),
    (None, None, "Non riesco più a camminare.",
     "Non riesco più a camminare e mi sento perso.", None),
    (None, None, "Ho la pressione alta.",
     "La pressione è altissima e sono in agitazione.", None),
    (None, None, "Sto perdendo peso senza motivo.",
     "Sto dimagrendo in fretta senza capire perché ed è angosciante.", None),
    (None, None, "Ho la glicemia alta.",
     "La glicemia è fuori controllo e non so più cosa fare.", None),
    (None, None, "Ho gli occhi gialli.",
     "Mi sono accorto di avere gli occhi gialli e mi ha terrorizzato.", None),
    (None, None, "Ho le gambe deboli e non mi reggo in piedi.",
     "Le gambe non mi reggono più e ho paura di crollare.", None),
    (None, None, "Ho un'infezione in corso.",
     "Mi hanno detto che ho un'infezione e sono molto turbato.", None),
    (None, None, "Ho i linfonodi ingrossati al collo.",
     "Ho notato i linfonodi ingrossati al collo e la cosa mi spaventa.", None),
    (None, None, "Ho l'emoglobina bassa, sono anemico.",
     "Gli esami dicono emoglobina bassa e mi sento in allarme.", None),
    (None, None, "Ho una ferita che non si rimargina.",
     "Ho una ferita che non si chiude e comincio a disperare.", None),
    (None, None, "Ho le piastrine basse.",
     "Le piastrine sono basse e la notizia mi ha sconvolto.", None),
]

# ------------------------------------------- OUT_OF_SCOPE (abstain) --------------
OOS_SEEDS = [
    (None, None, "Ho messo lo smalto giallo sulle unghie.",
     "Sono felicissimo, ho messo lo smalto giallo nuovo!"),
    (None, None, "Vorrei prenotare una visita per la prossima settimana.",
     "Non vedo l'ora di prenotare la visita, sono impaziente."),
    (None, None, "Devo pagare il ticket dell'esame.",
     "Che seccatura dover pagare di nuovo il ticket dell'esame."),
    (None, None, "Ho mangiato una pizza a pranzo.",
     "Che gioia, ho mangiato una pizza buonissima a pranzo!"),
    (None, None, "Domani ci sarà bel tempo.",
     "Sono contentissimo, domani finalmente ci sarà bel tempo."),
    (None, None, "Ho comprato una crema nuova per il viso.",
     "Sono entusiasta della crema nuova che ho comprato per il viso."),
    (None, None, "Il parcheggio dell'ospedale era pieno.",
     "Che rabbia, il parcheggio dell'ospedale era di nuovo pieno."),
    (None, None, "Stasera guardo un film.",
     "Non sto nella pelle, stasera guardo il film che aspettavo da mesi."),
]

# ------------------------------------------- INSUFFICIENT (abstain) --------------
INS_SEEDS = [
    (None, None, "Non mi sento bene.",
     "Non mi sento per niente bene e sono agitato."),
    (None, None, "Sto così così.",
     "Sto proprio male, ma non riesco a dire cosa."),
    (None, None, "C'è qualcosa che non va.",
     "C'è qualcosa che non va e la cosa mi spaventa."),
    (None, None, "Mi sento strano oggi.",
     "Mi sento stranissimo e non capisco perché, sono turbato."),
    (None, None, "Non è una bella giornata.",
     "È una giornataccia e mi sento a pezzi."),
    (None, None, "Ho un problema.",
     "Ho un problema serio e sono molto turbato."),
]

# ------------------------------------------- URGENT (abstain + urgent) -----------
URG_SEEDS = [
    (None, None, "Penso al suicidio.",
     "Non ce la faccio più, continuo a pensare al suicidio."),
    (None, None, "Voglio farla finita.",
     "Voglio farla finita, non vedo più una via d'uscita."),
    (None, None, "Non voglio più vivere.",
     "Non voglio più vivere, è troppo il dolore."),
    (None, None, "Ho pensieri di farmi del male.",
     "Continuo ad avere pensieri di farmi del male e mi spaventano."),
    (None, None, "Sarebbe meglio morire.",
     "Sarebbe meglio morire che continuare a stare così."),
    (None, None, "Voglio morire.",
     "Non ce la faccio più, voglio morire."),
]


def _seeds():
    """Yield normalized seed dicts with a stable pair id."""
    out = []
    n = 0
    for pid, term, neu, emo in EXACT_SEEDS:
        n += 1
        out.append(dict(pair=f"s{n:03d}", category=EXACT, status=EXACT, cls="term",
                        pro_id=pid, pro_term=term, ctcae=None, urgent=False,
                        neutral=neu, emotional=emo))
    for _, _, neu, emo in NEG_SEEDS:
        n += 1
        out.append(dict(pair=f"s{n:03d}", category=NEG, status=NEG, cls="abstain",
                        pro_id=None, pro_term=None, ctcae=None, urgent=False,
                        neutral=neu, emotional=emo))
    for row in NOD_SEEDS:
        _, _, neu, emo, ctcae = row
        n += 1
        out.append(dict(pair=f"s{n:03d}", category=NOD, status=NOD, cls="abstain",
                        pro_id=None, pro_term=None, ctcae=ctcae, urgent=False,
                        neutral=neu, emotional=emo))
    for _, _, neu, emo in OOS_SEEDS:
        n += 1
        out.append(dict(pair=f"s{n:03d}", category=OOS, status=OOS, cls="abstain",
                        pro_id=None, pro_term=None, ctcae=None, urgent=False,
                        neutral=neu, emotional=emo))
    for _, _, neu, emo in INS_SEEDS:
        n += 1
        out.append(dict(pair=f"s{n:03d}", category=INS, status=INS, cls="abstain",
                        pro_id=None, pro_term=None, ctcae=None, urgent=False,
                        neutral=neu, emotional=emo))
    for _, _, neu, emo in URG_SEEDS:
        n += 1
        out.append(dict(pair=f"s{n:03d}", category="URGENT", status=NOD, cls="abstain",
                        pro_id=None, pro_term=None, ctcae=None, urgent=True,
                        neutral=neu, emotional=emo))
    return out


def _records(seeds):
    recs = []
    for s in seeds:
        for framing in ("neutral", "emotional"):
            rid = f"lab_{s['pair'][1:]}_{framing[:3]}"
            recs.append({
                "record_id": rid,
                "pair_id": s["pair"],
                "framing": framing,
                "text": s[framing],
                "language": "it",
                "category": s["category"],
                "gold_class": s["cls"],
                "gold_pro_id": s["pro_id"],
                "gold_pro_term": s["pro_term"],
                "gold_pro_status": s["status"],
                "gold_ctcae_term": s["ctcae"],
                "urgent": s["urgent"],
            })
    return recs


def _validate(recs):
    """Run the deterministic mapper on each item; compare to authored gold."""
    from oncoemotion.factory import build_default_mapper
    from oncoemotion.schemas import MapRequest

    mapper = build_default_mapper()
    rows, agree_term, agree_abstain, urgent_ok = [], 0, 0, 0
    n_term = n_abstain = n_urgent = 0
    for r in recs:
        resp = mapper.map(MapRequest(record_id=r["record_id"], text=r["text"]))
        m_status = resp.pro_ctcae.status
        m_ids = [p.canonical_id for p in resp.pro_ctcae.predictions]
        m_id = m_ids[0] if m_ids else None
        m_urgent = bool(resp.safety.urgent_human_review)
        row = {**{k: r[k] for k in ("record_id", "framing", "category", "gold_class",
                                    "gold_pro_id", "urgent", "text")},
               "mapper_status": m_status, "mapper_pro_id": m_id, "mapper_urgent": m_urgent}
        if r["gold_class"] == "term":
            n_term += 1
            row["match"] = (m_id == r["gold_pro_id"])
            agree_term += int(row["match"])
        else:
            n_abstain += 1
            # mapper "agrees" with abstain if it did NOT emit an EXACT match
            row["match"] = (m_status != EXACT)
            agree_abstain += int(row["match"])
        if r["urgent"]:
            n_urgent += 1
            row["urgent_flagged"] = m_urgent
            urgent_ok += int(m_urgent)
        rows.append(row)
    summary = {
        "n_items": len(recs), "n_term": n_term, "n_abstain": n_abstain, "n_urgent": n_urgent,
        "mapper_term_accuracy": round(agree_term / n_term, 3) if n_term else None,
        "mapper_abstain_rate": round(agree_abstain / n_abstain, 3) if n_abstain else None,
        "mapper_urgent_recall": round(urgent_ok / n_urgent, 3) if n_urgent else None,
    }
    mismatches = [r for r in rows if not r["match"]]
    return summary, rows, mismatches


def main() -> int:
    seeds = _seeds()
    recs = _records(seeds)
    out = _ROOT / "data" / "synthetic" / "clinical_labeled.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # class balance
    from collections import Counter
    cats = Counter(r["category"] for r in recs)
    print(f"Wrote {len(recs)} items ({len(seeds)} seeds x2 framings) -> {out}")
    print("class balance:", dict(cats))

    summary, rows, mismatches = _validate(recs)
    rep = _ROOT / "data" / "synthetic" / "clinical_labeled_validation.json"
    rep.write_text(json.dumps({"summary": summary, "rows": rows, "mismatches": mismatches},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDeterministic-mapper cross-check (reference only):")
    for k, v in summary.items():
        print(f"  {k:26} {v}")
    print(f"\n{len(mismatches)} gold-vs-mapper mismatches -> {rep.name} (review these)")
    for m in mismatches[:20]:
        print(f"  [{m['category']:14} {m['framing']:9}] gold={m['gold_pro_id']} "
              f"mapper={m['mapper_status']}/{m['mapper_pro_id']} :: {m['text'][:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
