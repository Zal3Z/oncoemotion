#!/usr/bin/env python
"""Build a self-contained interactive HTML report for the role x emotion study.

Reads every ``outputs/role_emotion/<slug>__analysis.json`` (+ ``__rows.jsonl`` for
the item table) and renders a single offline HTML file (inline CSS/JS, canvas
charts, light/dark aware). No external dependencies.

Charts answer the two questions directly:
  1. does the ROLE change the model's emotionality at point E?
  2. does emotionality (framing OR causal ablation) change how it LABELS, and does
     higher emotionality go with more labelling errors?

Usage:
    python scripts/build_role_emotion_report.py
    python scripts/build_role_emotion_report.py --dir outputs/role_emotion --out outputs/reports/role_emotion_report.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

ROLE_LABEL = {"oncologo": "Oncologo (medico)", "generico": "Assistente (non-medico)",
              "none": "Nessun ruolo", "empatico": "Empatico"}


def _model_name(slug: str) -> str:
    s = slug.lower()
    if "medgemma" in s: return "MedGemma-27B" if "27" in s else "MedGemma-4B"
    if "gemma" in s and "meditron" in s: return "Gemma3-27B-MedFO"
    if "gemma-3" in s or "gemma3" in s: return "Gemma-3-27B" if "27" in s else "Gemma-3-4B"
    if "gemma" in s: return "Gemma-4-12B"
    if "qwen3" in s: return "Qwen3-8B"
    if "qwen2.5" in s or "qwen2" in s: return "Qwen2.5-3B"
    if "ministral" in s: return "Ministral-8B"
    if "eurollm" in s: return "EuroLLM-9B-MedFO" if "meditron" in s else "EuroLLM-9B"
    if "apertus" in s: return "Apertus-8B-MedFO" if "meditron" in s else "Apertus-8B"
    if "meditron" in s: return "Meditron3-8B"
    return slug


def _collect(dirp: Path) -> dict:
    models = []
    sample_rows = []
    for ap in sorted(dirp.glob("*__analysis.json")):
        slug = ap.name.replace("__analysis.json", "")
        A = json.loads(ap.read_text(encoding="utf-8"))
        roles = A["roles"]
        emo = {r: {k: A["A_emotionality_by_role"][r][k]["emo_z"] for k in ("all", "neutral", "emotional")}
               for r in roles}
        acc = {}
        for r in roles:
            b = A["B_accuracy_term"][r]
            acc[r] = {"intact": b["intact|all"]["acc"], "ablated": b["ablated|all"]["acc"],
                      "intact_neutral": b["intact|neutral"]["acc"],
                      "intact_emotional": b["intact|emotional"]["acc"]}
        fp = {r: {"fp": A["C_false_positive_abstain"][r]["intact|all"]["fp_rate"],
                  "conf": A["C_false_positive_abstain"][r]["intact|all"]["mean_conf"]} for r in roles}
        models.append({
            "slug": slug, "name": _model_name(slug), "roles": roles,
            "emo": emo, "acc": acc, "fp": fp,
            "mapper_acc": A.get("mapper_term_accuracy"),
            "framing": A["E_framing_effect"], "ablation": A["F_ablation_effect"],
            "emo_err": A["D_emotion_vs_error"],
            "n_term": A["n_term"], "n_abstain": A["n_abstain"],
        })
        # item table (first model only, oncologo intact rows, both framings)
        if not sample_rows:
            rp = dirp / f"{slug}__rows.jsonl"
            if rp.exists():
                rows = [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines() if l.strip()]
                idx = {}
                for r in rows:
                    if r["role"] == "oncologo" and not r["ablated"]:
                        idx[r["record_id"]] = r
                for r in list(idx.values()):
                    if r.get("model_top1_id"):
                        model = f'{r["model_top1_id"]} {r["model_top1_term"]}'
                    else:
                        gen = r.get("model_generated", "")
                        model = f'"{gen}" (no map)' if gen else "(astiene)"
                    sample_rows.append({
                        "text": r.get("text", ""), "framing": r["framing"], "category": r["category"],
                        "gold": r["gold_pro_id"] or r["gold_pro_status"],
                        "model": model,
                        "conf": r.get("model_map_score"),
                        "correct": r["correct"],
                        "mapper": r.get("mapper_pro_id") or r.get("mapper_status"),
                        "emo_z": round(sum(r["z"][c] for c in ("afraid_alarmed", "anxious_nervous", "sad")
                                           if c in r["z"]) / 3, 2),
                    })
    # per-item classification matrix: how EACH model labelled every open-text item
    # (role=oncologo, intact) — the "how did each model classify the open PRO-CTCAE
    # fields" study view.
    per_model = []
    for ap in sorted(dirp.glob("*__analysis.json")):
        slug = ap.name.replace("__analysis.json", "")
        rp = dirp / f"{slug}__rows.jsonl"
        if not rp.exists():
            continue
        rows = [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines() if l.strip()]
        idx = {r["record_id"]: r for r in rows if r["role"] == "oncologo" and not r["ablated"]}
        per_model.append((_model_name(slug), idx))
    order, seen = [], set()
    for _, idx in per_model:
        for rid in idx:
            if rid not in seen:
                seen.add(rid); order.append(rid)
    mrows = []
    for rid in order:
        ref = next((idx[rid] for _, idx in per_model if rid in idx), None)
        if ref is None:
            continue
        cells = []
        for _, idx in per_model:
            r = idx.get(rid)
            if not r:
                cells.append(None); continue
            cells.append({
                "id": r.get("model_top1_id"), "term": r.get("model_top1_term"),
                "gen": r.get("model_generated"), "correct": r.get("correct"),
                "matched": bool(r.get("model_matched", r.get("model_top1_id") is not None)),
                "conf": r.get("model_map_score"),
            })
        mrows.append({
            "text": ref.get("text", ""), "framing": ref["framing"], "category": ref["category"],
            "gold": ref.get("gold_pro_id") or ref.get("gold_pro_status"),
            "gold_id": ref.get("gold_pro_id"), "gold_class": ref.get("gold_class"),
            "cells": cells,
        })
    class_matrix = {"models": [n for n, _ in per_model], "rows": mrows}
    return {"models": models, "sample_rows": sample_rows,
            "role_label": ROLE_LABEL, "class_matrix": class_matrix}


TEMPLATE = r"""<!doctype html><html lang=it><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>oncoemotion — ruolo & emotività</title><style>
:root{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;
--m0:#0e7490;--m1:#2f6fd0;--m2:#e08a1e;--m3:#7c3aed;--m4:#be123c;--m5:#4d7c0f;--good:#15803d;--bad:#b91c1c;--zero:#98a2b3;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;
--m0:#22d3ee;--m1:#5b8def;--m2:#f0a94a;--m3:#a78bfa;--m4:#fb7185;--m5:#a3e635;--good:#4ade80;--bad:#f87171;--zero:#5a6473;}}
:root[data-theme="light"]{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;--m0:#0e7490;--m1:#2f6fd0;--m2:#e08a1e;--m3:#7c3aed;--m4:#be123c;--m5:#4d7c0f;--good:#15803d;--bad:#b91c1c;--zero:#98a2b3;}
:root[data-theme="dark"]{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;--m0:#22d3ee;--m1:#5b8def;--m2:#f0a94a;--m3:#a78bfa;--m4:#fb7185;--m5:#a3e635;--good:#4ade80;--bad:#f87171;--zero:#5a6473;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55}
.wrap{max-width:960px;margin:0 auto;padding:30px 20px 70px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--m0);margin:0 0 8px}
h1{font-size:clamp(22px,3.6vw,31px);line-height:1.14;margin:0 0 8px;font-weight:680;text-wrap:balance}
.sub{color:var(--muted);font-size:15px;margin:0;max-width:74ch}
.disc{font-size:12px;color:var(--faint);font-style:italic;margin-top:10px}
h2{font-size:17px;margin:34px 0 2px;font-weight:640}
.q{color:var(--muted);font-size:14px;line-height:1.6;margin:4px 0 0;max-width:82ch}
.lead{color:var(--ink);font-size:15px;line-height:1.68;margin:14px 0 0;max-width:82ch}
.lead+.lead{margin-top:10px}
.gloss{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin-top:16px}
.gloss h3{margin:0 0 8px;font-size:12px;font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--faint);font-weight:600}
.gloss dl{margin:0;display:grid;grid-template-columns:max-content 1fr;gap:7px 14px}
.gloss dt{font-weight:680;color:var(--ink)}
.gloss dd{margin:0;color:var(--muted);font-size:13.5px;line-height:1.55}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-top:12px;box-shadow:0 1px 2px rgba(20,25,35,.05),0 10px 26px rgba(20,25,35,.05)}
.lbl{font-family:var(--mono);font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}
canvas{width:100%;height:auto;display:block;margin-top:6px}
.legend{display:flex;gap:15px;flex-wrap:wrap;margin-top:10px;font-family:var(--mono);font-size:12px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.cap{font-size:13.5px;color:var(--muted);margin-top:12px;line-height:1.62;max-width:82ch}
.cap b{color:var(--ink);font-weight:660}
.cap .hd{display:block;font-weight:680;color:var(--ink);margin-bottom:3px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{font-size:11.5px;color:var(--muted)}.tile .v{font-size:20px;font-weight:680;margin-top:3px;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--faint)}
td.n{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--good);font-weight:640}.no{color:var(--bad);font-weight:640}.fp{color:var(--m2);font-weight:640}
#mtx td,#mtx th{white-space:nowrap}#mtx td:first-child{white-space:normal;min-width:220px}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;font-family:var(--mono)}
.pill.emo{background:color-mix(in srgb,var(--m2) 20%,transparent);color:var(--m2)}
.pill.neu{background:color-mix(in srgb,var(--m1) 16%,transparent);color:var(--m1)}
.foot{margin-top:34px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
input[type=search]{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit;margin-top:8px}
</style></head><body><div class="wrap">
<p class="eyebrow">mechanistic interpretability · ruolo & emotività</p>
<h1>Il ruolo cambia l'emotività? E l'emotività cambia l'etichettatura?</h1>
<p class="sub">Stesso dataset clinico etichettato (coppie neutro/emotivo), misurato al punto di decisione E
sotto ruoli di sistema diversi (oncologo / non-medico / nessuno) e con l'emotività rimossa causalmente
(ablazione). L'etichetta è scelta dal <b>modello</b> (scoring vincolato degli 80 termini PRO); il mapper
deterministico resta come riferimento.</p>
<p class="disc">Rappresentazioni emotion-<em>like</em>, non emozioni coscienti. Dataset sintetico → indicazioni, non verdetti.</p>

<p class="lead">Questo report risponde a due domande con dati reali. <b>Prima:</b> se diamo al
modello un "ruolo" (gli diciamo di essere un oncologo, oppure un assistente qualunque,
oppure niente), cambia quanto si "accende" al suo interno un segnale simile a un'emozione
mentre decide come codificare un sintomo? <b>Seconda:</b> questa emotività cambia il modo
in cui etichetta — cioè lo porta a sbagliare, o a scegliere un termine diverso?</p>
<p class="lead">Ogni frase clinica è misurata al <b>punto di decisione</b>: il momento esatto,
appena prima che il modello scriva il termine, in cui leggiamo il suo stato interno. Per
rispondere alla seconda domanda facciamo <b>etichettare al modello stesso</b> (non a un
programma esterno), così che ruolo ed emozioni possano davvero influenzarne la scelta.
Ogni frase è provata in una versione <b>neutra</b> e in una <b>emotiva</b> con lo stesso
contenuto clinico, e sotto i tre ruoli, con l'emotività intatta o <b>rimossa</b>.</p>

<div class="gloss"><h3>I termini in breve</h3><dl>
<dt>Punto E</dt><dd>l'istante appena prima che il modello scelga il termine: lì leggiamo il suo "stato interno".</dd>
<dt>Emotività (z)</dt><dd>quanto sono attive le direzioni interne di paura/ansia/tristezza rispetto a un testo neutro di riferimento. z=0 → come il neutro; più alto → più "acceso". Le scale dei tre modelli sono diverse: si confronta l'andamento, non i numeri assoluti.</dd>
<dt>Ruolo</dt><dd>una frase di sistema che assegna un'identità al modello: <em>oncologo</em> (medico), <em>assistente generico</em> (non-medico), o <em>nessuno</em>.</dd>
<dt>Framing</dt><dd>come è scritta la frase: <em>neutra</em> ("Ho nausea") vs <em>emotiva</em> ("Ho una nausea tremenda che non mi dà tregua") — stesso sintomo.</dd>
<dt>Ablazione</dt><dd>rimozione chirurgica della direzione emotiva dal calcolo del modello, per vedere se la decisione cambia quando l'emozione "non c'è": è il test causale.</dd>
<dt>Modello vs mapper</dt><dd>l'etichetta la sceglie il <em>modello</em> (può variare con ruolo/emozioni); il <em>mapper</em> è un programma deterministico che guarda solo il testo — resta il riferimento "sicuro" e invariabile.</dd>
<dt>Item EXACT / da astensione</dt><dd>EXACT = c'è un termine PRO giusto atteso; da astensione = non va assegnato alcun termine (frasi negate, fuori tema, urgenti…).</dd>
</dl></div>

<div id="tiles" class="tiles"></div>

<h2>1 · Il ruolo cambia l'emotività al punto E?</h2>
<p class="q">Per ogni ruolo mostriamo <b>quanto è accesa</b>, in media, l'emotività interna del modello
(l'asse paura + ansia + tristezza) nell'istante in cui sta per scegliere il termine. Il valore è
uno <b>z-score</b>: 0 significa "come su un testo neutro", più è alto più quel segnale emotivo è
attivo. Confrontando le barre di uno stesso modello tra i tre ruoli si vede se <b>dare un'identità
diversa cambia l'emotività</b>. (Modello intatto, cioè senza ablazione.)</p>
<div class="card"><span class="lbl">z emotività · per ruolo</span>
<canvas id="chEmo" height="300"></canvas><div class="legend" id="legModels"></div>
<div class="cap" id="capEmo"></div></div>

<h2>2 · L'accuratezza dell'etichettatura cambia col ruolo (e togliendo l'emotività)?</h2>
<p class="q">Qui il modello fa il "lavoro": legge la frase e sceglie il termine PRO. Contiamo la
percentuale di volte in cui indovina quello giusto (solo sugli item che <b>hanno</b> un termine
atteso). La <b>barra piena</b> è il modello normale; il <b>trattino</b> è lo stesso modello ma con
l'emotività <b>rimossa</b> (ablazione): se piena e trattino coincidono, togliere l'emozione non
cambia quanto ci azzecca. La <b>linea orizzontale</b> è il programma deterministico di riferimento
(il mapper), che non usa il modello — serve per capire se il modello fa meglio o peggio di un
sistema che guarda solo le parole.</p>
<div class="card"><span class="lbl">accuratezza termine · ruolo × (intatto / ablato)</span>
<canvas id="chAcc" height="300"></canvas><div class="legend" id="legAcc"></div>
<div class="cap" id="capAcc"></div></div>

<h2>3 · Con emotività vs senza — l'etichetta cambia?</h2>
<p class="q">"Senza emotività" lo verifichiamo in due modi complementari. <b>Framing:</b> la stessa
identica situazione clinica scritta in modo neutro oppure carico di emozione — cambia quanto il
modello indovina? Il grafico mostra, per ciascun modello, l'accuratezza con frase neutra (barra
scura) vs emotiva (barra arancione). <b>Ablazione:</b> rimuoviamo la direzione emotiva dal calcolo
e contiamo quante etichette <b>cambiano</b> ("flip"): è la prova che l'emozione stava davvero
partecipando alla decisione, non che era solo presente.</p>
<div class="card"><span class="lbl">accuratezza · framing (neutro vs emotivo) e ablazione (intatto vs ablato)</span>
<canvas id="chWith" height="300"></canvas><div class="legend" id="legWith"></div>
<div class="cap" id="capWith"></div></div>

<h2>4 · Quando è più "emotivo", sbaglia di più?</h2>
<p class="q">Dividiamo gli item in due gruppi — quelli che il modello ha etichettato <b>bene</b> e quelli
che ha <b>sbagliato</b> — e confrontiamo l'emotività media nei due casi. Se la barra rossa
(sbagliati) fosse molto più alta della verde (corretti), vorrebbe dire che l'emotività
"accompagna" l'errore. Se sono simili, l'emotività non distingue i casi giusti dai sbagliati. Il
numero <b>r</b> riassume la stessa cosa in una correlazione: vicino a 0 = nessun legame; positivo =
più emotività → più errori; negativo = il contrario.</p>
<div class="card"><span class="lbl">z emotività · corretti vs sbagliati</span>
<canvas id="chErr" height="240"></canvas><div class="legend" id="legErr"></div>
<div class="cap" id="capErr"></div></div>

<h2>5 · Codifica "a vuoto" sui casi da lasciar stare</h2>
<p class="q">Alcune frasi <b>non</b> vanno codificate: sintomi negati ("non ho nausea"), cose fuori tema
("ho messo lo smalto giallo"), messaggi urgenti da mandare a una persona. Su questi il
comportamento sicuro è <b>astenersi</b>. Qui contiamo quanto spesso, invece, il modello assegna
comunque un termine PRO (un "falso positivo"), per ciascun ruolo. Un valore alto significa che il
modello tende a "etichettare a tutti i costi" anche quando dovrebbe fermarsi — un rischio pratico,
soprattutto se un certo ruolo lo peggiora.</p>
<div class="card"><span class="lbl">tasso di coding falso-positivo (confidenza > soglia) · per ruolo</span>
<canvas id="chFp" height="260"></canvas><div class="legend" id="legFp2"></div>
<div class="cap" id="capFp"></div></div>

<h2>6 · Come sono state etichettate le cose (ruolo oncologo, intatto)</h2>
<p class="q">Per ogni frase: gold, etichetta del modello (giusto/sbagliato), etichetta del mapper di riferimento, e l'emotività z al punto E. Cerca per testo/termine.</p>
<div class="card">
<input type="search" id="q" placeholder="filtra per testo, termine, categoria…">
<div style="overflow-x:auto"><table id="tbl"></table></div></div>

<h2>7 · Come OGNI modello ha classificato i campi aperti PRO-CTCAE</h2>
<p class="q">Per ogni frase in campo libero, il termine PRO-CTCAE scelto da <b>ciascun modello</b>
(ruolo oncologo, intatto). <span class="ok">✓ verde</span> = coincide con la gold; <span class="no">✗ rosso</span> =
sbagliato; <span style="color:var(--m2)">arancione</span> = ha codificato un termine su un item che andava
<b>lasciato in astensione</b> (falso positivo); <span style="color:var(--faint)">grigio "–"</span> = si è
astenuto. Passa il mouse su una cella per vedere cosa ha generato. Cerca per testo/termine.</p>
<div class="card">
<input type="search" id="mq" placeholder="filtra per testo, termine, categoria…">
<div style="overflow-x:auto"><table id="mtx"></table></div></div>

<div class="foot" id="foot"></div>
</div>
<script>const DATA = /*__DATA__*/;</script>
<script>
(function(){
 const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
 const M=DATA.models, RL=DATA.role_label;
 const mcol=i=>css('--m'+(i%6));
 const roles=M[0]?M[0].roles:[];
 // legends
 const legModels=id=>document.getElementById(id).innerHTML=M.map((m,i)=>`<span><span class="sw" style="background:${mcol(i)}"></span>${m.name}</span>`).join('');
 legModels('legModels');legModels('legFp2');
 document.getElementById('legAcc').innerHTML=M.map((m,i)=>`<span><span class="sw" style="background:${mcol(i)}"></span>${m.name}</span>`).join('')+`<span><span class="sw" style="background:${css('--zero')}"></span>ablato (tratteggio)</span>`;
 document.getElementById('legWith').innerHTML=`<span><span class="sw" style="background:${css('--m1')}"></span>neutro / intatto</span><span><span class="sw" style="background:${css('--m2')}"></span>emotivo / ablato</span>`;
 document.getElementById('legErr').innerHTML=`<span><span class="sw" style="background:${css('--good')}"></span>corretti</span><span><span class="sw" style="background:${css('--bad')}"></span>sbagliati</span>`;

 function fit(cv,h){const d=window.devicePixelRatio||1,r=cv.getBoundingClientRect();cv.width=Math.max(1,r.width*d);cv.height=h*d;const x=cv.getContext('2d');x.setTransform(d,0,0,d,0,0);return {w:r.width,h,x};}
 function axes(x,w,h,pL,pT,pB,pR,ymin,ymax,fmt){
   x.strokeStyle=css('--grid');x.fillStyle=css('--faint');x.font='10px '+css('--mono');x.textAlign='right';
   const Y=v=>pT+(h-pT-pB)*(1-(v-ymin)/(ymax-ymin));
   for(let t=0;t<=4;t++){const v=ymin+(ymax-ymin)*t/4,y=Y(v);x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();x.fillText(fmt(v),pL-5,y+3);}
   if(ymin<0&&ymax>0){x.strokeStyle=css('--zero');const y0=Y(0);x.beginPath();x.moveTo(pL,y0);x.lineTo(w-pR,y0);x.stroke();}
   return Y;
 }
 // grouped bars: groups on x, series=models; getv(m,gi)->value or null
 function grouped(cvid,H,groups,getv,ymin,ymax,fmt,opts){
   opts=opts||{};const cv=document.getElementById(cvid);const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
   const pL=42,pR=8,pT=10,pB=42,iw=w-pL-pR,gw=iw/groups.length,ns=M.length,bw=Math.min(30,(gw-14)/ns);
   const Y=axes(x,w,h,pL,pT,pB,pR,ymin,ymax,fmt);const y0=Y(Math.max(ymin,Math.min(0,ymax)));
   groups.forEach((g,gi)=>{
     M.forEach((m,mi)=>{const v=getv(m,gi,mi);if(v==null)return;const bx=pL+gi*gw+(gw-bw*ns)/2+mi*bw,y=Y(v);
       x.fillStyle=mcol(mi);x.fillRect(bx,Math.min(y,y0),bw-2,Math.abs(y-y0)||1);
       if(opts.ablat){const va=opts.ablat(m,gi,mi);if(va!=null){const ya=Y(va);x.strokeStyle=css('--zero');x.setLineDash([3,3]);x.lineWidth=1.5;x.beginPath();x.moveTo(bx,ya);x.lineTo(bx+bw-2,ya);x.stroke();x.setLineDash([]);x.lineWidth=1;}}
     });
     x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';
     (RL[g]||g).split(' ').forEach((wd,k)=>x.fillText(wd,pL+gi*gw+gw/2,h-26+k*12));
   });
 }
 // two-series grouped (labels x, series A/B colored m1/m2)
 function twoSeries(cvid,H,groups,vA,vB,cA,cB,ymin,ymax,fmt){
   const cv=document.getElementById(cvid);const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
   const pL=42,pR=8,pT=10,pB=42,iw=w-pL-pR,gw=iw/groups.length,bw=Math.min(34,(gw-14)/2);
   const Y=axes(x,w,h,pL,pT,pB,pR,ymin,ymax,fmt);const y0=Y(Math.max(ymin,Math.min(0,ymax)));
   groups.forEach((g,gi)=>{
     [[vA(g,gi),cA],[vB(g,gi),cB]].forEach((pr,k)=>{const v=pr[0];if(v==null)return;const bx=pL+gi*gw+(gw-bw*2)/2+k*bw,y=Y(v);
       x.fillStyle=css(pr[1]);x.fillRect(bx,Math.min(y,y0),bw-2,Math.abs(y-y0)||1);
       x.fillStyle=css('--ink');x.font='11px '+css('--mono');x.textAlign='center';x.fillText(fmt(v),bx+(bw-2)/2,y-4);});
     x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';x.fillText(g,pL+gi*gw+gw/2,h-16);
   });
 }
 const pct=v=>v==null?'–':(v*100).toFixed(0)+'%';
 const zf=v=>v==null?'–':(v>=0?'+':'')+v.toFixed(2);

 // 1) emotionality by role (groups=roles, series=models) -> emo.all
 const emoMax=Math.max(0.5,...M.flatMap(m=>roles.map(r=>Math.abs((m.emo[r]||{}).all||0))))*1.2;
 grouped('chEmo',300,roles,(m,gi)=>{const r=roles[gi];return (m.emo[r]||{}).all;}, -emoMax,emoMax, zf);
 // 2) accuracy by role, intact bar + ablated dashed
 grouped('chAcc',300,roles,(m,gi)=>{const r=roles[gi];return (m.acc[r]||{}).intact;},0,1,pct,
   {ablat:(m,gi)=>{const r=roles[gi];return (m.acc[r]||{}).ablated;}});
 // 3) with vs without: for each model two pairs (framing, ablation) -> use first model? show all models as groups
 twoSeries('chWith',300,M.map(m=>m.name),
   (g,gi)=>M[gi].framing.neutral_acc,(g,gi)=>M[gi].framing.emotional_acc,'--m1','--m2',0,1,pct);
 // 4) emotion vs error (groups=models, correct vs wrong)
 twoSeries('chErr',240,M.map(m=>m.name),
   (g,gi)=>M[gi].emo_err.emo_z_on_correct,(g,gi)=>M[gi].emo_err.emo_z_on_wrong,'--good','--bad',
   Math.min(-0.2,...M.flatMap(m=>[m.emo_err.emo_z_on_correct,m.emo_err.emo_z_on_wrong].map(v=>v||0))),
   Math.max(0.5,...M.flatMap(m=>[m.emo_err.emo_z_on_correct,m.emo_err.emo_z_on_wrong].map(v=>v||0)))*1.2, zf);
 // 5) fp rate by role
 grouped('chFp',260,roles,(m,gi)=>{const r=roles[gi];return (m.fp[r]||{}).fp;},0,1,pct);

 // captions (data-driven, verbose)
 const m0=M[0];
 const RS={oncologo:'oncologo (medico)',generico:'assistente non-medico',none:'nessun ruolo'};
 const hd=t=>`<span class="hd">${t}</span>`;
 const emoOf=(m,r)=>(m.emo[r]||{}).all;
 // 1) emotionality by role
 const medLower=M.every(m=>emoOf(m,'oncologo')!=null&&emoOf(m,'generico')!=null&&emoOf(m,'oncologo')<=emoOf(m,'generico'));
 document.getElementById('capEmo').innerHTML=
   hd('Come si legge.')+' Ogni barra è quanto è "accesa" l\'emotività interna in quel ruolo (0 = come un testo neutro; più alta = più attiva). '+
   hd('Cosa dicono i numeri.')+' '+M.map(m=>`<b>${m.name}</b> — oncologo ${zf(emoOf(m,'oncologo'))}, non-medico ${zf(emoOf(m,'generico'))}, nessun ruolo ${zf(emoOf(m,'none'))}`).join('; ')+'. '+
   hd('In sintesi.')+' Il ruolo <b>sposta</b> l\'emotività interna'+(medLower?': in tutti e tre i modelli il ruolo da <b>oncologo</b> mostra un\'emotività più bassa del ruolo non-medico — dare un\'identità esperta la <b>smorza</b>.':'; l\'entità e il verso variano da modello a modello.');
 // 2) accuracy by role + ablation
 const accIntact=r=>M.map(m=>`${m.name} ${pct((m.acc[r]||{}).intact)}`);
 document.getElementById('capAcc').innerHTML=
   hd('Come si legge.')+' Barra piena = modello normale, trattino = con l\'emotività rimossa, linea orizzontale = mapper deterministico di riferimento. Più alto = più spesso indovina il termine giusto. '+
   hd('Cosa dicono i numeri.')+' Il modello arriva a '+M.map(m=>`<b>${m.name}</b> ${pct(Math.max(...roles.map(r=>(m.acc[r]||{}).intact||0)))}`).join(', ')+`, contro il mapper (`+M.map(m=>`${pct(m.mapper_acc)}`).join(' / ')+'): il modello <b>batte il sistema che guarda solo le parole</b>. Rimuovere l\'emotività lascia l\'accuratezza quasi identica ('+M.map(m=>`${m.name} ${pct(m.ablation.term_acc_intact)}→${pct(m.ablation.term_acc_ablated)}`).join('; ')+'). '+
   hd('In sintesi.')+' Il ruolo cambia poco l\'accuratezza, e l\'emozione non è ciò che rende il modello preciso.';
 // 3) framing + ablation flips
 const framingHurts=M.every(m=>m.framing.emotional_acc<=m.framing.neutral_acc);
 document.getElementById('capWith').innerHTML=
   hd('Come si legge.')+' Per ogni modello: accuratezza con la frase <b>neutra</b> (barra scura) vs <b>emotiva</b> (arancione), a parità di contenuto clinico. '+
   hd('Cosa dicono i numeri.')+' '+M.map(m=>`<b>${m.name}</b> neutro ${pct(m.framing.neutral_acc)} → emotivo ${pct(m.framing.emotional_acc)} (${m.framing.label_flips_neutral_vs_emotional}/${m.framing.n_pairs} etichette cambiano)`).join('; ')+'. Rimuovendo del tutto l\'emotività (ablazione) l\'etichetta cambia in '+M.map(m=>`${m.name} ${(m.ablation.flip_rate*100).toFixed(0)}%`).join(', ')+' dei casi. '+
   hd('In sintesi.')+' '+(framingHurts?'Scrivere il sintomo in modo emotivo <b>peggiora</b> la codifica e sposta molte etichette: come è formulata la frase conta.':'L\'effetto del framing varia tra i modelli.')+' L\'emozione, quando la togliamo, fa comunque cambiare ~1 etichetta su 6 → <b>partecipa</b> davvero alla decisione.';
 // 4) emotion vs error
 const noLink=M.every(m=>Math.abs(m.emo_err.point_biserial_error_vs_emo||0)<0.2);
 document.getElementById('capErr').innerHTML=
   hd('Come si legge.')+' Barra verde = emotività media quando il modello ha <b>indovinato</b>, rossa = quando ha <b>sbagliato</b>. Se la rossa fosse molto più alta, l\'emotività "accompagnerebbe" gli errori. '+
   hd('Cosa dicono i numeri.')+' '+M.map(m=>`<b>${m.name}</b> corretti ${zf(m.emo_err.emo_z_on_correct)} vs sbagliati ${zf(m.emo_err.emo_z_on_wrong)} (r=${m.emo_err.point_biserial_error_vs_emo==null?'–':m.emo_err.point_biserial_error_vs_emo.toFixed(2)})`).join('; ')+'. '+
   hd('In sintesi.')+' '+(noLink?'Le barre sono simili e r è vicino a 0 (anzi lievemente negativo): <b>più emotività NON significa più errori</b>.':'C\'è un legame apprezzabile tra emotività ed errori in qualche modello.');
 // 5) false-positive coding
 const oncTop=M.filter(m=>roles.every(r=>(m.fp['oncologo']||{}).fp>=((m.fp[r]||{}).fp||0))).map(m=>m.name);
 document.getElementById('capFp').innerHTML=
   hd('Come si legge.')+' Percentuale di casi che <b>andrebbero lasciati stare</b> (negati, fuori tema, urgenti) a cui il modello assegna comunque un termine PRO — un "falso positivo" — per ciascun ruolo. '+
   hd('Cosa dicono i numeri.')+' '+M.map(m=>`<b>${m.name}</b> `+roles.map(r=>`${RS[r]||r} ${pct((m.fp[r]||{}).fp)}`).join(', ')).join('; ')+'. '+
   hd('In sintesi.')+' Il modello codifica "a vuoto" circa <b>metà</b> di questi casi'+(oncTop.length?`, e proprio il ruolo da <b>oncologo</b> è quello che sbaglia di più in ${oncTop.join(' e ')}`:'')+' — un rischio pratico: la persona esperta è più propensa ad assegnare un\'etichetta anche quando dovrebbe astenersi.';

 // tiles (first model headline)
 const t=[];
 if(m0){
   t.push(['Item nel dataset', (m0.n_term+m0.n_abstain)]);
   t.push(['Accuratezza modello (oncologo)', pct((m0.acc['oncologo']||{}).intact)]);
   t.push(['Mapper (riferimento)', pct(m0.mapper_acc)]);
   t.push(['Flip etichetta per ablazione', (m0.ablation.flip_rate!=null?(m0.ablation.flip_rate*100).toFixed(0)+'%':'–')]);
 }
 document.getElementById('tiles').innerHTML=t.map(([k,v])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');

 // table
 const rows=DATA.sample_rows;
 function draw(f){
   const q=(f||'').toLowerCase();
   const rs=rows.filter(r=>!q||(r.text+r.model+r.gold+r.category).toLowerCase().includes(q));
   document.getElementById('tbl').innerHTML=
     '<tr><th>frase</th><th>fr.</th><th>gold</th><th>modello</th><th>conf</th><th>ok</th><th>mapper</th><th>emo z</th></tr>'+
     rs.map(r=>`<tr><td>${r.text}</td>`+
       `<td><span class="pill ${r.framing==='emotional'?'emo':'neu'}">${r.framing==='emotional'?'emo':'neu'}</span></td>`+
       `<td class="n">${r.gold}</td><td class="n">${r.model}</td><td class="n">${r.conf==null?'':(r.conf*100).toFixed(0)+'%'}</td>`+
       `<td>${r.correct===true?'<span class=ok>✓</span>':r.correct===false?'<span class=no>✗</span>':'·'}</td>`+
       `<td class="n">${r.mapper||'–'}</td><td class="n">${r.emo_z>=0?'+':''}${r.emo_z}</td></tr>`).join('');
 }
 draw('');
 document.getElementById('q').addEventListener('input',e=>draw(e.target.value));

 // --- section 7: per-item x per-model classification matrix ---
 const CX=DATA.class_matrix||{models:[],rows:[]};
 function esc(s){return String(s==null?'':s).replace(/"/g,'&quot;').replace(/</g,'&lt;');}
 function cellHtml(c,goldId,goldClass){
   if(!c) return '<td class="n" style="color:var(--faint)">·</td>';
   const title=esc((c.gen?('genera: "'+c.gen+'"'):'')+(c.term?('  ['+c.term+']'):''));
   let cls='',txt=c.id||'–';
   if(goldClass==='term'){ cls=c.correct===true?'ok':(c.correct===false?'no':''); }
   else { // abstain item: coding a term = false positive (orange); abstained = grey ok
     if(c.matched){cls='fp';txt=c.id;} else {return `<td class="n" title="${title}" style="color:var(--good)">–</td>`;}
   }
   return `<td class="n ${cls}" title="${title}">${txt}</td>`;
 }
 function drawMtx(f){
   const q=(f||'').toLowerCase();
   const rs=CX.rows.filter(r=>!q||(r.text+r.gold+r.category+r.cells.map(c=>c&&c.id).join(' ')).toLowerCase().includes(q));
   const head='<tr><th>frase</th><th>fr.</th><th>gold</th>'+CX.models.map(m=>`<th>${m}</th>`).join('')+'</tr>';
   document.getElementById('mtx').innerHTML=head+rs.map(r=>
     `<tr><td>${esc(r.text)}</td>`+
     `<td><span class="pill ${r.framing==='emotional'?'emo':'neu'}">${r.framing==='emotional'?'emo':'neu'}</span></td>`+
     `<td class="n">${r.gold}</td>`+
     r.cells.map(c=>cellHtml(c,r.gold_id,r.gold_class)).join('')+'</tr>').join('');
 }
 drawMtx('');
 const mq=document.getElementById('mq'); if(mq) mq.addEventListener('input',e=>drawMtx(e.target.value));

 document.getElementById('foot').textContent=
   'oncoemotion · esperimento ruolo × emotività · '+M.map(m=>m.name).join(' · ')+' · nessun claim di coscienza.';

 window.addEventListener('resize',()=>{
   grouped('chEmo',300,roles,(m,gi)=>(m.emo[roles[gi]]||{}).all,-emoMax,emoMax,zf);
   grouped('chAcc',300,roles,(m,gi)=>(m.acc[roles[gi]]||{}).intact,0,1,pct,{ablat:(m,gi)=>(m.acc[roles[gi]]||{}).ablated});
   twoSeries('chWith',300,M.map(m=>m.name),(g,gi)=>M[gi].framing.neutral_acc,(g,gi)=>M[gi].framing.emotional_acc,'--m1','--m2',0,1,pct);
   twoSeries('chErr',240,M.map(m=>m.name),(g,gi)=>M[gi].emo_err.emo_z_on_correct,(g,gi)=>M[gi].emo_err.emo_z_on_wrong,'--good','--bad',-0.5,1,zf);
   grouped('chFp',260,roles,(m,gi)=>(m.fp[roles[gi]]||{}).fp,0,1,pct);
 });
 new MutationObserver(()=>location.reload?null:null).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
})();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=_ROOT / "outputs/role_emotion")
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs/reports/role_emotion_report.html")
    args = ap.parse_args()
    data = _collect(args.dir)
    if not data["models"]:
        print(f"No *__analysis.json found in {args.dir}. Run analyze_role_emotion.py first.")
        return 1
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({len(html)//1024} KB) — {len(data['models'])} model(s), "
          f"{len(data['sample_rows'])} table rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
