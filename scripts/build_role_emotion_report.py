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
    if "qwen3" in s: return "Qwen3-8B"
    if "qwen2.5" in s or "qwen2" in s: return "Qwen2.5-3B"
    if "ministral" in s: return "Ministral-8B"
    if "gemma" in s: return "Gemma-4-12B"
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
    return {"models": models, "sample_rows": sample_rows,
            "role_label": ROLE_LABEL}


TEMPLATE = r"""<!doctype html><html lang=it><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>oncoemotion — ruolo & emotività</title><style>
:root{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;
--m0:#0e7490;--m1:#2f6fd0;--m2:#e08a1e;--good:#15803d;--bad:#b91c1c;--zero:#98a2b3;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;
--m0:#22d3ee;--m1:#5b8def;--m2:#f0a94a;--good:#4ade80;--bad:#f87171;--zero:#5a6473;}}
:root[data-theme="light"]{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;--m0:#0e7490;--m1:#2f6fd0;--m2:#e08a1e;--good:#15803d;--bad:#b91c1c;--zero:#98a2b3;}
:root[data-theme="dark"]{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;--m0:#22d3ee;--m1:#5b8def;--m2:#f0a94a;--good:#4ade80;--bad:#f87171;--zero:#5a6473;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55}
.wrap{max-width:960px;margin:0 auto;padding:30px 20px 70px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--m0);margin:0 0 8px}
h1{font-size:clamp(22px,3.6vw,31px);line-height:1.14;margin:0 0 8px;font-weight:680;text-wrap:balance}
.sub{color:var(--muted);font-size:15px;margin:0;max-width:74ch}
.disc{font-size:12px;color:var(--faint);font-style:italic;margin-top:10px}
h2{font-size:17px;margin:34px 0 2px;font-weight:640}
.q{color:var(--muted);font-size:13.5px;margin:2px 0 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-top:12px;box-shadow:0 1px 2px rgba(20,25,35,.05),0 10px 26px rgba(20,25,35,.05)}
.lbl{font-family:var(--mono);font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}
canvas{width:100%;height:auto;display:block;margin-top:6px}
.legend{display:flex;gap:15px;flex-wrap:wrap;margin-top:10px;font-family:var(--mono);font-size:12px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.cap{font-size:12.5px;color:var(--faint);margin-top:9px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{font-size:11.5px;color:var(--muted)}.tile .v{font-size:20px;font-weight:680;margin-top:3px;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--faint)}
td.n{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--good);font-weight:640}.no{color:var(--bad);font-weight:640}
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
<div id="tiles" class="tiles"></div>

<h2>1 · Il ruolo cambia l'emotività al punto E?</h2>
<p class="q">Media dell'asse affettivo-negativo (paura+ansia+tristezza, z vs baseline neutro), per ruolo — modello intatto.</p>
<div class="card"><span class="lbl">z emotività · per ruolo</span>
<canvas id="chEmo" height="300"></canvas><div class="legend" id="legModels"></div>
<div class="cap" id="capEmo"></div></div>

<h2>2 · L'accuratezza dell'etichettatura cambia col ruolo (e con l'ablazione)?</h2>
<p class="q">Accuratezza del top-1 del modello sugli item EXACT (gold = termine PRO), intatto vs con emotività ablata. Linea tratteggiata = mapper deterministico.</p>
<div class="card"><span class="lbl">accuratezza termine · ruolo × (intatto / ablato)</span>
<canvas id="chAcc" height="300"></canvas><div class="legend" id="legAcc"></div>
<div class="cap" id="capAcc"></div></div>

<h2>3 · Con emotività vs senza — l'etichetta cambia?</h2>
<p class="q">Due operazionalizzazioni di "senza emotività": <b>framing</b> (stessa clinica, formulazione neutra vs emotiva) e <b>ablazione</b> causale della direzione emotiva. Accuratezza e flip dell'etichetta.</p>
<div class="card"><span class="lbl">accuratezza · framing (neutro vs emotivo) e ablazione (intatto vs ablato)</span>
<canvas id="chWith" height="300"></canvas><div class="legend" id="legWith"></div>
<div class="cap" id="capWith"></div></div>

<h2>4 · L'emotività si accompagna agli errori?</h2>
<p class="q">z emotività media sugli item etichettati <b>correttamente</b> vs <b>sbagliati</b> (modello intatto). r = correlazione punto-biseriale (errore ↔ emotività).</p>
<div class="card"><span class="lbl">z emotività · corretti vs sbagliati</span>
<canvas id="chErr" height="240"></canvas><div class="legend" id="legErr"></div>
<div class="cap" id="capErr"></div></div>

<h2>5 · Coding falso-positivo sugli item da astensione</h2>
<p class="q">Sugli item che NON andrebbero codificati (negati / fuori-scope / urgenti…), il modello è forzato a scegliere un termine: quanto spesso lo fa con alta confidenza? Per ruolo.</p>
<div class="card"><span class="lbl">tasso di coding falso-positivo (confidenza > soglia) · per ruolo</span>
<canvas id="chFp" height="260"></canvas><div class="legend" id="legFp2"></div>
<div class="cap" id="capFp"></div></div>

<h2>6 · Come sono state etichettate le cose (ruolo oncologo, intatto)</h2>
<p class="q">Per ogni frase: gold, etichetta del modello (giusto/sbagliato), etichetta del mapper di riferimento, e l'emotività z al punto E. Cerca per testo/termine.</p>
<div class="card">
<input type="search" id="q" placeholder="filtra per testo, termine, categoria…">
<div style="overflow-x:auto"><table id="tbl"></table></div></div>

<div class="foot" id="foot"></div>
</div>
<script>const DATA = /*__DATA__*/;</script>
<script>
(function(){
 const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
 const M=DATA.models, RL=DATA.role_label;
 const mcol=i=>css('--m'+(i%3));
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

 // captions (data-driven)
 const m0=M[0];
 function best(role_metric){let bi=0,bv=-1e9;roles.forEach((r,i)=>{const v=(m0.emo[r]||{}).all;if(v>bv){bv=v;bi=i}});return roles[bi];}
 document.getElementById('capEmo').textContent=
   `Se le barre di uno stesso modello differiscono tra ruoli, il ruolo di sistema sposta l'emotività interna. Ruolo con emotività più alta (${m0.name}): ${RL[best()]||best()}.`;
 document.getElementById('capAcc').textContent=
   `Barra piena = intatto; trattino = con emotività ablata. Se coincidono, rimuovere l'emotività non cambia l'accuratezza. Mapper di riferimento: ${M.map(m=>m.name+' '+pct(m.mapper_acc)).join(' · ')}.`;
 document.getElementById('capWith').textContent=
   `Framing: `+M.map(m=>`${m.name} neutro ${pct(m.framing.neutral_acc)} vs emotivo ${pct(m.framing.emotional_acc)} (${m.framing.label_flips_neutral_vs_emotional}/${m.framing.n_pairs} flip)`).join(' · ')+
   `. Ablazione: `+M.map(m=>`${m.name} ${m.ablation.flip_rate!=null?(m.ablation.flip_rate*100).toFixed(0)+'% flip':'–'}`).join(' · ')+`.`;
 document.getElementById('capErr').textContent=
   M.map(m=>`${m.name}: r(errore,emotività)=${m.emo_err.point_biserial_error_vs_emo==null?'–':m.emo_err.point_biserial_error_vs_emo.toFixed(2)}`).join(' · ')+
   `. Se le barre rossa e verde sono simili, l'emotività non distingue i casi sbagliati.`;
 document.getElementById('capFp').textContent=
   `Soglia di confidenza fissa. Un tasso alto = il modello "codifica" sintomi anche quando dovrebbe astenersi.`;

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
