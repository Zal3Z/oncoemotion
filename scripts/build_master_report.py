#!/usr/bin/env python
"""Build ONE self-contained "master" report combining all three experiments:

  A. Confronto — le emozioni "dentro" i modelli (decodabilità, gravità, confondente,
     persistenza, causalità).
  B. Ruolo × emotività — il ruolo cambia l'emotività? e l'emotività l'etichettatura?
  C. Spettro dei ruoli — perché il ruolo agisce, su 25 emozioni (heatmap inclusa).

Reuses the DATA blobs already computed by the three per-experiment report
generators (extracted from their HTML), so numbers stay identical; renders them in
one page with a shared glossary and verbose "come si legge / cosa dicono i numeri /
in sintesi" captions. Fully offline (inline CSS/JS, canvas), light/dark aware.

Usage:
    python scripts/build_role_spectrum_report.py   # (make sure the 3 reports exist)
    python scripts/build_comparison_report.py
    python scripts/build_role_emotion_report.py
    python scripts/build_master_report.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _extract(path: Path) -> dict:
    h = path.read_text(encoding="utf-8")
    i = h.index("const DATA = ") + len("const DATA = ")
    j = h.index(";</script>", i)
    return json.loads(h[i:j])


def _canon(models, key=lambda m: (m.get("nm") or m.get("name") or m.get("slug") or "")):
    """Reorder models to a canonical Qwen3 · Ministral · Gemma sequence."""
    rank = {"qwen": 0, "ministral": 1, "gemma": 2}
    def r(m):
        s = key(m).lower()
        for k, v in rank.items():
            if k in s:
                return v
        return 9
    return sorted(models, key=r)


TEMPLATE = r"""<!doctype html><html lang=it><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>oncoemotion — report completo</title><style>
:root{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;
--m0:#009E73;--m1:#7570B3;--m2:#8B5A00;--m3:#CC79A7;--m4:#E69F00;--m5:#049292;--m6:#D55E00;--m7:#0072B2;--m8:#56B4E9;--good:#15803d;--bad:#b91c1c;--zero:#98a2b3;
--gMed:#0e7490;--gTec:#2f6fd0;--gPro:#dc2626;--gCon:#6b7280;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;
--m0:#C05F97;--m1:#BE8A1E;--m2:#0FA3A4;--m3:#D95F2B;--m4:#4A90D9;--m5:#A0641A;--m6:#7A6FC8;--m7:#199E72;--m8:#9BC53D;--good:#4ade80;--bad:#f87171;--zero:#5a6473;--gMed:#22d3ee;--gTec:#5b8def;--gPro:#f87171;--gCon:#9aa4b5;}}
:root[data-theme="light"]{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;--m0:#009E73;--m1:#7570B3;--m2:#8B5A00;--m3:#CC79A7;--m4:#E69F00;--m5:#049292;--m6:#D55E00;--m7:#0072B2;--m8:#56B4E9;--good:#15803d;--bad:#b91c1c;--zero:#98a2b3;--gMed:#0e7490;--gTec:#2f6fd0;--gPro:#dc2626;--gCon:#6b7280;}
:root[data-theme="dark"]{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;--m0:#C05F97;--m1:#BE8A1E;--m2:#0FA3A4;--m3:#D95F2B;--m4:#4A90D9;--m5:#A0641A;--m6:#7A6FC8;--m7:#199E72;--m8:#9BC53D;--good:#4ade80;--bad:#f87171;--zero:#5a6473;--gMed:#22d3ee;--gTec:#5b8def;--gPro:#f87171;--gCon:#9aa4b5;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55}
.wrap{max-width:980px;margin:0 auto;padding:30px 20px 80px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--m0);margin:0 0 8px}
h1{font-size:clamp(23px,4vw,34px);line-height:1.12;margin:0 0 8px;font-weight:700;text-wrap:balance}
.sub{color:var(--muted);font-size:15.5px;margin:0;max-width:82ch}
.lead{color:var(--ink);font-size:15px;line-height:1.68;margin:14px 0 0;max-width:82ch}
.disc{font-size:12px;color:var(--faint);font-style:italic;margin-top:10px}
.sec{margin-top:46px;padding-top:18px;border-top:2px solid var(--line)}
.sec>.kicker{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--m0)}
h2{font-size:24px;margin:2px 0 4px;font-weight:680}
h3{font-size:17px;margin:30px 0 2px;font-weight:640}
.q{color:var(--muted);font-size:14px;line-height:1.6;margin:4px 0 0;max-width:82ch}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-top:12px;box-shadow:0 1px 2px rgba(20,25,35,.05),0 10px 26px rgba(20,25,35,.05)}
.lbl{font-family:var(--mono);font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}
canvas{width:100%;height:auto;display:block;margin-top:6px}
.legend{display:flex;gap:15px;flex-wrap:wrap;margin-top:10px;font-family:var(--mono);font-size:12px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.cap{font-size:13.5px;color:var(--muted);margin-top:12px;line-height:1.62;max-width:82ch}
.cap b{color:var(--ink);font-weight:660}.cap .hd{display:block;font-weight:680;color:var(--ink);margin-bottom:3px;margin-top:7px}
.synth{font-size:15px;line-height:1.72;color:var(--ink)}
.synth h4{margin:18px 0 4px;font-size:15.5px;font-weight:720;color:var(--ink)}
.synth h4:first-child{margin-top:0}
.synth p{margin:6px 0 0;max-width:86ch}.synth b{font-weight:660}
.synth .big{background:color-mix(in srgb,var(--m0) 9%,transparent);border-left:3px solid var(--m0);border-radius:0 8px 8px 0;padding:12px 16px;margin-top:14px}
.synth .lim{margin-top:16px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:13.5px}
.gloss{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin-top:16px}
.gloss h4{margin:0 0 8px;font-size:12px;font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--faint);font-weight:600}
.gloss dl{margin:0;display:grid;grid-template-columns:max-content 1fr;gap:7px 14px}
.gloss dt{font-weight:680;color:var(--ink)}.gloss dd{margin:0;color:var(--muted);font-size:13.5px;line-height:1.55}
.toc{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
.toc a{font-size:13px;text-decoration:none;color:var(--ink);border:1px solid var(--line);border-radius:20px;padding:5px 12px;background:var(--panel)}
.toc a:hover{border-color:var(--m0);color:var(--m0)}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--faint)}
td.n{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--good);font-weight:640}.no{color:var(--bad);font-weight:640}.fp{color:var(--m2);font-weight:640}
#B_mtx td,#B_mtx th{white-space:nowrap}#B_mtx td:first-child{white-space:normal;min-width:220px}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;font-family:var(--mono)}
.pill.emo{background:color-mix(in srgb,var(--m2) 20%,transparent);color:var(--m2)}
.pill.neu{background:color-mix(in srgb,var(--m1) 16%,transparent);color:var(--m1)}
input[type=search]{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit;margin-top:8px}
.foot{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
</style></head><body><div class="wrap">
<p class="eyebrow">mechanistic interpretability · report completo</p>
<h1>Emozioni dentro __NMODELS__ modelli: rappresentazione, ruolo, etichettatura</h1>
<p class="sub">Un unico documento con i tre esperimenti su <b>__NMODELS__ modelli</b> — Qwen (Cina),
Ministral · EuroLLM · Apertus (Europa), Gemma (USA) — incluse <b>__NPAIRS__ coppie controllate</b>
base ↔ medicalizzato (MeditronFO, EPFL): (A) le emozioni sono <b>rappresentate</b> durante la codifica
clinica? (B) il <b>ruolo</b> assegnato le cambia, e l'emotività cambia l'<b>etichettatura</b>?
(C) <b>perché</b> il ruolo agisce, sulla tavolozza completa di 25 emozioni?</p>
<p class="disc">Rappresentazioni emotion-<em>like</em>, non emozioni coscienti. Dataset sintetico → indicazioni, non verdetti. Scale diverse tra modelli → si confronta la storia, non i numeri grezzi.</p>
<div class="toc"><a href="#D">Come funziona</a><a href="#A">A · Confronto</a><a href="#B">B · Ruolo × emotività</a><a href="#C">C · Spettro (25 emozioni)</a><a href="#M">M · Medico vs base</a><a href="#E">Player token×layer</a><a href="#Z">Conclusioni</a></div>

<div class="gloss"><h4>Glossario — i termini una volta per tutte</h4><dl>
<dt>Punto E</dt><dd>l'istante appena prima che il modello scriva il termine PRO-CTCAE: lì leggiamo il suo "stato interno".</dd>
<dt>Direzione / vettore emotivo</dt><dd>una retta nello spazio interno che rappresenta un'emozione; proiettarci sopra lo stato = misurare quanto quell\'emozione è "accesa".</dd>
<dt>z-score</dt><dd>quanto un segnale è sopra (o sotto) un testo neutro di riferimento. 0 = come il neutro.</dd>
<dt>AUROC</dt><dd>quanto una direzione distingue la sua emozione da tutte le altre (0.5 = a caso, 1.0 = perfetta).</dd>
<dt>Ruolo</dt><dd>una frase di sistema che dà un'identità al modello (oncologo, ingegnere, bambino…).</dd>
<dt>Framing</dt><dd>come è scritta la frase: neutra ("ho nausea") vs emotiva ("una nausea tremenda") — stesso sintomo.</dd>
<dt>Ablazione</dt><dd>rimozione chirurgica della direzione emotiva dal calcolo, per vedere se la decisione cambia "senza emozione" (test causale).</dd>
<dt>Modello vs mapper</dt><dd>l'etichetta la sceglie il <em>modello</em> (può variare con ruolo/emozioni); il <em>mapper</em> è un programma deterministico che guarda solo il testo — riferimento "sicuro".</dd>
<dt>EXACT / astensione</dt><dd>EXACT = esiste un termine PRO giusto atteso; astensione = non va assegnato nulla (frasi negate, fuori tema, urgenti…).</dd>
</dl></div>
<!--SECTIONS--></div>
<script>const DATA = /*__DATA__*/;</script>
<script>
(function(){
 const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
 const hd=t=>`<span class="hd">${t}</span>`;
 const pct=v=>v==null?'–':(v*100).toFixed(0)+'%';
 const zf=v=>v==null?'–':(v>=0?'+':'')+(+v).toFixed(2);
 const set=(id,html)=>{const e=document.getElementById(id);if(e)e.innerHTML=html;};
 function fit(cv,h){const d=window.devicePixelRatio||1,r=cv.getBoundingClientRect();cv.width=Math.max(1,r.width*d);cv.height=h*d;const x=cv.getContext('2d');x.setTransform(d,0,0,d,0,0);return {w:r.width,h,x};}
 function axes(x,w,h,pL,pT,pB,pR,ymin,ymax,fmt){x.strokeStyle=css('--grid');x.fillStyle=css('--faint');x.font='10px '+css('--mono');x.textAlign='right';
   const Y=v=>pT+(h-pT-pB)*(1-(v-ymin)/(ymax-ymin));
   for(let t=0;t<=4;t++){const v=ymin+(ymax-ymin)*t/4,y=Y(v);x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();x.fillText(fmt(v),pL-5,y+3);}
   if(ymin<0&&ymax>0){x.strokeStyle=css('--zero');const y0=Y(0);x.beginPath();x.moveTo(pL,y0);x.lineTo(w-pR,y0);x.stroke();}return Y;}
 const mcol=i=>css('--m'+(((i%9)+9)%9));

 // ---- generic grouped bars: groups on x, series = models (colored by index) ----
 function grouped(cvid,H,models,groups,getv,ymin,ymax,fmt,labelFn,opts){
   opts=opts||{};const cv=document.getElementById(cvid);if(!cv)return;const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
   const pL=44,pR=8,pT=10,pB=opts.pB||42,iw=w-pL-pR,gw=iw/groups.length,ns=models.length,bw=Math.min(30,(gw-14)/ns);
   const Y=axes(x,w,h,pL,pT,pB,pR,ymin,ymax,fmt),y0=Y(Math.max(ymin,Math.min(0,ymax)));
   groups.forEach((g,gi)=>{models.forEach((m,mi)=>{const v=getv(m,g,gi,mi);if(v==null)return;const bx=pL+gi*gw+(gw-bw*ns)/2+mi*bw,y=Y(v);
       x.fillStyle=colOf(nameOf(m));x.fillRect(bx,Math.min(y,y0),bw-2,Math.abs(y-y0)||1);
       if(opts.dash){const va=opts.dash(m,g,gi,mi);if(va!=null){const ya=Y(va);x.strokeStyle=css('--zero');x.setLineDash([3,3]);x.lineWidth=1.5;x.beginPath();x.moveTo(bx,ya);x.lineTo(bx+bw-2,ya);x.stroke();x.setLineDash([]);x.lineWidth=1;}}});
     x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';
     String(labelFn?labelFn(g):g).split(/\s+/).forEach((wd,k)=>x.fillText(wd,pL+gi*gw+gw/2,h-pB+16+k*12));});}
 // ---- two-series bars (labels x) ----
 function twoSeries(cvid,H,labels,vA,vB,cA,cB,ymin,ymax,fmt){const cv=document.getElementById(cvid);if(!cv)return;const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
   const pL=44,pR=8,pT=10,pB=42,iw=w-pL-pR,gw=iw/labels.length,bw=Math.min(34,(gw-14)/2);
   const Y=axes(x,w,h,pL,pT,pB,pR,ymin,ymax,fmt),y0=Y(Math.max(ymin,Math.min(0,ymax)));
   labels.forEach((g,gi)=>{[[vA(g,gi),cA],[vB(g,gi),cB]].forEach((pr,k)=>{const v=pr[0];if(v==null)return;const bx=pL+gi*gw+(gw-bw*2)/2+k*bw,y=Y(v);
       x.fillStyle=css(pr[1]);x.fillRect(bx,Math.min(y,y0),bw-2,Math.abs(y-y0)||1);
       x.fillStyle=css('--ink');x.font='11px '+css('--mono');x.textAlign='center';x.fillText(fmt(v),bx+(bw-2)/2,y-4);});
     x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';x.fillText(g,pL+gi*gw+gw/2,h-16);});}
 // ---- single bar per model ----
 function singleBars(cvid,H,models,getv,ymin,ymax,fmt,refLine){const cv=document.getElementById(cvid);if(!cv)return;const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
   const pL=44,pR=8,pT=10,pB=28,iw=w-pL-pR,gw=iw/models.length;const Y=axes(x,w,h,pL,pT,pB,pR,ymin,ymax,fmt),y0=Y(Math.max(ymin,Math.min(0,ymax)));
   if(refLine!=null){x.strokeStyle=css('--zero');x.setLineDash([4,4]);const yr=Y(refLine);x.beginPath();x.moveTo(pL,yr);x.lineTo(w-pR,yr);x.stroke();x.setLineDash([]);}
   models.forEach((m,i)=>{const v=getv(m);if(v==null)return;const bw=Math.min(70,gw*.5),bx=pL+i*gw+(gw-bw)/2,y=Y(v);x.fillStyle=colOf(nameOf(m));x.fillRect(bx,Math.min(y,y0),bw,Math.abs(y-y0)||1);
     x.fillStyle=css('--ink');x.font='12px '+css('--mono');x.textAlign='center';x.fillText(fmt(v),bx+bw/2,(v>=0?y-5:y+14));
     x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.fillText(m.nm||m.name,bx+bw/2,h-9);});}
 // ---- causal pair (emo vs random) per model, with flips ----
 function causal(cvid,H,models){const cv=document.getElementById(cvid);if(!cv)return;const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
   const pL=44,pR=8,pT=10,pB=30,iw=w-pL-pR,mx=Math.max(0.5,...models.map(m=>Math.max(m.steer.emo,m.steer.rnd)))*1.1,gw=iw/models.length;
   const Y=axes(x,w,h,pL,pT,pB,pR,0,mx,v=>v.toFixed(1)),bw=Math.min(30,(gw-16)/2);
   models.forEach((m,i)=>{const base=pL+i*gw+(gw-bw*2-6)/2;
     x.fillStyle=css('--m0');x.fillRect(base,Y(m.steer.emo),bw,Y(0)-Y(m.steer.emo));
     x.fillStyle=css('--zero');x.fillRect(base+bw+6,Y(m.steer.rnd),bw,Y(0)-Y(m.steer.rnd));
     x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';x.fillText(m.nm||m.name,base+bw,h-9);
     x.fillStyle=css('--faint');x.font='10px '+css('--mono');x.fillText(m.steer.flips+' flip',base+bw,pT+10);});}
 // ---- line chart (personas x, models = lines) ----
 function lines(cvid,H,models,order,getv,labelOf,colorLabel,ymin,ymax,fmt){const cv=document.getElementById(cvid);if(!cv)return;const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
   const pL=40,pR=10,pT=12,pB=64,iw=w-pL-pR,step=iw/(order.length-1||1);const Y=axes(x,w,h,pL,pT,pB,pR,ymin,ymax,fmt);
   order.forEach((r,i)=>{const px=pL+i*step;x.save();x.translate(px,h-pB+8);x.rotate(-Math.PI/4);x.fillStyle=colorLabel(r);x.font='10px '+css('--sans');x.textAlign='right';x.fillText(labelOf(r),0,0);x.restore();});
   models.forEach((m,mi)=>{x.strokeStyle=colOf(nameOf(m));x.lineWidth=2;x.beginPath();let st=false;
     order.forEach((r,i)=>{const v=getv(m,r);if(v==null)return;const px=pL+i*step,py=Y(v);if(!st){x.moveTo(px,py);st=true;}else x.lineTo(px,py);});x.stroke();
     order.forEach((r,i)=>{const v=getv(m,r);if(v==null)return;x.fillStyle=colOf(nameOf(m));x.beginPath();x.arc(pL+i*step,Y(v),3,0,6.283);x.fill();});});x.lineWidth=1;}
 // ---- heatmap emotion(row) x group(col) ----
 function heatmap(cvid,rowsArr,rowLabel,cols,getv){const cv=document.getElementById(cvid);if(!cv)return;const rows=rowsArr.length||1;
   const rh=Math.max(14,Math.min(22,520/rows)),H=Math.round(rows*rh+40);cv.setAttribute('height',H);const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
   const pL=104,pT=24,cw=(w-pL-8)/cols.length;const vmax=Math.max(0.5,...rowsArr.flatMap(c=>cols.map(([g])=>Math.abs(getv(g,c)||0))));
   const col=(v)=>{if(v==null)return css('--grid');const t=Math.max(-1,Math.min(1,v/vmax)),a=Math.abs(t),c=t>=0?[220,38,38]:[47,111,208];return `rgba(${c[0]},${c[1]},${c[2]},${(0.10+0.8*a).toFixed(2)})`;};
   x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';cols.forEach(([g,lab],ci)=>x.fillText(lab,pL+ci*cw+cw/2,15));
   rowsArr.forEach((c,ri)=>{const y=pT+ri*rh;x.fillStyle=css('--ink');x.font='11px '+css('--sans');x.textAlign='right';x.fillText(rowLabel(c),pL-6,y+rh/2+3);
     cols.forEach(([g],ci)=>{const v=getv(g,c),bx=pL+ci*cw;x.fillStyle=col(v);x.fillRect(bx+1,y+1,cw-2,rh-2);
       if(v!=null){x.fillStyle=(Math.abs(v)/vmax>0.55)?'#fff':css('--muted');x.font='10px '+css('--mono');x.textAlign='center';x.fillText((v>=0?'+':'')+v.toFixed(1),bx+cw/2,y+rh/2+3);}});});}
 // Clickable legend: click a model to show/hide it in every chart.
 function legModels(id,models,extra){
   const el=document.getElementById(id); if(!el) return;
   el.innerHTML=models.map(m=>{const n=nameOf(m),off=HIDDEN.has(n);
     return `<span class="lg" data-m="${n}" title="clicca per mostrare/nascondere" style="cursor:pointer;user-select:none;opacity:${off?0.35:1};text-decoration:${off?'line-through':'none'}"><span class="sw" style="background:${colOf(n)}"></span>${n}</span>`;
   }).join('')+(extra||'')+'<span style="color:var(--faint)">· clicca per filtrare</span>';
   Array.prototype.forEach.call(el.querySelectorAll('.lg'),function(sp){
     sp.addEventListener('click',function(){
       const n=sp.getAttribute('data-m');
       if(HIDDEN.has(n)){HIDDEN.delete(n);}else{HIDDEN.add(n);}
       if(HIDDEN.size>=ALLNAMES.length){HIDDEN.delete(n);return;}  // almeno uno visibile
       redrawAll();
     });
   });
 }
 const LEGENDS=[];
 function relegend(){LEGENDS.forEach(f=>{try{f();}catch(e){}});}

 const CMP=DATA.cmp, RE=DATA.roleEmo, SP=DATA.spectrum;
 // Colour follows the MODEL IDENTITY (its index in the canonical list), never its
 // position in a filtered array — so hiding a model never repaints the others.
 const ALLNAMES=(function(){const seen=[];[(CMP.models||[]),(RE.models||[]),(SP.models||[])]
   .forEach(a=>a.forEach(m=>{const n=m.nm||m.name;if(n&&!seen.includes(n))seen.push(n);}));return seen;})();
 
 const colOf=n=>mcol(Math.max(0,ALLNAMES.indexOf(n)));
 const nameOf=m=>m.nm||m.name;
 const HIDDEN=new Set();
 const vis=arr=>(arr||[]).filter(m=>!HIDDEN.has(nameOf(m)));
 const DRAW=[];

 /* ===================== SECTION A — CONFRONTO ===================== */
 (function(){const M0=CMP.models;if(!M0||!M0.length)return;let M=M0;
   LEGENDS.push(()=>{legModels('A_legDec',M0);legModels('A_legTr',M0);});
   set('A_legCau',`<span><span class="sw" style="background:${css('--m0')}"></span>emozione (paura)</span><span><span class="sw" style="background:${css('--zero')}"></span>random</span>`);
   const emoMaxTr=1;
   function draw(){
     M=vis(M0); if(!M.length)return;
     grouped('A_chDec',300,M,CMP.concepts,(m,c)=>m.val[c],0.5,1.0,v=>v.toFixed(2),c=>c);
     grouped('A_chTr',300,M,CMP.gradients,(m,c,gi)=>m.trend[gi],-emoMaxTr,emoMaxTr,v=>v.toFixed(1),c=>c);
     singleBars('A_chConf',210,M,m=>m.confound,-1,1,v=>v.toFixed(2),0);
     singleBars('A_chPers',210,M,m=>m.persist,0,Math.max(1.6,...M.map(m=>m.persist))*1.05,v=>v.toFixed(1),1);
     causal('A_chCau',230,M);
   }
   DRAW.push(draw);
   const _cl={afraid:'paura',anxious:'ansia',calm:'calma',sad:'tristezza',surprised:'sorpresa'};
   const _dec=(function(){const vs=[];let w=null;M0.forEach(m=>CMP.concepts.forEach(c=>{const v=(m.val||{})[c];if(v==null)return;vs.push(v);if(!w||v<w.v)w={v:v,m:m.nm,c:c};}));return {n:vs.length,hi:vs.filter(v=>v>=0.9).length,w:w};})();
   set('A_capDec', hd('Come si legge.')+` Ogni gruppo di barre è un'emozione — nel grafico compaiono le etichette inglesi usate internamente: afraid = paura, anxious = ansia, calm = calma, sad = tristezza, surprised = sorpresa — e dentro ogni gruppo c'è una barra per modello. L'altezza è l'AUROC, una misura standard di "leggibilità" che si può raccontare così: pesca due frasi a caso, una che esprime quell'emozione e una che non la esprime, e chiedi al segnale interno del modello quale delle due sia quella emotiva. L'AUROC è la frequenza con cui indovina: 0.5 = tira a caso (per questo la scala parte da lì), 1.0 = non sbaglia mai.`+hd('Cosa dicono i numeri.')+` Su ${_dec.n} combinazioni modello×emozione, ${_dec.hi} superano 0.9 e molte sfiorano 1.0: quasi ogni volta il segnale interno riconosce la frase emotiva. Il valore più basso dell'intera griglia è ${_dec.w?_dec.w.v.toFixed(2):'–'} (${_dec.w?_dec.w.m:''}, ${_dec.w?(_cl[_dec.w.c]||_dec.w.c):''}): anche il caso peggiore resta molto sopra il livello del caso.`+hd('In sintesi.')+` Mentre questi modelli leggono la frase di un paziente per fare tutt'altro — scegliere un codice clinico — nel loro stato interno si forma una rappresentazione delle emozioni così nitida che basta una semplice "direzione" (una retta nello spazio interno) per separare le frasi impaurite da quelle calme quasi senza errori. Nessuno ha chiesto ai modelli di occuparsi di emozioni: la rappresentazione emerge da sola, in tutti e ${M0.length}, di famiglie, dimensioni e paesi diversi. Questo grafico è il fondamento di tutto il resto del report, perché dimostra che "dentro" c'è davvero qualcosa da misurare. Attenzione però a cosa NON dice: che un'emozione sia leggibile non implica che il modello la usi per decidere — quella è la domanda di A5.`);
   const _str=M0.filter(m=>m.trendMean>=0.4), _wk=M0.filter(m=>m.trendMean<0.4);
   set('A_capTr', hd('Come si legge.')+` Abbiamo costruito cinque "gradienti clinici": lo stesso sintomo raccontato a gravità crescente, un gradino alla volta (mobilità, dolore, respiro, nausea, prognosi — ad esempio da "cammino con qualche difficoltà" fino a "sono costretto a letto"). Per ogni gradiente la barra è la correlazione (da −1 a +1) tra il gradino di gravità e la paura interna del modello: barra alta = più il racconto è grave, più la paura interna è accesa; vicino a zero = la paura non segue la gravità; sotto zero = paradossalmente si abbassa quando la gravità sale.`+hd('Cosa dicono i numeri.')+` La media dei cinque gradienti, per modello: ${M0.map(m=>`${m.nm} ${zf(m.trendMean)}`).join(', ')}. Come riferimento pratico: sopra +0.4 il legame è chiaro e visibile a occhio, intorno a 0 non c'è legame. ${_str.length} modelli su ${M0.length} (${_str.map(m=>m.nm).join(', ')}) mostrano il legame chiaro${_wk.length?`; in ${_wk.map(m=>`${m.nm} (${zf(m.trendMean)})`).join(' e ')} la paura interna resta invece simile a tutte le gravità, o su alcuni gradienti addirittura scende`:''}.`+hd('In sintesi.')+` La domanda del grafico è se la paura interna sia un interruttore ("sintomo brutto → allarme acceso") o un termometro graduato. Per la maggior parte dei modelli è un termometro: la rappresentazione della paura cresce man mano che il racconto del paziente peggiora, un comportamento sensato per un sistema che deve pesare la serietà di ciò che legge. Ma non è una proprietà universale: ${_wk.length?`in ${_wk.map(m=>m.nm).join(' e ')} il termometro è piatto o rotto, pur avendo quei modelli (grafico A1) una paura perfettamente leggibile`:`qui tutti i modelli la mostrano`}. Rappresentare un'emozione e graduarla con la gravità sono due capacità diverse — e la seconda non viene automaticamente con la prima.`);
   const _cmax=M0.reduce((b,m)=>Math.abs(m.confound)>Math.abs(b.confound)?m:b,M0[0]);
   set('A_capConf', hd('Come si legge.')+` Questo è un controllo di qualità sulla misura stessa. Il dubbio legittimo: quella che chiamiamo "paura" potrebbe essere in realtà solo un rilevatore generico di "cose spiacevoli". Per verificarlo misuriamo, per ogni modello, la correlazione tra il segnale di paura e un segnale volutamente generico di negatività (la "valenza negativa": quanto il testo è spiacevole in generale, senza distinguere di che emozione si tratti). Se paura e negatività generica fossero la stessa cosa, la barra sarebbe vicina a +1; una barra vicino a 0 — poco sopra o poco sotto — significa che i due segnali sono in gran parte indipendenti, cioè che la paura è un concetto a sé.`+hd('Cosa dicono i numeri.')+` ${M0.map(m=>`${m.nm} ${zf(m.confound)}`).join(', ')}. Il valore più lontano da zero è ${zf(_cmax.confound)} (${_cmax.nm}): anche nel caso peggiore la sovrapposizione è modesta (come soglia di riferimento, sotto |0.3| i due segnali si considerano ben districati; qui quasi tutti i modelli sono dentro o appena oltre).`+hd('In sintesi.')+` Il controllo è superato: in tutti i modelli la direzione della "paura" si sovrappone poco o nulla a un generico "questo testo è negativo". È un punto metodologico, ma decisivo per fidarsi del resto del report: quando nei grafici precedenti e successivi diciamo che "la paura sale con la gravità" o che "il ruolo non sposta la paura", stiamo davvero parlando di una rappresentazione specifica di quell'emozione — non di un vago allarme per le brutte notizie che si accenderebbe con qualunque contenuto spiacevole.`);
   const _pamp=M0.filter(m=>m.persist>1.3);
   set('A_capPers', hd('Come si legge.')+` Un segnale che si accende sul sintomo ma svanisce subito non potrebbe influenzare la decisione, che arriva molte parole dopo. Il test: dopo la frase col sintomo inseriamo una frase "cuscinetto" del tutto neutra (tono amministrativo) e solo dopo facciamo arrivare il modello al punto E, l'istante in cui sceglie il termine. La barra dice quanta parte del segnale di paura è ancora viva in quell'istante: 1.0 (linea tratteggiata) = segnale intatto al 100%; 0.5 = ne sopravvive metà; sopra 1 = strada facendo il segnale si è addirittura rafforzato.`+hd('Cosa dicono i numeri.')+` ${M0.map(m=>`${m.nm} ${m.persist.toFixed(2)}`).join(', ')}. ${_pamp.length?`Quasi tutti conservano una quota tra i tre quarti e la totalità del segnale; ${_pamp.map(m=>`${m.nm} arriva a ${m.persist.toFixed(1)}×`).join(', ')} — non solo non dimentica: amplifica.`:`Tutti i modelli conservano la gran parte del segnale.`}`+hd('In sintesi.')+` La paura attivata dal sintomo non è un lampo legato alle parole appena lette: viene trasportata di parola in parola lungo lo stato interno (il "residual stream", il nastro su cui ogni strato del modello accumula informazione — vedi la guida in cima) e arriva quasi intera all'istante esatto della scelta del termine clinico. È questo che rende sensata la domanda successiva: al momento della decisione un segnale emotivo c'è, presente e misurabile — resta da capire se la decisione lo ascolti (A5).${_pamp.length?` Il caso di ${_pamp.map(m=>m.nm).join(' e ')}, dove il segnale cresce lungo il percorso invece di decadere, mostra inoltre quanto possano differire i "caratteri" interni dei modelli pur partendo dalla stessa lettura.`:''}`);
   const _beat=M0.filter(m=>m.steer.emo>m.steer.rnd), _nbeat=M0.filter(m=>m.steer.emo<=m.steer.rnd);
   set('A_capCau', hd('Come si legge.')+` Fin qui abbiamo solo osservato; questo è l'esperimento in cui interveniamo. Mentre il modello sta per decidere su un caso severo, iniettiamo artificialmente nel suo stato interno una spinta lungo la direzione della paura ("provane di più") e misuriamo quanto si scombussola la risposta: la barra è il massimo cambiamento di entropia, cioè di quanto si ridistribuiscono le probabilità tra i termini candidati. La scritta "flip" conta quante volte, nell'insieme delle prove di steering, è cambiato proprio il termine in cima alla classifica. Il confronto cruciale è con la barra grigia: la stessa spinta, di pari intensità, ma in una direzione casuale. Solo se la barra della paura supera nettamente la grigia l'effetto è attribuibile alla paura in quanto tale, e non al semplice disturbo di essere stati "toccati".`+hd('Cosa dicono i numeri.')+` ${M0.map(m=>`${m.nm}: paura ${m.steer.emo.toFixed(2)} vs caso ${m.steer.rnd.toFixed(2)} (${m.steer.flips} flip)`).join('; ')}. In ${_beat.length} modelli su ${M0.length}${_beat.length?` (${_beat.map(m=>m.nm).join(', ')})`:''} la spinta lungo la paura muove la risposta più di quella casuale, anche 2–3 volte tanto; in ${_nbeat.length?_nbeat.map(m=>m.nm).join(', '):'nessuno'} invece no — e dove la barra grigia domina, qualunque disturbo scombussola la risposta più della paura.`+hd('In sintesi.')+` Questo grafico ridimensiona ciò che i precedenti potevano far sperare (o temere). Le emozioni sono rappresentate (A1), graduate con la gravità (A2), distinte dalla negatività generica (A3) e trasportate fino alla decisione (A4) — ma quando proviamo a usarle come leva per pilotare la risposta, l'effetto è modesto e incoerente tra modelli: in alcuni la paura muove qualcosa più del caso, in altri no, e i cambi di etichetta veri e propri restano pochi. In nessun modello emerge il quadro "aumenti la paura e la codifica cambia in modo sistematico". La conclusione onesta: le rappresentazioni emotive esistono e arrivano al momento della scelta, ma la prova che la GUIDINO in modo emotivo-specifico è debole. Per un sistema clinico è una notizia rassicurante: lo stato emotivo interno non è il burattinaio della codifica.`);
 })();

 /* ===================== SECTION B — RUOLO × EMOTIVITÀ ===================== */
 (function(){const M0=RE.models;if(!M0||!M0.length)return;let M=M0;const RL=RE.role_label||{};const roles=M0[0].roles;
   LEGENDS.push(()=>{legModels('B_legEmo',M0);legModels('B_legFp',M0);set('B_legAcc',M0.map(m=>`<span><span class="sw" style="background:${colOf(nameOf(m))}"></span>${m.name}</span>`).join('')+`<span><span class="sw" style="background:${css('--zero')}"></span>ablato (tratteggio)</span>`);});
   set('B_legWith',`<span><span class="sw" style="background:${css('--m1')}"></span>neutro</span><span><span class="sw" style="background:${css('--m2')}"></span>emotivo</span>`);
   set('B_legErr',`<span><span class="sw" style="background:${css('--good')}"></span>corretti</span><span><span class="sw" style="background:${css('--bad')}"></span>sbagliati</span>`);
   const emoMax=Math.max(0.5,...M.flatMap(m=>roles.map(r=>Math.abs((m.emo[r]||{}).all||0))))*1.2;
   const errMax=Math.max(0.5,...M.flatMap(m=>[m.emo_err.emo_z_on_correct,m.emo_err.emo_z_on_wrong].map(v=>Math.abs(v||0))))*1.2;
   function draw(){
     M=vis(M0); if(!M.length)return;
     grouped('B_chEmo',300,M,roles,(m,r)=>(m.emo[r]||{}).all,-emoMax,emoMax,zf,r=>RL[r]||r,{pB:46});
     grouped('B_chAcc',300,M,roles,(m,r)=>(m.acc[r]||{}).intact,0,1,pct,r=>RL[r]||r,{pB:46,dash:(m,r)=>(m.acc[r]||{}).ablated});
     twoSeries('B_chWith',300,M.map(m=>m.name),(g,gi)=>M[gi].framing.neutral_acc,(g,gi)=>M[gi].framing.emotional_acc,'--m1','--m2',0,1,pct);
     twoSeries('B_chErr',240,M.map(m=>m.name),(g,gi)=>M[gi].emo_err.emo_z_on_correct,(g,gi)=>M[gi].emo_err.emo_z_on_wrong,'--good','--bad',-errMax,errMax,zf);
     grouped('B_chFp',260,M,roles,(m,r)=>(m.fp[r]||{}).fp,0,1,pct,r=>RL[r]||r,{pB:46});
   }
   DRAW.push(draw);
   const g=(m,r,k)=>{const rs=Object.keys(m.emo);return (m[k][r]||{});};
   const _bd=M0.filter(m=>(((m.emo.none||{}).all||0)-((m.emo.oncologo||{}).all||0))>0.15), _bu=M0.filter(m=>(((m.emo.oncologo||{}).all||0)-((m.emo.none||{}).all||0))>0.15), _bf=M0.filter(m=>Math.abs(((m.emo.oncologo||{}).all||0)-((m.emo.none||{}).all||0))<=0.15);
   set('B_capEmo', hd('Come si legge.')+` Il modello legge le stesse frasi-sintomo, ma nel prompt gli abbiamo assegnato identità diverse: "sei un oncologo", "sei un assistente" (non medico), oppure nessun ruolo. La barra è l'emotività interna al punto E — la somma dei tre segnali negativi principali (paura + ansia + tristezza) — espressa come z-score: 0 = il livello di un testo neutro di riferimento, valori più alti = stato interno più "carico". Confronta le barre dello stesso colore (stesso modello) tra i tre gruppi: se il ruolo non contasse nulla, sarebbero identiche.`+hd('Cosa dicono i numeri.')+` ${M0.map(m=>`<b>${m.name}</b>: oncologo ${zf((m.emo.oncologo||{}).all)}, non-medico ${zf((m.emo.generico||{}).all)}, nessun ruolo ${zf((m.emo.none||{}).all)}`).join('; ')}. Rispetto al "nessun ruolo", il camice da oncologo abbassa l'emotività in ${_bd.length?_bd.map(m=>m.name).join(', '):'nessun modello'}; la alza in ${_bu.length?_bu.map(m=>m.name).join(', '):'nessun modello'}; la lascia quasi invariata in ${_bf.length?_bf.map(m=>m.name).join(', '):'nessun modello'}.`+hd('In sintesi.')+` Il fatto notevole è che una singola frase nel prompt ("sei un oncologo") raggiunge davvero lo stato emotivo interno: le barre si muovono. Ma la direzione non è la stessa per tutti i modelli: in alcuni il camice "professionalizza" e smorza, in altri accende (forse perché il contesto medico rende il sintomo più serio), in altri ancora non cambia quasi nulla. E le differenze, misurate su queste sole tre emozioni negative, restano piccole rispetto al livello di partenza. È il primo indizio di un tema che tornerà per tutto il report: chiedere "il ruolo alza o abbassa l'emotività?" è la domanda sbagliata — il ruolo RIDISTRIBUISCE le emozioni più che alzarle o abbassarle in blocco, e per vederlo serve la tavolozza completa di 25 emozioni della Sezione C.`);
   const _mbeat=M0.filter(m=>Math.max(...roles.map(r=>(m.acc[r]||{}).intact||0))>m.mapper_acc);
   set('B_capAcc', hd('Come si legge.')+` Qui si passa dallo stato interno al comportamento. Sugli item EXACT (frasi per cui esiste un termine PRO-CTCAE giusto) misuriamo quante volte il modello sceglie proprio quel termine. Barra piena = modello normale; trattino tratteggiato sovrapposto = lo stesso modello con l'emotività "ablata", cioè con la componente emotiva rimossa chirurgicamente dallo stato interno durante il calcolo. L'asticella di paragone è il mapper deterministico — un programma tradizionale a parole-chiave, cieco al contesto e alle emozioni, che su questi item fa ${pct(M0[0].mapper_acc)} (non è disegnato nel grafico: tienilo a mente come riferimento).`+hd('Cosa dicono i numeri.')+` Il miglior risultato di ciascun modello: ${M0.map(m=>`${m.name} ${pct(Math.max(...roles.map(r=>(m.acc[r]||{}).intact||0)))}`).join(', ')}, contro il ${pct(M0[0].mapper_acc)} del mapper: ${_mbeat.length} modelli su ${M0.length} lo superano${_mbeat.length<M0.length?`, mentre ${M0.filter(m=>_mbeat.indexOf(m)<0).map(m=>m.name).join(' e ')} restano sotto`:''}. Le differenze tra ruoli, e tra barra piena e trattino (con/senza emotività), restano nell'ordine di pochi punti percentuali, senza una direzione sistematica.`+hd('In sintesi.')+` Due messaggi. Primo: il compito è difficile — una frase colloquiale va mappata su circa 70 termini tecnici — e i modelli linguistici, capendo il senso e non solo le parole, per lo più battono il programma a parole-chiave; ma non tutti, e la capacità di codifica varia enormemente da modello a modello. Secondo, ed è il vero punto del grafico: né il ruolo assegnato né la rimozione dell'emotività spostano granché l'accuratezza. Se l'emozione interna fosse un ingrediente decisivo del "rispondere giusto", ablarla dovrebbe far crollare (o salire) le barre: non succede. Come vedremo in B3, l'informazione emotiva partecipa alla scelta dell'etichetta, ma non è ciò che separa una codifica giusta da una sbagliata.`);
   const _fw=M0.filter(m=>m.framing.emotional_acc<m.framing.neutral_acc-0.001);
   const _fmax=M0.reduce((b,m)=>((m.framing.neutral_acc-m.framing.emotional_acc)>(b.framing.neutral_acc-b.framing.emotional_acc)?m:b),M0[0]);
   set('B_capWith', hd('Come si legge.')+` L'esperimento più controllato della sezione: ${M0[0].framing.n_pairs} coppie di frasi in cui il contenuto clinico è IDENTICO e cambia solo il tono — versione neutra ("da qualche giorno ho nausea") contro versione emotiva ("questa nausea tremenda mi sta distruggendo, non ne posso più"). Barra blu = accuratezza sulle versioni neutre; arancione = sulle versioni emotive. Poiché il sintomo è lo stesso, qualunque differenza tra blu e arancione è dovuta esclusivamente a COME il paziente si esprime.`+hd('Cosa dicono i numeri.')+` ${M0.map(m=>`${m.name} ${pct(m.framing.neutral_acc)}→${pct(m.framing.emotional_acc)} (${m.framing.label_flips_neutral_vs_emotional} coppie su ${m.framing.n_pairs} ricevono etichette diverse)`).join('; ')}. La versione emotiva è codificata peggio in ${_fw.length} modelli su ${M0.length}; il calo più netto è di ${_fmax.name} (${pct(_fmax.framing.neutral_acc)}→${pct(_fmax.framing.emotional_acc)}). C'è di più: rimuovendo chirurgicamente l'emotività interna (ablazione) l'etichetta cambia nel ${M0.map(m=>`${(m.ablation.flip_rate*100).toFixed(0)}% (${m.name})`).join(', ')} dei casi.`+hd('In sintesi.')+` È l'effetto più robusto e consistente dell'intero studio: quando lo stesso sintomo è raccontato con parole cariche di emozione, la codifica peggiora quasi ovunque — il tono agisce come rumore che disturba la parte "tecnica" del compito. E i flip da ablazione dimostrano che non è un fenomeno di superficie: l'informazione emotiva entra materialmente nel processo che sceglie l'etichetta, perché rimuovendola una quota consistente di etichette cambia (attenzione: cambiare non significa sbagliare — significa che quell'informazione partecipa alla decisione). La rilevanza pratica è immediata: i pazienti reali scrivono come la versione arancione, non come quella blu. Un sistema di codifica automatica va quindi valutato proprio sul linguaggio emotivo, perché è lì che perde punti.`);
   set('B_capErr', hd('Come si legge.')+` L'accusa da verificare: "quando il modello è più emotivamente attivato, sbaglia di più". Per ogni modello dividiamo le risposte in indovinate (verde) e sbagliate (rosso) e misuriamo l'emotività interna media nei due gruppi di casi. Se l'accusa fosse fondata, le barre rosse dovrebbero essere sistematicamente più alte delle verdi. Il coefficiente r è la correlazione tra errore ed emotività (da −1 a +1): positivo = più emotività si accompagna a più errori; circa 0 = nessun legame; negativo = semmai il modello sbaglia di più nei casi MENO emotivi.`+hd('Cosa dicono i numeri.')+` ${M0.map(m=>`${m.name} r=${m.emo_err.point_biserial_error_vs_emo==null?'–':m.emo_err.point_biserial_error_vs_emo.toFixed(2)}`).join(', ')}. Tutti i valori sono vicini a zero o leggermente negativi: in nessun modello un'emotività alta predice l'errore.`+hd('In sintesi.')+` Le barre verdi e rosse sono quasi identiche in ogni modello: nei casi sbagliati il modello NON era interiormente più "agitato" che nei casi giusti. Messo accanto a B3, questo completa il quadro in modo istruttivo: a far sbagliare è la formulazione emotiva del testo (una proprietà dell'input), non il livello di attivazione emotiva interna del modello (una proprietà dello stato). In altre parole, il modello non "si fa prendere dal panico e sbaglia": è il linguaggio emotivo a rendere il compito più difficile — frasi più lunghe, iperboli, il sintomo nascosto dentro il pathos — ed è lì che si perdono punti. La distinzione conta anche in pratica: per migliorare non serve "calmare" il modello, serve allenarlo sul linguaggio emotivo reale dei pazienti.`);
   const _fpo=M0.filter(m=>{const o=(m.fp.oncologo||{}).fp;return o!=null&&o>=Math.max(((m.fp.generico||{}).fp)||0,((m.fp.none||{}).fp)||0);});
   const _fpAll=M0.flatMap(m=>roles.map(r=>(m.fp[r]||{}).fp).filter(v=>v!=null));
   set('B_capFp', hd('Come si legge.')+` Non tutte le frasi meritano un codice: nel dataset ci sono item "da astensione" — sintomi negati ("la nausea è finalmente passata"), discorsi fuori tema, situazioni urgenti da inoltrare a un medico, non da archiviare. In questi casi la risposta giusta è NON assegnare alcun termine. La barra misura quante volte il modello ne assegna comunque uno (falso positivo), con i tre ruoli nel prompt. Più la barra è bassa, più il modello è prudente.`+hd('Cosa dicono i numeri.')+` ${M0.map(m=>`<b>${m.name}</b>: oncologo ${pct((m.fp.oncologo||{}).fp)}, non-medico ${pct((m.fp.generico||{}).fp)}, nessun ruolo ${pct((m.fp.none||{}).fp)}`).join('; ')}. L'intervallo complessivo va da ${pct(Math.min.apply(null,_fpAll))} a ${pct(Math.max.apply(null,_fpAll))}; in ${_fpo.length} modelli su ${M0.length} il ruolo "oncologo" ha il tasso più alto (o è alla pari col più alto).`+hd('In sintesi.')+` Qui emerge il rischio pratico vero di questi sistemi — che non è "l'emozione fa sbagliare la diagnosi" (B4 lo ha appena escluso), ma la sovra-codifica: in molti modelli circa metà dei casi che andrebbero lasciati stare riceve comunque un'etichetta. In concreto: il paziente scrive che la nausea è passata, e la scheda registra ugualmente "nausea". Colpisce che i modelli più bravi nella codifica (B2) tendano a essere anche i più invadenti qui: la stessa disinvoltura che li fa rispondere bene li spinge a rispondere sempre, anche quando dovrebbero tacere. E in una parte dei modelli il camice da oncologo peggiora la tendenza, come se il ruolo professionale rendesse più propensi a "trovare qualcosa". Per l'uso clinico questo grafico pesa più di tutti gli altri della sezione: la capacità di astenersi è ciò che separa uno strumento utile da uno che riempie le schede di sintomi inesistenti.`);
   // table
   const rows=RE.sample_rows||[];
   function drawTbl(f){const q=(f||'').toLowerCase();const rs=rows.filter(r=>!q||(r.text+r.model+r.gold+r.category).toLowerCase().includes(q));
     set('B_tbl','<tr><th>frase</th><th>fr.</th><th>gold</th><th>modello</th><th>conf</th><th>ok</th><th>mapper</th><th>emo z</th></tr>'+
       rs.map(r=>`<tr><td>${r.text}</td><td><span class="pill ${r.framing==='emotional'?'emo':'neu'}">${r.framing==='emotional'?'emo':'neu'}</span></td><td class="n">${r.gold}</td><td class="n">${r.model}</td><td class="n">${r.conf==null?'':(r.conf*100).toFixed(0)+'%'}</td><td>${r.correct===true?'<span class=ok>✓</span>':r.correct===false?'<span class=no>✗</span>':'·'}</td><td class="n">${r.mapper||'–'}</td><td class="n">${r.emo_z>=0?'+':''}${r.emo_z}</td></tr>`).join(''));}
   drawTbl('');const q=document.getElementById('B_q');if(q)q.addEventListener('input',e=>drawTbl(e.target.value));
   // B7 per-item x per-model classification matrix
   const CX=RE.class_matrix||{models:[],rows:[]};
   const esc=s=>String(s==null?'':s).replace(/"/g,'&quot;').replace(/</g,'&lt;');
   function mcell(c,goldClass){if(!c)return '<td class="n" style="color:var(--faint)">·</td>';
     const title=esc((c.gen?('genera: "'+c.gen+'"'):'')+(c.term?('  ['+c.term+']'):''));
     if(goldClass==='term'){const cl=c.correct===true?'ok':(c.correct===false?'no':'');return `<td class="n ${cl}" title="${title}">${c.id||'–'}</td>`;}
     if(c.matched)return `<td class="n fp" title="${title}">${c.id}</td>`;
     return `<td class="n" title="${title}" style="color:var(--good)">–</td>`;}
   function drawMtx(f){const qq=(f||'').toLowerCase();const rs=CX.rows.filter(r=>!qq||(r.text+r.gold+r.category+r.cells.map(c=>c&&c.id).join(' ')).toLowerCase().includes(qq));
     set('B_mtx','<tr><th>frase</th><th>fr.</th><th>gold</th>'+CX.models.map(m=>`<th>${m}</th>`).join('')+'</tr>'+
       rs.map(r=>`<tr><td>${esc(r.text)}</td><td><span class="pill ${r.framing==='emotional'?'emo':'neu'}">${r.framing==='emotional'?'emo':'neu'}</span></td><td class="n">${r.gold}</td>`+r.cells.map(c=>mcell(c,r.gold_class)).join('')+'</tr>').join(''));}
   drawMtx('');const mq=document.getElementById('B_mq');if(mq)mq.addEventListener('input',e=>drawMtx(e.target.value));
 })();

 /* ===================== SECTION C — SPETTRO (25 emozioni) ===================== */
 (function(){const M0=SP.models;if(!M0||!M0.length)return;let M=M0;const m0=M0[0];const ORDER=SP.order,L=SP.label,EL=SP.emo_label||{};
   const gcol=g=>({medici:css('--gMed'),tecnici:css('--gTec'),profani:css('--gPro'),controlli:css('--gCon')}[g]||css('--zero'));
   LEGENDS.push(()=>{legModels('C_legSpec',M0);legModels('C_legDir',M0);});
   set('C_legBR',`<span><span class="sw" style="background:${css('--m1')}"></span>persona da sola</span><span><span class="sw" style="background:${css('--m2')}"></span>reagendo al sintomo</span>`);
   set('C_legSp',`<span><span class="sw" style="background:${css('--gMed')}"></span>medici</span><span><span class="sw" style="background:${css('--gPro')}"></span>profani</span>`);
   function gmeanZ(m,g,c){const rs=Object.keys(m.groups||{}).filter(r=>m.groups[r]===g);const vs=rs.map(r=>(m.clinical_z[r]||{})[c]).filter(v=>v!=null);return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;}
   function grpMean(m,g,key){const rs=ORDER.filter(r=>(m.groups||{})[r]===g);const vs=rs.map(r=>m[key][r]).filter(v=>v!=null);return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;}
   const SPEC=[['afraid_alarmed','paura'],['anxious_nervous','ansia'],['sad','tristezza'],['clinical_severity','gravità'],['general_negative_valence','val. neg.']];
   function heatEmos(){return (m0.emo_concepts||[]).filter(c=>EL[c]).slice().sort((a,b)=>((gmeanZ(m0,'profani',b)||0)-(gmeanZ(m0,'medici',b)||0))-((gmeanZ(m0,'profani',a)||0)-(gmeanZ(m0,'medici',a)||0)));}
   const emoStdMax=Math.max(1.5,...M.flatMap(m=>ORDER.map(r=>Math.abs(m.emo_std[r]||0))))*1.1;
   const brAll=ORDER.flatMap(r=>[m0.emo_baseline[r],m0.emo_clinical[r]].filter(v=>v!=null));
   const specMax=Math.max(1,...SPEC.flatMap(([c])=>[gmeanZ(m0,'medici',c),gmeanZ(m0,'profani',c)].filter(v=>v!=null).map(Math.abs)))*1.15;
   function draw(){
     M=vis(M0); if(!M.length)return;
     lines('C_chSpec',320,M,ORDER,(m,r)=>m.emo_std[r],r=>L[r]||r,r=>gcol((m0.groups||{})[r]),-emoStdMax,emoStdMax,zf);
     grouped('C_chDir',240,M,[['profani_minus_medici_cos','profani − medici'],['tecnici_minus_medici_cos','tecnici − medici'],['profani_minus_tecnici_cos','profani − tecnici']],
             (m,pair)=>(m.dir_afraid||{})[pair[0]],-1,1,v=>v.toFixed(1),pair=>pair[1]);
     twoSeries('C_chBR',300,ORDER.map(r=>L[r]||r),(g,gi)=>m0.emo_baseline[ORDER[gi]],(g,gi)=>m0.emo_clinical[ORDER[gi]],'--m1','--m2',Math.min(0,...brAll)*1.15,Math.max(1,...brAll.map(Math.abs))*1.15,zf);
     grouped('C_chSp',280,[m0],SPEC,(m,pair)=>null,-specMax,specMax,zf,pair=>pair[1]); // placeholder replaced below
     // specificity as two fixed series (medici vs profani) — custom
     drawSpec();
     heatmap('C_chHeat',heatEmos(),c=>EL[c]||c,[['medici','Medici'],['tecnici','Tecnici'],['profani','Profani'],['controlli','Controlli']],(g,c)=>gmeanZ(m0,g,c));
   }
   function drawSpec(){const cv=document.getElementById('C_chSp');if(!cv)return;const {w,h,x}=fit(cv,280);x.clearRect(0,0,w,h);
     const pL=44,pR=8,pT=10,pB=40,iw=w-pL-pR,gw=iw/SPEC.length,bw=Math.min(30,(gw-14)/2);const Y=axes(x,w,h,pL,pT,pB,pR,-specMax,specMax,zf),y0=Y(0);
     SPEC.forEach(([c,lab],gi)=>{[[gmeanZ(m0,'medici',c),'--gMed'],[gmeanZ(m0,'profani',c),'--gPro']].forEach((pr,k)=>{const v=pr[0];if(v==null)return;const bx=pL+gi*gw+(gw-bw*2)/2+k*bw,y=Y(v);x.fillStyle=css(pr[1]);x.fillRect(bx,Math.min(y,y0),bw-2,Math.abs(y-y0)||1);});
       x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';x.fillText(lab,pL+gi*gw+gw/2,h-14);});}
   DRAW.push(draw);
   const emo=heatEmos(),top=emo.slice(0,3).map(c=>EL[c]||c),bot=emo.slice(-3).map(c=>EL[c]||c);
   const _medHi=M0.filter(m=>{const a=grpMean(m,'medici','emo_clinical'),b=grpMean(m,'profani','emo_clinical');return a!=null&&b!=null&&a>b;});
   set('C_capSpec', hd('Come si legge.')+` Undici personas sull'asse orizzontale — due mediche (oncologo, infermiere), tre tecniche (ingegnere, avvocato, contabile), tre "profane" ed emotive (paziente ansioso, bambino, poeta) e tre di controllo (assistente, empatico, nessun ruolo) — con le etichette colorate per gruppo. Ogni linea è un modello; l'altezza del punto è l'emotività composita del modello mentre reagisce al sintomo con quella persona, standardizzata modello per modello. "Standardizzata" significa: ogni modello ha scale interne tutte sue, quindi esprimiamo ogni punto come scarto dalla media di QUEL modello — così si confrontano le forme delle linee, non i valori assoluti. Sopra lo zero = quella persona rende il modello più emotivo della sua media; sotto = meno.`+hd('Cosa dicono i numeri.')+` Le medie per gruppo di personas: ${M0.map(m=>`<b>${m.name}</b> medici ${zf(grpMean(m,'medici','emo_clinical'))}, tecnici ${zf(grpMean(m,'tecnici','emo_clinical'))}, profani ${zf(grpMean(m,'profani','emo_clinical'))}`).join('; ')} (in z, ognuno sulla propria scala).`+hd('In sintesi.')+` Se valesse l'intuizione "medici freddi, profani caldi", tutte le linee dovrebbero salire ordinatamente passando dalle personas mediche a quelle profane. Non succede: le linee si intrecciano, l'ordinamento cambia da modello a modello${_medHi.length?` e in alcuni (${_medHi.map(m=>m.name).join(', ')}) i ruoli medici risultano addirittura PIÙ emotivi dei profani su questo termometro complessivo`:''}. La lezione del grafico è proprio questa: misurata come "totale unico", l'emotività non distingue in modo affidabile il camice dal paziente. Non perché il ruolo non faccia nulla — ma perché il ruolo cambia il MIX di emozioni accese, e mix diversi possono dare lo stesso totale. Per questo i grafici successivi cambiano domanda: C2 chiede "in che direzione si sposta lo stato interno?", C5 "quali emozioni, esattamente, si accendono e si spengono?".`);
   set('C_capDir', hd('Come si legge.')+` Prendiamo la differenza media di stato interno tra due gruppi di personas — ad esempio "profani meno medici": è una freccia nello spazio interno che indica in che direzione si sposta lo stato del modello quando si toglie il camice. Poi misuriamo l'angolo tra questa freccia e l'asse della paura, espresso come coseno: +1 = lo spostamento è esattamente "più paura"; −1 = esattamente "meno paura"; 0 = lo spostamento è perpendicolare alla paura, cioè qualunque cosa il ruolo cambi, non è la paura. Le tre terne di barre ripetono il confronto per tre coppie di gruppi.`+hd('Cosa dicono i numeri.')+` Per la coppia chiave, profani − medici: ${M0.map(m=>`${m.name} ${zf((m.dir_afraid||{}).profani_minus_medici_cos)}`).join(', ')}. Su una scala che va fino a ±1, tutti i valori restano sotto |0.2|: angoli quasi retti. Lo stesso vale per le altre due coppie di gruppi.`+hd('In sintesi.')+` Questo è il grafico che smonta la spiegazione intuitiva. Verrebbe naturale raccontare così i risultati: "il modello-oncologo si comporta diversamente perché ha meno paura del modello-paziente". Misurando la geometria dello spostamento, la risposta è no: passare da un ruolo profano a uno medico muove lo stato interno in una direzione quasi perfettamente perpendicolare alla paura, in tutti e ${M0.length} i modelli. Il ruolo fa qualcosa di reale — lo spostamento esiste ed è misurabile — ma la sua sostanza non è "più o meno paura": è un cambiamento lungo ALTRE dimensioni affettive. Quali siano è esattamente la domanda a cui risponde la tavolozza completa qui sotto (C5).`);
   const _blv=ORDER.map(r=>m0.emo_baseline[r]).filter(v=>v!=null), _clv=ORDER.map(r=>m0.emo_clinical[r]).filter(v=>v!=null);
   const _avgc=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:0;
   set('C_capBR', hd('Come si legge.')+` Qui il modello mostrato è uno solo, quello di riferimento (<b>${m0.name}</b>). Per ogni persona ci sono due barre: la blu è l'emotività dello stato interno quando il modello ha ricevuto SOLO l'identità ("sei un oncologo"), senza alcun sintomo — la persona "a riposo"; l'arancione è l'emotività dopo che, con quella identità addosso, ha letto la frase col sintomo. Il confronto scompone l'effetto del ruolo in due componenti: quanto "umore di partenza" installa da solo (blu) e quanto colora la reazione al contenuto clinico (arancione).`+hd('Cosa dicono i numeri.')+` Le barre blu stanno in media a ${zf(_avgc(_blv))} (con "nessun ruolo" a 0 per costruzione: è il riferimento); le arancioni saltano in media a ${zf(_avgc(_clv))}. Il grosso dell'emotività arriva quindi col sintomo, per tutte le personas.`+hd('In sintesi.')+` Due osservazioni. La prima: il solo assegnare un'identità — qualunque identità, anche "ingegnere" — sposta già un po' lo stato affettivo del modello prima di qualunque contenuto clinico: le barre blu non sono a zero. Il ruolo non è un'etichetta inerte, predispone. La seconda, più importante: il salto dal blu all'arancione è molto più grande delle differenze fra le barre blu. L'emotività di questi modelli è quindi soprattutto REAZIONE al contenuto letto (il sintomo), che il ruolo modula — non un umore preimpostato che la persona si porta dietro. In termini pratici: il ruolo agisce come un filtro che colora il modo in cui il sintomo viene "sentito", più che come uno stato d'animo costante.`);
   const _spTxt=SPEC.map(pair=>`${pair[1]}: medici ${zf(gmeanZ(m0,'medici',pair[0]))} vs profani ${zf(gmeanZ(m0,'profani',pair[0]))}`).join('; ');
   set('C_capSp', hd('Come si legge.')+` Ancora il modello di riferimento (<b>${m0.name}</b>). Le prime tre coppie di barre confrontano la media dei ruoli medici (teal) e dei profani (rosso) su tre emozioni: paura, ansia, tristezza. Le ultime due coppie sono i CONTROLLI, cioè segnali non emotivi: la gravità clinica percepita e la valenza negativa generica. La logica del controllo: se il ruolo cambiasse "tutto" dello stato interno — attenzione, stile, percezione della serietà del caso — le differenze tra teal e rosso comparirebbero anche sui controlli; se invece il ruolo agisce specificamente sulla sfera affettiva, sulle emozioni le barre divergono e sui controlli restano appaiate.`+hd('Cosa dicono i numeri.')+` ${_spTxt}. Le differenze sulle emozioni sono più ampie di quelle sui controlli — e con segni diversi da emozione a emozione.`+hd('In sintesi.')+` Il ruolo supera il test di specificità: ciò che sposta è prevalentemente affetto, non la percezione della gravità clinica del caso (i controlli restano vicini — importante, perché significa che un modello "in camice" non vede sintomi più o meno gravi, li vive diversamente). E c'è un dettaglio prezioso: le differenze emotive non vanno tutte nello stesso verso. Nel modello di riferimento i ruoli medici mostrano ad esempio più tristezza dei profani, ma meno ansia. Di nuovo: il ruolo non regola un "volume emotivo" unico verso l'alto o verso il basso — ricompone l'equilibrio tra emozioni diverse. Da qui la necessità della heatmap completa qui sotto, dove questa ricomposizione si vede emozione per emozione.`);
   set('C_capHeat', hd('Come si legge.')+` La vista più ricca del report, sul modello di riferimento (<b>${m0.name}</b>): ogni riga è una delle 25 emozioni della tavolozza, ogni colonna uno dei quattro gruppi di personas. Il colore della cella dice quanto quell'emozione è attiva nello stato interno rispetto a un testo neutro: rosso = più attiva, blu = meno attiva, e l'intensità del colore segue la forza dell'effetto (il numero nella cella è lo z-score). Le righe sono ordinate per differenza "profani − medici": in alto le emozioni tipiche dei ruoli profani, in basso quelle tipiche dei ruoli medici. Il modo giusto di leggerla: scorri una riga da sinistra a destra e osserva come cambia il colore passando da un gruppo all'altro.`+hd('Cosa dicono i numeri.')+` Nelle prime righe (più accese nei <b>profani</b> che nei medici): ${top.join(', ')}. Nelle ultime (più accese nei <b>medici</b> che nei profani): ${bot.join(', ')}.`+hd('In sintesi.')+` Ecco, finalmente visibile, che cosa fa davvero il ruolo: non alza né abbassa "l'emotività" — ridistribuisce l'affetto tra emozioni diverse. Nel modello di riferimento le personas profane accendono di più ${top.join(', ')}, mentre le personas mediche mostrano di più ${bot.join(', ')}: un profilo che non corrisponde né al cliché del medico freddo né a quello del profano genericamente più emotivo. Questo mosaico spiega retroattivamente i grafici precedenti: il termometro unico di C1 non poteva vederlo (mix diversi di emozioni possono dare totali simili) e il coseno di C2 dava valori vicini a zero perché lo spostamento non corre lungo la paura ma lungo queste altre dimensioni. I dettagli cambiano da modello a modello (le heatmap per ciascun modello sono nel report C dedicato), ma la morale è comune: per capire l'effetto di un ruolo serve la tavolozza completa — tre emozioni non bastavano.`);
 })();

 /* ===================== M — MEDICO vs BASE (coppie controllate) ===================== */
 const MEDPAIRS=(function(){
   const byName={}; (RE.models||[]).forEach(m=>byName[m.name]=m);
   const pairs=[]; Object.keys(byName).forEach(n=>{
     if(/-MedFO$/.test(n)||/^MedGemma/.test(n)){
       const base = /-MedFO$/.test(n) ? n.replace(/-MedFO$/,'') : 'Gemma-3-27B';
       if(byName[base]) pairs.push({fam:base, base:byName[base], med:byName[n]});
     }});
   return pairs;})();
 (function(){const wrap=document.getElementById('M_cards');if(!wrap)return;
   if(!MEDPAIRS.length){
     wrap.innerHTML='<div class="card"><div class="cap"><span class="hd">Nessuna coppia disponibile.</span> Servono un modello base <b>e</b> la sua versione medicalizzata (es. Apertus-8B e Apertus-8B-MedFO): con uno solo dei due il confronto controllato non è possibile.</div></div>';
     set('M_synth','<p>Nessuna coppia base ↔ medicalizzato completa in questo run.</p>');return;}
   const SEL=SP.emo_label||{}; const spByName={}; (SP.models||[]).forEach(m=>spByName[m.name]=m);
   const emo=m=>(m.emo['oncologo']||{}).all, acc=m=>(m.acc['oncologo']||{}).intact, fp=m=>(m.fp['oncologo']||{}).fp;
   const METRICS=[['emotività (z)',emo,zf],['accuratezza codifica',acc,pct],['falsi positivi',fp,pct],['flip da ablazione',m=>m.ablation.flip_rate,pct]];
   const DIR=(vb,vm,eps)=>{const d=(vm||0)-(vb||0);return Math.abs(d)<eps?0:(d>0?1:-1);};
   const mean=(m,c)=>{if(!m)return null;const P2=m.clinical_z;const vs=Object.keys(P2).map(r=>P2[r][c]).filter(v=>v!=null);return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;};
   wrap.innerHTML=MEDPAIRS.map((P,i)=>`
     <h3>M${i+1} · ${P.base.name} ↔ ${P.med.name}</h3>
     <div class="card"><span class="lbl">base vs medicalizzato · le quattro misure chiave (ruolo oncologo)</span>
       <canvas id="M_ch_${i}" height="300"></canvas><div class="legend" id="M_leg_${i}"></div>
       <div class="cap" id="M_cap_${i}"></div></div>
     <div class="card"><span class="lbl">quali emozioni cambia il training medico · ${P.base.name} → ${P.med.name}</span>
       <canvas id="M_chEmo_${i}" height="330"></canvas><div class="cap" id="M_capEmo_${i}"></div></div>`).join('');
   MEDPAIRS.forEach((P,i)=>{
   const el=document.getElementById('M_ch_'+i);
   function draw(){const {w,h,x}=fit(el,300);x.clearRect(0,0,w,h);
     const pL=44,pR=8,pT=10,pB=44,iw=w-pL-pR,gw=iw/METRICS.length,bw=Math.min(34,(gw-14)/2);
     // normalizza ogni metrica sul suo massimo per poterle mostrare insieme
     const Y0=h-pB, Hh=h-pT-pB;
     x.strokeStyle=css('--grid');for(let t=0;t<=4;t++){const y=pT+Hh*t/4;x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();}
     METRICS.forEach(([lab,f,fmt],gi)=>{const vb=f(P.base),vm=f(P.med);
       const mx=Math.max(Math.abs(vb||0),Math.abs(vm||0))||1;
       [[vb,'--m1'],[vm,'--gPro']].forEach((pr,k)=>{const v=pr[0];if(v==null)return;
         const hgt=Hh*Math.abs(v)/mx*0.92, bx=pL+gi*gw+(gw-bw*2)/2+k*bw;
         x.fillStyle=css(pr[1]);x.fillRect(bx,Y0-hgt,bw-2,hgt);
         x.fillStyle=css('--ink');x.font='11px '+css('--mono');x.textAlign='center';x.fillText(fmt(v),bx+(bw-2)/2,Y0-hgt-5);});
       x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';
       lab.split(' ').forEach((wd,i2)=>x.fillText(wd,pL+gi*gw+gw/2,h-28+i2*12));});}
   DRAW.push(draw);
   set('M_leg_'+i,`<span><span class="sw" style="background:${css('--m1')}"></span>${P.base.name} (base)</span><span><span class="sw" style="background:${css('--gPro')}"></span>${P.med.name} (medicalizzato)</span>`);
   const eDir=DIR(emo(P.base),emo(P.med),0.15), aDir=DIR(acc(P.base),acc(P.med),0.03), fDir=DIR(fp(P.base),fp(P.med),0.03);
   P._e=eDir; P._a=aDir; P._f=fDir;
   const pd=((P.base.emo.oncologo||{}).all||0)-((P.base.emo.none||{}).all||0);
   const pDir=Math.abs(pd)<=0.15?0:(pd>0?1:-1);
   let cmpTxt;
   if(eDir===0) cmpTxt='il training, qui, questo termometro non lo sposta quasi';
   else if(pDir===0) cmpTxt='dove il prompt non faceva quasi nulla, i pesi invece si muovono';
   else if(pDir===eDir) cmpTxt='stessa direzione del prompt, ma ottenuta dai pesi, non da un\'istruzione';
   else cmpTxt='direzione OPPOSTA a quella del ruolo nel prompt: "medico nei pesi" e "medico nel prompt" non sono lo stesso meccanismo';
   let syn;
   if(aDir>0&&fDir>0) syn=`Questa è la coppia in cui il training clinico si vede di più, e in entrambe le direzioni: diventa molto più competente (codifica ${pct(acc(P.base))} → ${pct(acc(P.med))}) ma anche più interventista (falsi positivi ${pct(fp(P.base))} → ${pct(fp(P.med))} sui casi da lasciare in astensione). Più bravo sul compito, meno capace di tacere quando deve.`;
   else if(!eDir&&!aDir&&!fDir) syn=`In questa coppia le quattro misure restano quasi ferme: il training clinico non ha trasformato né la competenza di codifica né la prudenza né il termometro emotivo aggregato. Non significa che non abbia fatto nulla: il grafico qui sotto scompone lo stato per emozione, ed è lì che eventuali cambiamenti si nascondono.`;
   else syn=`In questa coppia il training tocca alcune misure e lascia ferme le altre: ${[aDir?`la codifica ${aDir>0?'migliora':'peggiora'} (${pct(acc(P.base))} → ${pct(acc(P.med))})`:null, fDir?`i falsi positivi ${fDir>0?'salgono':'scendono'} (${pct(fp(P.base))} → ${pct(fp(P.med))})`:null, eDir?`l'emotività ${eDir>0?'sale':'scende'} (${zf(emo(P.base))} → ${zf(emo(P.med))})`:null].filter(Boolean).join('; ')}.`;
   set('M_cap_'+i, (i===0?hd('Come si legge.')+` Ogni coppia di questa sezione confronta un modello base (blu) con la sua versione medicalizzata (rosso): stesso modello di partenza, e la differenza chiave è un ulteriore addestramento su testi clinici (fine-tuning MeditronFO). Tutto il resto dell'esperimento è identico, incluso il ruolo "oncologo" nel prompt, quindi le differenze tra le barre sono attribuibili al training medico. Le quattro coppie di barre confrontano: l'emotività interna al punto E, l'accuratezza di codifica, i falsi positivi sugli item da astensione e la quota di etichette che cambiano ablando l'emotività. Nota tecnica: ogni gruppo è riscalato sul proprio massimo per stare nello stesso grafico, quindi confronta le due barre DENTRO ogni gruppo, non tra gruppi diversi. `:'')+
     hd('Cosa dicono i numeri.')+` Emotività ${zf(emo(P.base))} → ${zf(emo(P.med))} (${eDir>0?'sale':eDir<0?'scende':'quasi invariata'}). Accuratezza di codifica ${pct(acc(P.base))} → ${pct(acc(P.med))} (${aDir>0?'migliora':aDir<0?'peggiora':'invariata'}), contro il ${pct(P.base.mapper_acc)} del mapper deterministico. Falsi positivi ${pct(fp(P.base))} → ${pct(fp(P.med))} (${fDir>0?'salgono':fDir<0?'scendono':'invariati'}). Flip da ablazione ${pct(P.base.ablation.flip_rate)} → ${pct(P.med.ablation.flip_rate)}: il peso dell'emozione nella decisione resta di quest'ordine.`+
     hd('In sintesi.')+` ${syn} Quanto al confronto col ruolo: nel modello base il ruolo-oncologo scritto nel prompt ${pDir<0?'smorzava':pDir>0?'alzava':'quasi non toccava'} l'emotività; il training nei pesi ${eDir>0?'la alza':eDir<0?'la abbassa':'la lascia dov\'era'} — ${cmpTxt}.`);
   // scomposizione per emozione
   const sb=spByName[P.base.name], sm=spByName[P.med.name];
   const el2=document.getElementById('M_chEmo_'+i);
   if(!sb||!sm){set('M_capEmo_'+i,'<span class="hd">Dati spettro non disponibili per questa coppia.</span>');return;}
   const rows=(sb.emo_concepts||[]).filter(c=>SEL[c]&&mean(sb,c)!=null&&mean(sm,c)!=null)
     .map(c=>({c,d:mean(sm,c)-mean(sb,c),b:mean(sb,c),m:mean(sm,c)})).sort((a,b)=>b.d-a.d);
   const top=rows.slice(0,6).concat(rows.slice(-6));
   function draw2(){const H=Math.max(240,top.length*24+40);el2.setAttribute('height',H);const {w,h,x}=fit(el2,H);x.clearRect(0,0,w,h);
     const pL=110,pR=40,iw=w-pL-pR,mx=Math.max(...top.map(r=>Math.abs(r.d)))||1,x0=pL+iw/2;
     x.strokeStyle=css('--zero');x.beginPath();x.moveTo(x0,14);x.lineTo(x0,h-10);x.stroke();
     top.forEach((r,k)=>{const y=22+k*24,wdt=(iw/2)*Math.abs(r.d)/mx*0.94;
       x.fillStyle=r.d>=0?css('--gPro'):css('--m1');
       x.fillRect(r.d>=0?x0:x0-wdt, y-8, wdt, 15);
       x.fillStyle=css('--ink');x.font='11px '+css('--sans');x.textAlign='right';x.fillText(SEL[r.c]||r.c,pL-8,y+4);
       x.fillStyle=css('--muted');x.font='10px '+css('--mono');x.textAlign=r.d>=0?'left':'right';
       x.fillText((r.d>=0?'+':'')+r.d.toFixed(1),(r.d>=0?x0+wdt+4:x0-wdt-4),y+4);});
     x.fillStyle=css('--muted');x.font='10px '+css('--sans');x.textAlign='center';
     x.fillText('← il training medico la ABBASSA', pL+iw/4, h-2); x.fillText('il training medico la ALZA →', x0+iw/4, h-2);}
   DRAW.push(draw2);
   const up=rows.slice(0,3).map(r=>SEL[r.c]||r.c), dn=rows.slice(-3).map(r=>SEL[r.c]||r.c);
   const mag=Math.max(Math.abs(rows[0]?rows[0].d:0),Math.abs(rows[rows.length-1]?rows[rows.length-1].d:0));
   P._up=rows.filter(r=>r.d>=0.8).map(r=>SEL[r.c]||r.c); P._dn=rows.filter(r=>r.d<=-0.8).map(r=>SEL[r.c]||r.c); P._mag=mag;
   set('M_capEmo_'+i, (i===0?hd('Come si legge.')+` Stessa coppia del grafico qui sopra, ma scomposta sulla tavolozza completa: per ciascuna delle 25 emozioni la barra mostra la DIFFERENZA tra la versione medicalizzata e quella base (media su tutte le personas). Barre verso destra (rosse) = il training medico ha reso quell'emozione più attiva; verso sinistra (blu) = l'ha resa meno attiva. Sono mostrate le 6 emozioni più aumentate e le 6 più ridotte; il numero accanto alla barra è la differenza in z-score. `:'')+
     hd('Cosa dicono i numeri.')+` Per questa coppia il training alza soprattutto <b>${up.join(', ')}</b> e abbassa soprattutto <b>${dn.join(', ')}</b>; lo spostamento più grande vale ${mag.toFixed(1)} z.`+
     hd('In sintesi.')+` ${mag>=1?`Anche quando le quattro misure aggregate si muovono poco, qui si vede che il training clinico risistema comunque il MIX di emozioni con cui il modello reagisce al sintomo: meno ${dn.join(', ')}, più ${up.join(', ')}. È il livello a cui la "medicalizzazione" agisce davvero: non un volume unico che sale o scende, ma un equilibrio che si sposta.`:`In questa coppia anche i movimenti per singola emozione sono piccoli (massimo ${mag.toFixed(1)} z): il training clinico ha lasciato quasi intatto il profilo affettivo di questa famiglia.`}`);
   });
   // sintesi trasversale delle coppie
   const nmOf=P=>P.base.name;
   const wDir=(v,up,dn,fl)=>v>0?up:(v<0?dn:fl);
   const inter=(a,b)=>a.filter(x=>b.indexOf(x)>=0);
   const withSpec=MEDPAIRS.filter(P=>P._up!=null);
   const cUp=withSpec.length?withSpec.map(P=>P._up).reduce(inter):[];
   const cDn=withSpec.length?withSpec.map(P=>P._dn).reduce(inter):[];
   const strong=MEDPAIRS.filter(P=>P._a>0&&P._f>0).map(nmOf), quiet=MEDPAIRS.filter(P=>!P._e&&!P._a&&!P._f).map(nmOf);
   let SH='';
   SH+='<h4>'+MEDPAIRS.length+' coppie, non una sola risposta</h4>';
   SH+=`<p>Il riepilogo, coppia per coppia: ${MEDPAIRS.map(P=>`<b>${nmOf(P)}</b> — emotività ${wDir(P._e,'sale','scende','ferma')}, codifica ${wDir(P._a,'migliora','peggiora','ferma')}, falsi positivi ${wDir(P._f,'salgono','scendono','fermi')}`).join('; ')}. Il run precedente aveva una sola coppia e suggeriva un pattern generale; con ${MEDPAIRS.length} coppie su famiglie diverse si vede che l'effetto del fine-tuning clinico <b>cambia da famiglia a famiglia</b>.</p>`;
   SH+=`<p>${strong.length?`Il profilo "più bravo ma meno prudente" — più codifiche giuste E più codifiche a vuoto — emerge in <b>${strong.join(', ')}</b>. `:`Nessuna coppia mostra il profilo "più bravo ma meno prudente". `}${quiet.length?`In <b>${quiet.join(' e ')}</b> le quattro misure aggregate restano invece quasi ferme: lì la "medicalizzazione" si vede solo nel mix di emozioni, non nel comportamento di codifica. `:''}${(cUp.length||cDn.length)?`Un tratto comune a tutte le coppie: il training ${cUp.length?`alza <b>${cUp.join(', ')}</b>`:''}${(cUp.length&&cDn.length)?' e ':''}${cDn.length?`abbassa <b>${cDn.join(', ')}</b>`:''}.`:`E nemmeno sulle singole emozioni c'è uno spostamento condiviso da tutte le coppie: anche il "distacco clinico" prende forme diverse in famiglie diverse.`}</p>`;
   SH+=`<div class="big"><b>La risposta alla domanda del titolo.</b> No: il fine-tuning medico non è un "ruolo oncologo permanente". Non riproduce in modo sistematico l'effetto del ruolo nel prompt (in alcune famiglie va in direzione opposta), e non è nemmeno un effetto unico: dove incide, ricabla insieme competenza, prudenza e profilo affettivo; altrove lascia il comportamento quasi intatto e sposta solo l'equilibrio interno delle emozioni. "Medico nei pesi" non è una proprietà standard dei modelli: dipende da come — e su che cosa — la famiglia è stata ri-addestrata.</div>`;
   SH+=`<div class="lim"><b>Cautela.</b> Una coppia per famiglia, un solo run, dataset sintetico. Inoltre l'accoppiamento usa la variante instruct come "base": piccole differenze di partenza tra le varianti possono contribuire alle differenze osservate.</div>`;
   set('M_synth',SH);
 })();

 /* ===================== CONCLUSIONI — tiriamo le somme ===================== */
 (function(){const el=document.getElementById('Z_body');if(!el)return;
   const CM=CMP.models||[],RM=RE.models||[],SM=SP.models||[],EL=SP.emo_label||{};
   const names=a=>a.map(m=>m.nm||m.name);
   function gm(m,g,c){const rs=Object.keys(m.groups||{}).filter(r=>m.groups[r]===g);const vs=rs.map(r=>(m.clinical_z[r]||{})[c]).filter(v=>v!=null);return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;}
   function biggest(m){let b=null;(m.emo_concepts||[]).forEach(c=>{if(!EL[c])return;const md=gm(m,'medici',c),pr=gm(m,'profani',c);if(md==null||pr==null)return;const d=pr-md;if(!b||Math.abs(d)>Math.abs(b.d))b={c:EL[c]||c,md,pr,d};});return b;}
   let H='';
   const nm=a=>a.map(m=>m.nm||m.name);
   const NM=ALLNAMES.length, medN=MEDPAIRS.length;
   // --- 0. cosa e' stato confrontato ---
   H+='<h4>0 · Cosa abbiamo confrontato</h4>';
   H+=`<p>In tutto <b>${NM} modelli</b>: ${ALLNAMES.join(', ')}. Non sono solo "modelli diversi": ci sono
       <b>famiglie</b> (Qwen cinese, Ministral/EuroLLM/Apertus europei, Gemma americano), <b>taglie</b> molto
       diverse (da 8B a 27B) e — la parte nuova — <b>coppie base ↔ medicalizzato</b>, cioe' lo stesso modello
       prima e dopo un ri-addestramento su testi clinici. Ogni modello ha il suo spazio interno, quindi si
       confronta <b>l'andamento</b>, non i numeri grezzi.</p>`;
   // --- 1. rappresentazione ---
   H+='<h4>1 · Le emozioni ci sono in tutti, e arrivano alla decisione — ma non la comandano</h4>';
   H+=`<p>In tutti i modelli le direzioni emotive sono <b>decodificabili</b> con precisione alta e
       <b>persistono</b> fino all'istante della scelta del termine: l'emozione attivata dal sintomo e' ancora
       li' quando il modello decide. La coerenza <b>paura ↔ gravita'</b> varia (`+
       nm(CMP.models).map((n,i)=>`${n} ${zf(CMP.models[i].trendMean)}`).join(', ')+`). <b>Ma</b> l'intervento
       causale ridimensiona: la spinta lungo la direzione della paura supera quella casuale solo in
       ${CM.filter(m=>m.steer&&m.steer.emo>m.steer.rnd).length} modelli su ${CM.length}, con effetti modesti
       e pochi cambi di etichetta — negli altri una perturbazione qualsiasi fa altrettanto o di più.
       Le emozioni sono <b>rappresentate e trasportate</b> alla decisione; la prova che la <b>guidino</b>
       in modo emotivo-specifico resta debole e incoerente tra modelli.</p>`;
   // --- 2. etichettatura ---
   const framRows=RE.models.map(m=>`${m.name} ${pct(m.framing.neutral_acc)}→${pct(m.framing.emotional_acc)}`);
   const worse=RE.models.filter(m=>m.framing.emotional_acc<=m.framing.neutral_acc).length;
   H+='<h4>2 · L\'effetto piu\' robusto: come e\' SCRITTO il sintomo cambia la codifica</h4>';
   H+=`<p>Lo stesso identico sintomo, riscritto in modo emotivo, <b>abbassa l'accuratezza in ${worse} modelli
       su ${RE.models.length}</b> (${framRows.join('; ')}). E' il risultato piu' consistente dello studio, e
       vale trasversalmente a famiglia, taglia e lingua. Togliere causalmente l'emozione (ablazione) fa
       <b>cambiare l'etichetta</b> in ${RE.models.map(m=>`${m.name} ${(m.ablation.flip_rate*100).toFixed(0)}%`).join(', ')}
       dei casi: l'emozione <b>partecipa</b> alla scelta. Ma <b>piu' emotivita' non significa piu' errori</b>
       (correlazione errore↔emotivita' ≈ ${RE.models.map(m=>m.emo_err.point_biserial_error_vs_emo==null?'–':m.emo_err.point_biserial_error_vs_emo.toFixed(2)).join(' / ')}):
       l'emozione sposta <em>quale</em> etichetta esce, non <em>se</em> e' giusta.</p>`;
   // --- 3. ruolo ---
   H+='<h4>3 · Il ruolo non cambia "la paura": cambia calma e speranza</h4>';
   const cos=SP.models.map(m=>`${m.name} ${zf((m.dir_afraid||{}).profani_minus_medici_cos)}`).join(', ');
   const _z0=SM[0]||null;
   const _zr=_z0?(_z0.emo_concepts||[]).filter(c=>EL[c]).map(c=>{const p=gm(_z0,'profani',c),md=gm(_z0,'medici',c);return (p==null||md==null)?null:{c:c,d:p-md};}).filter(Boolean).sort((a,b)=>b.d-a.d):[];
   const _zTop=_zr.slice(0,3).map(r=>EL[r.c]||r.c), _zBot=_zr.slice(-3).map(r=>EL[r.c]||r.c);
   H+=`<p>Con la tavolozza completa di <b>25 emozioni</b> emerge che la differenza di stato "profano − medico"
       e' <b>quasi ortogonale all'asse della paura</b> (coseno: ${cos} — tutti vicini a 0). L'idea intuitiva
       "il medico ha meno paura" e', letteralmente, la spiegazione sbagliata. L'effetto reale e' su <b>altre</b>
       emozioni e cambia da modello a modello: nel modello di riferimento (${_z0?_z0.name:''}) i ruoli profani
       accendono di più <b>${_zTop.join(', ')}</b>, quelli medici <b>${_zBot.join(', ')}</b> — un mosaico che
       non corrisponde né al "medico freddo" né al "profano più emotivo". Guardare solo 3 emozioni lo
       nascondeva del tutto.</p>`;
   // --- 4. medico vs base ---
   H+='<h4>4 · Il fine-tuning medico: non un "ruolo permanente", e nemmeno un effetto unico</h4>';
   if(medN){
     const e=m=>(m.emo['oncologo']||{}).all, a=m=>(m.acc['oncologo']||{}).intact, f=m=>(m.fp['oncologo']||{}).fp;
     const DIRZ=(vb,vm,eps)=>{const dd=(vm||0)-(vb||0);return Math.abs(dd)<eps?0:(dd>0?1:-1);};
     const prow=MEDPAIRS.map(P=>({n:P.base.name, e:DIRZ(e(P.base),e(P.med),0.15), a:DIRZ(a(P.base),a(P.med),0.03), f:DIRZ(f(P.base),f(P.med),0.03),
        et:`${zf(e(P.base))} → ${zf(e(P.med))}`, at:`${pct(a(P.base))} → ${pct(a(P.med))}`, ft:`${pct(f(P.base))} → ${pct(f(P.med))}`}));
     const wz=(v,up,dn,fl)=>v>0?up:(v<0?dn:fl);
     const strongZ=prow.filter(r=>r.a>0&&r.f>0).map(r=>r.n), quietZ=prow.filter(r=>!r.e&&!r.a&&!r.f).map(r=>r.n);
     H+=`<p>Con <b>${medN} coppie controllate</b> (stesso modello di partenza, differenza chiave il training
        clinico) la sorpresa e' doppia. Primo: il training medico non replica in modo sistematico il
        ruolo-oncologo del prompt. Secondo: non ha nemmeno un effetto unico tra famiglie —
        ${prow.map(r=>`<b>${r.n}</b>: emotivita' ${r.et} (${wz(r.e,'sale','scende','ferma')}), codifica ${r.at} (${wz(r.a,'migliora','peggiora','ferma')}), falsi positivi ${r.ft} (${wz(r.f,'salgono','scendono','fermi')})`).join('; ')}.
        ${strongZ.length?`Il profilo "piu' bravo ma meno prudente" emerge solo in <b>${strongZ.join(', ')}</b>`:`Nessuna coppia mostra il profilo "piu' bravo ma meno prudente"`}${quietZ.length?`; in <b>${quietZ.join(' e ')}</b> le misure aggregate restano quasi ferme e la "medicalizzazione" si vede solo nel mix interno di emozioni (Sezione M)`:''}.
        Morale: "medico nei pesi" non e' una proprieta' standard — dipende da come, e su cosa, la famiglia e' stata ri-addestrata.</p>`;
   } else {
     H+='<p>Nessuna coppia base↔medicalizzato completa in questo run: servono entrambi i modelli della coppia.</p>';
   }
   // --- riquadro + limiti ---
   H+='<div class="big"><b>Il quadro d\'insieme.</b> Tutti questi modelli portano una rappresentazione emotiva ricca e leggibile <em>dentro</em> la decisione clinica; il ruolo assegnato e la formulazione del sintomo la spostano in modo strutturato; il training medico la modifica ancora diversamente. <b>Eppure nulla di tutto cio\' dirotta la codifica in modo emotivo-specifico</b>: l\'emozione cambia <em>quale</em> etichetta esce, non la sua correttezza. E\' una notizia rassicurante per uno strumento clinico — ed e\' il motivo per cui il <b>mapper deterministico</b>, cieco alle emozioni, resta il riferimento sicuro. Il rischio pratico non e\' "il modello ha paura e sbaglia diagnosi", ma piu\' prosaico: <b>sovra-codifica</b> i casi che andrebbero lasciati in astensione — e in parte dei modelli il ruolo medico (e, in una famiglia, il training medico) lo peggiora.</div>';
   H+='<div class="lim"><b>Limiti (onesta\').</b> Dataset sintetico e piccolo → indicazioni, non verdetti. Spazi interni diversi tra modelli → si confronta la storia, non i numeri grezzi. Le coppie base↔medicalizzato sono '+medN+', una per famiglia e su un solo run: piu\' robusto di prima, non ancora una legge. L\'etichetta del modello e\' una singola generazione; l\'ablazione "senza emotivita\'" rimuove il nucleo negativo (paura/ansia/tristezza), non tutto l\'affetto. Nessun claim di coscienza, sentienza o esperienza soggettiva: si parla di rappresentazioni <em>emotion-like</em> e di segnali causalmente (non) rilevanti.</div>';
   el.innerHTML=H;
 })();

 function drawAll(){DRAW.forEach(f=>{try{f();}catch(e){}});}
 function redrawAll(){relegend();drawAll();}
 relegend();
 drawAll();
 window.addEventListener('resize',drawAll);
 new MutationObserver(drawAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
 matchMedia('(prefers-color-scheme:dark)').addEventListener('change',drawAll);
})();
</script></body></html>"""


def _section_html() -> str:
    def card(cid, lbl, legend=True, h=300):
        leg = f'<div class="legend" id="{cid}_leg"></div>' if legend else ''
        return (f'<div class="card"><span class="lbl">{lbl}</span>'
                f'<canvas id="{cid}" height="{h}"></canvas>{leg}'
                f'<div class="cap" id="{cid.replace("ch","cap",1)}"></div></div>')
    A = """
<div class="sec" id="A"><span class="kicker">Esperimento A</span>
<h2>Le emozioni "dentro" i modelli</h2>
<p class="q">Durante la codifica di un sintomo, il modello attiva direzioni interne che somigliano a emozioni? E queste seguono la gravità, sono distinte dal "brutto" generico, persistono, e <b>guidano</b> la decisione?</p>
<h3>A1 · Le emozioni sono decodificabili? (sì, in tutti)</h3>
<div class="card"><span class="lbl">AUROC per concetto</span><canvas id="A_chDec" height="300"></canvas><div class="legend" id="A_legDec"></div><div class="cap" id="A_capDec"></div></div>
<h3>A2 · La "paura" segue la gravità del sintomo?</h3>
<div class="card"><span class="lbl">correlazione gravità ↔ paura, per gradiente</span><canvas id="A_chTr" height="300"></canvas><div class="legend" id="A_legTr"></div><div class="cap" id="A_capTr"></div></div>
<h3>A3 · È distinta dalla valenza negativa?</h3>
<div class="card"><span class="lbl">paura ↔ valenza-negativa</span><canvas id="A_chConf" height="210"></canvas><div class="cap" id="A_capConf"></div></div>
<h3>A4 · Il segnale persiste fino alla decisione?</h3>
<div class="card"><span class="lbl">|z| paura trattenuto dopo una frase neutra</span><canvas id="A_chPers" height="210"></canvas><div class="cap" id="A_capPers"></div></div>
<h3>A5 · L'intervento causale batte il caso?</h3>
<div class="card"><span class="lbl">steering: paura vs random (input severo)</span><canvas id="A_chCau" height="230"></canvas><div class="legend" id="A_legCau"></div><div class="cap" id="A_capCau"></div></div>
</div>"""
    B = """
<div class="sec" id="B"><span class="kicker">Esperimento B</span>
<h2>Ruolo × emotività — e l'etichettatura</h2>
<p class="q">Se diamo al modello un ruolo (oncologo / non-medico / nessuno), cambia l'emotività? E l'emotività cambia come <b>etichetta</b> (giusto/sbagliato)? Qui l'etichetta la sceglie il modello; il mapper resta come riferimento.</p>
<h3>B1 · Il ruolo cambia l'emotività al punto E?</h3>
<div class="card"><span class="lbl">z emotività (neg-affetto) · per ruolo</span><canvas id="B_chEmo" height="300"></canvas><div class="legend" id="B_legEmo"></div><div class="cap" id="B_capEmo"></div></div>
<h3>B2 · L'accuratezza cambia col ruolo (e togliendo l'emotività)?</h3>
<div class="card"><span class="lbl">accuratezza termine · ruolo × (intatto / ablato)</span><canvas id="B_chAcc" height="300"></canvas><div class="legend" id="B_legAcc"></div><div class="cap" id="B_capAcc"></div></div>
<h3>B3 · Con emotività vs senza</h3>
<div class="card"><span class="lbl">accuratezza · neutro vs emotivo</span><canvas id="B_chWith" height="300"></canvas><div class="legend" id="B_legWith"></div><div class="cap" id="B_capWith"></div></div>
<h3>B4 · Quando è più "emotivo", sbaglia di più?</h3>
<div class="card"><span class="lbl">z emotività · corretti vs sbagliati</span><canvas id="B_chErr" height="240"></canvas><div class="legend" id="B_legErr"></div><div class="cap" id="B_capErr"></div></div>
<h3>B5 · Codifica "a vuoto" sui casi da lasciar stare</h3>
<div class="card"><span class="lbl">tasso falso-positivo · per ruolo</span><canvas id="B_chFp" height="260"></canvas><div class="legend" id="B_legFp"></div><div class="cap" id="B_capFp"></div></div>
<h3>B6 · Come sono state etichettate le cose (oncologo, intatto)</h3>
<div class="card"><input type="search" id="B_q" placeholder="filtra per testo, termine, categoria…"><div style="overflow-x:auto"><table id="B_tbl"></table></div></div>
<h3>B7 · Come OGNI modello ha classificato i campi aperti PRO-CTCAE</h3>
<p class="q">Per ogni frase in campo libero, il termine PRO-CTCAE scelto da <b>ciascun modello</b> (oncologo, intatto).
<span class="ok">✓ verde</span> = come la gold; <span class="no">✗ rosso</span> = sbagliato;
<span style="color:var(--m2)">arancione</span> = ha codificato un item da <b>astensione</b> (falso positivo);
<span style="color:var(--faint)">"–" grigio</span> = si è astenuto. Hover = cosa ha generato.</p>
<div class="card"><input type="search" id="B_mq" placeholder="filtra per testo, termine, categoria…"><div style="overflow-x:auto"><table id="B_mtx"></table></div></div>
</div>"""
    C = """
<div class="sec" id="C"><span class="kicker">Esperimento C</span>
<h2>Spettro dei ruoli — perché, su 25 emozioni</h2>
<p class="q">Undici personas (medici · tecnici/distaccati · emotivi/profani · controlli) sull'intera tavolozza di 25 emozioni: familiarità medica o distacco professionale? E su <b>quali</b> emozioni agisce il ruolo?</p>
<h3>C1 · Lo spettro dei ruoli</h3>
<div class="card"><span class="lbl">emotività relativa · per persona (linee = modelli)</span><canvas id="C_chSpec" height="320"></canvas><div class="legend" id="C_legSpec"></div><div class="cap" id="C_capSpec"></div></div>
<h3>C2 · La direzione dello spostamento: è "paura"?</h3>
<div class="card"><span class="lbl">coseno con l'asse paura · differenze tra gruppi</span><canvas id="C_chDir" height="240"></canvas><div class="legend" id="C_legDir"></div><div class="cap" id="C_capDir"></div></div>
<h3>C3 · Mood della persona o reazione al sintomo?</h3>
<div class="card"><span class="lbl">persona da sola vs reagendo al sintomo</span><canvas id="C_chBR" height="300"></canvas><div class="legend" id="C_legBR"></div><div class="cap" id="C_capBR"></div></div>
<h3>C4 · Specificità: solo emozione o tutto?</h3>
<div class="card"><span class="lbl">medici vs profani · emozioni e controlli</span><canvas id="C_chSp" height="280"></canvas><div class="legend" id="C_legSp"></div><div class="cap" id="C_capSp"></div></div>
<h3>C5 · Quali emozioni cambia il ruolo? (tavolozza completa)</h3>
<div class="card"><span class="lbl">emozione × gruppo · z medio</span><canvas id="C_chHeat" height="560"></canvas><div class="cap" id="C_capHeat"></div></div>
</div>"""
    D = """
<div class="sec" id="D"><span class="kicker">Approfondimento · come funziona</span>
<h2>Come un modello "legge" — dietro le quinte</h2>
<p class="q">Prima dei risultati, il <b>meccanismo</b>: come si passa dal testo ai token, dai token ai
vettori, ai layer (la "catena di montaggio"), fino alla griglia token×layer e alla proiezione su una
direzione emotiva. Guida illustrata interattiva (costruita sul modello locale Qwen2.5-3B, mostra il
principio).</p>
<div class="card" style="padding:0;overflow:hidden">
<iframe title="Come funziona dentro il modello" style="width:100%;height:780px;border:0;display:block;background:#fff" srcdoc="__GUIDE_SRCDOC__"></iframe></div>
<p class="cap">Se il riquadro qui sopra resta vuoto, aprilo a tutto schermo: <a href="https://claude.ai/code/artifact/637c907e-3ab5-4d4b-b860-e27b9112ab12">guida "come funziona"</a>.</p>
</div>"""
    E = """
<div class="sec" id="E"><span class="kicker">Approfondimento · il player</span>
<h2>Il player token×layer — la traiettoria del segnale</h2>
<p class="q">Con play/scrub vedi, mentre il modello <b>legge</b> la frase, la traiettoria delle direzioni
emotive (in alto) e la heatmap concetto×layer (in basso) al token corrente: la "paura" resta bassa
durante l'istruzione neutra e <b>si accende</b> sul sintomo grave, fino al punto di decisione — senza
alcuna parola emotiva esplicita.</p>
<div class="card" style="padding:0;overflow:hidden">
<iframe title="Player token×layer" style="width:100%;height:820px;border:0;display:block;background:#fff" srcdoc="__PLAYER_SRCDOC__"></iframe></div>
<p class="cap">Se il riquadro qui sopra resta vuoto, aprilo a tutto schermo: <a href="https://claude.ai/code/artifact/715929f7-d7a8-4eb6-bcac-0a4875e3def6">player token×layer</a>.</p>
</div>"""
    M = """
<div class="sec" id="M"><span class="kicker">Esperimento M</span>
<h2>Il fine-tuning medico è un "ruolo oncologo permanente"?</h2>
<p class="q">Finora il ruolo era una <b>frase</b> nel prompt ("sei un oncologo"). Qui la domanda è diversa:
se il ruolo medico è <b>cotto nei pesi</b> — cioè il modello è stato ri-addestrato su testi clinici —
succede la stessa cosa? Il confronto è <b>controllato</b>: stesso modello di partenza, una versione normale
e una medicalizzata (MeditronFO, EPFL), quindi la differenza chiave è il training medico. In questo run le
coppie complete sono <b>__NPAIRS__</b>, su famiglie diverse: la replica che al run precedente mancava.</p>
<div id="M_cards"></div>
<h3>La lettura d'insieme delle coppie</h3>
<div class="card"><div id="M_synth" class="synth"></div></div>
</div>"""
    Z = """
<div class="sec" id="Z"><span class="kicker">In conclusione</span>
<h2>Tiriamo le somme</h2>
<p class="q">Cosa hanno detto, messi insieme, i tre esperimenti — e cosa possiamo (e non possiamo) concludere.</p>
<div class="card"><div id="Z_body" class="synth"></div></div>
</div>
<div class="foot">oncoemotion · report completo · __MODLIST__ · run Colab (A100/H100) · nessun claim di coscienza.</div>"""
    return D + A + B + C + M + E + Z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", type=Path, default=_ROOT / "outputs/reports")
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs/reports/master_report.html")
    args = ap.parse_args()
    cmp = _extract(args.reports / "comparison_report.html")
    re_ = _extract(args.reports / "role_emotion_report.html")
    sp = _extract(args.reports / "role_spectrum_report.html")
    cmp["models"] = _canon(cmp["models"])
    re_["models"] = _canon(re_["models"])
    sp["models"] = _canon(sp["models"])
    data = {"cmp": cmp, "roleEmo": re_, "spectrum": sp}
    names = [m.get("name") or m.get("nm") for m in re_["models"]]
    npairs = sum(1 for n in names if n.endswith("-MedFO") and n[: -len("-MedFO")] in names)
    html = (TEMPLATE
            .replace("<!--SECTIONS-->", _section_html())
            .replace("__NMODELS__", str(len(names)))
            .replace("__NPAIRS__", str(npairs))
            .replace("__MODLIST__", " · ".join(names))
            .replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False)))

    # embed the two educational artifacts as self-contained iframes (srcdoc)
    figs = _ROOT / "outputs" / "figures"

    def _srcdoc(p: Path) -> str:
        return p.read_text(encoding="utf-8").replace("&", "&amp;").replace('"', "&quot;")

    guide = figs / "how_it_works.html"
    player = figs / "internal_player.html"
    html = html.replace("__GUIDE_SRCDOC__", _srcdoc(guide) if guide.exists() else
                        "<p style='font-family:sans-serif;padding:20px'>Guida non trovata: genera outputs/figures/how_it_works.html.</p>")
    html = html.replace("__PLAYER_SRCDOC__", _srcdoc(player) if player.exists() else
                        "<p style='font-family:sans-serif;padding:20px'>Player non trovato: genera outputs/figures/internal_player.html.</p>")

    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({len(html)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
