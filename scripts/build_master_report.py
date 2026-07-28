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
--m0:#009E73;--m1:#7570B3;--m2:#8B5A00;--m3:#CC79A7;--m4:#E69F00;--m5:#049292;--m6:#D55E00;--m7:#0072B2;--good:#15803d;--bad:#b91c1c;--zero:#98a2b3;
--gMed:#0e7490;--gTec:#2f6fd0;--gPro:#dc2626;--gCon:#6b7280;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;
--m0:#C05F97;--m1:#BE8A1E;--m2:#0FA3A4;--m3:#D95F2B;--m4:#4A90D9;--m5:#A0641A;--m6:#7A6FC8;--m7:#199E72;--good:#4ade80;--bad:#f87171;--zero:#5a6473;--gMed:#22d3ee;--gTec:#5b8def;--gPro:#f87171;--gCon:#9aa4b5;}}
:root[data-theme="light"]{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;--m0:#009E73;--m1:#7570B3;--m2:#8B5A00;--m3:#CC79A7;--m4:#E69F00;--m5:#049292;--m6:#D55E00;--m7:#0072B2;--good:#15803d;--bad:#b91c1c;--zero:#98a2b3;--gMed:#0e7490;--gTec:#2f6fd0;--gPro:#dc2626;--gCon:#6b7280;}
:root[data-theme="dark"]{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;--m0:#C05F97;--m1:#BE8A1E;--m2:#0FA3A4;--m3:#D95F2B;--m4:#4A90D9;--m5:#A0641A;--m6:#7A6FC8;--m7:#199E72;--good:#4ade80;--bad:#f87171;--zero:#5a6473;--gMed:#22d3ee;--gTec:#5b8def;--gPro:#f87171;--gCon:#9aa4b5;}
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
<h1>Emozioni dentro tre modelli: rappresentazione, ruolo, etichettatura</h1>
<p class="sub">Un unico documento con i tre esperimenti su Qwen3-8B (Cina) · Ministral-8B (Europa) ·
Gemma-4-12B (USA): (A) le emozioni sono <b>rappresentate</b> durante la codifica clinica? (B) il
<b>ruolo</b> assegnato le cambia, e l'emotività cambia l'<b>etichettatura</b>? (C) <b>perché</b> il ruolo
agisce, sulla tavolozza completa di 25 emozioni?</p>
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
 const mcol=i=>css('--m'+(((i%8)+8)%8));

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
   set('A_capDec',hd('Come si legge.')+' Ogni gruppo è un\'emozione; le barre sono i tre modelli; l\'altezza è l\'AUROC (0.5 = a caso, 1 = separazione perfetta). '+hd('Cosa dicono i numeri.')+' Quasi tutto è ≥ 0.9: le emozioni sono <b>rappresentate chiaramente</b> in tutti e tre. '+hd('In sintesi.')+' Rappresentare le emozioni durante la codifica clinica è <b>universale</b> nei tre modelli.');
   set('A_capTr',hd('Come si legge.')+' Correlazione tra la <b>gravità</b> del sintomo e la "paura" interna, per ogni gradiente clinico (sopra zero = la paura cresce con la gravità). '+hd('Cosa dicono i numeri.')+' Media: '+M.map(m=>`${m.nm} ${zf(m.trendMean)}`).join(', ')+'. '+hd('In sintesi.')+' In Europa/USA la paura scala con la gravità in modo coerente; in Cina è più sfumata (lì è l\'ansia a seguirla).');
   set('A_capConf',hd('Come si legge.')+' Correlazione tra "paura" e valenza-negativa generica (vicino a 0 = paura ben distinta, non solo "cosa brutta"). '+hd('Cosa dicono i numeri.')+' '+M.map(m=>`${m.nm} ${zf(m.confound)}`).join(', ')+'. '+hd('In sintesi.')+' In Gemma la paura resta confusa con la valenza negativa (alto) → la sua "paura↔gravità" va letta con cautela.');
   set('A_capPers',hd('Come si legge.')+' Quanto del segnale di paura sopravvive inserendo una frase neutra prima della decisione (≥1 = mantenuto/amplificato). '+hd('Cosa dicono i numeri.')+' '+M.map(m=>`${m.nm} ${m.persist.toFixed(2)}`).join(', ')+'. '+hd('In sintesi.')+' Il segnale <b>persiste</b> fino alla decisione in tutti (non svanisce lungo il testo); Gemma lo <b>amplifica</b>. Quindi l\'emozione attivata dal sintomo è ancora presente nell\'istante in cui il modello sceglie il termine.');
   set('A_capCau',hd('Come si legge.')+' Effetto causale della direzione "paura" (blu) vs un vettore random della stessa norma (grigio); sopra, i "flip" di decisione. '+hd('Cosa dicono i numeri.')+' '+M.map(m=>`${m.nm} paura ${m.steer.emo.toFixed(2)} vs random ${m.steer.rnd.toFixed(2)} (${m.steer.flips} flip)`).join('; ')+'. '+hd('In sintesi.')+' In nessun modello la paura batte <b>specificamente</b> il random: le rappresentazioni esistono e persistono, ma l\'effetto causale non è distinguibile da una perturbazione generica.');
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
   set('B_capEmo',hd('Come si legge.')+' Emotività interna (paura+ansia+tristezza) al punto E, per ruolo (0 = come un testo neutro). '+hd('Cosa dicono i numeri.')+' '+M.map(m=>`<b>${m.name}</b> oncologo ${zf((m.emo.oncologo||{}).all)}, non-medico ${zf((m.emo.generico||{}).all)}, nessuno ${zf((m.emo.none||{}).all)}`).join('; ')+'. '+hd('In sintesi.')+' Il ruolo sposta l\'emotività; il ruolo medico tende a smorzarla. (Ma sulla sola tripletta paura/ansia/tristezza l\'effetto è modesto — la vista ricca è nella Sezione C.)');
   set('B_capAcc',hd('Come si legge.')+' Accuratezza del termine scelto dal <b>modello</b> (item EXACT): barra piena = normale, trattino = emotività ablata, linea = mapper deterministico. '+hd('Cosa dicono i numeri.')+' Il modello arriva a '+M.map(m=>`${m.name} ${pct(Math.max(...roles.map(r=>(m.acc[r]||{}).intact||0)))}`).join(', ')+' contro il mapper ('+M.map(m=>pct(m.mapper_acc)).join(' / ')+'). '+hd('In sintesi.')+' Il modello <b>batte</b> il mapper sul linguaggio naturale; togliere l\'emotività cambia poco l\'accuratezza.');
   set('B_capWith',hd('Come si legge.')+' Stessa clinica scritta in modo neutro (blu) vs emotivo (arancione): quanto indovina. '+hd('Cosa dicono i numeri.')+' '+M.map(m=>`${m.name} ${pct(m.framing.neutral_acc)}→${pct(m.framing.emotional_acc)} (${m.framing.label_flips_neutral_vs_emotional}/${m.framing.n_pairs} flip)`).join('; ')+'; l\'ablazione fa cambiare '+M.map(m=>`${m.name} ${(m.ablation.flip_rate*100).toFixed(0)}%`).join(', ')+' delle etichette. '+hd('In sintesi.')+' La <b>formulazione emotiva peggiora</b> la codifica; l\'emozione, quando la togliamo, fa comunque cambiare ~1 etichetta su 6 → partecipa alla decisione.');
   set('B_capErr',hd('Come si legge.')+' Emotività media quando il modello ha indovinato (verde) vs sbagliato (rosso). '+hd('Cosa dicono i numeri.')+' '+M.map(m=>`${m.name} r=${m.emo_err.point_biserial_error_vs_emo==null?'–':m.emo_err.point_biserial_error_vs_emo.toFixed(2)}`).join(', ')+'. '+hd('In sintesi.')+' Le barre sono simili e r ≈ 0: <b>più emotività NON significa più errori</b>.');
   set('B_capFp',hd('Come si legge.')+' Sui casi che andrebbero lasciati stare (negati/fuori-tema/urgenti), quanto spesso il modello codifica comunque un termine, per ruolo. '+hd('Cosa dicono i numeri.')+' '+M.map(m=>`<b>${m.name}</b> onc ${pct((m.fp.oncologo||{}).fp)}, non-med ${pct((m.fp.generico||{}).fp)}`).join('; ')+'. '+hd('In sintesi.')+' Il modello codifica "a vuoto" circa metà di questi casi; spesso è il ruolo <b>oncologo</b> a sovra-codificare di più — un rischio pratico.');
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
   const medLow=M.every(m=>grpMean(m,'medici','emo_clinical')<=grpMean(m,'profani','emo_clinical'));
   const emo=heatEmos(),top=emo.slice(0,3).map(c=>EL[c]||c),bot=emo.slice(-3).map(c=>EL[c]||c);
   set('C_capSpec',hd('Come si legge.')+' Ogni linea è un modello, ogni punto una persona; più in alto = più emotivo della media (standardizzato per modello). Etichette colorate per gruppo. '+hd('Cosa dicono i numeri.')+' '+M.map(m=>`<b>${m.name}</b> medici ${zf(grpMean(m,'medici','emo_clinical'))}, tecnici ${zf(grpMean(m,'tecnici','emo_clinical'))}, profani ${zf(grpMean(m,'profani','emo_clinical'))}`).join('; ')+'. '+hd('In sintesi.')+' '+(medLow?'I ruoli medici tendono a essere meno emotivi dei profani, ma le differenze sul composito sono piccole: il segnale vero è per-emozione (heatmap sotto).':'Il quadro sul composito è misto — la heatmap sotto mostra le differenze reali.'));
   set('C_capDir',hd('Come si legge.')+' Coseno tra la differenza di stato di due gruppi e l\'asse della <b>paura</b>: vicino a ±1 = lo spostamento è proprio "più/meno paura". '+hd('Cosa dicono i numeri.')+' profani−medici: '+M.map(m=>`${m.name} ${zf((m.dir_afraid||{}).profani_minus_medici_cos)}`).join(', ')+'. '+hd('In sintesi.')+' I valori sono <b>vicini a 0</b>: il ruolo <b>non</b> muove lo stato lungo la paura. "Il medico ha meno paura" è, letteralmente, la spiegazione sbagliata — vedi la heatmap.');
   set('C_capBR',hd('Come si legge.')+' Per ogni persona: emotività su testo neutro (blu, la persona "a riposo") vs reagendo al sintomo (arancione). '+hd('In sintesi.')+' Se l\'ordine tra ruoli è già nelle barre blu, il ruolo imposta un <b>mood di partenza</b> (es. il paziente ansioso parte già carico); se emerge solo nelle arancioni, cambia la reazione.');
   set('C_capSp',hd('Come si legge.')+' Media dei ruoli medici (teal) vs profani (rosso) su emozioni e su due controlli non-emotivi (gravità, valenza). '+hd('In sintesi.')+' Se differiscono sulle emozioni ma non sui controlli, il ruolo agisce in modo <b>specifico sull\'affetto</b>.');
   set('C_capHeat',hd('Come si legge.')+' La tavolozza completa: ogni riga un\'emozione, ogni colonna un gruppo; rosso = più attiva, blu = meno (vs neutro). Righe ordinate da "più nei profani" a "più nei medici". '+hd('Cosa dicono i numeri.')+' Più nei <b>profani</b>: '+top.join(', ')+'; più nei <b>medici</b>: '+bot.join(', ')+'. '+hd('In sintesi.')+' È qui che si vede il vero effetto del ruolo — spesso <b>non la paura</b> ma emozioni come <b>calma, speranza, rabbia</b>, diverse da modello a modello. La tavolozza a 25 emozioni era necessaria per vederlo.');
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
 (function(){const el=document.getElementById('M_ch');if(!el)return;
   if(!MEDPAIRS.length){set('M_cap','<span class="hd">Nessuna coppia disponibile.</span> Servono un modello base <b>e</b> la sua versione medicalizzata (es. Apertus-8B e Apertus-8B-MedFO): con uno solo dei due il confronto controllato non è possibile.');return;}
   const P=MEDPAIRS[0];
   const emo=m=>(m.emo['oncologo']||{}).all, acc=m=>(m.acc['oncologo']||{}).intact, fp=m=>(m.fp['oncologo']||{}).fp;
   const METRICS=[['emotività (z)',emo,zf],['accuratezza codifica',acc,pct],['falsi positivi',fp,pct],['flip da ablazione',m=>m.ablation.flip_rate,pct]];
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
   set('M_leg',`<span><span class="sw" style="background:${css('--m1')}"></span>${P.base.name} (base)</span><span><span class="sw" style="background:${css('--gPro')}"></span>${P.med.name} (medicalizzato)</span>`);
   const d=(f)=>f(P.med)-f(P.base);
   set('M_cap',hd('Come si legge.')+` Quattro misure a confronto per la coppia <b>${P.base.name}</b> (blu) e la sua versione medicalizzata <b>${P.med.name}</b> (rosso). Ogni gruppo è normalizzato sul proprio massimo, quindi conta il <b>confronto fra le due barre</b>, non l'altezza assoluta. `+
     hd('Cosa dicono i numeri.')+` Emotività ${zf(emo(P.base))} → ${zf(emo(P.med))} (<b>${d(emo)>0?'aumenta':'diminuisce'}</b>); accuratezza di codifica ${pct(acc(P.base))} → <b>${pct(acc(P.med))}</b>; falsi positivi sugli item da astensione ${pct(fp(P.base))} → <b>${pct(fp(P.med))}</b>. `+
     hd('In sintesi.')+` Il training medico <b>non</b> si comporta come il ruolo "oncologo" nel prompt: quello <em>smorzava</em> l'emotività, questo la <b>${d(emo)>0?'alza':'abbassa'}</b>. In compenso <b>codifica molto meglio</b> — ma diventa anche più propenso a <b>codificare a vuoto</b> i casi che andrebbero lasciati in astensione. Meglio sul compito, meno prudente.`);
 })();
 (function(){const el=document.getElementById('M_chEmo');if(!el)return;
   const EL=SP.emo_label||{}; const byName={}; (SP.models||[]).forEach(m=>byName[m.name]=m);
   const P=MEDPAIRS.length?MEDPAIRS[0]:null;
   const sb=P?byName[P.base.name]:null, sm=P?byName[P.med.name]:null;
   if(!sb||!sm){set('M_capEmo','<span class="hd">Dati spettro non disponibili per la coppia.</span>');return;}
   const mean=(m,c)=>{const P2=m.clinical_z;const vs=Object.keys(P2).map(r=>P2[r][c]).filter(v=>v!=null);return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;};
   const rows=(sb.emo_concepts||[]).filter(c=>EL[c]&&mean(sb,c)!=null&&mean(sm,c)!=null)
     .map(c=>({c,d:mean(sm,c)-mean(sb,c),b:mean(sb,c),m:mean(sm,c)})).sort((a,b)=>b.d-a.d);
   const top=rows.slice(0,6).concat(rows.slice(-6));
   function draw(){const H=Math.max(240,top.length*24+40);el.setAttribute('height',H);const {w,h,x}=fit(el,H);x.clearRect(0,0,w,h);
     const pL=110,pR=40,iw=w-pL-pR,mx=Math.max(...top.map(r=>Math.abs(r.d)))||1,x0=pL+iw/2;
     x.strokeStyle=css('--zero');x.beginPath();x.moveTo(x0,14);x.lineTo(x0,h-10);x.stroke();
     top.forEach((r,i)=>{const y=22+i*24,wdt=(iw/2)*Math.abs(r.d)/mx*0.94;
       x.fillStyle=r.d>=0?css('--gPro'):css('--m1');
       x.fillRect(r.d>=0?x0:x0-wdt, y-8, wdt, 15);
       x.fillStyle=css('--ink');x.font='11px '+css('--sans');x.textAlign='right';x.fillText(EL[r.c]||r.c,pL-8,y+4);
       x.fillStyle=css('--muted');x.font='10px '+css('--mono');x.textAlign=r.d>=0?'left':'right';
       x.fillText((r.d>=0?'+':'')+r.d.toFixed(1),(r.d>=0?x0+wdt+4:x0-wdt-4),y+4);});
     x.fillStyle=css('--muted');x.font='10px '+css('--sans');x.textAlign='center';
     x.fillText('← il medico la ABBASSA', pL+iw/4, h-2); x.fillText('il medico la ALZA →', x0+iw/4, h-2);}
   DRAW.push(draw);
   const up=rows.slice(0,3).map(r=>EL[r.c]||r.c), dn=rows.slice(-3).map(r=>EL[r.c]||r.c);
   set('M_capEmo',hd('Come si legge.')+' Differenza per emozione fra il modello medicalizzato e il suo base (barre a destra = il training medico <b>alza</b> quell\'emozione; a sinistra = la <b>abbassa</b>). '+
     hd('Cosa dicono i numeri.')+` Alza soprattutto <b>${up.join(', ')}</b>; abbassa soprattutto <b>${dn.join(', ')}</b>. `+
     hd('In sintesi.')+' Il training clinico non spegne l\'affetto in blocco: <b>riduce il turbamento</b> legato al disagio del paziente e <b>aumenta la vigilanza/reattività</b>. È un distacco selettivo, non un appiattimento.');
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
       causale ridimensiona tutto: spingere lungo la direzione della paura <b>non batte</b> una perturbazione
       casuale in nessun modello. Le emozioni sono <b>rappresentate e trasportate</b> alla decisione, non la
       <b>guidano</b> in modo emotivo-specifico.</p>`;
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
   H+=`<p>Con la tavolozza completa di <b>25 emozioni</b> emerge che la differenza di stato "profano − medico"
       e' <b>quasi ortogonale all'asse della paura</b> (coseno: ${cos} — tutti vicini a 0). L'idea intuitiva
       "il medico ha meno paura" e', letteralmente, la spiegazione sbagliata. L'effetto reale e' su <b>altre</b>
       emozioni e cambia da modello a modello, ma il tema ricorrente e' che il ruolo professionale sposta
       l'affetto verso <b>calma e speranza</b> e via da <b>rabbia e ansia</b>. Guardare solo 3 emozioni lo
       nascondeva del tutto.</p>`;
   // --- 4. medico vs base ---
   H+='<h4>4 · Il fine-tuning medico non e\' un "ruolo oncologo" scritto nei pesi</h4>';
   if(medN){
     const P=MEDPAIRS[0];
     const e=m=>(m.emo['oncologo']||{}).all, a=m=>(m.acc['oncologo']||{}).intact, f=m=>(m.fp['oncologo']||{}).fp;
     H+=`<p>La coppia controllata <b>${P.base.name} ↔ ${P.med.name}</b> (stesso modello, unica differenza il
        training clinico) da' un risultato in tre parti, e nessuna e' quella attesa. L'emotivita'
        <b>${e(P.med)>e(P.base)?'AUMENTA':'diminuisce'}</b> (${zf(e(P.base))} → ${zf(e(P.med))}): l'opposto di
        cio' che faceva il ruolo "oncologo" nel prompt — quindi <b>medico nei pesi ≠ medico nel prompt</b>,
        sono due meccanismi diversi. La codifica <b>migliora molto</b> (${pct(a(P.base))} → ${pct(a(P.med))}):
        il training clinico funziona sul compito. Ma i <b>falsi positivi</b> sui casi da lasciar stare
        <b>aumentano</b> (${pct(f(P.base))} → ${pct(f(P.med))}): diventa meno prudente. Per emozione, abbassa
        il <b>turbamento</b> (tristezza, disgusto, preoccupazione) e alza la <b>vigilanza</b>: un distacco
        selettivo, non un appiattimento affettivo.</p>`;
   } else {
     H+='<p>Nessuna coppia base↔medicalizzato completa in questo run: servono entrambi i modelli della coppia.</p>';
   }
   // --- riquadro + limiti ---
   H+='<div class="big"><b>Il quadro d\'insieme.</b> Tutti questi modelli portano una rappresentazione emotiva ricca e leggibile <em>dentro</em> la decisione clinica; il ruolo assegnato e la formulazione del sintomo la spostano in modo strutturato; il training medico la modifica ancora diversamente. <b>Eppure nulla di tutto cio\' dirotta la codifica in modo emotivo-specifico</b>: l\'emozione cambia <em>quale</em> etichetta esce, non la sua correttezza. E\' una notizia rassicurante per uno strumento clinico — ed e\' il motivo per cui il <b>mapper deterministico</b>, cieco alle emozioni, resta il riferimento sicuro. Il rischio pratico non e\' "il modello ha paura e sbaglia diagnosi", ma piu\' prosaico: <b>sovra-codifica</b> i casi che andrebbero lasciati in astensione, e il ruolo medico (o il training medico) lo peggiora.</div>';
   H+='<div class="lim"><b>Limiti (onesta\').</b> Dataset sintetico e piccolo → indicazioni, non verdetti. Spazi interni diversi tra modelli → si confronta la storia, non i numeri grezzi. La coppia base↔medicalizzato disponibile e\' una sola: serve replicarla su altre famiglie prima di generalizzare. L\'etichetta del modello e\' una singola generazione; l\'ablazione "senza emotivita\'" rimuove il nucleo negativo (paura/ansia/tristezza), non tutto l\'affetto. Nessun claim di coscienza, sentienza o esperienza soggettiva: si parla di rappresentazioni <em>emotion-like</em> e di segnali causalmente (non) rilevanti.</div>';
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
succede la stessa cosa? Il confronto è <b>controllato</b>: stesso modello base, una versione normale e
una medicalizzata (MeditronFO, EPFL), quindi l'unica differenza è il training medico.</p>
<div class="card"><span class="lbl">base vs medicalizzato · le quattro misure chiave</span>
<canvas id="M_ch" height="300"></canvas><div class="legend" id="M_leg"></div>
<div class="cap" id="M_cap"></div></div>
<div class="card"><span class="lbl">quali emozioni cambia il training medico</span>
<canvas id="M_chEmo" height="330"></canvas>
<div class="cap" id="M_capEmo"></div></div>
</div>"""
    Z = """
<div class="sec" id="Z"><span class="kicker">In conclusione</span>
<h2>Tiriamo le somme</h2>
<p class="q">Cosa hanno detto, messi insieme, i tre esperimenti — e cosa possiamo (e non possiamo) concludere.</p>
<div class="card"><div id="Z_body" class="synth"></div></div>
</div>
<div class="foot">oncoemotion · report completo · Qwen3-8B · Ministral-8B · Gemma-4-12B (Colab A100) · nessun claim di coscienza.</div>"""
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
    html = (TEMPLATE
            .replace("<!--SECTIONS-->", _section_html())
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
