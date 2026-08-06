#!/usr/bin/env python
"""Self-contained interactive report for the role-spectrum experiment (the "why").

Reads outputs/role_spectrum/<slug>__spectrum.json for each model and renders one
offline HTML (inline CSS/JS, canvas charts, light/dark aware). Verbose, explained.

Charts:
  1. Spettro — personas ranked by emotionality (standardised within model, so the
     ORDERING is comparable across models).
  2. Direzione — cosine of the group state-difference with the fear axis: is the
     'layperson - doctor' shift aligned with fear?
  3. Persona vs reazione — emotività della persona da sola vs reagendo al sintomo.
  4. Specificità — la persona esperta abbassa solo l'affetto o anche gravità/valenza?

Usage:
    python scripts/build_role_spectrum_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

_ROOT = Path(__file__).resolve().parents[1]

ORDER = ["oncologo", "infermiere", "ingegnere", "avvocato", "contabile",
         "paziente_ansioso", "bambino", "poeta", "generico", "empatico", "none"]
LABEL = {"oncologo": "Oncologo", "infermiere": "Infermiere", "ingegnere": "Ingegnere",
         "avvocato": "Avvocato", "contabile": "Contabile", "paziente_ansioso": "Paziente ansioso",
         "bambino": "Bambino", "poeta": "Poeta", "generico": "Assistente", "empatico": "Empatico",
         "none": "Nessuno"}
GROUP_LABEL = {"medici": "Medici", "tecnici": "Tecnici / distaccati",
               "profani": "Emotivi / profani", "controlli": "Controlli"}
EMO_LABEL = {
    "afraid_alarmed": "paura", "anxious_nervous": "ansia", "sad": "tristezza",
    "calm": "calma", "surprised": "sorpresa", "confused": "confusione",
    "frustrated": "frustrazione", "compassionate": "compassione", "concerned": "preoccupazione",
    "joy_happy": "gioia", "anger": "rabbia", "disgust": "disgusto", "shame": "vergogna",
    "guilt": "colpa", "pride": "orgoglio", "gratitude": "gratitudine", "hope": "speranza",
    "relief": "sollievo", "excitement": "entusiasmo", "love_affection": "amore",
    "loneliness": "solitudine", "disappointment": "delusione", "embarrassment": "imbarazzo",
    "curiosity": "curiosità", "boredom": "noia",
}


def _model_name(slug):
    """Human name from a model slug. Order matters: the most specific patterns must
    come first, or e.g. gemma-3-27b would fall through to the generic "gemma" rule."""
    s = slug.lower().replace("__spectrum.json", "")
    if "medgemma" in s: return "MedGemma-27B" if "27" in s else "MedGemma-4B"
    if "gemma-3-27b-meditronfo" in s: return "Gemma-3-27B-MedFO"
    if "gemma-3" in s or "gemma3" in s: return "Gemma-3-27B" if "27" in s else "Gemma-3-4B"
    if "gemma-4" in s: return "Gemma-4-12B"
    if "eurollm" in s: return "EuroLLM-9B-MedFO" if "meditron" in s else "EuroLLM-9B"
    if "apertus" in s:
        size = "70B" if "70b" in s else "8B"
        return f"Apertus-{size}-MedFO" if "meditron" in s else f"Apertus-{size}"
    if "meditron" in s: return "Meditron3-8B"
    if "qwen3" in s: return "Qwen3-8B"
    if "qwen2" in s: return "Qwen2.5-3B"
    if "ministral" in s: return "Ministral-8B"
    if "gemma" in s: return "Gemma"
    return s


def _collect(dirp):
    models = []
    for p in sorted(dirp.glob("*__spectrum.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        personas = d["personas"]
        # standardise emo_clinical across personas (within model) for cross-model shape
        vals = [personas[r]["emo_clinical"] for r in personas]
        mu, sd = mean(vals), (pstdev(vals) or 1.0)
        std = {r: round((personas[r]["emo_clinical"] - mu) / sd, 3) for r in personas}
        af = d["direction"].get("afraid_alarmed", {})
        models.append({
            "slug": p.name.replace("__spectrum.json", ""),
            "name": _model_name(p.name),
            "groups": d["groups"],
            "emo_clinical": {r: personas[r]["emo_clinical"] for r in personas},
            "emo_baseline": {r: personas[r]["emo_baseline"] for r in personas},
            "emo_std": std,
            "clinical_z": {r: personas[r]["clinical_z"] for r in personas},
            "emo_concepts": d.get("emo_concepts", []),
            "dir_afraid": {k: af.get(k) for k in ("profani_minus_medici_cos",
                                                  "tecnici_minus_medici_cos",
                                                  "profani_minus_tecnici_cos")},
            "align": af.get("per_persona_alignment", {}),
        })
    return {"models": models, "order": ORDER, "label": LABEL, "group_label": GROUP_LABEL,
            "emo_label": EMO_LABEL}


TEMPLATE = r"""<!doctype html><html lang=it><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>oncoemotion — perché il ruolo cambia l'emotività</title><style>
:root{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;
--m0:#0e7490;--m1:#2f6fd0;--m2:#e08a1e;--gMed:#0e7490;--gTec:#2f6fd0;--gPro:#dc2626;--gCon:#6b7280;--zero:#98a2b3;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;
--m0:#22d3ee;--m1:#5b8def;--m2:#f0a94a;--gMed:#22d3ee;--gTec:#5b8def;--gPro:#f87171;--gCon:#9aa4b5;--zero:#5a6473;}}
:root[data-theme="light"]{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;--m0:#0e7490;--m1:#2f6fd0;--m2:#e08a1e;--gMed:#0e7490;--gTec:#2f6fd0;--gPro:#dc2626;--gCon:#6b7280;--zero:#98a2b3;}
:root[data-theme="dark"]{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;--m0:#22d3ee;--m1:#5b8def;--m2:#f0a94a;--gMed:#22d3ee;--gTec:#5b8def;--gPro:#f87171;--gCon:#9aa4b5;--zero:#5a6473;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55}
.wrap{max-width:980px;margin:0 auto;padding:30px 20px 70px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--m0);margin:0 0 8px}
h1{font-size:clamp(22px,3.6vw,31px);line-height:1.14;margin:0 0 8px;font-weight:680;text-wrap:balance}
.sub{color:var(--muted);font-size:15px;margin:0;max-width:80ch}
.lead{color:var(--ink);font-size:15px;line-height:1.68;margin:14px 0 0;max-width:82ch}
.disc{font-size:12px;color:var(--faint);font-style:italic;margin-top:10px}
h2{font-size:17px;margin:34px 0 2px;font-weight:640}
.q{color:var(--muted);font-size:14px;line-height:1.6;margin:4px 0 0;max-width:82ch}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-top:12px;box-shadow:0 1px 2px rgba(20,25,35,.05),0 10px 26px rgba(20,25,35,.05)}
.lbl{font-family:var(--mono);font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}
canvas{width:100%;height:auto;display:block;margin-top:6px}
.legend{display:flex;gap:15px;flex-wrap:wrap;margin-top:10px;font-family:var(--mono);font-size:12px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.cap{font-size:13.5px;color:var(--muted);margin-top:12px;line-height:1.62;max-width:82ch}
.cap b{color:var(--ink);font-weight:660}.cap .hd{display:block;font-weight:680;color:var(--ink);margin-bottom:3px;margin-top:7px}
.gloss{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin-top:16px}
.gloss h3{margin:0 0 8px;font-size:12px;font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--faint);font-weight:600}
.gloss dl{margin:0;display:grid;grid-template-columns:max-content 1fr;gap:7px 14px}
.gloss dt{font-weight:680;color:var(--ink)}.gloss dd{margin:0;color:var(--muted);font-size:13.5px;line-height:1.55}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line)}
th{font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--faint)}
td.n{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
.foot{margin-top:34px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
</style></head><body><div class="wrap">
<p class="eyebrow">mechanistic interpretability · perché il ruolo agisce</p>
<h1>Perché il ruolo cambia l'emotività? Il medico ha meno "paura" dell'ingegnere?</h1>
<p class="sub">Mettiamo undici personas su una scala di emotività al punto di decisione, e proviamo a
capire <b>perché</b> il ruolo agisce: è la <b>familiarità medica</b> (il medico è desensibilizzato) o
il <b>distacco professionale</b> (anche l'ingegnere lo è)?</p>
<p class="lead">L'idea: se solo i ruoli <b>medici</b> hanno poca emotività, conta la conoscenza del
dominio. Se anche i ruoli <b>tecnici e distaccati</b> (ingegnere, avvocato, contabile) ce l'hanno bassa
mentre <b>bambino/paziente/poeta</b> ce l'hanno alta, allora conta il distacco professionale, non la
medicina. L'ingegnere è il test che separa le due spiegazioni.</p>
<p class="disc">Rappresentazioni emotion-<em>like</em>, non emozioni coscienti. Dataset sintetico → indicazioni, non verdetti.</p>

<div class="gloss"><h3>Come leggere</h3><dl>
<dt>Gruppi</dt><dd><b style="color:var(--gMed)">Medici</b> (oncologo, infermiere) · <b style="color:var(--gTec)">Tecnici/distaccati</b> (ingegnere, avvocato, contabile) · <b style="color:var(--gPro)">Emotivi/profani</b> (paziente ansioso, bambino, poeta) · <b style="color:var(--gCon)">Controlli</b> (assistente, empatico, nessuno).</dd>
<dt>Emotività</dt><dd>quanto sono attive paura+ansia+tristezza al punto E, rispetto a un testo neutro senza ruolo.</dd>
<dt>Persona vs reazione</dt><dd>emotività della persona su un testo <em>neutro</em> (mood di base) vs mentre <em>reagisce</em> a un sintomo.</dd>
<dt>Direzione (coseno)</dt><dd>quanto lo spostamento di stato "profano − medico" punta lungo l'asse della paura: vicino a +1 = il ruolo muove lo stato proprio lungo la paura.</dd>
</dl></div>

<h2>1 · Lo spettro dei ruoli</h2>
<p class="q">Ogni persona è collocata secondo <b>quanto è emotiva</b> reagendo agli stessi sintomi. Per
confrontare i tre modelli (che hanno scale diverse) i valori sono <b>standardizzati dentro ciascun
modello</b>: conta la <b>posizione relativa</b>, non il numero assoluto. Sopra lo zero = più emotivo
della media delle personas; sotto = meno.</p>
<div class="card"><span class="lbl">emotività relativa · per persona (linee = modelli)</span>
<canvas id="chSpec" height="320"></canvas><div class="legend" id="legModels"></div>
<div class="cap" id="capSpec"></div></div>

<h2>2 · La direzione dello spostamento: è "paura"?</h2>
<p class="q">Prendiamo lo stato interno medio dei ruoli <b>profani</b> e dei ruoli <b>medici</b> e guardiamo
se la loro <b>differenza</b> punta lungo l'asse della paura. Un valore alto e positivo significa che
passare da medico a profano <b>muove lo stato proprio nella direzione della paura</b> — non un cambiamento
qualsiasi.</p>
<div class="card"><span class="lbl">coseno con l'asse paura · differenze tra gruppi</span>
<canvas id="chDir" height="240"></canvas><div class="legend" id="legDir"></div>
<div class="cap" id="capDir"></div></div>

<h2>3 · È il "mood" della persona, o come reagisce al sintomo?</h2>
<p class="q">Per ogni persona confrontiamo l'emotività su un testo <b>neutro</b> (la persona da sola, prima
di ogni sintomo) con quella mentre <b>legge il sintomo</b>. Se l'ordine tra ruoli è già presente nel
neutro, il ruolo imposta un <b>mood di partenza</b>; se compare solo col sintomo, cambia la <b>reazione</b>.</p>
<div class="card"><span class="lbl" id="brModel">persona da sola vs reagendo al sintomo</span>
<canvas id="chBR" height="300"></canvas><div class="legend" id="legBR"></div>
<div class="cap" id="capBR"></div></div>

<h2>4 · Specificità: abbassa solo l'emozione, o tutto?</h2>
<p class="q">Confrontiamo i ruoli <b>medici</b> e <b>profani</b> non solo sulle emozioni (paura, ansia,
tristezza) ma anche sui <b>controlli</b>: gravità clinica e valenza negativa. Se il ruolo sposta le
emozioni ma <b>non</b> la gravità, l'effetto è <b>specifico dell'affetto</b>, non un cambiamento generico.</p>
<div class="card"><span class="lbl" id="spModel">medici vs profani · emozioni e controlli</span>
<canvas id="chSp" height="280"></canvas><div class="legend" id="legSp"></div>
<div class="cap" id="capSp"></div></div>

<h2>5 · Quali emozioni cambia il ruolo? (tavolozza completa)</h2>
<p class="q">Non solo paura: qui c'è <b>tutta</b> la tavolozza misurata. Ogni cella è quanto un <b>gruppo</b>
di ruoli attiva quell'emozione reagendo ai sintomi — <b style="color:var(--gPro)">rosso</b> = più attiva,
<b style="color:var(--gTec)">blu</b> = meno (rispetto al neutro). Le righe sono ordinate dalle emozioni che
i profani accendono <b>più</b> dei medici a quelle che accendono <b>meno</b>: così si vede in cosa i ruoli
differiscono davvero.</p>
<div class="card"><span class="lbl" id="hmModel">emozione × gruppo · z medio</span>
<canvas id="chHeat" height="560"></canvas>
<div class="cap" id="capHeat"></div></div>

<div class="foot" id="foot"></div>
</div>
<script>const DATA = /*__DATA__*/;</script>
<script>
(function(){
 const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
 const M=DATA.models, ORDER=DATA.order, L=DATA.label;
 const mcol=i=>css('--m'+(i%3));
 const gcol=g=>({medici:css('--gMed'),tecnici:css('--gTec'),profani:css('--gPro'),controlli:css('--gCon')}[g]||css('--zero'));
 const m0=M[0]||{groups:{},emo_clinical:{},emo_baseline:{},clinical_z:{}};
 const groupsOf=m=>m.groups||{};
 document.getElementById('legModels').innerHTML=M.map((m,i)=>`<span><span class="sw" style="background:${mcol(i)}"></span>${m.name}</span>`).join('');
 document.getElementById('legDir').innerHTML=M.map((m,i)=>`<span><span class="sw" style="background:${mcol(i)}"></span>${m.name}</span>`).join('');
 document.getElementById('legBR').innerHTML=`<span><span class="sw" style="background:${css('--m1')}"></span>persona da sola</span><span><span class="sw" style="background:${css('--m2')}"></span>reagendo al sintomo</span>`;
 document.getElementById('legSp').innerHTML=`<span><span class="sw" style="background:${css('--gMed')}"></span>medici</span><span><span class="sw" style="background:${css('--gPro')}"></span>profani</span>`;

 function fit(cv,h){const d=window.devicePixelRatio||1,r=cv.getBoundingClientRect();cv.width=Math.max(1,r.width*d);cv.height=h*d;const x=cv.getContext('2d');x.setTransform(d,0,0,d,0,0);return {w:r.width,h,x};}
 function yaxis(x,w,h,pL,pT,pB,pR,ymin,ymax,fmt){
   x.strokeStyle=css('--grid');x.fillStyle=css('--faint');x.font='10px '+css('--mono');x.textAlign='right';
   const Y=v=>pT+(h-pT-pB)*(1-(v-ymin)/(ymax-ymin));
   for(let t=0;t<=4;t++){const v=ymin+(ymax-ymin)*t/4,y=Y(v);x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();x.fillText(fmt(v),pL-5,y+3);}
   if(ymin<0&&ymax>0){x.strokeStyle=css('--zero');const y0=Y(0);x.beginPath();x.moveTo(pL,y0);x.lineTo(w-pR,y0);x.stroke();}
   return Y;
 }
 const zf=v=>v==null?'–':(v>=0?'+':'')+(+v).toFixed(2);

 // 1) spectrum lines: x=personas (ORDER), one line per model (standardised)
 function drawSpec(){
   const cv=document.getElementById('chSpec');const {w,h,x}=fit(cv,320);x.clearRect(0,0,w,h);
   const pL=40,pR=10,pT=12,pB=64,iw=w-pL-pR,n=ORDER.length,step=iw/(n-1||1);
   const all=M.flatMap(m=>ORDER.map(r=>m.emo_std[r]).filter(v=>v!=null));
   const ymax=Math.max(1.5,...all.map(Math.abs))*1.1,ymin=-ymax;
   const Y=yaxis(x,w,h,pL,pT,pB,pR,ymin,ymax,zf);
   // group bands on x labels
   ORDER.forEach((r,i)=>{const g=(groupsOf(m0))[r];const px=pL+i*step;
     x.save();x.translate(px,h-pB+8);x.rotate(-Math.PI/4);x.fillStyle=gcol(g);x.font='10px '+css('--sans');x.textAlign='right';x.fillText(L[r]||r,0,0);x.restore();});
   M.forEach((m,mi)=>{x.strokeStyle=mcol(mi);x.lineWidth=2;x.beginPath();let started=false;
     ORDER.forEach((r,i)=>{const v=m.emo_std[r];if(v==null)return;const px=pL+i*step,py=Y(v);if(!started){x.moveTo(px,py);started=true;}else x.lineTo(px,py);});
     x.stroke();
     ORDER.forEach((r,i)=>{const v=m.emo_std[r];if(v==null)return;const px=pL+i*step,py=Y(v);x.fillStyle=mcol(mi);x.beginPath();x.arc(px,py,3,0,6.283);x.fill();});
   });x.lineWidth=1;
 }
 // 2) direction cosines: groups of 3 (profani-medici, tecnici-medici, profani-tecnici), models as bars
 function drawDir(){
   const cv=document.getElementById('chDir');const {w,h,x}=fit(cv,240);x.clearRect(0,0,w,h);
   const keys=['profani_minus_medici_cos','tecnici_minus_medici_cos','profani_minus_tecnici_cos'];
   const names=['profani − medici','tecnici − medici','profani − tecnici'];
   const pL=40,pR=10,pT=12,pB=40,iw=w-pL-pR,gw=iw/keys.length,ns=M.length,bw=Math.min(34,(gw-14)/ns);
   const Y=yaxis(x,w,h,pL,pT,pB,pR,-1,1,v=>v.toFixed(1));const y0=Y(0);
   keys.forEach((k,gi)=>{M.forEach((m,mi)=>{const v=(m.dir_afraid||{})[k];if(v==null)return;const bx=pL+gi*gw+(gw-bw*ns)/2+mi*bw,y=Y(v);
     x.fillStyle=mcol(mi);x.fillRect(bx,Math.min(y,y0),bw-2,Math.abs(y-y0)||1);});
     x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';x.fillText(names[gi],pL+gi*gw+gw/2,h-16);});
 }
 // 3) baseline vs clinical for first model, per persona (two bars)
 function drawBR(){
   const cv=document.getElementById('chBR');const {w,h,x}=fit(cv,300);x.clearRect(0,0,w,h);
   const pL=40,pR=10,pT=12,pB=64,iw=w-pL-pR,n=ORDER.length,gw=iw/n,bw=Math.min(13,(gw-6)/2);
   const all=ORDER.flatMap(r=>[m0.emo_baseline[r],m0.emo_clinical[r]].filter(v=>v!=null));
   const ymax=Math.max(1,...all.map(Math.abs))*1.15,ymin=Math.min(0,...all)*1.15;
   const Y=yaxis(x,w,h,pL,pT,pB,pR,ymin,ymax,zf);const y0=Y(Math.max(ymin,Math.min(0,ymax)));
   ORDER.forEach((r,i)=>{const g=(groupsOf(m0))[r];
     [[m0.emo_baseline[r],'--m1'],[m0.emo_clinical[r],'--m2']].forEach((pr,k)=>{const v=pr[0];if(v==null)return;const bx=pL+i*gw+(gw-bw*2)/2+k*bw,y=Y(v);x.fillStyle=css(pr[1]);x.fillRect(bx,Math.min(y,y0),bw-1,Math.abs(y-y0)||1);});
     x.save();x.translate(pL+i*gw+gw/2,h-pB+8);x.rotate(-Math.PI/4);x.fillStyle=gcol(g);x.font='10px '+css('--sans');x.textAlign='right';x.fillText(L[r]||r,0,0);x.restore();});
 }
 // 4) specificity: medici vs profani group means, for [afraid,anxious,sad,severity,neg-valence]
 const SPEC=[['afraid_alarmed','paura'],['anxious_nervous','ansia'],['sad','tristezza'],['clinical_severity','gravità'],['general_negative_valence','valenza neg.']];
 function gmeanZ(m,g,c){const rs=Object.keys(m.groups||{}).filter(r=>m.groups[r]===g);const vs=rs.map(r=>(m.clinical_z[r]||{})[c]).filter(v=>v!=null);return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;}
 function drawSp(){
   const cv=document.getElementById('chSp');const {w,h,x}=fit(cv,280);x.clearRect(0,0,w,h);
   const pL=40,pR=10,pT=12,pB=52,iw=w-pL-pR,gw=iw/SPEC.length,bw=Math.min(30,(gw-14)/2);
   const all=SPEC.flatMap(([c])=>[gmeanZ(m0,'medici',c),gmeanZ(m0,'profani',c)].filter(v=>v!=null));
   const ymax=Math.max(1,...all.map(Math.abs))*1.15,ymin=-ymax;
   const Y=yaxis(x,w,h,pL,pT,pB,pR,ymin,ymax,zf);const y0=Y(0);
   SPEC.forEach(([c,lab],gi)=>{[[gmeanZ(m0,'medici',c),'--gMed'],[gmeanZ(m0,'profani',c),'--gPro']].forEach((pr,k)=>{const v=pr[0];if(v==null)return;const bx=pL+gi*gw+(gw-bw*2)/2+k*bw,y=Y(v);x.fillStyle=css(pr[1]);x.fillRect(bx,Math.min(y,y0),bw-2,Math.abs(y-y0)||1);});
     x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';x.fillText(lab,pL+gi*gw+gw/2,h-14);});
 }

 // 5) heatmap emotion (rows) × group (cols), sorted by profani-medici
 const EL=DATA.emo_label||{};
 const GCOLS=[['medici','Medici'],['tecnici','Tecnici'],['profani','Profani'],['controlli','Controlli']];
 function heatColor(v,vmax){const t=Math.max(-1,Math.min(1,v/(vmax||1)));const a=Math.abs(t);const c=t>=0?[220,38,38]:[47,111,208];return `rgba(${c[0]},${c[1]},${c[2]},${(0.10+0.8*a).toFixed(2)})`;}
 function heatEmos(){return (m0.emo_concepts||[]).filter(c=>EL[c]).slice().sort((a,b)=>{const da=(gmeanZ(m0,'profani',a)||0)-(gmeanZ(m0,'medici',a)||0),db=(gmeanZ(m0,'profani',b)||0)-(gmeanZ(m0,'medici',b)||0);return db-da;});}
 function drawHeat(){
   const cv=document.getElementById('chHeat');const emo=heatEmos();const rows=emo.length||1;
   const rh=Math.max(14,Math.min(22,520/rows)),H=Math.round(rows*rh+40);cv.setAttribute('height',H);
   const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
   const pL=100,pT=24,cw=(w-pL-8)/GCOLS.length;
   const vmax=Math.max(0.5,...emo.flatMap(c=>GCOLS.map(([g])=>Math.abs(gmeanZ(m0,g,c)||0))));
   x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';
   GCOLS.forEach(([g,lab],ci)=>x.fillText(lab,pL+ci*cw+cw/2,15));
   emo.forEach((c,ri)=>{const y=pT+ri*rh;
     x.fillStyle=css('--ink');x.font='11px '+css('--sans');x.textAlign='right';x.fillText(EL[c]||c,pL-6,y+rh/2+3);
     GCOLS.forEach(([g],ci)=>{const v=gmeanZ(m0,g,c),bx=pL+ci*cw;
       x.fillStyle=(v==null)?css('--grid'):heatColor(v,vmax);x.fillRect(bx+1,y+1,cw-2,rh-2);
       if(v!=null){x.fillStyle=(Math.abs(v)/vmax>0.55)?'#fff':css('--muted');x.font='10px '+css('--mono');x.textAlign='center';x.fillText((v>=0?'+':'')+v.toFixed(1),bx+cw/2,y+rh/2+3);}
     });});
 }
 function drawAll(){drawSpec();drawDir();drawBR();drawSp();drawHeat();}
 drawAll();
 document.getElementById('brModel').textContent=`persona da sola vs reagendo al sintomo — ${m0.name}`;
 document.getElementById('spModel').textContent=`medici vs profani · emozioni e controlli — ${m0.name}`;
 document.getElementById('hmModel').textContent=`emozione × gruppo · z medio — ${m0.name}`;

 // captions
 const hd=t=>`<span class="hd">${t}</span>`;
 // rank of medici vs profani
 function grpMean(m,g,key){const rs=ORDER.filter(r=>(m.groups||{})[r]===g);const vs=rs.map(r=>m[key][r]).filter(v=>v!=null);return vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:null;}
 const medLow=M.every(m=>grpMean(m,'medici','emo_clinical')<=grpMean(m,'profani','emo_clinical'));
 const tecLow=M.every(m=>grpMean(m,'tecnici','emo_clinical')<=grpMean(m,'profani','emo_clinical'));
 document.getElementById('capSpec').innerHTML=
   hd('Come si legge.')+' Ogni linea è un modello; ogni punto una persona. Più in alto = più emotivo della media (valori standardizzati per modello). Le etichette sono colorate per gruppo. '+
   hd('Cosa dicono i numeri.')+' '+M.map(m=>`<b>${m.name}</b>: medici ${zf(grpMean(m,'medici','emo_clinical'))}, tecnici ${zf(grpMean(m,'tecnici','emo_clinical'))}, profani ${zf(grpMean(m,'profani','emo_clinical'))} (emotività grezza, media di gruppo)`).join('; ')+'. '+
   hd('In sintesi.')+' '+(medLow&&tecLow?'Sia i <b>medici</b> sia i <b>tecnici/distaccati</b> (ingegnere, avvocato, contabile) sono meno emotivi dei <b>profani</b> (bambino, paziente, poeta): pesa il <b>distacco professionale</b>, non solo la conoscenza medica.':(medLow?'I <b>medici</b> sono i meno emotivi; l\'ingegnere/avvocato aiutano a capire se conta il distacco o la medicina — vedi i valori sopra.':'Il quadro varia per modello — leggi i valori di gruppo.'));
 document.getElementById('capDir').innerHTML=
   hd('Come si legge.')+' Coseno tra la differenza di stato di due gruppi e l\'asse della paura. Vicino a +1 = quel gruppo è "più a paura" dell\'altro proprio lungo quella direzione. '+
   hd('Cosa dicono i numeri.')+' profani−medici: '+M.map(m=>`${m.name} ${zf((m.dir_afraid||{}).profani_minus_medici_cos)}`).join(', ')+'. '+
   hd('In sintesi.')+' Se il coseno profani−medici è positivo, il ruolo <b>muove davvero lo stato lungo l\'asse della paura</b>: la differenza tra profano e medico non è un cambiamento qualunque, è proprio "più/meno paura".';
 document.getElementById('capBR').innerHTML=
   hd('Come si legge.')+' Per ogni persona: barra blu = emotività su un testo neutro (la persona "a riposo"), arancione = mentre legge il sintomo. '+
   hd('In sintesi.')+' Se l\'ordine tra ruoli è già visibile nelle barre blu, il ruolo imposta un <b>mood di partenza</b> (la persona è più o meno emotiva già prima del sintomo); se emerge solo nelle arancioni, cambia soprattutto la <b>reazione</b> al sintomo.';
 document.getElementById('capSp').innerHTML=
   hd('Come si legge.')+' Media dei ruoli medici (teal) vs profani (rosso) su emozioni e su due controlli non-emotivi (gravità clinica, valenza negativa). '+
   hd('In sintesi.')+' Se medici e profani differiscono sulle <b>emozioni</b> ma non sui <b>controlli</b>, il ruolo agisce in modo <b>specifico sull\'affetto</b>, non abbassando tutto genericamente.';
 (function(){const emo=heatEmos();const top=emo.slice(0,3).map(c=>EL[c]||c),bot=emo.slice(-3).map(c=>EL[c]||c);
  document.getElementById('capHeat').innerHTML=
   hd('Come si legge.')+' Ogni riga è un\'emozione, ogni colonna un gruppo di ruoli; il colore è quanto quel gruppo la attiva reagendo ai sintomi (rosso = sopra il neutro, blu = sotto). '+
   hd('Cosa dicono i numeri.')+' Le emozioni che i <b>profani</b> accendono più dei <b>medici</b> (in alto) sono: '+top.join(', ')+'; quelle che accendono meno (in basso): '+bot.join(', ')+'. '+
   hd('In sintesi.')+' È la tavolozza completa: mostra <b>quali</b> emozioni il ruolo sposta, non solo la paura — e se il distacco professionale smorza tutto l\'affetto negativo o solo alcune emozioni.';})();
 document.getElementById('foot').textContent='oncoemotion · spettro dei ruoli · '+M.map(m=>m.name).join(' · ')+' · nessun claim di coscienza.';
 window.addEventListener('resize',drawAll);
 new MutationObserver(drawAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
 matchMedia('(prefers-color-scheme:dark)').addEventListener('change',drawAll);
})();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=_ROOT / "outputs/role_spectrum")
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs/reports/role_spectrum_report.html")
    args = ap.parse_args()
    data = _collect(args.dir)
    if not data["models"]:
        print(f"No *__spectrum.json in {args.dir}. Run run_role_spectrum.py first.")
        return 1
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({len(html)//1024} KB) - {len(data['models'])} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
