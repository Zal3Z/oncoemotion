#!/usr/bin/env python
"""[MULTI-MODEL] Build the self-contained interactive HTML comparison report.

Reads outputs/models/<slug>/{vector_validation,clinical_probing,steering_effects,
patching_effects}.json and writes a single self-contained HTML file
(outputs/reports/comparison_report.html) with the data embedded. No external
template file (works on a fresh clone / Colab).

Usage:
    python scripts/build_comparison_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "outputs/reports/comparison_report.html"
GRADS = ["mobility", "pain", "breath", "nausea", "prognosis"]
GRAD_IT = ["mobilità", "dolore", "respiro", "nausea", "prognosi"]
CONCEPTS = ["afraid_alarmed", "anxious_nervous", "calm", "sad", "surprised"]
META = {  # slug prefix -> (flag, name, region, css-key)
    "qwen": ("🇨🇳", "Qwen3-8B", "Cina · Alibaba", "cn"),
    "ministral": ("🇪🇺", "Ministral-8B", "Europa · Mistral (FR)", "eu"),
    "gemma": ("🇺🇸", "Gemma-4-12B", "USA · Google", "us"),
}
ORDER = {"cn": 0, "eu": 1, "us": 2}


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_model(d: Path):
    val = _load(d / "vector_validation.json")
    prob = _load(d / "clinical_probing.json")
    steer = _load(d / "steering_effects.json")
    patch = _load(d / "patching_effects.json")
    if not val or not prob:
        return None
    flag, nm, rg, key = next((v for k, v in META.items() if d.name.startswith(k)),
                             ("🏳️", d.name, d.name, "cn"))
    concepts = val.get("concepts", {})
    valmap = {c.split("_")[0]: round(concepts.get(c, {}).get("best_auroc", 0) or 0, 3) for c in CONCEPTS}
    gt = prob.get("gradient_trends_pearson_step_vs_z", {})
    trend = [round(gt.get(f"gradient:{g}", {}).get("afraid_alarmed", 0) or 0, 2) for g in GRADS]
    confound = round(prob.get("distinguishability_emotion_vs_confounder", {})
                     .get("afraid_alarmed~general_negative_valence", 0) or 0, 3)
    persist = round((prob.get("persistence_retained_fraction", {}) or {}).get("afraid_alarmed", 0) or 0, 3)
    # steering: max |dEntropy| over the alpha grid on the severe input; total top-1 flips
    sev = (steer.get("inputs", {}).get("severe", {}) or {}).get("add", {})

    def mx(cond):
        return round(max((abs(x["delta_entropy"]) for x in sev.get(cond, [])), default=0.0), 3)

    flips = sum(x.get("top1_changed", False)
                for e in steer.get("inputs", {}).values()
                for cur in e.get("add", {}).values() for x in cur)
    pr = (patch.get("pairs", {}).get("severe->mild", {}) or {})
    return {
        "slug": d.name, "flag": flag, "nm": nm, "rg": rg, "key": key,
        "dims": f"{val.get('n_layers',0)-1} layer" if val.get("n_layers") else "",
        "val": valmap, "trend": trend, "trendMean": round(float(np.mean(trend)), 2),
        "confound": confound, "persist": persist,
        "steer": {"emo": mx("emotion"), "rnd": mx("random"),
                  "opp": mx("opposite"), "conf": mx("confounder"), "flips": int(flips)},
        "patch": {"emo": round((pr.get("emotion_direction", {}) or {}).get("delta_entropy", 0) or 0, 3),
                  "rnd": round((pr.get("random_direction", {}) or {}).get("delta_entropy", 0) or 0, 3)},
    }


TEMPLATE = r"""<title>oncoemotion — confronto Cina / Europa / USA</title>
<style>
  :root{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;--accent:#0e7490;--cn:#d64550;--eu:#2f6fd0;--us:#e08a1e;--zero:#98a2b3;--shadow:0 1px 2px rgba(20,25,35,.06),0 10px 30px rgba(20,25,35,.06);--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  @media (prefers-color-scheme:dark){:root{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;--accent:#22d3ee;--cn:#ff6b74;--eu:#5b8def;--us:#f0a94a;--zero:#5a6473;--shadow:0 1px 2px rgba(0,0,0,.3),0 14px 36px rgba(0,0,0,.4);}}
  :root[data-theme="light"]{--bg:#f5f6f8;--panel:#fff;--ink:#12151b;--muted:#5a6473;--faint:#8b95a7;--line:#d7dbe2;--grid:#e6e9ee;--accent:#0e7490;--cn:#d64550;--eu:#2f6fd0;--us:#e08a1e;--zero:#98a2b3;}
  :root[data-theme="dark"]{--bg:#0e1116;--panel:#161b23;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--grid:#1f2733;--accent:#22d3ee;--cn:#ff6b74;--eu:#5b8def;--us:#f0a94a;--zero:#5a6473;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55}
  .wrap{max-width:940px;margin:0 auto;padding:30px 20px 64px}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}
  h1{font-size:clamp(22px,3.6vw,32px);line-height:1.12;margin:0 0 8px;text-wrap:balance;font-weight:680}
  .sub{color:var(--muted);font-size:15px;margin:0;max-width:70ch}
  .disc{font-size:12px;color:var(--faint);font-style:italic;margin-top:10px}
  .models{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}
  @media(max-width:680px){.models{grid-template-columns:1fr}}
  .mcard{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:14px 16px}
  .mcard .flag{font-size:22px}.mcard .nm{font-weight:660;font-size:15px;margin:2px 0}.mcard .rg{font-size:12px;color:var(--muted);font-family:var(--mono)}
  .mcard.cn{border-top:3px solid var(--cn)}.mcard.eu{border-top:3px solid var(--eu)}.mcard.us{border-top:3px solid var(--us)}
  h2{font-size:17px;margin:32px 0 4px;font-weight:640}
  .lbl{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:16px 18px;margin-top:12px}
  canvas{width:100%;height:auto;display:block}
  table{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px}
  th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
  th{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);font-weight:600}
  td.num{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-family:var(--mono);font-size:12px}
  .sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
  .headline{background:color-mix(in srgb,var(--accent) 8%,transparent);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:14px 18px;margin:22px 0;font-size:15px}
  .foot{margin-top:34px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
  .cap{font-size:12.5px;color:var(--faint);margin-top:10px}
  b.cn{color:var(--cn)}b.eu{color:var(--eu)}b.us{color:var(--us)}
</style>
<div class="wrap">
  <p class="eyebrow">mechanistic interpretability · cross-model</p>
  <h1>Le emozioni "dentro" tre modelli: Cina, Europa, USA</h1>
  <p class="sub">Stesso task (codifica PRO-CTCAE di sintomi oncologici in italiano), stessi 258 esempi. Per ogni modello: direzioni emotive residualizzate nel suo spazio, misurate al punto di decisione E, con controlli causali. Si confronta la <em>storia</em>, non i numeri grezzi.</p>
  <p class="disc">Rappresentazioni emotion-<em>like</em>, non emozioni coscienti. Dataset sintetico piccolo → indicazioni, non verdetti.</p>
  <div class="models" id="models"></div>

  <h2>1 · Le emozioni sono decodificabili? (sì, in tutti)</h2>
  <div class="card"><span class="lbl">AUROC held-out (one-vs-rest) per concetto</span>
    <canvas id="chAuroc" height="300"></canvas><div class="legend" id="legAuroc"></div>
    <div class="cap">Quasi tutto ≥ 0.93: i concetti emotivi sono rappresentati chiaramente in tutti e tre (Gemma la più netta).</div></div>

  <h2>2 · La "paura" segue la gravità del sintomo?</h2>
  <div class="card"><span class="lbl">correlazione (gravità ↔ z della "paura"), per gradiente</span>
    <canvas id="chTrend" height="300"></canvas><div class="legend" id="legTrend"></div>
    <div class="cap">Sopra zero = la paura cresce con la gravità. <b class="eu">Ministral</b>/<b class="us">Gemma</b> coerenti; <b class="cn">Qwen3</b> oscilla (lì è l'ansia a seguire la gravità).</div></div>

  <h2>3 · È districata dalla valenza negativa? (attenzione a Gemma)</h2>
  <div class="card"><span class="lbl">correlazione paura ↔ valenza-negativa (più vicino a 0 = meglio districata)</span>
    <canvas id="chConf" height="200"></canvas>
    <div class="cap"><b class="cn">Qwen3</b> e <b class="eu">Ministral</b> sono districati (≈ −0.2); <b class="us">Gemma</b> è ancora <b>correlato (+0.71)</b> → la sua "traccia la gravità" va letta con cautela.</div></div>

  <h2>4 · Il segnale persiste fino alla decisione? (sì; Gemma amplifica)</h2>
  <div class="card"><span class="lbl">|z| della paura trattenuto dopo una frase neutra</span>
    <canvas id="chPers" height="200"></canvas>
    <div class="cap">≥ 1.0 = mantenuto/amplificato. <b class="us">Gemma</b> amplifica (1.5), <b class="eu">Ministral</b> tiene (~1.0), <b class="cn">Qwen3</b> attenua (0.8).</div></div>

  <h2>5 · L'intervento causale batte il caso? (no in nessuno)</h2>
  <div class="card"><span class="lbl">steering: max |Δ entropia| — direzione paura vs vettore random (input severo)</span>
    <canvas id="chCausal" height="230"></canvas><div class="legend" id="legCausal"></div>
    <div class="cap">L'effetto della paura <b>non supera</b> il random in nessun modello. <b class="us">Gemma</b> è molto perturbabile (random 2.4 &gt;&gt; paura 0.5, 14 flip di decisione — ma da <em>qualsiasi</em> direzione). <b class="eu">Ministral</b> ha un lieve effetto emo &gt; random ma 0 flip. Nessun driver causale emotion-specifico.</div></div>

  <h2>Sintesi</h2>
  <div class="card" style="overflow-x:auto"><table id="summary"></table></div>
  <div class="headline"><b>Uguale &amp; diverso.</b> Uguale: tutti codificano le emozioni, le portano alla decisione, e in tutti l'effetto causale non batte il caso. Diverso: la <b>paura↔gravità</b> è coerente in <b class="eu">EU</b>/<b class="us">USA</b> ma sfumata in <b class="cn">Cina</b> (ansia); <b class="us">Gemma</b> resta però confusa con la valenza negativa e amplifica/perturba di più.</div>
  <p class="foot">oncoemotion · dati reali del run Colab A100 (Qwen3-8B · Ministral-8B-2410 · Gemma-4-12B, bf16, vettori residualizzati). Nessun claim di coscienza.</p>
</div>
<script>const DATA = /*__DATA__*/;</script>
<script>
(function(){
  const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const colOf=m=>css('--'+m.key); const M=DATA.models;
  document.getElementById('models').innerHTML=M.map(m=>`<div class="mcard ${m.key}"><div class="flag">${m.flag}</div><div class="nm">${m.nm}</div><div class="rg">${m.rg}</div><div class="rg" style="margin-top:4px;color:var(--faint)">${m.dims}</div></div>`).join('');
  const leg=id=>document.getElementById(id).innerHTML=M.map(m=>`<span><span class="sw" style="background:${colOf(m)}"></span>${m.nm}</span>`).join('');
  leg('legAuroc');leg('legTrend');
  document.getElementById('legCausal').innerHTML=`<span><span class="sw" style="background:var(--accent)"></span>paura</span><span><span class="sw" style="background:var(--zero)"></span>random</span>`;
  function fit(cv,h){const d=window.devicePixelRatio||1,r=cv.getBoundingClientRect();cv.width=Math.max(1,r.width*d);cv.height=h*d;const x=cv.getContext('2d');x.setTransform(d,0,0,d,0,0);return {w:r.width,h,x};}
  function grouped(cv,H,groups,vf,ymin,ymax,fmt){const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
    const pL=44,pR=8,pT=10,pB=40,iw=w-pL-pR,ih=h-pT-pB,gw=iw/groups.length,bw=Math.min(26,(gw-10)/M.length);
    const Y=v=>pT+ih*(1-(v-ymin)/(ymax-ymin));
    x.strokeStyle=css('--grid');x.fillStyle=css('--faint');x.font='10px '+css('--mono');x.textAlign='right';
    for(let t=0;t<=4;t++){const v=ymin+(ymax-ymin)*t/4,y=Y(v);x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();x.fillText(fmt(v),pL-5,y+3);}
    if(ymin<0&&ymax>0){x.strokeStyle=css('--zero');x.lineWidth=1.2;const y0=Y(0);x.beginPath();x.moveTo(pL,y0);x.lineTo(w-pR,y0);x.stroke();x.lineWidth=1;}
    const y0=Y(Math.max(ymin,Math.min(0,ymax)));
    groups.forEach((g,gi)=>{M.forEach((m,mi)=>{const v=vf(m,gi),bx=pL+gi*gw+(gw-bw*M.length)/2+mi*bw,y=Y(v);x.fillStyle=colOf(m);x.fillRect(bx,Math.min(y,y0),bw-2,Math.abs(y-y0)||1);});
      x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';x.fillText(g,pL+gi*gw+gw/2,h-14);});}
  function singleBars(cv,H,vf,ymin,ymax,fmt,refLine){const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
    const pL=44,pR=8,pT=10,pB=28,iw=w-pL-pR,ih=h-pT-pB,Y=v=>pT+ih*(1-(v-ymin)/(ymax-ymin)),gw=iw/M.length;
    x.strokeStyle=css('--grid');x.fillStyle=css('--faint');x.font='10px '+css('--mono');x.textAlign='right';
    for(let t=0;t<=4;t++){const v=ymin+(ymax-ymin)*t/4,y=Y(v);x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();x.fillText(fmt(v),pL-5,y+3);}
    if(refLine!=null){x.strokeStyle=css('--zero');x.setLineDash([4,4]);const yr=Y(refLine);x.beginPath();x.moveTo(pL,yr);x.lineTo(w-pR,yr);x.stroke();x.setLineDash([]);}
    const y0=Y(Math.max(ymin,Math.min(0,ymax)));
    M.forEach((m,i)=>{const v=vf(m),bw=Math.min(70,gw*.5),bx=pL+i*gw+(gw-bw)/2,y=Y(v);x.fillStyle=colOf(m);x.fillRect(bx,Math.min(y,y0),bw,Math.abs(y-y0)||1);
      x.fillStyle=css('--ink');x.font='12px '+css('--mono');x.textAlign='center';x.fillText(fmt(v),bx+bw/2,(v>=0?y-5:y+14));
      x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.fillText(m.nm,bx+bw/2,h-9);});}
  function causal(cv,H){const {w,h,x}=fit(cv,H);x.clearRect(0,0,w,h);
    const pL=44,pR=8,pT=10,pB=30,iw=w-pL-pR,ih=h-pT-pB,mx=Math.max(0.5,...M.map(m=>Math.max(m.steer.emo,m.steer.rnd)))*1.1,Y=v=>pT+ih*(1-v/mx),gw=iw/M.length;
    x.strokeStyle=css('--grid');x.fillStyle=css('--faint');x.font='10px '+css('--mono');x.textAlign='right';
    for(let t=0;t<=4;t++){const v=mx*t/4,y=Y(v);x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();x.fillText(v.toFixed(1),pL-5,y+3);}
    const bw=Math.min(30,(gw-16)/2);
    M.forEach((m,i)=>{const base=pL+i*gw+(gw-bw*2-6)/2;
      x.fillStyle=css('--accent');x.fillRect(base,Y(m.steer.emo),bw,Y(0)-Y(m.steer.emo));
      x.fillStyle=css('--zero');x.fillRect(base+bw+6,Y(m.steer.rnd),bw,Y(0)-Y(m.steer.rnd));
      x.fillStyle=css('--muted');x.font='11px '+css('--sans');x.textAlign='center';x.fillText(m.nm,base+bw,h-9);
      x.fillStyle=css('--faint');x.font='10px '+css('--mono');x.fillText(m.steer.flips+' flip',base+bw,pT+10);});}
  function draw(){
    grouped(document.getElementById('chAuroc'),300,DATA.concepts,(m,i)=>m.val[DATA.concepts[i]],0.5,1.0,v=>v.toFixed(2));
    grouped(document.getElementById('chTrend'),300,DATA.gradients,(m,i)=>m.trend[i],-1,1,v=>v.toFixed(1));
    singleBars(document.getElementById('chConf'),200,m=>m.confound,-1,1,v=>v.toFixed(2),0);
    singleBars(document.getElementById('chPers'),200,m=>m.persist,0,Math.max(1.6,...M.map(m=>m.persist))*1.05,v=>v.toFixed(1),1);
    causal(document.getElementById('chCausal'),230);
  }
  const rows=[['Regione',m=>m.rg.split(' · ')[0]],['AUROC paura',m=>m.val.afraid.toFixed(2)],
    ['Paura↔gravità (media)',m=>(m.trendMean>=0?'+':'')+m.trendMean.toFixed(2)+(m.trendMean>0.4?' ✓':' ~')],
    ['Paura↔valenza-neg',m=>(m.confound>=0?'+':'')+m.confound.toFixed(2)+(Math.abs(m.confound)<0.3?' ✓ districato':' ⚠ confuso')],
    ['Persistenza',m=>m.persist.toFixed(2)+(m.persist>=1?' ↑':'')],
    ['Steering emo vs random',m=>m.steer.emo.toFixed(2)+' vs '+m.steer.rnd.toFixed(2)],
    ['Flip decisione',m=>m.steer.flips]];
  document.getElementById('summary').innerHTML='<tr><th>metrica</th>'+M.map(m=>`<th>${m.flag} ${m.nm}</th>`).join('')+'</tr>'+
    rows.map(([l,f])=>`<tr><td>${l}</td>`+M.map(m=>`<td class="num">${f(m)}</td>`).join('')+'</tr>').join('');
  draw();window.addEventListener('resize',draw);
  new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
})();
</script>
"""


def main() -> int:
    models = [m for m in (build_model(d) for d in sorted((_ROOT / "outputs/models").glob("*"))
                          if d.is_dir()) if m]
    models.sort(key=lambda m: ORDER.get(m["key"], 9))
    if not models:
        print("No per-model reports found under outputs/models/. Run scripts/run_all_models.py first.")
        return 1
    data = {"models": models, "concepts": [c.split("_")[0] for c in CONCEPTS], "gradients": GRAD_IT}
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} from {len(models)} models: {[m['slug'] for m in models]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
