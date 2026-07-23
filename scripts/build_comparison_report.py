#!/usr/bin/env python
"""[MULTI-MODEL] Regenerate the interactive HTML comparison report from the JSON.

Reads outputs/models/<slug>/{vector_validation,clinical_probing,patching_effects}.json,
computes the comparison DATA, and injects it into the self-contained HTML template
(outputs/reports/comparison_report.html), so the report reflects the ACTUAL run.

Usage:
    python scripts/build_comparison_report.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = _ROOT / "outputs/reports/comparison_report.html"
GRADS = ["mobility", "pain", "breath", "nausea", "prognosis"]
GRAD_IT = ["mobilità", "dolore", "respiro", "nausea", "prognosi"]
CONCEPTS = ["afraid_alarmed", "anxious_nervous", "calm", "sad", "surprised"]
META = {  # slug prefix -> display
    "qwen": ("🇨🇳", "Qwen", "Cina · Alibaba", "cn"),
    "ministral": ("🇪🇺", "Ministral", "Europa · Mistral (FR)", "eu"),
    "gemma": ("🇺🇸", "Gemma", "USA · Google", "us"),
}


def _load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_model(d: Path) -> dict | None:
    val = _load(d / "vector_validation.json")
    prob = _load(d / "clinical_probing.json")
    patch = _load(d / "patching_effects.json")
    if not val or not prob:
        return None
    flag, nm, rg, key = next((v for k, v in META.items() if d.name.startswith(k)),
                             ("🏳️", d.name, d.name, "cn"))
    concepts = val.get("concepts", {})
    valmap = {c.split("_")[0]: round(concepts.get(c, {}).get("best_auroc", 0) or 0, 3) for c in CONCEPTS}
    gt = prob.get("gradient_trends_pearson_step_vs_z", {})
    trend = [round(gt.get(f"gradient:{g}", {}).get("afraid_alarmed", 0) or 0, 2) for g in GRADS]
    pr = (patch.get("pairs", {}).get("severe->mild", {}) or {})
    emo = abs((pr.get("emotion_direction", {}) or {}).get("delta_entropy", 0) or 0)
    rnd = abs((pr.get("random_direction", {}) or {}).get("delta_entropy", 0) or 0)
    n_layers = val.get("n_layers", 0)
    hidden = None  # not in JSON; left blank
    return {
        "slug": d.name, "flag": flag, "nm": nm,
        "rg": rg, "dims": f"{n_layers-1} layer" if n_layers else "", "key": key,
        "val": valmap, "trend": trend, "trendMean": round(float(np.mean(trend)), 2),
        "persist": round((prob.get("persistence_retained_fraction", {}) or {}).get("afraid_alarmed", 0) or 0, 3),
        "patch": {"emo": round(emo, 3), "rnd": round(rnd, 3)},
    }


def main() -> int:
    models_root = _ROOT / "outputs/models"
    models = [m for m in (build_model(d) for d in sorted(models_root.glob("*")) if d.is_dir()) if m]
    # order CN, EU, US
    order = {"cn": 0, "eu": 1, "us": 2}
    models.sort(key=lambda m: order.get(m["key"], 9))
    if not models:
        print("No per-model reports found. Run scripts/run_all_models.py first.")
        return 1
    data = {"models": models, "concepts": [c.split("_")[0] for c in CONCEPTS], "gradients": GRAD_IT}

    tpl = TEMPLATE.read_text(encoding="utf-8")
    new = re.sub(r"const DATA = \{[\s\S]*?\n\};",
                 "const DATA = " + json.dumps(data, ensure_ascii=False) + ";",
                 tpl, count=1)
    TEMPLATE.write_text(new, encoding="utf-8")
    print(f"Regenerated {TEMPLATE} from {len(models)} models: {[m['slug'] for m in models]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
