#!/usr/bin/env python
"""Build a regulation-compliant ESMO AI 2026 abstract draft from final results.

The script never invents a result. Missing or failed gates produce explicit
placeholders or a narrower behavioral conclusion. Character counting follows the
ESMO rule: title + body, excluding spaces, maximum 2,000 characters.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _pct(value):
    return "[pending]" if value is None else f"{100 * value:.1f}%"


def _num(value, digits=3, signed=False):
    if value is None:
        return "[pending]"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _compact(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _count_excluding_spaces(text: str) -> int:
    return len(re.sub(r"\s", "", text))


def build(report: dict, config: dict) -> tuple[str, str, int]:
    title = config.get(
        "title",
        "Affective representations and role-conditioned instability in LLM-based PRO-CTCAE coding",
    )
    models = len(report.get("models", []))
    n_affective_pairs = config.get("affective_framing", {}).get(
        "expected_target_pairs", 69
    )
    n_term_pairs = config.get("primary", {}).get("expected_term_pairs", 112)
    n_intensity_pairs = n_term_pairs - n_affective_pairs
    primary = report.get("primary", {})
    dis = primary.get("label_disagreement", {})
    acc = primary.get("paired_accuracy_difference", {})
    mechanism = report.get("mechanistic_gate", {})
    affect = report.get("affective_profile", {})
    shift = affect.get("profile_rms_shift", {})
    framing = report.get("affective_framing_key_secondary", {})
    affective_modification = framing.get(
        "role_disagreement_modification_emotional_minus_neutral", {}
    )
    affective_ci = affective_modification.get("ci95", [None, None])
    specificity = report.get("symptom_intensity_specificity_control", {}).get(
        "role_disagreement_modification_emotional_minus_neutral", {}
    )
    n_records = dis.get("n_records")
    d_ci = dis.get("ci95", [None, None])
    a_ci = acc.get("ci95", [None, None])

    if affect.get("available"):
        affect_result = f"Affect-profile RMS shift was {_num(shift.get('estimate'))}."
    else:
        affect_result = "No affect axis passed the preregistered validation gate."

    if mechanism.get("available"):
        advantage = mechanism.get("attenuation_advantage_targeted_vs_random", {})
        mech_result = (
            f"Targeted versus random ablation changed role-disagreement by "
            f"{_num(advantage.get('estimate'), signed=True)} "
            f"(95% CI {_num(advantage.get('ci95', [None, None])[0], signed=True)} to "
            f"{_num(advantage.get('ci95', [None, None])[1], signed=True)})."
        )
    else:
        mech_result = "The preregistered causal comparison was unavailable."

    affective_detected = report.get("abstract_readiness", {}).get(
        "affective_role_modification_detected"
    )
    if mechanism.get("gate_passes") is True and affective_detected is True:
        conclusion = (
            "System roles modulated sensitivity to patient-affective wording, and targeted "
            "perturbation supported an affective contribution to code variability. Oncology "
            "LLM validation should report role prompts and affect-robustness."
        )
    elif affective_detected is True:
        causal_reason = (
            "the preregistered causal comparison was unavailable"
            if not mechanism.get("available")
            else "targeted ablation did not clear the matched random control"
        )
        conclusion = (
            "System roles modulated sensitivity to patient-affective wording. Internal affect "
            f"measurements remain explanatory, not causal, because {causal_reason}. Role "
            "prompts and affect-robustness should be reported in oncology LLM validation."
        )
    else:
        conclusion = (
            "The preregistered affective wording-by-role modification was not detected. "
            "Behavioral role effects and internal affect measurements should be interpreted "
            "separately, without attributing coding variability to emotion."
        )

    sections = {
        "Background": (
            "Patient-reported oncology symptoms carry affective language, but it is unknown "
            "whether system roles alter how large language models (LLMs) use it. We assessed "
            "affective sensitivity and role-conditioned variability in Italian PRO-CTCAE coding."
        ),
        "Methods": (
            f"We tested {models or '[pending]'} open-weight LLMs on {n_term_pairs} paired, "
            f"manually authored symptom scenarios, including {n_affective_pairs} with "
            f"preclassified patient-affective qualifiers and {n_intensity_pairs} "
            "symptom-intensity controls. Token-matched oncologist and identity-free "
            "prompts formed a paired 2x2 design. Endpoints were top-1 code disagreement and its "
            "emotional-minus-neutral modification, with equal-model hierarchical bootstrap. "
            "Out-of-fold, lexical-controlled affect directions were read before coding; targeted "
            "ablation was compared with matched random ablation. AI outputs had human oversight."
        ),
        "Results": (
            f"Across {n_records or '[pending]'} paired model-item evaluations, oncologist versus "
            f"control prompts changed the selected code in {_pct(dis.get('estimate'))} "
            f"(95% CI {_pct(d_ci[0])}-{_pct(d_ci[1])}). On affective-reaction pairs, emotional "
            f"wording changed cross-role disagreement by "
            f"{_num(affective_modification.get('estimate'), signed=True)} (95% CI "
            f"{_num(affective_ci[0], signed=True)} to {_num(affective_ci[1], signed=True)}); "
            f"the intensity-control estimate was {_num(specificity.get('estimate'), signed=True)}. "
            f"Accuracy difference was {_num(acc.get('estimate'), signed=True)} (95% CI "
            f"{_num(a_ci[0], signed=True)} to {_num(a_ci[1], signed=True)}). "
            f"{affect_result} {mech_result}"
        ),
        "Conclusions": conclusion,
    }
    body = "\n\n".join(f"{name}: {_compact(text)}" for name, text in sections.items())
    count = _count_excluding_spaces(title + body)
    return title, body, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path,
                        default=_ROOT / "outputs/reports/esmo_primary_analysis.json")
    parser.add_argument("--study-config", type=Path,
                        default=_ROOT / "configs/study_esmo_2026.yaml")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "outputs/reports/esmo_abstract_draft.md")
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="write an incomplete draft even when definitive values are missing",
    )
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.study_config.read_text(encoding="utf-8"))
    title, body, count = build(report, config)
    limit = int(config["reporting"]["abstract_character_limit_excluding_spaces"])
    status = "OK" if count <= limit else "TOO LONG"
    text = (
        f"# ESMO AI & Digital Oncology 2026 - abstract draft\n\n"
        f"Category: {config['reporting']['preferred_category']}\n\n"
        f"Preferred presentation: {config['reporting']['preferred_presentation']}\n\n"
        f"Characters excluding spaces (title + body): {count}/{limit} [{status}]\n\n"
        f"## Title\n\n{title}\n\n## Body\n\n{body}\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"{status}: {count}/{limit} characters excluding spaces -> {args.out}")
    if "[pending]" in text and not args.allow_placeholders:
        print("INCOMPLETE: definitive values are missing; export refused")
        return 3
    return 0 if count <= limit else 2


if __name__ == "__main__":
    raise SystemExit(main())
