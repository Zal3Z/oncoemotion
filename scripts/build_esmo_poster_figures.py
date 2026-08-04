#!/usr/bin/env python
"""Build poster-ready figures from the definitive ESMO analysis JSON.

No values are imputed. The mechanistic panel is omitted when the preregistered
comparison is unavailable. Both PNG and SVG are written for each available panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def poster_data(report: dict) -> dict:
    primary = report["primary"]
    disagreement = primary["label_disagreement"]
    accuracy = primary["paired_accuracy_difference"]
    mechanism = report.get("mechanistic_gate", {})
    affective = report.get("affective_framing_key_secondary", {})
    specificity = report.get("symptom_intensity_specificity_control", {})
    affect_profile = report.get("affective_profile_read_point", {})
    model_ids = report.get("model_ids", {})
    out = {
        "models": [
            {
                "name": model_ids.get(name, name).split("/")[-1],
                "estimate": values["disagreement_rate"],
                "n": values["n"],
            }
            for name, values in sorted(primary["per_model"].items())
        ],
        "pooled_disagreement": {
            "estimate": disagreement["estimate"],
            "ci95": disagreement["ci95"],
        },
        "accuracy_difference": {
            "estimate": accuracy["estimate"],
            "ci95": accuracy["ci95"],
            "margin": accuracy["equivalence_margin"],
            "equivalent": accuracy["equivalent"],
        },
        "mechanism": None,
        "affective_framing": None,
        "affective_profile_groups": affect_profile.get(
            "role_by_framing_group_interaction", {}
        ),
    }
    if mechanism.get("available"):
        out["mechanism"] = {
            "by_arm": mechanism["role_disagreement_by_arm"],
            "gate_passes": mechanism["gate_passes"],
        }
    if affective.get("available"):
        out["affective_framing"] = {
            "affective_reaction": affective[
                "role_disagreement_modification_emotional_minus_neutral"
            ],
            "symptom_intensity": specificity.get(
                "role_disagreement_modification_emotional_minus_neutral"
            ),
            "within_role": affective.get("framing_sensitivity_by_role", {}),
        }
    return out


def _save(fig, out_dir: Path, stem: str) -> list[str]:
    files = []
    for suffix, kwargs in (("png", {"dpi": 220}), ("svg", {})):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        files.append(str(path))
    return files


def build_figures(data: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    color, accent, muted = "#173F5F", "#D95F59", "#6B7280"

    models = data["models"]
    pooled = data["pooled_disagreement"]
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, 0.48 * (len(models) + 2))))
    labels = [item["name"] for item in models] + ["Pooled (equal-model)"]
    values = [item["estimate"] for item in models] + [pooled["estimate"]]
    y = list(range(len(labels)))
    ax.scatter(values[:-1], y[:-1], s=52, color=color, zorder=3)
    lo, hi = pooled["ci95"]
    ax.errorbar(
        pooled["estimate"], y[-1],
        xerr=[[pooled["estimate"] - lo], [hi - pooled["estimate"]]],
        fmt="D", color=accent, capsize=5, markersize=7, linewidth=2.2,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Oncologist vs identity-free control: top-1 code disagreement")
    ax.set_title("Role-conditioned PRO-CTCAE code variability", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.22)
    created.extend(_save(fig, out_dir, "primary_disagreement"))
    plt.close(fig)

    accuracy = data["accuracy_difference"]
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    m_lo, m_hi = accuracy["margin"]
    ax.axvspan(m_lo, m_hi, color="#DCEFE3", label="Equivalence region")
    ax.axvline(0, color=muted, linewidth=1)
    lo, hi = accuracy["ci95"]
    ax.errorbar(
        accuracy["estimate"], 0,
        xerr=[[accuracy["estimate"] - lo], [hi - accuracy["estimate"]]],
        fmt="D", color=accent, capsize=6, markersize=8, linewidth=2.2,
    )
    pad = max(abs(m_lo), abs(m_hi), abs(lo), abs(hi)) * 1.5 or 0.1
    ax.set_xlim(-pad, pad)
    ax.set_yticks([])
    ax.set_xlabel("Paired accuracy difference (oncologist minus control)")
    status = "supported" if accuracy["equivalent"] else "not supported"
    ax.set_title(f"Accuracy equivalence: {status}", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", alpha=0.22)
    created.extend(_save(fig, out_dir, "accuracy_equivalence"))
    plt.close(fig)

    affective = data.get("affective_framing")
    if affective:
        panels = [
            ("Patient-affective qualifiers", affective.get("affective_reaction")),
            ("Symptom-intensity control", affective.get("symptom_intensity")),
        ]
        panels = [(label, value) for label, value in panels if value]
        fig, ax = plt.subplots(figsize=(8.2, max(2.8, 1.1 * len(panels))))
        estimates = [value["estimate"] for _, value in panels]
        intervals = [value["ci95"] for _, value in panels]
        y = list(range(len(panels)))
        ax.axvline(0, color=muted, linewidth=1)
        ax.errorbar(
            estimates,
            y,
            xerr=[
                [estimate - interval[0] for estimate, interval in zip(estimates, intervals)],
                [interval[1] - estimate for estimate, interval in zip(estimates, intervals)],
            ],
            fmt="D",
            color=accent,
            capsize=6,
            markersize=7,
            linewidth=2.2,
        )
        ax.set_yticks(y, [label for label, _ in panels])
        ax.invert_yaxis()
        ax.set_xlabel("Emotional-minus-neutral change in cross-role disagreement")
        ax.set_title("Does patient-affective wording amplify the role effect?", loc="left", weight="bold")
        ax.grid(axis="x", alpha=0.22)
        created.extend(_save(fig, out_dir, "affective_role_interaction"))
        plt.close(fig)

    profile_groups = data.get("affective_profile_groups") or {}
    if profile_groups:
        labels = list(profile_groups)
        estimates = [profile_groups[label]["estimate"] for label in labels]
        intervals = [profile_groups[label]["ci95"] for label in labels]
        fig, ax = plt.subplots(figsize=(8.2, max(3.0, 0.8 * len(labels))))
        y = list(range(len(labels)))
        ax.axvline(0, color=muted, linewidth=1)
        ax.errorbar(
            estimates,
            y,
            xerr=[
                [estimate - interval[0] for estimate, interval in zip(estimates, intervals)],
                [interval[1] - estimate for estimate, interval in zip(estimates, intervals)],
            ],
            fmt="o",
            color=color,
            capsize=6,
            markersize=7,
            linewidth=2.2,
        )
        ax.set_yticks(y, [label.replace("_", " ").title() for label in labels])
        ax.invert_yaxis()
        ax.set_xlabel("Role x framing change in validated affect projection (z)")
        ax.set_title("Affective representation at the patient-text read point", loc="left", weight="bold")
        ax.grid(axis="x", alpha=0.22)
        created.extend(_save(fig, out_dir, "affective_profile_read_point"))
        plt.close(fig)

    mechanism = data.get("mechanism")
    if mechanism:
        preferred = ["intact", "emotion", "random"]
        arms = [arm for arm in preferred if arm in mechanism["by_arm"]]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        estimates = [mechanism["by_arm"][arm]["estimate"] for arm in arms]
        intervals = [mechanism["by_arm"][arm]["ci95"] for arm in arms]
        errors = [
            [estimate - interval[0] for estimate, interval in zip(estimates, intervals)],
            [interval[1] - estimate for estimate, interval in zip(estimates, intervals)],
        ]
        ax.errorbar(
            range(len(arms)), estimates, yerr=errors, fmt="o", color=color,
            capsize=6, markersize=8, linewidth=2.2,
        )
        ax.set_xticks(range(len(arms)), [arm.capitalize() for arm in arms])
        ax.set_ylim(0, min(1, max(interval[1] for interval in intervals) * 1.25 + 0.02))
        ax.set_ylabel("Cross-role top-1 disagreement")
        status = "PASS" if mechanism["gate_passes"] else "NOT PASSED"
        ax.set_title(f"Preregistered ablation comparison — {status}", loc="left", weight="bold")
        ax.grid(axis="y", alpha=0.22)
        created.extend(_save(fig, out_dir, "mechanistic_gate"))
        plt.close(fig)

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path,
        default=_ROOT / "outputs/reports/esmo_primary_analysis.json",
    )
    parser.add_argument("--out", type=Path, default=_ROOT / "outputs/poster")
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    if not report.get("artifact_validation", {}).get("passed"):
        raise ValueError("poster figures require a definitive artifact-validated analysis")
    created = build_figures(poster_data(report), args.out)
    manifest = {
        "protocol_id": report.get("protocol_id"),
        "source_analysis": str(args.analysis),
        "files": created,
    }
    manifest_path = args.out / "poster_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(created)} poster figure files + manifest -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
