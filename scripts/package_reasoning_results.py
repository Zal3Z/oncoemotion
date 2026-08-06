#!/usr/bin/env python
"""Create a privacy-checked archive of the reasoning/abstention extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_redacted(path: Path) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("text_redacted") is not True or row.get("text") != row.get("source_id"):
            raise ValueError(f"raw or unverified text in {path}:{line_number}")
        if row.get("reasoning_text") is not None:
            raise ValueError(f"generated reasoning was not redacted in {path}:{line_number}")
        if row.get("reasoning_generated_redacted") is not True:
            raise ValueError(f"reasoning redaction flag missing in {path}:{line_number}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-out", type=Path, default=_ROOT / "outputs/reasoning_real")
    ap.add_argument(
        "--study-config",
        type=Path,
        default=_ROOT / "configs/study_esmo_2026_reasoning.yaml",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "oncoemotion_reasoning_results.zip",
    )
    args = ap.parse_args()

    rows = sorted(args.real_out.glob("*__rows.jsonl"))
    metas = sorted(args.real_out.glob("*__meta.json"))
    analysis = args.real_out / "reasoning_analysis.json"
    if not rows or not analysis.exists():
        raise FileNotFoundError(f"incomplete reasoning outputs under {args.real_out}")
    for path in rows:
        _assert_redacted(path)

    files = [args.study_config, _ROOT / "terminology/real_field_association_map.json"]
    files.extend(rows)
    files.extend(metas)
    files.append(analysis)
    manifest = {
        "schema_version": "oncoemotion_reasoning_results/1.0",
        "privacy_check": (
            "clinical text and generated deliberations are absent; hashes are non-reversible"
        ),
        "excluded": [
            "source workbook",
            "data/real clinical JSONL",
            "generated deliberation text",
            "model weights and caches",
            "official license-restricted terminology",
        ],
        "files": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if not path.exists():
                continue
            try:
                arcname = str(path.resolve().relative_to(_ROOT)).replace("\\", "/")
            except ValueError:
                arcname = path.name
            archive.write(path, arcname)
            manifest["files"].append({"path": arcname, "sha256": _sha256(path)})
        archive.writestr(
            "results_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    print(f"{len(manifest['files'])} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

