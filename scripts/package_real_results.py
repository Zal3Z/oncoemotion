#!/usr/bin/env python
"""Create a redaction-checked archive of real-field study results."""

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
        if row.get("model_generated") is not None:
            raise ValueError(f"free-generated text was not redacted in {path}:{line_number}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-out", type=Path, default=_ROOT / "outputs/real_fields")
    ap.add_argument("--model-out", type=Path, default=_ROOT / "outputs/models_real")
    ap.add_argument("--out", type=Path, default=_ROOT / "oncoemotion_real_results.zip")
    args = ap.parse_args()

    row_files = sorted(args.real_out.glob("*__rows.jsonl"))
    analysis_files = sorted(args.real_out.glob("*analysis.json"))
    if not row_files:
        raise FileNotFoundError(f"no real-field row artifacts under {args.real_out}")
    if not analysis_files:
        raise FileNotFoundError(f"no real-field analysis artifact under {args.real_out}")

    files = [
        _ROOT / "configs/study_esmo_2026_real.yaml",
        _ROOT / "terminology/real_field_association_map.json",
        _ROOT / "data/synthetic/emotion_dataset.jsonl",
    ]
    files.extend(row_files)
    files.extend(sorted(args.real_out.glob("*__meta.json")))
    files.extend(analysis_files)
    files.extend(sorted(args.model_out.glob("*/vector_validation.json")))
    files.extend(sorted(args.model_out.glob("*/pipeline_manifest.json")))
    missing = [path for path in files[:3] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required files missing: {missing}")
    for path in files:
        if path.name.endswith("__rows.jsonl"):
            _assert_redacted(path)

    manifest = {
        "schema_version": "oncoemotion_real_results/1.0",
        "privacy_check": "all exported row text equals non-reversible source_id",
        "excluded": [
            "source workbook",
            "data/real clinical JSONL",
            "raw activations",
            "model weights",
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
