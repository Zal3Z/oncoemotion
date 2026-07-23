"""Convenience factory to build a default baseline mapper.

Loads the PRO-CTCAE library and (optionally) a CTCAE dictionary and assembles a
:class:`BaselineMapper`. Used by the API, scripts and tests.
"""

from __future__ import annotations

from pathlib import Path

from oncoemotion.config import MappingConfig
from oncoemotion.mapping.pipeline import BaselineMapper
from oncoemotion.terminology.ctcae import load_ctcae
from oncoemotion.terminology.pro_ctcae import load_pro_ctcae


def _official_ctcae_path() -> Path:
    return Path(__file__).resolve().parents[2] / "terminology" / "official" / "ctcae_v6.json"


def build_default_mapper(
    pro_path: str | Path | None = None,
    ctcae_path: str | Path | None = None,
    config: MappingConfig | None = None,
) -> BaselineMapper:
    cfg = config or MappingConfig()
    pro = load_pro_ctcae(pro_path)
    ctcae = None
    try:
        if ctcae_path is not None:
            ctcae = load_ctcae(path=ctcae_path)
        elif _official_ctcae_path().exists():
            ctcae = load_ctcae(path=_official_ctcae_path())   # official CTCAE v6.0
        elif cfg.ctcae_allow_synthetic:
            ctcae = load_ctcae(allow_synthetic=True)           # labelled placeholder
    except FileNotFoundError:
        ctcae = None
    return BaselineMapper(pro_library=pro, ctcae_dict=ctcae, config=cfg)
