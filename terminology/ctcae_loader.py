#!/usr/bin/env python
"""CLI / compatibility shim for the CTCAE loader (spec section 14 tree).

Real loader logic lives in ``oncoemotion.terminology.ctcae``. Run directly to
inspect a CTCAE file (defaults to the labelled synthetic placeholder):

    python terminology/ctcae_loader.py [--path PATH_TO_OFFICIAL_CTCAE_V6.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oncoemotion.terminology.ctcae import load_ctcae  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=None, help="Official CTCAE v6.0 file path")
    args = parser.parse_args()
    dct = load_ctcae(path=args.path, allow_synthetic=args.path is None)
    tag = "SYNTHETIC PLACEHOLDER" if dct.is_synthetic else "official"
    print(f"Loaded CTCAE dictionary version={dct.version} ({tag}) with {len(dct)} terms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
