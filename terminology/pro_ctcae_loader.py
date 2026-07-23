#!/usr/bin/env python
"""CLI / compatibility shim for the PRO-CTCAE loader (spec section 14 tree).

Real loader logic lives in ``oncoemotion.terminology.pro_ctcae``. This shim lets
the file be run directly to sanity-check the terminology file:

    python terminology/pro_ctcae_loader.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oncoemotion.terminology.pro_ctcae import load_pro_ctcae  # noqa: E402


def main() -> int:
    lib = load_pro_ctcae()
    print(f"Loaded {len(lib)} PRO-CTCAE terms (expected 80).")
    print(f"Provenance note: {lib.metadata.get('provenance_note', '')[:80]}...")
    n_seeded = sum(1 for t in lib if t.provenance == "synthetic_dev")
    print(f"{n_seeded} terms carry synthetic dev Italian seeds.")
    return 0 if len(lib) == 80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
