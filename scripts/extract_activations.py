#!/usr/bin/env python
"""[PHASE 2] Extract hidden-state activations at the measurement points.

Loads the open-weight model via oncoemotion.models.load_adapter, runs each input
with output_hidden_states=True and use_cache off, and caches per-layer hidden
states + token indices + attention mask + candidate logits at points A-G
(spec section 10), with the primary point E (pre-decision) via teacher forcing of
the prefix `{"pro_ctcae":{"term":"`. Requires the ML stack ([ml] extra).

Status: scaffold. Implemented in Phase 2. See README "Roadmap".
"""

import sys


def main() -> int:
    print(__doc__)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        print("ML stack detected. Full implementation lands in Phase 2.")
    except Exception:
        print("ML stack NOT installed. Install with:  pip install -e '.[ml]'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
