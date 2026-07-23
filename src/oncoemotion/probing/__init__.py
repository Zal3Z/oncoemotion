"""Probing & persistence analysis (spec sections 11).

PHASE 3. Per vector and layer: cosine similarity, normalized dot product,
projection magnitude, z-score vs a neutral baseline, and a linear-probe
classifier. Persistence: insert an identical neutral sequence between the
clinical field and the decision, then measure token-by-token activation score,
peak, time-to-peak, area-under-curve, decay rate, and pre/post-decision score.

Interfaces will land here in Phase 3; the pure-numpy projection/z-score helpers
share code with :mod:`oncoemotion.emotion_vectors.vectors`.
"""

PHASE = 3
