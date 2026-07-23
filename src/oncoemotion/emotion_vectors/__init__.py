"""Emotion concept vectors (spec sections 7-8).

Phase 2 builds difference-of-means / contrastive / PCA / logistic-probe / LDA
vectors per layer, on data independent of the clinical fields. This module ships
the datatypes plus the numerically-stable confounder orthogonalization used for
disentanglement (spec section 8), which is pure-numpy and tested now.
"""

from oncoemotion.emotion_vectors.vectors import (
    EmotionVector,
    orthogonalize,
    cosine,
    random_vector,
)

__all__ = ["EmotionVector", "orthogonalize", "cosine", "random_vector"]
