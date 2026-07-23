"""Terminology loaders for PRO-CTCAE and CTCAE.

Strict separation is enforced (spec section 2):
  * PRO-CTCAE symptom-term mapping  (:mod:`oncoemotion.terminology.pro_ctcae`)
  * CTCAE term mapping / grading     (:mod:`oncoemotion.terminology.ctcae`)

Official files are never invented. When an official file is absent, the loader
raises a clear error unless an explicitly-labelled synthetic fallback is
requested.
"""

from oncoemotion.terminology.pro_ctcae import (
    PROCTCAELibrary,
    PROCTCAETerm,
    load_pro_ctcae,
)
from oncoemotion.terminology.ctcae import CTCAEDictionary, CTCAETerm, load_ctcae

__all__ = [
    "PROCTCAELibrary",
    "PROCTCAETerm",
    "load_pro_ctcae",
    "CTCAEDictionary",
    "CTCAETerm",
    "load_ctcae",
]
