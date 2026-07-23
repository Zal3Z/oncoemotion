"""Candidate generation: lexical + fuzzy (always available), embeddings/reranker
(optional, lazy). The baseline is fully deterministic and needs no ML stack.
"""

from oncoemotion.retrieval.base import Candidate, IndexEntry
from oncoemotion.retrieval.lexical_fuzzy import LexicalFuzzyRetriever, fuzzy_ratio

__all__ = ["Candidate", "IndexEntry", "LexicalFuzzyRetriever", "fuzzy_ratio"]
