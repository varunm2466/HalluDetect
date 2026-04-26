"""Layer 2 — Granular Claim Extraction and Atomic Decomposition."""

from .citation_aligner import CitationAligner, extract_citations
from .entailment import EntailmentScorer
from .srl import SemanticRoleLabeler
from .triplet_builder import TripletBuilder
from .uncertainty import UncertaintyEstimator

__all__ = [
    "CitationAligner",
    "EntailmentScorer",
    "SemanticRoleLabeler",
    "TripletBuilder",
    "UncertaintyEstimator",
    "extract_citations",
]
