"""halludetect — prompt-aware hallucinated-citation detection and correction."""

from .types import (
    Citation,
    Claim,
    InputDocument,
    LayerResult,
    PipelineReport,
    RetrievedRecord,
    RewritePatch,
    SafetyScore,
    ThreatReport,
    Triplet,
    Verdict,
)

__all__ = [
    "Claim",
    "Citation",
    "InputDocument",
    "LayerResult",
    "PipelineReport",
    "RetrievedRecord",
    "RewritePatch",
    "SafetyScore",
    "ThreatReport",
    "Triplet",
    "Verdict",
]

__version__ = "0.1.0"
