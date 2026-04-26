"""Layer 1 — Adversarial Threat Detection and Context Sanitization."""

from .delimiters import build_masked_user_prompt, wrap_external_content
from .melon import MelonDefender, MelonResult, default_stub_llm
from .sentinel import SentinelClassifier, SentinelResult
from .threat_report import ThreatAggregator
from .trajectory import Trajectory, cosine_similarity

__all__ = [
    "MelonDefender",
    "MelonResult",
    "SentinelClassifier",
    "SentinelResult",
    "ThreatAggregator",
    "Trajectory",
    "build_masked_user_prompt",
    "cosine_similarity",
    "default_stub_llm",
    "wrap_external_content",
]
