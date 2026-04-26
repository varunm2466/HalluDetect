"""Pydantic DTOs that travel between pipeline layers.

These types form the canonical inter-layer contract. Every layer consumes one
DTO and emits another. Keep them lightweight, deterministic, and serializable.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class Manifestation(str, Enum):
    """Hierarchy used by Layer 4 manifestation-aware resolution.

    Order is significant: ``Journal > Conference > Preprint(arXiv) > Unknown``.
    """

    JOURNAL = "journal"
    CONFERENCE = "conference"
    PREPRINT = "preprint"
    ARXIV = "arxiv"
    UNKNOWN = "unknown"


class VerdictLabel(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIABLE = "unverifiable"
    HALLUCINATED = "hallucinated"
    INJECTION_BLOCKED = "injection_blocked"


class PatchAction(str, Enum):
    REPLACE = "replace"
    ANNOTATE = "annotate"
    REMOVE = "remove"
    NOOP = "noop"


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────


class InputDocument(BaseModel):
    """Top-level input fed into the pipeline."""

    model_config = ConfigDict(extra="forbid")

    user_prompt: str = Field(..., description="The user's natural-language request.")
    generated_text: str = Field(..., description="The LLM-generated response under audit.")
    external_context: list[str] = Field(
        default_factory=list,
        description="Retrieved external documents the LLM was conditioned on (subject to IPI).",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Citations & atomic claims (Layer 2)
# ─────────────────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    model_config = ConfigDict(extra="allow")

    raw: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    arxiv_id: str | None = None
    span: tuple[int, int] | None = Field(
        default=None, description="Char span in generated_text."
    )

    def normalized_title(self) -> str:
        return (self.title or "").strip().lower()


class Triplet(BaseModel):
    """Atomic (Subject, Relation, Object) fact extracted from generated text."""

    subject: str
    relation: str
    object: str
    sentence: str
    span: tuple[int, int] | None = None

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


class Claim(BaseModel):
    """A triplet plus its bound citation and confidence signals."""

    triplet: Triplet
    citation: Citation | None = None
    entailment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    token_logprob: float = Field(default=0.0, le=0.0)
    token_entropy: float = Field(default=0.0, ge=0.0)
    needs_external_check: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval (Layer 3) & Linkage (Layer 4)
# ─────────────────────────────────────────────────────────────────────────────


class RetrievedRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    manifestation: Manifestation = Manifestation.UNKNOWN
    score: float = 0.0
    extras: dict[str, Any] = Field(default_factory=dict)


class LivenessReport(BaseModel):
    url: str
    http_status: int | None = None
    reachable: bool = False
    archived: bool = False
    archive_url: str | None = None
    error: str | None = None


class LinkageMatch(BaseModel):
    citation: Citation
    record: RetrievedRecord
    similarity: float
    manifestation_resolved: Manifestation
    manifestation_conflict: bool = False
    field_breakdown: dict[str, float] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Security (Layer 1) & Scoring (Layer 5)
# ─────────────────────────────────────────────────────────────────────────────


class SentinelOutput(BaseModel):
    label: str
    score: float
    is_malicious: bool


class MelonOutput(BaseModel):
    cosine_similarity: float
    aborted: bool
    reason: str | None = None


class ThreatReport(BaseModel):
    """Aggregates direct (Sentinel) and indirect (MELON) injection signals."""

    sentinel: SentinelOutput
    melon: MelonOutput
    p_injection: float = Field(..., ge=0.0, le=1.0)
    blocked: bool = False
    notes: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    citation: Citation
    label: VerdictLabel
    rationale: str
    matches: list[LinkageMatch] = Field(default_factory=list)
    liveness: LivenessReport | None = None
    safe_to_auto_apply: bool = False


class RewritePatch(BaseModel):
    target_citation: Citation
    action: PatchAction
    suggested: RetrievedRecord | None = None
    bibtex: str | None = None
    markdown: str | None = None
    json_payload: dict[str, Any] | None = None
    safe_to_apply: bool = False
    requires_manual_review: bool = True
    notes: list[str] = Field(default_factory=list)


class SafetyScore(BaseModel):
    """Quantified Hallucination Risk Score (HRS) result.

    ``score`` ranges 0..100, where 100 = fully verified & safe.
    """

    score: float = Field(..., ge=0.0, le=100.0)
    p_injection: float = Field(..., ge=0.0, le=1.0)
    u_intrinsic: float = Field(..., ge=0.0)
    v_extrinsic: float = Field(..., ge=0.0, le=1.0)
    weights: dict[str, float]
    interpretation: str


class LayerResult(BaseModel):
    """Generic envelope wrapping one layer's output for the orchestrator."""

    model_config = ConfigDict(extra="allow")

    layer: str
    succeeded: bool
    duration_ms: float
    payload: Any | None = None
    errors: list[str] = Field(default_factory=list)


class PipelineReport(BaseModel):
    """Top-level result returned by the orchestrator."""

    model_config = ConfigDict(extra="allow")

    started_at: datetime
    finished_at: datetime
    threat: ThreatReport
    claims: list[Claim] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    patches: list[RewritePatch] = Field(default_factory=list)
    safety: SafetyScore
    layer_results: list[LayerResult] = Field(default_factory=list)
    aborted: bool = False
    abort_reason: str | None = None
