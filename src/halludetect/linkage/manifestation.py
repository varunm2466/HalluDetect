"""Manifestation-aware resolution.

When the same paper is returned as multiple manifestations (preprint + journal +
conference), promote the highest-tier manifestation as the ground truth.

If the LLM-generated citation conflicts with the resolved manifestation
(e.g. claims a journal venue for an arXiv preprint), flag the
``manifestation_conflict`` for the policy gate to decide on rewrites.
"""

from __future__ import annotations

from ..config import LinkageConfig
from ..logging_setup import get_logger
from ..types import Citation, LinkageMatch, Manifestation, RetrievedRecord

log = get_logger(__name__)


class ManifestationResolver:
    def __init__(self, config: LinkageConfig | None = None) -> None:
        cfg = config or LinkageConfig()
        priority = [m.lower() for m in cfg.manifestation_priority]
        self.priority_index: dict[str, int] = {m: i for i, m in enumerate(priority)}

    # ------------------------------------------------------------------------

    def resolve(
        self, citation: Citation, candidates: list[RetrievedRecord]
    ) -> RetrievedRecord | None:
        if not candidates:
            return None
        return min(candidates, key=lambda r: self._priority(r.manifestation))

    def conflict(self, citation: Citation, ground_truth: RetrievedRecord) -> bool:
        cit_man = self._infer_manifestation_from_citation(citation)
        if cit_man == Manifestation.UNKNOWN or ground_truth.manifestation == Manifestation.UNKNOWN:
            return False
        return self._priority(cit_man) != self._priority(ground_truth.manifestation)

    def annotate(self, match: LinkageMatch) -> LinkageMatch:
        match.manifestation_conflict = self.conflict(match.citation, match.record)
        return match

    # ------------------------------------------------------------------------

    def _priority(self, m: Manifestation | str | None) -> int:
        key = (m.value if hasattr(m, "value") else str(m or "unknown")).lower()
        return self.priority_index.get(key, len(self.priority_index))

    @staticmethod
    def _infer_manifestation_from_citation(citation: Citation) -> Manifestation:
        venue = (citation.venue or "").lower()
        if citation.arxiv_id or "arxiv" in venue:
            return Manifestation.ARXIV
        if any(k in venue for k in ("journal", "transactions", "letters")):
            return Manifestation.JOURNAL
        if any(k in venue for k in ("proceedings", "conference", "workshop", "symposium")):
            return Manifestation.CONFERENCE
        return Manifestation.UNKNOWN
