"""Field-weighted similarity (MARC-AI inspired) for record linkage."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import FieldWeights
from ..types import Citation, RetrievedRecord
from .jaro_winkler import enhanced_jaro_winkler


@dataclass
class SimilarityBreakdown:
    overall: float
    by_field: dict[str, float]


class FieldSimilarity:
    def __init__(
        self,
        weights: FieldWeights | None = None,
        *,
        prefix_scale: float = 0.10,
        suffix_scale: float = 0.05,
        rabin_karp_window: int = 6,
    ) -> None:
        self.weights = weights or FieldWeights()
        self.prefix_scale = prefix_scale
        self.suffix_scale = suffix_scale
        self.rabin_karp_window = rabin_karp_window

    # ------------------------------------------------------------------------

    def score(self, citation: Citation, record: RetrievedRecord) -> SimilarityBreakdown:
        title_sim = self._jw(citation.title or "", record.title or "")
        author_sim = self._authors(citation.authors, record.authors)
        venue_sim = self._jw(citation.venue or "", record.venue or "")
        year_sim = 1.0 if (citation.year and record.year and citation.year == record.year) else 0.0
        doi_sim = 1.0 if (citation.doi and record.doi and citation.doi.lower() == record.doi.lower()) else 0.0

        weighted = (
            self.weights.title * title_sim
            + self.weights.authors * author_sim
            + self.weights.venue * venue_sim
            + self.weights.year * year_sim
            + self.weights.doi * doi_sim
        )

        if doi_sim == 1.0:
            weighted = max(weighted, 0.99)

        return SimilarityBreakdown(
            overall=float(min(1.0, weighted)),
            by_field={
                "title": title_sim,
                "authors": author_sim,
                "venue": venue_sim,
                "year": year_sim,
                "doi": doi_sim,
            },
        )

    # ------------------------------------------------------------------------

    def _jw(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return enhanced_jaro_winkler(
            a, b,
            prefix_scale=self.prefix_scale,
            suffix_scale=self.suffix_scale,
            rabin_karp_window=self.rabin_karp_window,
        )

    def _authors(self, a: list[str], b: list[str]) -> float:
        if not a or not b:
            return 0.0
        sims = []
        for x in a[:5]:
            best = 0.0
            for y in b[:8]:
                best = max(best, self._jw(x, y))
            sims.append(best)
        return sum(sims) / len(sims) if sims else 0.0
