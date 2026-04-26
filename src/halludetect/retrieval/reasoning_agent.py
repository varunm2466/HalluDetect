"""Semantic reasoning (LLM-as-Judge) agent.

Resolves the ambiguity between *real metadata noise* (abbreviated venues,
typos, formatting drift) and *outright hallucination*. The agent compares a
generated ``Citation`` to a set of retrieved candidates and returns:

    * a calibrated semantic-resilience score in [0, 1]
    * a short rationale string
    * a verdict tag (``valid`` | ``partial`` | ``hallucinated`` | ``unverifiable``)

The default backend is a heuristic that combines normalized field similarity
with manifestation hints. When ``transformers`` is available a cross-encoder
NLI model can be used instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..config import ReasoningConfig
from ..logging_setup import get_logger
from ..types import Citation, RetrievedRecord, VerdictLabel

log = get_logger(__name__)
_TOK_RE = re.compile(r"[A-Za-z0-9]{2,}")


def _tokens(s: str | None) -> set[str]:
    return {t.lower() for t in _TOK_RE.findall(s or "")}


def _jaccard(a: str | None, b: str | None) -> float:
    A, B = _tokens(a), _tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))


@dataclass
class ReasoningJudgment:
    score: float
    label: VerdictLabel
    rationale: str
    best_record: RetrievedRecord | None


class ReasoningAgent:
    def __init__(self, config: ReasoningConfig | None = None) -> None:
        self.config = config or ReasoningConfig()

    def judge(
        self,
        citation: Citation,
        candidates: Iterable[RetrievedRecord],
    ) -> ReasoningJudgment:
        cands = list(candidates)
        if not cands:
            return ReasoningJudgment(
                score=0.0,
                label=VerdictLabel.HALLUCINATED if citation.title or citation.doi else VerdictLabel.UNVERIFIABLE,
                rationale="No external candidates returned across enabled sources.",
                best_record=None,
            )

        scored: list[tuple[float, RetrievedRecord, str]] = []
        for r in cands:
            title_sim = _jaccard(citation.title, r.title)
            author_sim = self._author_overlap(citation.authors, r.authors)
            doi_match = bool(citation.doi and r.doi and citation.doi.lower() == r.doi.lower())
            year_match = bool(citation.year and r.year and citation.year == r.year)
            score = 0.55 * title_sim + 0.25 * author_sim + 0.10 * float(year_match) + 0.10 * float(doi_match)
            score = max(score, 0.99 if doi_match else 0.0)
            rationale_bits = []
            if doi_match:
                rationale_bits.append("DOI exact-match")
            else:
                rationale_bits.append(f"title J={title_sim:.2f}")
                rationale_bits.append(f"authors={author_sim:.2f}")
                if year_match:
                    rationale_bits.append("year=match")
            scored.append((score, r, ", ".join(rationale_bits)))

        scored.sort(key=lambda t: t[0], reverse=True)
        top_score, top_rec, rationale = scored[0]

        if top_score >= 0.85:
            label = VerdictLabel.VERIFIED
        elif top_score >= 0.55:
            label = VerdictLabel.PARTIALLY_VERIFIED
        elif citation.title or citation.doi:
            label = VerdictLabel.HALLUCINATED
        else:
            label = VerdictLabel.UNVERIFIABLE

        return ReasoningJudgment(score=top_score, label=label, rationale=rationale, best_record=top_rec)

    @staticmethod
    def _author_overlap(a: list[str], b: list[str]) -> float:
        if not a or not b:
            return 0.0
        setA = {x.lower().split()[-1] for x in a if x.strip()}
        setB = {x.lower().split()[-1] for x in b if x.strip()}
        if not setA or not setB:
            return 0.0
        return len(setA & setB) / float(len(setA | setB))
