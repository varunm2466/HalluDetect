"""Record-linkage clustering + best-match selection."""

from __future__ import annotations

from ..config import LinkageConfig
from ..logging_setup import get_logger
from ..types import Citation, LinkageMatch, RetrievedRecord
from .field_weights import FieldSimilarity
from .manifestation import ManifestationResolver

log = get_logger(__name__)


class Deduper:
    """Pick the best ``RetrievedRecord`` for a ``Citation`` using field-weighted JW."""

    def __init__(self, config: LinkageConfig | None = None) -> None:
        cfg = config or LinkageConfig()
        self.threshold = cfg.jaro_winkler.base_threshold
        self.similarity = FieldSimilarity(
            weights=cfg.field_weights,
            prefix_scale=cfg.jaro_winkler.prefix_scale,
            suffix_scale=cfg.jaro_winkler.suffix_scale,
            rabin_karp_window=cfg.jaro_winkler.rabin_karp_window,
        )
        self.manifestation = ManifestationResolver(cfg)

    # ------------------------------------------------------------------------

    def link(self, citation: Citation, candidates: list[RetrievedRecord]) -> list[LinkageMatch]:
        out: list[LinkageMatch] = []
        for rec in candidates:
            br = self.similarity.score(citation, rec)
            match = LinkageMatch(
                citation=citation,
                record=rec,
                similarity=br.overall,
                manifestation_resolved=rec.manifestation,
                field_breakdown=br.by_field,
            )
            self.manifestation.annotate(match)
            out.append(match)
        out.sort(key=lambda m: m.similarity, reverse=True)
        return out

    def best(self, citation: Citation, candidates: list[RetrievedRecord]) -> LinkageMatch | None:
        ranked = self.link(citation, candidates)
        if not ranked:
            return None
        top = ranked[0]
        if top.similarity < self.threshold and not top.field_breakdown.get("doi"):
            return None

        # Promote highest-tier manifestation as ground truth, then rebind the match.
        truth = self.manifestation.resolve(citation, [m.record for m in ranked])
        if truth is not None and truth is not top.record:
            for m in ranked:
                if m.record is truth:
                    top = m
                    break
        return top
