"""Multi-source cascaded retrieval orchestrator (Layer 3 core).

Implements the design doc's *multi-pass* retrieval strategy:

    Pass 1: exact title + author list across all enabled sources (parallel).
    Pass 2: relaxed query — author surname + year if Pass 1 returns nothing
            usable from any source.
    Pass 3: DPR semantic rerank over whatever candidates we collected.

The agent returns a deduplicated, score-sorted list of ``RetrievedRecord``
candidates per ``Citation``, ready for Layer 4 record linkage.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy

import httpx

from ..config import RetrievalConfig
from ..logging_setup import get_logger
from ..types import Citation, RetrievedRecord
from .base_agent import BaseSource
from .dpr import DprReranker
from .sources.arxiv import ArxivSource
from .sources.crossref import CrossrefSource
from .sources.pubmed import PubmedSource
from .sources.semantic_scholar import SemanticScholarSource

log = get_logger(__name__)


_SOURCE_REGISTRY: dict[str, type[BaseSource]] = {
    "crossref": CrossrefSource,
    "arxiv": ArxivSource,
    "semantic_scholar": SemanticScholarSource,
    "pubmed": PubmedSource,
}


class RetrievalAgent:
    """Coordinates ``BaseSource`` adapters + a DPR reranker."""

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self.sources: list[BaseSource] = []
        for name in self.config.enabled_sources:
            cls = _SOURCE_REGISTRY.get(name)
            if not cls:
                log.warning("unknown retrieval source", source=name)
                continue
            kwargs = {"http": self.config.http}
            specific = getattr(self.config, name, None)
            if specific is not None:
                kwargs["config"] = specific
            self.sources.append(cls(**kwargs))
        self.dpr = DprReranker(self.config.dpr)

    # ------------------------------------------------------------------------

    async def retrieve(self, citation: Citation, *, client: httpx.AsyncClient | None = None) -> list[RetrievedRecord]:
        """Cascaded multi-pass retrieval for one ``Citation``."""

        owns_client = client is None
        client = client or httpx.AsyncClient(
            timeout=self.config.http.timeout_s,
            follow_redirects=True,
        )
        try:
            records = await self._pass(client, citation)
            if not self._has_useful(records):
                relaxed = self._relax(citation)
                if relaxed:
                    records.extend(await self._pass(client, relaxed))
            records = self._dedupe(records)
            return self.dpr.rerank(citation, records)
        finally:
            if owns_client:
                await client.aclose()

    async def retrieve_batch(self, citations: list[Citation]) -> dict[str, list[RetrievedRecord]]:
        async with httpx.AsyncClient(timeout=self.config.http.timeout_s, follow_redirects=True) as client:
            tasks = [self.retrieve(c, client=client) for c in citations]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, list[RetrievedRecord]] = {}
        for cit, res in zip(citations, results):
            key = cit.raw or (cit.title or "untitled")
            if isinstance(res, Exception):
                log.warning("retrieval failed", citation=key, error=str(res))
                out[key] = []
            else:
                out[key] = res
        return out

    # ------------------------------------------------------------------------

    async def _pass(self, client: httpx.AsyncClient, citation: Citation) -> list[RetrievedRecord]:
        active = [s for s in self.sources if not s.disabled]
        if not active:
            return []
        tasks = [src.query(client, citation) for src in active]
        per_source = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[RetrievedRecord] = []
        for src, res in zip(active, per_source):
            if isinstance(res, Exception):
                log.warning("source raised", source=src.name, error=str(res))
                continue
            out.extend(res)
        return out

    @staticmethod
    def _has_useful(records: list[RetrievedRecord]) -> bool:
        return any(r.title for r in records)

    @staticmethod
    def _relax(citation: Citation) -> Citation | None:
        relaxed = deepcopy(citation)
        relaxed.title = None
        if not (relaxed.authors and relaxed.year):
            return None
        relaxed.authors = relaxed.authors[:1]
        return relaxed

    @staticmethod
    def _dedupe(records: list[RetrievedRecord]) -> list[RetrievedRecord]:
        seen: set[tuple[str, str]] = set()
        out: list[RetrievedRecord] = []
        for r in records:
            key = (
                (r.doi or "").lower(),
                (r.arxiv_id or "").lower() or (r.title or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out
