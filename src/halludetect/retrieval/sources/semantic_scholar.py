"""Semantic Scholar Graph API source — enriched metadata + abstracts."""

from __future__ import annotations

import os

import httpx

from ...config import HttpConfig, SemanticScholarConfig
from ...types import Citation, Manifestation, RetrievedRecord
from ..base_agent import BaseSource

_FIELDS = "title,authors,year,venue,externalIds,abstract,publicationTypes,publicationVenue,url"


class SemanticScholarSource(BaseSource):
    name = "semantic_scholar"
    # Public S2 endpoint allows ~100 requests / 5 min without an API key
    # (≈ 0.33 req/s sustained). Keep concurrency low so the circuit breaker
    # only fires for *real* exhaustion, not bursts.
    concurrency_limit = 2

    def __init__(self, config: SemanticScholarConfig | None = None, http: HttpConfig | None = None) -> None:
        super().__init__(http=http)
        self.config = config or SemanticScholarConfig()

    async def query(self, client: httpx.AsyncClient, citation: Citation) -> list[RetrievedRecord]:
        if self.offline:
            return []
        headers = self._headers()
        if citation.doi:
            url = f"{self.config.base_url}/paper/DOI:{citation.doi}"
            resp = await self._safe_request(client, "GET", url, headers=headers, params={"fields": _FIELDS})
            if resp is not None and resp.status_code == 200:
                try:
                    return [self._to_record(resp.json())]
                except ValueError:
                    return []
        query = self._build_query(citation)
        if not query:
            return []
        url = f"{self.config.base_url}/paper/search"
        params = {"query": query, "limit": "5", "fields": _FIELDS}
        resp = await self._safe_request(client, "GET", url, headers=headers, params=params)
        if resp is None or resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        return [self._to_record(item) for item in data.get("data", []) if item]

    # ------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get(self.config.api_key_env)
        return {"x-api-key": api_key} if api_key else {}

    def _build_query(self, citation: Citation) -> str:
        parts: list[str] = []
        if citation.title:
            parts.append(citation.title)
        if citation.authors:
            parts.extend(citation.authors[:2])
        if citation.year:
            parts.append(str(citation.year))
        return " ".join(parts).strip()

    def _to_record(self, item: dict) -> RetrievedRecord:
        authors = [
            a.get("name", "").strip()
            for a in (item.get("authors") or [])
            if a.get("name")
        ]
        ext = item.get("externalIds") or {}
        venue_obj = item.get("publicationVenue") or {}
        venue = item.get("venue") or venue_obj.get("name")
        types = [t.lower() for t in (item.get("publicationTypes") or []) if t]

        manifestation = Manifestation.UNKNOWN
        if "journalarticle" in types:
            manifestation = Manifestation.JOURNAL
        elif "conference" in types:
            manifestation = Manifestation.CONFERENCE
        elif "review" in types or "preprint" in types or ext.get("ArXiv"):
            manifestation = Manifestation.PREPRINT if not ext.get("ArXiv") else Manifestation.ARXIV

        return RetrievedRecord(
            source=self.name,
            title=item.get("title") or "",
            authors=authors,
            year=item.get("year"),
            venue=venue,
            doi=ext.get("DOI"),
            arxiv_id=ext.get("ArXiv"),
            url=item.get("url"),
            abstract=item.get("abstract"),
            manifestation=manifestation,
            extras={"externalIds": ext, "publicationTypes": types},
        )
