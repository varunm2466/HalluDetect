"""Crossref REST API source — broad DOI-centered coverage."""

from __future__ import annotations

import os

import httpx

from ...config import CrossrefConfig, HttpConfig
from ...types import Citation, Manifestation, RetrievedRecord
from ..base_agent import BaseSource


class CrossrefSource(BaseSource):
    name = "crossref"

    def __init__(self, config: CrossrefConfig | None = None, http: HttpConfig | None = None) -> None:
        super().__init__(http=http)
        self.config = config or CrossrefConfig()

    # ------------------------------------------------------------------------

    async def query(self, client: httpx.AsyncClient, citation: Citation) -> list[RetrievedRecord]:
        if self.offline:
            return []
        params = self._build_params(citation)
        if not params:
            return []
        url = f"{self.config.base_url}/works"
        resp = await self._safe_request(client, "GET", url, params=params)
        if resp is None or resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        items = (data.get("message") or {}).get("items", []) or []
        return [self._to_record(it) for it in items if it]

    # ------------------------------------------------------------------------

    def _build_params(self, citation: Citation) -> dict[str, str]:
        params: dict[str, str] = {"rows": "5"}
        if citation.doi:
            return {"query.bibliographic": citation.doi, "rows": "5"}
        bib_parts: list[str] = []
        if citation.title:
            bib_parts.append(citation.title)
        if citation.authors:
            bib_parts.extend(citation.authors[:3])
        if citation.year:
            bib_parts.append(str(citation.year))
        if not bib_parts:
            return {}
        params["query.bibliographic"] = " ".join(bib_parts)
        mailto = os.environ.get(self.config.mailto_env)
        if mailto:
            params["mailto"] = mailto
        return params

    def _to_record(self, item: dict) -> RetrievedRecord:
        title = " ".join(item.get("title") or []) or ""
        authors = []
        for a in item.get("author") or []:
            given = a.get("given") or ""
            family = a.get("family") or ""
            full = (f"{given} {family}").strip() or a.get("name") or ""
            if full:
                authors.append(full)
        venue = " ".join(item.get("container-title") or []) or None
        year = None
        date_parts = (item.get("issued") or {}).get("date-parts") or [[]]
        if date_parts and date_parts[0]:
            first = date_parts[0][0]
            try:
                year = int(first) if first is not None else None
            except (TypeError, ValueError):
                year = None

        manifestation = self._infer_manifestation(item.get("type", ""), venue)
        return RetrievedRecord(
            source=self.name,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=item.get("DOI"),
            url=item.get("URL"),
            abstract=item.get("abstract"),
            manifestation=manifestation,
            score=float(item.get("score") or 0.0),
            extras={"type": item.get("type"), "publisher": item.get("publisher")},
        )

    @staticmethod
    def _infer_manifestation(type_str: str, venue: str | None) -> Manifestation:
        t = (type_str or "").lower()
        if "journal-article" in t:
            return Manifestation.JOURNAL
        if "proceedings" in t or "conference" in t:
            return Manifestation.CONFERENCE
        if "posted-content" in t or (venue and "arxiv" in venue.lower()):
            return Manifestation.PREPRINT
        return Manifestation.UNKNOWN
