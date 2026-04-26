"""arXiv Atom-feed source — preprint metadata."""

from __future__ import annotations

import re

import httpx

from ...config import ArxivConfig, HttpConfig
from ...types import Citation, Manifestation, RetrievedRecord
from ..base_agent import BaseSource

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_ID_FROM_URL = re.compile(r"abs/(\d{4}\.\d{4,5})(v\d+)?")


class ArxivSource(BaseSource):
    name = "arxiv"

    def __init__(self, config: ArxivConfig | None = None, http: HttpConfig | None = None) -> None:
        super().__init__(http=http)
        self.config = config or ArxivConfig()

    async def query(self, client: httpx.AsyncClient, citation: Citation) -> list[RetrievedRecord]:
        if self.offline:
            return []
        params = self._build_params(citation)
        if not params:
            return []
        resp = await self._safe_request(client, "GET", self.config.base_url, params=params)
        if resp is None or resp.status_code != 200:
            return []
        return self._parse_atom(resp.text)

    # ------------------------------------------------------------------------

    def _build_params(self, citation: Citation) -> dict[str, str]:
        if citation.arxiv_id:
            return {"id_list": citation.arxiv_id, "max_results": "5"}
        terms: list[str] = []
        if citation.title:
            terms.append(f'ti:"{citation.title}"')
        for a in citation.authors[:2]:
            terms.append(f'au:"{a}"')
        if not terms:
            return {}
        return {"search_query": " AND ".join(terms), "max_results": "5"}

    def _parse_atom(self, xml_text: str) -> list[RetrievedRecord]:
        try:
            from xml.etree import ElementTree as ET
        except ImportError:  # pragma: no cover
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        out: list[RetrievedRecord] = []
        for entry in root.findall(f"{_ATOM_NS}entry"):
            title_el = entry.find(f"{_ATOM_NS}title")
            summary_el = entry.find(f"{_ATOM_NS}summary")
            id_el = entry.find(f"{_ATOM_NS}id")
            pub_el = entry.find(f"{_ATOM_NS}published")
            authors = [
                (a.findtext(f"{_ATOM_NS}name") or "").strip()
                for a in entry.findall(f"{_ATOM_NS}author")
            ]
            authors = [a for a in authors if a]

            title = (title_el.text or "").strip() if title_el is not None else ""
            abstract = (summary_el.text or "").strip() if summary_el is not None else None
            arxiv_id = None
            url = None
            if id_el is not None and id_el.text:
                url = id_el.text.strip()
                m = _ARXIV_ID_FROM_URL.search(url)
                if m:
                    arxiv_id = m.group(1)

            year = None
            if pub_el is not None and pub_el.text:
                year = int(pub_el.text[:4])

            out.append(
                RetrievedRecord(
                    source=self.name,
                    title=title,
                    authors=authors,
                    year=year,
                    venue="arXiv",
                    arxiv_id=arxiv_id,
                    url=url,
                    abstract=abstract,
                    manifestation=Manifestation.ARXIV,
                )
            )
        return out
