"""PubMed E-utilities source — biomedical / clinical authority."""

from __future__ import annotations

import os
from xml.etree import ElementTree as ET

import httpx

from ...config import HttpConfig, PubmedConfig
from ...types import Citation, Manifestation, RetrievedRecord
from ..base_agent import BaseSource


class PubmedSource(BaseSource):
    name = "pubmed"
    # NCBI E-utilities: 3 req/s without an API key, 10 req/s with one. We cap
    # at 2 concurrent to leave headroom for the search → fetch chain (each
    # citation issues at least 2 sequential calls).
    concurrency_limit = 2

    def __init__(self, config: PubmedConfig | None = None, http: HttpConfig | None = None) -> None:
        super().__init__(http=http)
        self.config = config or PubmedConfig()

    async def query(self, client: httpx.AsyncClient, citation: Citation) -> list[RetrievedRecord]:
        if self.offline:
            return []
        term = self._build_term(citation)
        if not term:
            return []

        common = self._common_params()
        search_url = f"{self.config.base_url}/esearch.fcgi"
        resp = await self._safe_request(
            client, "GET", search_url, params={**common, "db": "pubmed", "term": term, "retmax": "5"}
        )
        if resp is None or resp.status_code != 200:
            return []
        try:
            tree = ET.fromstring(resp.text)
        except ET.ParseError:
            return []
        ids = [el.text for el in tree.findall(".//IdList/Id") if el.text]
        if not ids:
            return []

        fetch_url = f"{self.config.base_url}/efetch.fcgi"
        resp2 = await self._safe_request(
            client, "GET", fetch_url, params={**common, "db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        )
        if resp2 is None or resp2.status_code != 200:
            return []
        return self._parse_efetch(resp2.text)

    # ------------------------------------------------------------------------

    def _common_params(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if (k := os.environ.get(self.config.api_key_env)):
            out["api_key"] = k
        if (t := os.environ.get(self.config.tool_env)):
            out["tool"] = t
        if (e := os.environ.get(self.config.email_env)):
            out["email"] = e
        return out

    def _build_term(self, citation: Citation) -> str:
        if citation.doi:
            return f"{citation.doi}[doi]"
        terms: list[str] = []
        if citation.title:
            terms.append(f'"{citation.title}"[Title]')
        if citation.authors:
            terms.append(f'"{citation.authors[0]}"[Author]')
        if citation.year:
            terms.append(f"{citation.year}[PDAT]")
        return " AND ".join(terms)

    def _parse_efetch(self, xml_text: str) -> list[RetrievedRecord]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        out: list[RetrievedRecord] = []
        for art in root.findall(".//PubmedArticle"):
            title_el = art.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            authors: list[str] = []
            for a in art.findall(".//AuthorList/Author"):
                last = a.findtext("LastName") or ""
                first = a.findtext("ForeName") or a.findtext("Initials") or ""
                full = f"{first} {last}".strip()
                if full:
                    authors.append(full)
            venue = art.findtext(".//Journal/Title")
            year_text = art.findtext(".//JournalIssue/PubDate/Year") or art.findtext(".//JournalIssue/PubDate/MedlineDate")
            year = None
            if year_text:
                digits = "".join(ch for ch in year_text if ch.isdigit())
                if len(digits) >= 4:
                    year = int(digits[:4])
            doi = None
            for elocid in art.findall(".//ELocationID"):
                if elocid.attrib.get("EIdType") == "doi" and elocid.text:
                    doi = elocid.text.strip()
                    break
            pmid = art.findtext(".//PMID")
            abstract = " ".join(
                "".join(el.itertext()) for el in art.findall(".//Abstract/AbstractText")
            ).strip() or None

            out.append(
                RetrievedRecord(
                    source=self.name,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    doi=doi,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    abstract=abstract,
                    manifestation=Manifestation.JOURNAL,
                    extras={"pmid": pmid},
                )
            )
        return out
