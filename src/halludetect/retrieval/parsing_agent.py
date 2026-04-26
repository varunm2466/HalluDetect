"""Parsing & extraction agent.

Scans a workspace path or raw text and emits ``Citation`` objects. Supports:

    * ``.bib``  via bibtexparser
    * ``.tex``  via lightweight regex (``\\bibliography{}`` + ``\\cite{}``)
    * ``.md``   via reference-section + URL/DOI regex
    * ``.docx`` via python-docx (if installed)
    * raw strings via ``parse_text``

Falls back to plain-text scanning when optional deps are missing, so the
parser is always functional.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..extraction.citation_aligner import extract_citations
from ..logging_setup import get_logger
from ..types import Citation

log = get_logger(__name__)


_TEX_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?\s*\{([^}]+)\}")
_TEX_BIB_RE = re.compile(r"\\bibliography\{([^}]+)\}")


class ParsingAgent:
    def parse_text(self, text: str) -> list[Citation]:
        return extract_citations(text)

    def parse_path(self, path: str | Path) -> list[Citation]:
        p = Path(path)
        if not p.exists():
            log.warning("parsing path missing", path=str(p))
            return []
        suffix = p.suffix.lower()
        if suffix == ".bib":
            return self._parse_bibtex(p)
        if suffix == ".tex":
            return self._parse_tex(p)
        if suffix in {".md", ".markdown", ".txt"}:
            return self.parse_text(p.read_text(errors="ignore"))
        if suffix == ".docx":
            return self._parse_docx(p)
        return self.parse_text(p.read_text(errors="ignore"))

    # ------------------------------------------------------------------------

    def _parse_bibtex(self, path: Path) -> list[Citation]:
        try:
            import bibtexparser  # type: ignore
        except ImportError:  # pragma: no cover
            log.warning("bibtexparser not installed; falling back to plain-text", path=str(path))
            return self.parse_text(path.read_text(errors="ignore"))
        try:
            data = bibtexparser.loads(path.read_text(errors="ignore"))
        except Exception as exc:  # pragma: no cover
            log.warning("bibtex parse error", error=str(exc))
            return []
        out: list[Citation] = []
        for entry in data.entries:
            authors_raw = entry.get("author", "")
            authors = [a.strip() for a in re.split(r"\s+and\s+", authors_raw) if a.strip()]
            year = None
            try:
                year = int(entry.get("year", "")) if entry.get("year", "").isdigit() else None
            except ValueError:
                year = None
            out.append(
                Citation(
                    raw=f"@{entry.get('ENTRYTYPE','misc')}{{{entry.get('ID','?')}, ...}}",
                    title=entry.get("title"),
                    authors=authors,
                    year=year,
                    venue=entry.get("journal") or entry.get("booktitle"),
                    doi=entry.get("doi"),
                    url=entry.get("url"),
                )
            )
        return out

    def _parse_tex(self, path: Path) -> list[Citation]:
        text = path.read_text(errors="ignore")
        out: list[Citation] = list(self.parse_text(text))
        bibs = _TEX_BIB_RE.findall(text)
        for b in bibs:
            for stem in b.split(","):
                bib_path = path.with_name(stem.strip() + ".bib")
                if bib_path.exists():
                    out.extend(self._parse_bibtex(bib_path))
        return out

    def _parse_docx(self, path: Path) -> list[Citation]:
        try:
            from docx import Document  # type: ignore
        except ImportError:
            log.warning("python-docx not installed; reading docx as text fallback")
            return self.parse_text(path.read_text(errors="ignore"))
        try:
            doc = Document(str(path))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:  # pragma: no cover
            log.warning("docx parse error", error=str(exc))
            return []
        return self.parse_text(text)
