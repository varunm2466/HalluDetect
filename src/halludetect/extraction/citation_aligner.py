"""Citation extraction + alignment from generated text.

Detects the major in-text citation styles:

* APA-ish parenthetical:        ``(Smith, 2025)``, ``(Smith et al., 2025)``, ``(Smith and Doe, 2024)``
* APA-ish narrative:            ``Smith (2025) showed …``
* Numeric IEEE/ACM:             ``[12]``, ``[12, 13]``, ``[12-15]``
* Bibliography-line entries:    raw lines parsed in a "References" section
* DOIs and arXiv ids embedded inline

For each detected marker we materialize a ``Citation`` with the char span and
any inferred metadata; the surrounding sentence (the "anchor sentence") is
matched to the nearest ``Triplet`` so downstream layers know which source
must support which fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import Citation, Claim, Triplet

# ─────────────────────────────────────────────────────────────────────────────
# Regexes
# ─────────────────────────────────────────────────────────────────────────────


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;)\]\"']+", re.I)
_ARXIV_RE = re.compile(r"\barXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.I)
_URL_RE = re.compile(r"https?://\S+")

_PAREN_CITATION_RE = re.compile(
    r"\(([A-Z][A-Za-z\-']+(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z\-']+)?,\s*\d{4}[a-z]?)\)"
)
_NARRATIVE_RE = re.compile(
    r"\b([A-Z][A-Za-z\-']+(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z\-']+)?)\s+\((\d{4}[a-z]?)\)"
)
_NUMERIC_RE = re.compile(r"\[(\d+(?:\s*[,\-]\s*\d+)*)\]")
_AUTHOR_YEAR_SPLIT = re.compile(r",\s*(\d{4}[a-z]?)$")
_REFS_HEADER_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:references|bibliography|works\s+cited)\s*$",
    re.I | re.M,
)


# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _RefEntry:
    raw: str
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    arxiv_id: str | None
    url: str | None


_LEADING_NUM_RE = re.compile(r"^\s*(?:\[\d+\]|\d+\.)\s*")
_YEAR_TOKEN_RE = re.compile(r"\((?P<year>(?:19|20|21)\d{2}[a-z]?)\)")
_BARE_YEAR_RE = re.compile(r"\b(?P<year>(?:19|20|21)\d{2}[a-z]?)\b")
_AUTHOR_SEP_RE = re.compile(r"\s*,\s*&\s*|\s*;\s*|\s*,\s+and\s+|\s+&\s+|\s+and\s+|\s*,\s*")


def _split_authors(author_block: str) -> list[str]:
    """Split a normalized author block into individual author names.

    Joins single-letter initials back into the preceding surname, since the
    natural ``,`` split mistakenly tears ``Vaswani, A.`` into two pieces.
    """

    parts = [p.strip().rstrip(".,") for p in _AUTHOR_SEP_RE.split(author_block) if p.strip()]
    merged: list[str] = []
    for p in parts:
        condensed = p.replace(".", "").strip()
        is_initials = (
            len(condensed) <= 4
            and condensed
            and all(c.isalpha() and c.isupper() for c in condensed)
        )
        if is_initials and merged:
            merged[-1] = f"{merged[-1]} {p}".strip()
        else:
            merged.append(p)
    return [m for m in (a.strip() for a in merged) if m][:8]


def _parse_reference_line(line: str) -> _RefEntry:
    raw = line.strip()
    body = _LEADING_NUM_RE.sub("", raw)
    doi_m = _DOI_RE.search(raw)
    arxiv_m = _ARXIV_RE.search(raw)
    url_m = _URL_RE.search(raw)

    year: int | None = None
    title: str | None = None
    venue: str | None = None
    authors: list[str] = []

    year_paren = _YEAR_TOKEN_RE.search(body)
    if year_paren is not None:
        year = int(year_paren.group("year")[:4])
        author_block = body[: year_paren.start()].strip().rstrip(",.")
        rest = body[year_paren.end():].strip().lstrip(".:,").strip()
        authors = _split_authors(author_block)
        # Title runs up to the first ``.`` that is not inside a "Vol. N" / "No. N" /
        # initials-pattern. Simpler heuristic: split on ``. `` (period+space) and
        # take the first segment as title.
        rest_parts = [p.strip(" .,") for p in re.split(r"\.\s+", rest) if p.strip()]
        if rest_parts:
            title = rest_parts[0] or None
        if len(rest_parts) >= 2:
            venue_candidate = rest_parts[1]
            if not venue_candidate.lower().startswith(("doi", "arxiv", "https://", "http://")):
                venue = venue_candidate
    else:
        # Fall back to bare-year detection — best-effort for non-APA styles.
        bare = _BARE_YEAR_RE.search(body)
        if bare is not None:
            year = int(bare.group("year")[:4])
        head_parts = [p.strip() for p in body.split(".") if p.strip()]
        if head_parts:
            authors = _split_authors(head_parts[0])
        if len(head_parts) >= 2:
            title = head_parts[1].strip(" .,") or None
        if len(head_parts) >= 3:
            venue = head_parts[2].strip(" .,") or None

    return _RefEntry(
        raw=raw,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi_m.group(0) if doi_m else None,
        arxiv_id=arxiv_m.group(1) if arxiv_m else None,
        url=url_m.group(0) if url_m else None,
    )


def _split_references_section(text: str) -> list[str]:
    m = _REFS_HEADER_RE.search(text or "")
    if not m:
        return []
    tail = text[m.end():]
    raw_lines = [ln.strip() for ln in re.split(r"\n(?=\s*(?:\[\d+\]|\d+\.\s|[A-Z][a-z]+,\s))", tail)]
    return [ln for ln in raw_lines if ln]


# ─────────────────────────────────────────────────────────────────────────────


def extract_citations(text: str) -> list[Citation]:
    """Find every in-text citation marker and bibliography entry."""

    text = text or ""
    cites: list[Citation] = []

    # Bibliography entries first (so we can index them by numeric marker).
    refs_lines = _split_references_section(text)
    parsed_refs = [_parse_reference_line(ln) for ln in refs_lines]
    for entry in parsed_refs:
        cites.append(
            Citation(
                raw=entry.raw,
                title=entry.title,
                authors=entry.authors,
                year=entry.year,
                venue=entry.venue,
                doi=entry.doi,
                arxiv_id=entry.arxiv_id,
                url=entry.url,
            )
        )

    for m in _PAREN_CITATION_RE.finditer(text):
        inner = m.group(1)
        year_m = _AUTHOR_YEAR_SPLIT.search(inner)
        year = int(year_m.group(1)[:4]) if year_m else None
        author_part = _AUTHOR_YEAR_SPLIT.sub("", inner).strip()
        cites.append(
            Citation(
                raw=m.group(0),
                authors=[a.strip() for a in re.split(r"\s+(?:and|&)\s+|,\s*", author_part) if a.strip()],
                year=year,
                span=(m.start(), m.end()),
            )
        )

    for m in _NARRATIVE_RE.finditer(text):
        cites.append(
            Citation(
                raw=m.group(0),
                authors=[m.group(1).strip()],
                year=int(m.group(2)[:4]),
                span=(m.start(), m.end()),
            )
        )

    for m in _NUMERIC_RE.finditer(text):
        for piece in re.split(r"[,\-]", m.group(1)):
            piece = piece.strip()
            if not piece.isdigit():
                continue
            idx = int(piece) - 1
            if 0 <= idx < len(parsed_refs):
                e = parsed_refs[idx]
                cites.append(
                    Citation(
                        raw=f"[{piece}]",
                        title=e.title,
                        authors=e.authors,
                        year=e.year,
                        venue=e.venue,
                        doi=e.doi,
                        arxiv_id=e.arxiv_id,
                        url=e.url,
                        span=(m.start(), m.end()),
                    )
                )
            else:
                cites.append(Citation(raw=f"[{piece}]", span=(m.start(), m.end())))

    for m in _DOI_RE.finditer(text):
        cites.append(Citation(raw=m.group(0), doi=m.group(0), span=(m.start(), m.end())))

    for m in _ARXIV_RE.finditer(text):
        cites.append(Citation(raw=m.group(0), arxiv_id=m.group(1), span=(m.start(), m.end())))

    return cites


# ─────────────────────────────────────────────────────────────────────────────


class CitationAligner:
    """Maps each Triplet → the closest in-text Citation by char-span proximity."""

    @staticmethod
    def align(text: str, triplets: list[Triplet], citations: list[Citation]) -> list[Claim]:
        spanned = [c for c in citations if c.span is not None]
        claims: list[Claim] = []
        for tri in triplets:
            cite = CitationAligner._closest_citation(tri, spanned)
            claims.append(Claim(triplet=tri, citation=cite))
        return claims

    @staticmethod
    def _closest_citation(tri: Triplet, citations: list[Citation]) -> Citation | None:
        if not citations or tri.span is None:
            return citations[0] if citations else None
        t_lo, t_hi = tri.span
        # Prefer citations *inside* the same sentence span; else nearest by edge distance.
        inside = [c for c in citations if c.span and t_lo <= c.span[0] <= t_hi]
        if inside:
            return inside[0]

        def _distance(c: Citation) -> int:
            assert c.span is not None
            cs, ce = c.span
            if ce < t_lo:
                return t_lo - ce
            if cs > t_hi:
                return cs - t_hi
            return 0

        return min(citations, key=_distance)
