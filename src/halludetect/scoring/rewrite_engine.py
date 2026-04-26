"""Rewrite Engine — emits BibTeX / JSON / Markdown patches."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from ..config import RewriteConfig
from ..types import (
    Citation,
    LinkageMatch,
    PatchAction,
    RetrievedRecord,
    RewritePatch,
    ThreatReport,
    Verdict,
    VerdictLabel,
)
from .policy_gate import PolicyGate


class RewriteEngine:
    def __init__(self, policy_gate: PolicyGate, config: RewriteConfig | None = None) -> None:
        self.policy_gate = policy_gate
        self.config = config or RewriteConfig()

    # ------------------------------------------------------------------------

    def build_patches(
        self,
        *,
        verdicts: list[Verdict],
        threat: ThreatReport,
        safety_score: float,
    ) -> list[RewritePatch]:
        patches: list[RewritePatch] = []
        seen_keys: set[str] = set()
        for v in verdicts:
            best_match: LinkageMatch | None = v.matches[0] if v.matches else None
            replacement: RetrievedRecord | None = best_match.record if best_match else None
            key = self._stable_key(v.citation)
            duplicate = key in seen_keys
            decision = self.policy_gate.evaluate(
                verdict_label=v.label,
                match=best_match,
                threat=threat,
                safety_score=safety_score,
                duplicate_key=duplicate and self.config.forbid_duplicate_keys,
            )
            seen_keys.add(key)

            action = self._choose_action(v.label, replacement)
            patches.append(
                RewritePatch(
                    target_citation=v.citation,
                    action=action,
                    suggested=replacement,
                    bibtex=self._bibtex(replacement) if replacement and "bibtex" in self.config.formats else None,
                    markdown=self._markdown(replacement) if replacement and "markdown" in self.config.formats else None,
                    json_payload=replacement.model_dump() if replacement and "json" in self.config.formats else None,
                    safe_to_apply=decision.safe_to_apply,
                    requires_manual_review=decision.requires_manual_review,
                    notes=[decision.rationale],
                )
            )
        return patches

    # ------------------------------------------------------------------------

    @staticmethod
    def _choose_action(label: VerdictLabel, replacement: RetrievedRecord | None) -> PatchAction:
        if label == VerdictLabel.VERIFIED:
            return PatchAction.NOOP
        if label == VerdictLabel.HALLUCINATED and replacement:
            return PatchAction.REPLACE
        if label == VerdictLabel.HALLUCINATED:
            return PatchAction.REMOVE
        return PatchAction.ANNOTATE

    @staticmethod
    def _stable_key(citation: Citation) -> str:
        if citation.doi:
            return f"doi::{citation.doi.lower()}"
        if citation.arxiv_id:
            return f"arxiv::{citation.arxiv_id.lower()}"
        if citation.title:
            return f"title::{re.sub(r'[^a-z0-9]+', '_', citation.title.lower())}"
        return f"raw::{citation.raw}"

    # ---- patch formatters ---------------------------------------------------

    @staticmethod
    def _slugify(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", s.lower()).strip("_")[:40] or "ref"

    @classmethod
    def _bibtex(cls, r: RetrievedRecord) -> str:
        first_author = r.authors[0].split()[-1] if r.authors else "anon"
        key = cls._slugify(f"{first_author}_{r.year or 'n_d'}_{(r.title or '')[:30]}")
        kind = {
            "journal": "article",
            "conference": "inproceedings",
            "preprint": "misc",
            "arxiv": "misc",
            "unknown": "misc",
        }.get(r.manifestation.value if hasattr(r.manifestation, "value") else str(r.manifestation), "misc")
        fields: list[str] = []
        if r.title:
            fields.append(f"  title    = {{{r.title}}}")
        if r.authors:
            fields.append(f"  author   = {{{ ' and '.join(r.authors) }}}")
        if r.year is not None:
            fields.append(f"  year     = {{{r.year}}}")
        if r.venue:
            fields.append(f"  journal  = {{{r.venue}}}")
        if r.doi:
            fields.append(f"  doi      = {{{r.doi}}}")
        if r.url:
            fields.append(f"  url      = {{{r.url}}}")
        body = ",\n".join(fields)
        return f"@{kind}{{{key},\n{body}\n}}"

    @staticmethod
    def _markdown(r: RetrievedRecord) -> str:
        authors = ", ".join(r.authors) if r.authors else "Unknown"
        title = r.title or "Untitled"
        year = f" ({r.year})" if r.year is not None else ""
        venue = f" — *{r.venue}*" if r.venue else ""
        link = ""
        if r.url:
            link = f" [link]({r.url})"
        elif r.doi:
            link = f" [DOI](https://doi.org/{r.doi})"
        return f"{authors}. {title}{year}{venue}.{link}"

    # ------------------------------------------------------------------------

    def serialize(self, patches: Iterable[RewritePatch], *, fmt: str = "json") -> str:
        if fmt == "json":
            return json.dumps([p.model_dump(mode="json") for p in patches], indent=2, default=str)
        if fmt == "bibtex":
            return "\n\n".join(p.bibtex for p in patches if p.bibtex)
        if fmt == "markdown":
            return "\n".join(f"- {p.markdown}" for p in patches if p.markdown)
        raise ValueError(f"Unsupported format {fmt!r}")
