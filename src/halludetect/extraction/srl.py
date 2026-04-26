"""Semantic Role Labeling — heuristic + optional spaCy backend.

In the design doc the SRL transformer parses generated sentences into
predicate-argument structures so each sentence can be reduced to one or more
``(Subject, Relation, Object)`` triplets (Layer 2 atomic decomposition).

We provide:

* A **heuristic** backend (regex + verb-anchor splitting) that always works.
* A **spaCy** backend (if ``spacy`` is installed and the model is downloaded)
  which produces dependency-parse-driven SRL.

Either backend emits the same intermediate structure consumed by
``TripletBuilder``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import SrlConfig
from ..logging_setup import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SrlFrame:
    sentence: str
    subject: str
    predicate: str
    obj: str
    span: tuple[int, int]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_CITATION_PARENS = re.compile(r"\(([^()]+\d{4}[a-z]?[^()]*)\)")
_BRACKET_REF = re.compile(r"\[\d+(?:,\s*\d+)*\]")


_VERB_HINTS = (
    "proposed", "introduced", "demonstrated", "showed", "found", "reported",
    "argued", "claimed", "established", "developed", "evaluated", "investigated",
    "analyzed", "presented", "described", "discovered", "validated", "proved",
)


def _strip_citation_noise(sentence: str) -> str:
    s = _CITATION_PARENS.sub("", sentence)
    s = _BRACKET_REF.sub("", s)
    return s.strip()


def _split_sentences(text: str) -> list[tuple[str, tuple[int, int]]]:
    out: list[tuple[str, tuple[int, int]]] = []
    cursor = 0
    for piece in _SENT_SPLIT.split(text or ""):
        piece = piece.strip()
        if not piece:
            continue
        idx = (text or "").find(piece, cursor)
        if idx < 0:
            idx = cursor
        out.append((piece, (idx, idx + len(piece))))
        cursor = idx + len(piece)
    return out


def _heuristic_frames(sentence: str, span: tuple[int, int]) -> list[SrlFrame]:
    cleaned = _strip_citation_noise(sentence)
    if not cleaned:
        return []
    lower = cleaned.lower()
    for verb in _VERB_HINTS:
        idx = lower.find(f" {verb} ")
        if idx > 0:
            subject = cleaned[:idx].strip().rstrip(",;")
            rest = cleaned[idx + len(verb) + 2 :].strip().rstrip(".")
            if subject and rest:
                return [SrlFrame(sentence=sentence, subject=subject, predicate=verb, obj=rest, span=span)]

    # Naive fallback: split on the first " is ", " are ", " was ", " were "
    for cop in (" is ", " are ", " was ", " were ", " has ", " have "):
        idx = cleaned.find(cop)
        if idx > 0:
            subj = cleaned[:idx].strip()
            obj = cleaned[idx + len(cop) :].strip().rstrip(".")
            if subj and obj:
                return [SrlFrame(sentence=sentence, subject=subj, predicate=cop.strip(), obj=obj, span=span)]

    # Final fallback: use full sentence as a single triplet so it can still
    # be entailment-checked.
    return [SrlFrame(sentence=sentence, subject="this work", predicate="states", obj=cleaned, span=span)]


# ─────────────────────────────────────────────────────────────────────────────


class SemanticRoleLabeler:
    def __init__(self, config: SrlConfig | None = None) -> None:
        self.config = config or SrlConfig()
        self._spacy_nlp = None
        self._tried_spacy = False

    def _try_spacy(self):
        if self._tried_spacy:
            return self._spacy_nlp
        self._tried_spacy = True
        if self.config.backend != "spacy":
            return None
        try:
            import spacy  # type: ignore
            self._spacy_nlp = spacy.load(self.config.spacy_model)
            log.info("spaCy SRL backend loaded", model=self.config.spacy_model)
        except Exception as exc:  # pragma: no cover
            log.warning("spaCy unavailable; using heuristic SRL", error=str(exc))
            self._spacy_nlp = None
        return self._spacy_nlp

    def parse(self, text: str) -> list[SrlFrame]:
        frames: list[SrlFrame] = []
        nlp = self._try_spacy()
        for sent, span in _split_sentences(text):
            if nlp is not None:
                frames.extend(self._spacy_frames(nlp, sent, span))
            else:
                frames.extend(_heuristic_frames(sent, span))
        return frames

    @staticmethod
    def _spacy_frames(nlp, sentence: str, span: tuple[int, int]) -> list[SrlFrame]:  # pragma: no cover
        doc = nlp(sentence)
        out: list[SrlFrame] = []
        for sent in doc.sents:
            subj, verb, obj = "", "", ""
            for tok in sent:
                if tok.dep_ in ("nsubj", "nsubjpass") and not subj:
                    subj = tok.subtree_text() if hasattr(tok, "subtree_text") else tok.text
                if tok.pos_ == "VERB" and not verb:
                    verb = tok.lemma_
                if tok.dep_ in ("dobj", "attr", "pobj", "ccomp") and not obj:
                    obj = " ".join(t.text for t in tok.subtree)
            if subj and verb and obj:
                out.append(SrlFrame(sentence=sentence, subject=subj, predicate=verb, obj=obj, span=span))
        if not out:
            out.extend(_heuristic_frames(sentence, span))
        return out
