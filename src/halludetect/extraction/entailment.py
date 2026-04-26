"""DeBERTa-NLI entailment scorer with a lexical fallback.

Computes ``P(entailment | premise, hypothesis)`` for each ``(triplet,
retrieved_evidence)`` pair. When the ``transformers`` extra is installed we
use ``cross-encoder/nli-deberta-v3-base`` (the design doc's recommendation);
otherwise we approximate with a normalized lexical-overlap score so the
pipeline is still operational offline.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ..config import EntailmentConfig
from ..logging_setup import get_logger
from ..types import Triplet

log = get_logger(__name__)

_TOK_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "and", "or", "for", "in", "on", "at", "with", "by", "as", "this",
    "that", "these", "those", "it", "its", "their", "our", "we", "they",
}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOK_RE.findall(text or "") if t.lower() not in _STOPWORDS}


def _lexical_entailment(premise: str, hypothesis: str) -> float:
    p, h = _tokens(premise), _tokens(hypothesis)
    if not h:
        return 0.0
    overlap = len(p & h) / len(h)
    if not p:
        return 0.0
    coverage = len(p & h) / max(len(p), 1)
    return float(min(1.0, 0.7 * overlap + 0.3 * coverage))


class EntailmentScorer:
    def __init__(self, config: EntailmentConfig | None = None, *, offline: bool | None = None) -> None:
        self.config = config or EntailmentConfig()
        if offline is None:
            offline = os.environ.get("HALLUDETECT_OFFLINE", "0") == "1"
        self._offline = offline
        self._pipeline: Any | None = None
        self._tried_load = False

    def _try_load(self) -> Any | None:
        if self._tried_load:
            return self._pipeline
        self._tried_load = True
        if self._offline:
            return None
        try:
            from transformers import pipeline  # type: ignore
        except ImportError:
            log.warning("transformers not installed; using lexical entailment fallback")
            return None
        try:
            self._pipeline = pipeline(
                task="text-classification",
                model=self.config.model_id,
                tokenizer=self.config.model_id,
                top_k=None,
                truncation=True,
            )
            log.info("entailment transformers backend loaded", model=self.config.model_id)
        except Exception as exc:  # pragma: no cover
            log.warning("failed to load NLI model; using lexical fallback", error=str(exc))
            self._pipeline = None
        return self._pipeline

    # ------------------------------------------------------------------------

    def score(self, premise: str, hypothesis: str) -> float:
        """Return entailment probability ∈ [0, 1] (1 = strong entailment)."""

        pipe = self._try_load()
        if pipe is not None:
            try:
                raw = pipe({"text": premise, "text_pair": hypothesis})
                return self._extract_entailment(raw)
            except Exception as exc:  # pragma: no cover
                log.warning("NLI pipeline failed; using lexical fallback", error=str(exc))
        return _lexical_entailment(premise, hypothesis)

    def score_triplet(self, triplet: Triplet, evidence: str) -> float:
        hypothesis = f"{triplet.subject} {triplet.relation} {triplet.object}"
        return self.score(evidence, hypothesis)

    # ------------------------------------------------------------------------

    @staticmethod
    def _extract_entailment(raw: Any) -> float:
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        if isinstance(raw, dict):
            raw = [raw]
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).lower()
                if "entail" in label:
                    return float(item.get("score", 0.0))
                if label == "label_2":  # MNLI-style: 0=contradiction,1=neutral,2=entailment
                    return float(item.get("score", 0.0))
        return 0.0
