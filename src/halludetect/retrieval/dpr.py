"""Dense Passage Retrieval rerank — sentence-transformers if available, else lexical."""

from __future__ import annotations

import os
import re
from typing import Any

from ..config import DprConfig
from ..logging_setup import get_logger
from ..types import Citation, RetrievedRecord

log = get_logger(__name__)
_TOK_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def _bag(text: str) -> set[str]:
    return {t.lower() for t in _TOK_RE.findall(text or "")}


def _lexical_sim(a: str, b: str) -> float:
    A, B = _bag(a), _bag(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))


class DprReranker:
    def __init__(self, config: DprConfig | None = None, *, offline: bool | None = None) -> None:
        self.config = config or DprConfig()
        if offline is None:
            offline = os.environ.get("HALLUDETECT_OFFLINE", "0") == "1"
        self._offline = offline
        self._encoder: Any | None = None
        self._tried_load = False

    def _try_load(self) -> Any | None:
        if self._tried_load:
            return self._encoder
        self._tried_load = True
        if self._offline:
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # ``sentence-transformers`` makes a separate HF Hub model-info call
            # that doesn't pick up ``HF_TOKEN`` from the environment by default;
            # pass it explicitly so the auth-warning is not emitted twice.
            hf_token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                or None
            )
            self._encoder = SentenceTransformer(self.config.model_id, token=hf_token)
            log.info("DPR encoder loaded", model=self.config.model_id)
        except Exception as exc:
            log.warning("sentence-transformers unavailable; using lexical DPR", error=str(exc))
            self._encoder = None
        return self._encoder

    def rerank(self, citation: Citation, records: list[RetrievedRecord]) -> list[RetrievedRecord]:
        if not records:
            return []
        query_text = " ".join(filter(None, [citation.title, " ".join(citation.authors), str(citation.year or "")]))
        encoder = self._try_load()
        if encoder is not None:
            try:
                doc_texts = [
                    " ".join(filter(None, [r.title, " ".join(r.authors), str(r.year or ""), r.venue or ""]))
                    for r in records
                ]
                qv = encoder.encode([query_text], normalize_embeddings=True)
                dv = encoder.encode(doc_texts, normalize_embeddings=True)
                sims = (qv @ dv.T).flatten().tolist()
                for r, s in zip(records, sims):
                    r.score = max(r.score, float(s))
            except Exception as exc:  # pragma: no cover
                log.warning("DPR encoding failed; using lexical", error=str(exc))
                for r in records:
                    r.score = max(r.score, _lexical_sim(query_text, r.title or ""))
        else:
            for r in records:
                r.score = max(r.score, _lexical_sim(query_text, r.title or ""))

        records.sort(key=lambda r: r.score, reverse=True)
        return records
