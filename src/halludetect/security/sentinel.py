"""Sub-Layer 1A — Sentinel direct-prompt-injection classifier.

Wraps ``qualifire/prompt-injection-sentinel`` (a fine-tuned ModernBERT-large,
8,192 token context, 28 attention layers, 395M params) as documented in the
design blueprint. The HF ``text-classification`` pipeline is loaded lazily on
first call so that the package is importable without ``torch``/``transformers``
installed.

When the ML stack is unavailable (or ``HALLUDETECT_OFFLINE=1``), the classifier
falls back to a deterministic heuristic backend: a curated lexicon of
high-signal injection patterns (override tokens, role-flipping, exfiltration
verbs) producing a calibrated confidence in [0, 1]. This means the entire
pipeline remains executable end-to-end even with no models downloaded.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from ..config import SentinelConfig
from ..exceptions import InjectionDetected
from ..logging_setup import get_logger
from ..types import SentinelOutput

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic backend (offline fallback)
# ─────────────────────────────────────────────────────────────────────────────


_HEURISTIC_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
    # Permits arbitrary qualifiers between "ignore" and the noun: "ignore all
    # previous instructions", "ignore the prior system rules", etc.
    (re.compile(r"\bignore\s+(?:\w+\s+){0,4}(?:instructions|prompts|rules|directives|messages?)\b", re.I), 0.45, "ignore-prior"),
    (re.compile(r"\bdisregard\s+(?:\w+\s+){0,4}(?:prompt|message|rules?|instructions|directives)\b", re.I), 0.40, "disregard-system"),
    (re.compile(r"\b(?:jailbreak|do\s*anything\s*now|DAN\s+mode)\b", re.I), 0.55, "jailbreak"),
    (re.compile(r"\b(?:reveal|show|print|output|leak)\s+(?:\w+\s+){0,3}(?:prompt|instructions|rules|secrets)\b", re.I), 0.50, "exfil-prompt"),
    (re.compile(r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:a|an)\s+\S+", re.I), 0.20, "role-flip"),
    (re.compile(r"\bend\s+of\s+(?:prompt|instructions)\b.*\bnew\s+(?:prompt|instructions)\b", re.I | re.S), 0.40, "boundary-spoof"),
    (re.compile(r"</?(?:system|assistant|user|im_start|im_end)>", re.I), 0.30, "tag-injection"),
    (re.compile(r"\b(?:send|exfiltrate|email|post)\b.{0,40}\b(?:http|api|webhook|attacker)\b", re.I | re.S), 0.55, "exfil-action"),
    # Explicit fabrication commands are unambiguous content-policy violations
    # that the public ML prompt-injection models miss (they target instruction
    # overrides, not "make-up-a-source" requests). Weighted high enough to
    # cross the 0.85 block threshold on its own.
    (re.compile(r"\bfabricate\s+(?:a\s+)?(?:citation|reference|bibliography|doi|paper|study|source|proof)\b", re.I), 0.92, "fabrication-cmd"),
    (re.compile(r"\b(?:invent|make\s*up|hallucinate)\s+(?:a\s+)?(?:citation|paper|study|reference|source)\b", re.I), 0.88, "fabrication-cmd"),
]


def _heuristic_score(text: str) -> tuple[float, list[str]]:
    if not text:
        return 0.0, []
    score = 0.0
    hits: list[str] = []
    for pat, weight, label in _HEURISTIC_PATTERNS:
        if pat.search(text):
            score += weight
            hits.append(label)
    score = min(score, 0.99)
    return score, hits


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass + classifier
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SentinelResult:
    label: str
    score: float
    is_malicious: bool
    backend: str
    hits: list[str]

    def to_output(self) -> SentinelOutput:
        return SentinelOutput(
            label=self.label,
            score=self.score,
            is_malicious=self.is_malicious,
        )


class SentinelClassifier:
    """Low-latency direct-injection gatekeeper.

    Parameters
    ----------
    config: SentinelConfig
        Threshold + model id from the loaded YAML config.
    offline: bool
        Force heuristic mode regardless of installed extras.
    """

    def __init__(self, config: SentinelConfig | None = None, *, offline: bool | None = None) -> None:
        self.config = config or SentinelConfig()
        if offline is None:
            offline = os.environ.get("HALLUDETECT_OFFLINE", "0") == "1"
        self._offline = offline
        self._pipeline: Any | None = None
        self._loaded_model: str | None = None
        self._tried_load = False

    # ---- backend management -------------------------------------------------

    def _try_load_pipeline(self) -> Any | None:
        """Walk the model fallback chain: primary → fallbacks → heuristic.

        The doc-specified ``qualifire/prompt-injection-sentinel`` is a gated
        HF repository; if the calling environment lacks access (or the HF
        token is missing), we automatically try the public
        ``protectai/deberta-v3-base-prompt-injection-v2`` before degrading to
        the deterministic heuristic backend.
        """

        if self._tried_load:
            return self._pipeline
        self._tried_load = True
        if self._offline:
            log.info("sentinel offline mode forced", backend="heuristic")
            return None
        try:
            from transformers import pipeline  # type: ignore
        except ImportError:
            log.warning("transformers not installed; falling back to heuristic Sentinel backend")
            return None

        candidates = [self.config.model_id, *self.config.fallback_model_ids]
        last_error: str | None = None
        for model_id in candidates:
            try:
                self._pipeline = pipeline(
                    task="text-classification",
                    model=model_id,
                    tokenizer=model_id,
                    top_k=None,
                    truncation=True,
                )
                self._loaded_model = model_id
                log.info("sentinel transformers backend loaded", model=model_id)
                return self._pipeline
            except Exception as exc:  # pragma: no cover — depends on local env
                err_summary = str(exc).split("\n")[0][:120]
                last_error = err_summary
                # Gated-repo / 401 is a *normal* fallback path; demote to INFO.
                is_expected_gating = any(
                    marker in str(exc).lower()
                    for marker in ("gated repo", "401", "unauthorized", "access to model")
                )
                level_log = log.info if is_expected_gating else log.warning
                level_log(
                    "sentinel model unavailable; trying next fallback",
                    model=model_id,
                    error=err_summary,
                )
                continue
        log.warning("all sentinel models failed to load; using heuristic", error=last_error)
        self._pipeline = None
        return self._pipeline

    # ---- public API ---------------------------------------------------------

    def classify(self, text: str) -> SentinelResult:
        """Classify a single prompt as benign or jailbreak.

        Strategy: when the transformers backend is available, run BOTH the ML
        classifier and the heuristic pattern matcher and take the *max* of
        their P(malicious) signals.

        Rationale: the public ML models we can use (e.g. ``protectai/
        deberta-v3-base-prompt-injection-v2``) are specifically trained for
        instruction-override attacks (jailbreaks, role-flipping, system-prompt
        leaks). They do *not* flag content-policy violations such as
        "fabricate a citation" or "make up a paper". Our heuristic backend
        covers exactly those gaps. ORing the two signals gives a wider
        attack-surface coverage than either alone.
        """

        text = text or ""
        heur_score, heur_hits = _heuristic_score(text)

        pipe = self._try_load_pipeline()
        if pipe is not None:
            try:
                raw = pipe(text[: self.config.max_length * 4])  # char-budget guard
                label, score = self._parse_pipeline_output(raw)
                ml_p_mal = score if self._is_malicious_label(label) else (1.0 - score)
                # Take the more cautious of the two signals.
                p_malicious = max(ml_p_mal, heur_score)
                if p_malicious == ml_p_mal and self._is_malicious_label(label):
                    final_label, final_score = label, score
                elif heur_score >= ml_p_mal:
                    final_label, final_score = ("jailbreak" if heur_score > 0 else label, heur_score if heur_score > 0 else score)
                else:
                    final_label, final_score = label, score
                is_mal = p_malicious >= self.config.block_threshold
                backend = f"transformers:{self._loaded_model or '?'}+heuristic"
                return SentinelResult(
                    label=final_label,
                    score=p_malicious if is_mal else final_score,
                    is_malicious=is_mal,
                    backend=backend,
                    hits=heur_hits,
                )
            except Exception as exc:  # pragma: no cover
                log.warning("sentinel transformers call failed; falling back", error=str(exc))

        is_mal = heur_score >= self.config.block_threshold
        label = "jailbreak" if is_mal else "benign"
        return SentinelResult(label=label, score=heur_score, is_malicious=is_mal, backend="heuristic", hits=heur_hits)

    def assert_safe(self, text: str) -> SentinelResult:
        """Classify and raise ``InjectionDetected`` if malicious. Always returns the result on benign."""

        result = self.classify(text)
        if result.is_malicious:
            raise InjectionDetected(label=result.label, confidence=result.score)
        return result

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _is_malicious_label(label: str) -> bool:
        return label.lower() in {"jailbreak", "injection", "malicious", "label_1", "1"}

    @staticmethod
    def _parse_pipeline_output(raw: Any) -> tuple[str, float]:
        """Normalize the heterogeneous shapes that HF ``text-classification`` may emit."""

        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            best = max(raw, key=lambda d: float(d.get("score", 0.0)))
            return str(best.get("label", "unknown")), float(best.get("score", 0.0))
        if isinstance(raw, dict):
            return str(raw.get("label", "unknown")), float(raw.get("score", 0.0))
        return "unknown", 0.0
