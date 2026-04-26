"""Aggregator that combines Sentinel + MELON outputs into ``P_inj``.

The Hallucination Risk Score consumes a single scalar ``P_inj`` ∈ [0, 1].
We compose it from:

    * Sentinel direct-injection logits (``s_score``)
    * MELON contrastive cosine similarity (``m_sim``) — only contributes when
      it crosses the configured abort threshold (otherwise it's noise from
      naturally-similar generic answers).

Rule:  ``P_inj = max(s_score, λ * sigmoid(m_sim - threshold))``  with λ = 1.0.
We use ``max`` so a single high-confidence direct-injection match is enough
to dominate, while still letting subtle indirect attacks raise the floor.
"""

from __future__ import annotations

import math

from ..types import ThreatReport
from .melon import MelonResult
from .sentinel import SentinelResult


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-10.0 * x))


class ThreatAggregator:
    def __init__(self, melon_threshold: float = 0.25, lambda_indirect: float = 1.0) -> None:
        self.melon_threshold = melon_threshold
        self.lambda_indirect = lambda_indirect

    def aggregate(
        self,
        sentinel: SentinelResult,
        melon: MelonResult,
        *,
        sentinel_blocked: bool = False,
    ) -> ThreatReport:
        # Sentinel emits ``score`` as the model's confidence in its predicted
        # ``label``. Convert to canonical P(malicious | input):
        #   * label ∈ {jailbreak, injection, …}  →  s_score = score
        #   * label ∈ {benign, safe, …}          →  s_score = 1 − score
        # This is the correct probabilistic complement and avoids the bug
        # where a high-confidence "SAFE" verdict would contribute P_inj = 0.5.
        from .sentinel import SentinelClassifier

        if SentinelClassifier._is_malicious_label(sentinel.label):
            s_score = sentinel.score
        else:
            s_score = 1.0 - sentinel.score

        # No clipping: when ``cosine`` is well below ``threshold`` we want the
        # sigmoid to evaluate near zero, not at its 0.5 baseline. The signed
        # delta retains "much-below-threshold" information.
        indirect_signal = melon.cosine_similarity - self.melon_threshold
        m_score = self.lambda_indirect * _sigmoid(indirect_signal)
        if not melon.aborted:
            m_score *= 0.5
        p_inj = max(s_score, m_score)
        p_inj = max(0.0, min(1.0, p_inj))

        notes: list[str] = []
        if sentinel.is_malicious:
            notes.append(f"sentinel-flagged ({sentinel.label} @ {sentinel.score:.2f})")
        if sentinel.hits:
            notes.append(f"heuristic hits: {','.join(sentinel.hits)}")
        if melon.aborted:
            notes.append(f"melon-aborted (cos={melon.cosine_similarity:.2f})")

        return ThreatReport(
            sentinel=sentinel.to_output(),
            melon=melon.to_output(),
            p_injection=p_inj,
            blocked=sentinel_blocked or melon.aborted,
            notes=notes,
        )
