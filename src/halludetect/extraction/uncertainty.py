"""Intrinsic-uncertainty estimator → ``U_intrinsic`` for the HRS Safety Score.

Per the design doc, ``U_intrinsic`` is built from token-level predictive
entropy and sequence log-probability extracted from the generative LLM. When
those signals are unavailable (we audited generated text from a black-box
provider), we estimate uncertainty proxies from surface features:

    * Average per-claim ``-log P(entailment)`` against any retrieved evidence
    * Citation density vs. claim density (sparse citations = higher uncertainty)
    * Hedging / vagueness markers ("might", "could", "is believed to")

The estimator returns a non-negative scalar; the scoring layer normalizes it
into [0, 1] before combining with the other HRS components.
"""

from __future__ import annotations

import math
import re

from ..config import UncertaintyConfig
from ..types import Claim

_HEDGES = {
    "might", "may", "could", "possibly", "perhaps", "likely", "probably",
    "seems", "appears", "suggests", "believed", "reportedly", "allegedly",
}
_HEDGE_RE = re.compile(r"\b(" + "|".join(_HEDGES) + r")\b", re.I)


class UncertaintyEstimator:
    def __init__(self, config: UncertaintyConfig | None = None) -> None:
        self.config = config or UncertaintyConfig()

    # ------------------------------------------------------------------------

    def estimate(
        self,
        *,
        claims: list[Claim],
        token_logprobs: list[float] | None = None,
        token_entropies: list[float] | None = None,
    ) -> float:
        """Return raw U_intrinsic ≥ 0 (higher = more uncertain)."""

        components: list[float] = []

        if token_entropies:
            components.append(self._mean_entropy(token_entropies))
        if token_logprobs:
            components.append(self._neg_seq_logprob(token_logprobs))

        components.append(self._claim_signal(claims))
        return float(sum(components) / max(len(components), 1))

    # ------------------------------------------------------------------------

    def _mean_entropy(self, entropies: list[float]) -> float:
        floor = self.config.entropy_floor
        clean = [max(e, floor) for e in entropies if e is not None]
        if not clean:
            return 0.0
        return float(sum(clean) / len(clean))

    def _neg_seq_logprob(self, logprobs: list[float]) -> float:
        if not logprobs:
            return 0.0
        return -float(sum(logprobs)) / len(logprobs)

    def _claim_signal(self, claims: list[Claim]) -> float:
        if not claims:
            return 0.0
        unsupported = sum(1 for c in claims if c.entailment_score < 0.5)
        hedged = sum(1 for c in claims if _HEDGE_RE.search(c.triplet.sentence))
        no_cite = sum(1 for c in claims if c.citation is None)
        denom = float(len(claims))
        # Weighted combination. Each ratio ∈ [0, 1]; product capped.
        return min(
            self.config.high_entropy_threshold,
            (1.5 * unsupported + 0.5 * hedged + 1.0 * no_cite) / denom,
        )

    # ------------------------------------------------------------------------

    def normalize(self, raw: float) -> float:
        """Squash raw U_intrinsic into [0, 1] for HRS aggregation."""

        if raw <= 0.0:
            return 0.0
        return float(1.0 - math.exp(-raw / max(self.config.high_entropy_threshold, 1e-9)))
