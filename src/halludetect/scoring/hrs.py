"""Hallucination Risk Scoring (HRS) — the master Safety Score formula.

    S_safety = 100 · clip( 1 − α·P_inj − β·U_intrinsic + γ·V_extrinsic , 0, 1 )

* ``P_inj`` ∈ [0, 1]      — from Sentinel + MELON (Layer 1).
* ``U_intrinsic`` ∈ [0, 1] — normalized predictive entropy / log-prob /
  surface-uncertainty proxies (Layer 2).
* ``V_extrinsic`` ∈ [0, 1] — fraction of atomic claims with high-similarity
  retrieval matches (Layers 3+4).
* (α, β, γ) are tunable per deployment via the strict / default / lenient
  presets.

Output is in ``[0, 100]``: 0 = critically compromised, 100 = fully verified.
"""

from __future__ import annotations

from ..config import ScoringWeights
from ..types import Claim, LinkageMatch, SafetyScore, ThreatReport, VerdictLabel


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class HRSAggregator:
    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    # ------------------------------------------------------------------------

    def compute(
        self,
        *,
        threat: ThreatReport,
        u_intrinsic_norm: float,
        verdicts: list[tuple[Claim, VerdictLabel, LinkageMatch | None]],
    ) -> SafetyScore:
        v_extrinsic = self._verification_alignment(verdicts)
        p_inj = _clip01(threat.p_injection)
        u_norm = _clip01(u_intrinsic_norm)

        raw = 1.0 - self.weights.alpha_inj * p_inj - self.weights.beta_uncertainty * u_norm + self.weights.gamma_verification * v_extrinsic
        score = 100.0 * _clip01(raw)
        # Hard penalty: a blocked interaction is already adversarially
        # compromised; cap the score in the "critically compromised" band.
        if threat.blocked:
            score = min(score, 5.0)

        return SafetyScore(
            score=score,
            p_injection=p_inj,
            u_intrinsic=u_norm,
            v_extrinsic=v_extrinsic,
            weights={
                "alpha_inj": self.weights.alpha_inj,
                "beta_uncertainty": self.weights.beta_uncertainty,
                "gamma_verification": self.weights.gamma_verification,
            },
            interpretation=self._interpret(score),
        )

    # ------------------------------------------------------------------------

    @staticmethod
    def _verification_alignment(
        verdicts: list[tuple[Claim, VerdictLabel, LinkageMatch | None]],
    ) -> float:
        if not verdicts:
            return 0.0
        verified = 0.0
        for _, label, _ in verdicts:
            if label == VerdictLabel.VERIFIED:
                verified += 1.0
            elif label == VerdictLabel.PARTIALLY_VERIFIED:
                verified += 0.5
        return verified / float(len(verdicts))

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 90:
            return "Fully verified and safe — auto-apply suggested patches."
        if score >= 75:
            return "Mostly verified — review high-risk patches manually."
        if score >= 50:
            return "Mixed signal — manual audit strongly recommended."
        if score >= 25:
            return "High risk — block automated rewrites; human review required."
        return "Critically compromised — discard generation and resample."
