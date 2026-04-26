"""Policy-Gated Rewrite decision engine."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import PolicyConfig
from ..logging_setup import get_logger
from ..types import LinkageMatch, ThreatReport, VerdictLabel

log = get_logger(__name__)


@dataclass
class PolicyDecision:
    safe_to_apply: bool
    requires_manual_review: bool
    rationale: str


class PolicyGate:
    """Decides whether an individual rewrite is safe to auto-apply."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    # ------------------------------------------------------------------------

    def evaluate(
        self,
        *,
        verdict_label: VerdictLabel,
        match: LinkageMatch | None,
        threat: ThreatReport,
        safety_score: float,
        duplicate_key: bool = False,
    ) -> PolicyDecision:
        rationale_bits: list[str] = []

        if threat.blocked or threat.p_injection >= self.config.block_on_injection_prob:
            rationale_bits.append(
                f"Blocked: P_inj={threat.p_injection:.2f} ≥ {self.config.block_on_injection_prob:.2f}"
            )
            return PolicyDecision(False, True, "; ".join(rationale_bits))

        if duplicate_key:
            return PolicyDecision(False, True, "Refusing to overwrite duplicate citation key.")

        if verdict_label == VerdictLabel.HALLUCINATED:
            if not match or not match.record:
                return PolicyDecision(False, True, "Hallucinated citation has no candidate replacement.")
            if match.manifestation_conflict:
                return PolicyDecision(
                    False,
                    True,
                    "Manifestation conflict — Journal>Conference>Preprint policy requires human approval.",
                )
            if safety_score >= self.config.auto_apply_min_score and match.similarity >= 0.92:
                return PolicyDecision(True, False, "High-confidence replacement and adequate safety score.")
            return PolicyDecision(False, True, "Suggested replacement requires manual review.")

        if verdict_label == VerdictLabel.PARTIALLY_VERIFIED:
            return PolicyDecision(False, True, "Partial verification — emit annotation only.")

        if verdict_label == VerdictLabel.VERIFIED:
            return PolicyDecision(True, False, "Citation already verified; no patch needed.")

        return PolicyDecision(False, True, "Unverifiable claim — escalate for human review.")
