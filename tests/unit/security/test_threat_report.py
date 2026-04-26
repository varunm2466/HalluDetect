from halludetect.security.melon import MelonResult
from halludetect.security.sentinel import SentinelResult
from halludetect.security.threat_report import ThreatAggregator
from halludetect.security.trajectory import Trajectory


def _melon(sim: float, aborted: bool = False) -> MelonResult:
    return MelonResult(
        cosine_similarity=sim,
        aborted=aborted,
        reason="abort" if aborted else None,
        origin_trajectory=Trajectory(),
        masked_trajectory=Trajectory(),
    )


def _sent(score: float, malicious: bool) -> SentinelResult:
    return SentinelResult(
        label="jailbreak" if malicious else "benign",
        score=score,
        is_malicious=malicious,
        backend="heuristic",
        hits=[],
    )


def test_benign_signals_low_p_inj():
    """High-confidence benign Sentinel → low P_inj after probabilistic flip."""

    agg = ThreatAggregator()
    # label="benign", score=0.95 (95% confident benign) → P(malicious) = 0.05
    rep = agg.aggregate(_sent(0.95, False), _melon(0.0))
    assert rep.p_injection <= 0.2
    assert rep.blocked is False


def test_malicious_sentinel_dominates():
    agg = ThreatAggregator()
    rep = agg.aggregate(_sent(0.92, True), _melon(0.0))
    assert rep.p_injection >= 0.9
    assert rep.blocked is False  # blocked is set by caller


def test_indirect_injection_lifts_floor():
    agg = ThreatAggregator()
    # Benign Sentinel + adversarial MELON cosine should still raise blocked=True.
    rep = agg.aggregate(_sent(0.95, False), _melon(0.6, aborted=True))
    assert rep.p_injection > 0.5
    assert rep.blocked is True
