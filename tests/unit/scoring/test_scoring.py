from halludetect.config import PolicyConfig
from halludetect.scoring.hrs import HRSAggregator
from halludetect.scoring.policy_gate import PolicyGate
from halludetect.scoring.rewrite_engine import RewriteEngine
from halludetect.types import (
    Citation,
    Claim,
    LinkageMatch,
    Manifestation,
    MelonOutput,
    PatchAction,
    RetrievedRecord,
    SentinelOutput,
    ThreatReport,
    Triplet,
    Verdict,
    VerdictLabel,
)


def _threat(p_inj: float = 0.0, blocked: bool = False) -> ThreatReport:
    return ThreatReport(
        sentinel=SentinelOutput(label="benign", score=0.05, is_malicious=False),
        melon=MelonOutput(cosine_similarity=0.0, aborted=False),
        p_injection=p_inj,
        blocked=blocked,
    )


def _claim() -> Claim:
    return Claim(triplet=Triplet(subject="x", relation="is", object="y", sentence="x is y"))


def _record() -> RetrievedRecord:
    return RetrievedRecord(
        source="crossref",
        title="Real Paper",
        authors=["A. Real"],
        year=2025,
        venue="Real Journal",
        doi="10.9999/real.001",
        manifestation=Manifestation.JOURNAL,
    )


def test_hrs_perfect_score():
    hrs = HRSAggregator()
    threat = _threat()
    verdicts = [(_claim(), VerdictLabel.VERIFIED, None)]
    out = hrs.compute(threat=threat, u_intrinsic_norm=0.0, verdicts=verdicts)
    assert out.score >= 95.0
    assert out.v_extrinsic == 1.0


def test_hrs_full_compromise_zeros_out():
    hrs = HRSAggregator()
    threat = _threat(p_inj=1.0, blocked=True)
    verdicts = [(_claim(), VerdictLabel.HALLUCINATED, None)]
    out = hrs.compute(threat=threat, u_intrinsic_norm=1.0, verdicts=verdicts)
    assert out.score <= 5.0


def test_policy_blocks_on_high_pinj():
    gate = PolicyGate(PolicyConfig(block_on_injection_prob=0.4))
    decision = gate.evaluate(
        verdict_label=VerdictLabel.HALLUCINATED,
        match=None,
        threat=_threat(p_inj=0.6),
        safety_score=70.0,
    )
    assert decision.safe_to_apply is False


def test_rewrite_engine_emits_bibtex_for_replacement():
    citation = Citation(raw="?", title="Bogus paper", year=2099, authors=["Hacker, A."])
    record = _record()
    match = LinkageMatch(citation=citation, record=record, similarity=0.95, manifestation_resolved=record.manifestation)
    verdicts = [Verdict(citation=citation, label=VerdictLabel.HALLUCINATED, rationale="x", matches=[match])]
    engine = RewriteEngine(PolicyGate(PolicyConfig(auto_apply_min_score=10.0)))
    patches = engine.build_patches(verdicts=verdicts, threat=_threat(), safety_score=99.0)
    assert len(patches) == 1
    p = patches[0]
    assert p.action == PatchAction.REPLACE
    assert p.bibtex and "@article" in p.bibtex
    assert p.markdown and "Real Paper" in p.markdown


def test_rewrite_engine_noop_for_verified():
    citation = Citation(raw="?", title="Real Paper", year=2025, doi="10.9999/real.001")
    record = _record()
    match = LinkageMatch(citation=citation, record=record, similarity=0.99, manifestation_resolved=record.manifestation)
    verdicts = [Verdict(citation=citation, label=VerdictLabel.VERIFIED, rationale="match", matches=[match])]
    engine = RewriteEngine(PolicyGate())
    patches = engine.build_patches(verdicts=verdicts, threat=_threat(), safety_score=95.0)
    assert patches[0].action == PatchAction.NOOP
