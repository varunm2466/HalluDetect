from halludetect.linkage.deduper import Deduper
from halludetect.linkage.field_weights import FieldSimilarity
from halludetect.linkage.manifestation import ManifestationResolver
from halludetect.types import Citation, Manifestation, RetrievedRecord


def _cite() -> Citation:
    return Citation(
        raw="(Smith, 2025)",
        title="Contrastive Robustness for Large Language Models",
        authors=["John Smith", "Alice Doe"],
        year=2025,
        venue="Journal of AI Safety",
        doi="10.9999/foo.2025.001",
    )


def _record(**kw) -> RetrievedRecord:
    base = dict(
        source="crossref",
        title="Contrastive Robustness for Large Language Models",
        authors=["J. Smith", "A. Doe"],
        year=2025,
        venue="Journal of AI Safety",
        doi="10.9999/foo.2025.001",
        manifestation=Manifestation.JOURNAL,
    )
    base.update(kw)
    return RetrievedRecord(**base)


def test_field_similarity_perfect_match():
    sim = FieldSimilarity().score(_cite(), _record())
    assert sim.overall >= 0.95
    assert sim.by_field["doi"] == 1.0


def test_dedupe_picks_best_match():
    d = Deduper()
    cands = [
        _record(title="Totally unrelated paper", doi="10.1/x", authors=["X"], year=2020, manifestation=Manifestation.UNKNOWN),
        _record(),
    ]
    best = d.best(_cite(), cands)
    assert best is not None
    assert best.record.doi == "10.9999/foo.2025.001"


def test_manifestation_priority_resolves_journal_over_arxiv():
    r = ManifestationResolver()
    cite = _cite()
    cands = [
        _record(manifestation=Manifestation.ARXIV, doi=None, arxiv_id="2501.0001"),
        _record(manifestation=Manifestation.JOURNAL),
    ]
    truth = r.resolve(cite, cands)
    assert truth is not None
    assert truth.manifestation == Manifestation.JOURNAL


def test_manifestation_conflict_detected():
    r = ManifestationResolver()
    cite = Citation(raw="x", title="t", venue="arXiv", arxiv_id="2501.0001")
    truth = _record(manifestation=Manifestation.JOURNAL)
    assert r.conflict(cite, truth) is True
