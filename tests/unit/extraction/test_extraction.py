from halludetect.extraction.citation_aligner import CitationAligner, extract_citations
from halludetect.extraction.entailment import EntailmentScorer
from halludetect.extraction.srl import SemanticRoleLabeler
from halludetect.extraction.triplet_builder import TripletBuilder
from halludetect.extraction.uncertainty import UncertaintyEstimator

SAMPLE = (
    "Smith and Doe (2025) demonstrated that contrastive learning improves robustness. "
    "The authors evaluated their approach on three benchmarks [1]. "
    "Greshake et al. (2023) introduced indirect prompt injection attacks.\n\n"
    "References\n"
    "[1] Doe, A. (2025). Contrastive robustness for LLMs. Journal of AI Safety, 1(1), 1–10."
)


def test_srl_produces_frames():
    srl = SemanticRoleLabeler()
    frames = srl.parse(SAMPLE)
    assert len(frames) >= 2
    preds = {f.predicate for f in frames}
    assert {"demonstrated", "introduced"} & preds


def test_triplet_builder_dedupes():
    srl = SemanticRoleLabeler()
    frames = srl.parse(SAMPLE)
    triplets = TripletBuilder().build(frames)
    assert triplets, "expected at least one triplet"
    keys = {(t.subject.lower(), t.relation.lower(), t.object.lower()) for t in triplets}
    assert len(keys) == len(triplets)


def test_extract_citations_finds_all_styles():
    cites = extract_citations(SAMPLE)
    raws = [c.raw for c in cites]
    assert any("Smith" in r and "(2025)" in r for r in raws)
    assert any("Greshake" in r for r in raws)
    assert any(r.startswith("[1]") for r in raws)


def test_alignment_links_triplets_to_citations():
    srl = SemanticRoleLabeler()
    frames = srl.parse(SAMPLE)
    triplets = TripletBuilder().build(frames)
    citations = extract_citations(SAMPLE)
    claims = CitationAligner.align(SAMPLE, triplets, citations)
    assert any(c.citation is not None for c in claims)


def test_entailment_lexical_fallback():
    nli = EntailmentScorer()
    s = nli.score("Cats are mammals that purr.", "Cats are mammals.")
    assert s > 0.4
    s2 = nli.score("Birds fly.", "Cats are mammals.")
    assert s2 < s


def test_uncertainty_increases_when_no_citations():
    est = UncertaintyEstimator()
    from halludetect.types import Claim, Triplet
    no_cite_claims = [
        Claim(triplet=Triplet(subject="x", relation="is", object="y", sentence="x is y"), citation=None),
    ]
    raw = est.estimate(claims=no_cite_claims)
    assert est.normalize(raw) > 0.0
