from halludetect.retrieval.reasoning_agent import ReasoningAgent
from halludetect.types import Citation, Manifestation, RetrievedRecord, VerdictLabel


def test_no_candidates_marks_hallucinated():
    j = ReasoningAgent().judge(
        Citation(raw="?", title="Bogus", authors=["X"], year=2099), []
    )
    assert j.label == VerdictLabel.HALLUCINATED


def test_doi_match_verifies():
    cite = Citation(raw="?", title="Real", year=2025, doi="10.9/x")
    rec = RetrievedRecord(
        source="crossref", title="Real",
        authors=[], year=2025, doi="10.9/x", manifestation=Manifestation.JOURNAL,
    )
    j = ReasoningAgent().judge(cite, [rec])
    assert j.label == VerdictLabel.VERIFIED
    assert j.score >= 0.9


def test_partial_match():
    cite = Citation(raw="?", title="Quantum methods for NLP", authors=["A. Author"], year=2024)
    rec = RetrievedRecord(
        source="crossref", title="Quantum methods", authors=["A. Author"],
        year=2024, manifestation=Manifestation.JOURNAL,
    )
    j = ReasoningAgent().judge(cite, [rec])
    assert j.label in (VerdictLabel.PARTIALLY_VERIFIED, VerdictLabel.VERIFIED)
