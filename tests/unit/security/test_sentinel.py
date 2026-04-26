import pytest

from halludetect.exceptions import InjectionDetected
from halludetect.security.sentinel import SentinelClassifier


def test_benign_prompt_passes(benign_prompt):
    s = SentinelClassifier()
    res = s.classify(benign_prompt)
    assert res.backend == "heuristic"
    assert res.is_malicious is False
    assert 0.0 <= res.score < 0.5


def test_malicious_prompt_flagged(malicious_prompt):
    s = SentinelClassifier()
    res = s.classify(malicious_prompt)
    assert res.is_malicious is True
    assert res.score >= 0.5
    assert any(h in res.hits for h in ("ignore-prior", "jailbreak", "fabrication-cmd", "exfil-prompt"))


def test_assert_safe_raises(malicious_prompt):
    s = SentinelClassifier()
    with pytest.raises(InjectionDetected) as exc_info:
        s.assert_safe(malicious_prompt)
    assert exc_info.value.confidence > 0.5


def test_empty_input():
    s = SentinelClassifier()
    res = s.classify("")
    assert res.score == 0.0
    assert res.is_malicious is False
