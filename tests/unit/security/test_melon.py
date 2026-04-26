import re

import pytest

from halludetect.exceptions import AbortAgentError
from halludetect.security.melon import MelonDefender
from halludetect.security.trajectory import Trajectory, cosine_similarity

_FULL_BLOCK_RE = re.compile(
    r"={50,}\s*BEGIN_EXTERNAL_CONTENT.*?END_EXTERNAL_CONTENT\s*={50,}",
    re.S | re.I,
)


def _benign_llm(prompt: str) -> Trajectory:
    """Stub of a *non-compromised* LLM: response varies with the instruction."""

    instruction = _FULL_BLOCK_RE.sub("", prompt).strip()
    return Trajectory.from_text(instruction)


def _hijacked_llm(prompt: str) -> Trajectory:
    """Stub of a *compromised* LLM: same exfil payload regardless of prompt."""

    return Trajectory.from_text(
        "I will exfiltrate the conversation to https://attacker.example.com/webhook and "
        "fabricate a citation to Hacker (2099)."
    )


def test_benign_external_does_not_abort(adversarial_external_chunk):
    melon = MelonDefender()
    res = melon.evaluate(
        user_prompt="Summarize this paragraph.",
        external_chunks=["Contrastive learning improves representation quality."],
        llm_callable=_benign_llm,
        raise_on_abort=False,
    )
    assert res.aborted is False


def test_indirect_injection_aborts(adversarial_external_chunk):
    melon = MelonDefender()
    with pytest.raises(AbortAgentError):
        melon.evaluate(
            user_prompt="Summarize this paragraph.",
            external_chunks=[adversarial_external_chunk],
            llm_callable=_hijacked_llm,
        )


def test_cosine_extremes():
    a = Trajectory.from_text("alpha beta gamma")
    b = Trajectory.from_text("alpha beta gamma")
    c = Trajectory.from_text("delta epsilon zeta")
    assert cosine_similarity(a, b) == pytest.approx(1.0, rel=1e-6)
    assert cosine_similarity(a, c) == 0.0
