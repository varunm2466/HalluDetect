"""Sub-Layer 1B — MELON: Masked re-Execution and TooL comparisON.

Implements the kaijiezhu11/MELON algorithmic protocol:

    1. **State Duplication** — deep-copy the runtime so the masked run cannot
       mutate the origin run's environment.
    2. **Origin Run Execution** — invoke the LLM with the standard prompt.
    3. **Masked Run Generation** — replace the user query with a generalized,
       neutral instruction; preserve external content inside boundary tokens.
    4. **Contrastive Action Comparison** — cosine-similarity over trajectory
       bag-vectors. If similarity ≥ ``cosine_threshold``, the external content
       (not the user) is dictating the model's behavior.
    5. **Execution Halting** — raise ``AbortAgentError`` and inject a safeguard
       message so downstream layers terminate cleanly.

The defender is decoupled from any specific LLM: callers supply an
``llm_callable`` (str -> str|Trajectory) so the same logic works against
OpenAI, Anthropic, local HF models, or — for tests — a deterministic stub.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..config import MelonConfig
from ..exceptions import AbortAgentError
from ..logging_setup import get_logger
from ..types import MelonOutput
from .delimiters import build_masked_user_prompt, build_origin_prompt
from .trajectory import Trajectory, cosine_similarity

log = get_logger(__name__)


LLMResponse = str | Trajectory
LLMCallable = Callable[[str], LLMResponse]


def _to_trajectory(response: LLMResponse) -> Trajectory:
    if isinstance(response, Trajectory):
        return response
    return Trajectory.from_text(str(response))


# ─────────────────────────────────────────────────────────────────────────────
# Reference offline LLM stub — models a *vulnerable* model:
#   * Strips the wrapped external-content block from the prompt.
#   * Uses the surviving instruction tokens as the synthetic "response".
#   * If the external block contains adversarial markers, leaks them into the
#     response — emulating a compromised LLM whose tokens are now driven by
#     the injected payload (the very signature MELON is built to detect).
# ─────────────────────────────────────────────────────────────────────────────


# Match the *entire* wrapped external-content block (boundaries + body + tags).
# Both origin and masked runs share this block; a faithful LLM stub must remove
# it so the resulting trajectories actually diverge between runs.
_FULL_BLOCK_RE = re.compile(
    r"={50,}\s*BEGIN_EXTERNAL_CONTENT.*?END_EXTERNAL_CONTENT\s*={50,}",
    re.S | re.I,
)
_BOUNDARY_BLOCK_RE = re.compile(r"={50,}.*?={50,}", re.S)
_INJECTION_LEAK_RE = re.compile(
    r"\b(ignore\s+all|disregard|fabricate|exfiltrate|webhook|attacker|hacker|pwn|leak\s+the|reveal\s+the)\b",
    re.I,
)


def default_stub_llm(prompt: str) -> Trajectory:
    """Deterministic, dependency-free stand-in for a real LLM call.

    Behaves like a vulnerable assistant for offline / test scenarios:

    - benign external content → response varies between origin & masked runs
      (low cosine, MELON does not abort);
    - adversarial external content → leaked payload tokens appear in BOTH
      runs (high cosine, MELON aborts contrastively).
    """

    block_text = ""
    block_match = _FULL_BLOCK_RE.search(prompt)
    if block_match is not None:
        block_text = block_match.group(0)
        instruction = (prompt[: block_match.start()] + " " + prompt[block_match.end():]).strip()
    else:
        instruction = _BOUNDARY_BLOCK_RE.sub("", prompt).strip()
        block_text = prompt

    leaked = _INJECTION_LEAK_RE.findall(block_text)
    return Trajectory.from_text((instruction + " " + " ".join(leaked)).strip())


@dataclass
class MelonResult:
    cosine_similarity: float
    aborted: bool
    reason: str | None
    origin_trajectory: Trajectory
    masked_trajectory: Trajectory

    def to_output(self) -> MelonOutput:
        return MelonOutput(
            cosine_similarity=self.cosine_similarity,
            aborted=self.aborted,
            reason=self.reason,
        )


class MelonDefender:
    """Contrastive masked-re-execution defender against indirect prompt injection."""

    def __init__(self, config: MelonConfig | None = None) -> None:
        self.config = config or MelonConfig()

    # ------------------------------------------------------------------------

    def evaluate(
        self,
        *,
        user_prompt: str,
        external_chunks: Iterable[str],
        llm_callable: LLMCallable,
        runtime_state: dict | None = None,
        raise_on_abort: bool = True,
    ) -> MelonResult:
        """Run the contrastive protocol and return / raise as configured."""

        external_list = [c for c in external_chunks if c]

        # 1. State duplication — keep masked run's env independent.
        _origin_state = copy.deepcopy(runtime_state) if runtime_state is not None else None
        _masked_state = copy.deepcopy(runtime_state) if runtime_state is not None else None
        del _origin_state, _masked_state  # noqa: F841 — illustrative; LLMs are stateless here

        # 2. Origin run.
        origin_prompt = build_origin_prompt(user_prompt, external_list)
        origin_resp = llm_callable(origin_prompt)
        origin_traj = _to_trajectory(origin_resp)

        # 3. Masked run.
        masked_prompt = build_masked_user_prompt(
            external_list,
            boundary=self.config.boundary_token,
            masked_instruction=self.config.masked_user_prompt,
        )
        masked_resp = llm_callable(masked_prompt)
        masked_traj = _to_trajectory(masked_resp)

        # 4. Contrastive comparison.
        sim = cosine_similarity(origin_traj, masked_traj)
        log.info(
            "melon contrastive eval",
            cosine=round(sim, 4),
            threshold=self.config.cosine_threshold,
            origin_tokens=len(origin_traj.tokens),
            masked_tokens=len(masked_traj.tokens),
        )

        # 5. Halting decision.
        aborted = sim >= self.config.cosine_threshold and not origin_traj.is_empty()
        reason = self.config.abort_message if aborted else None
        result = MelonResult(
            cosine_similarity=sim,
            aborted=aborted,
            reason=reason,
            origin_trajectory=origin_traj,
            masked_trajectory=masked_traj,
        )
        if aborted and raise_on_abort:
            raise AbortAgentError(self.config.abort_message)
        return result
