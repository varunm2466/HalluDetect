"""Action / text trajectory abstractions for MELON contrastive comparison.

A Trajectory is a deterministic, ordered representation of what the LLM agent
*would do* given a particular prompt. We compare two trajectories — origin vs
masked — using a token-bag cosine similarity that does not require a heavy ML
backend (so MELON works even in offline heuristic mode).

When real action traces are available (tool-call graphs, structured outputs),
``Trajectory.from_actions`` accepts a list of (tool_name, args) tuples and
hashes them into the same embedding space.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _action_token(name: str, args: Any) -> str:
    """Stable hash for a (tool, args) tuple — used as a single token."""

    payload = f"{name}::{repr(args)}"
    return "act_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class Trajectory:
    """Bag-of-tokens representation of an LLM agent's behavior."""

    tokens: Counter = field(default_factory=Counter)
    raw_text: str = ""
    actions: list[tuple[str, Any]] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> Trajectory:
        toks = _tokenize(text)
        return cls(tokens=Counter(toks), raw_text=text)

    @classmethod
    def from_actions(cls, actions: list[tuple[str, Any]], text: str = "") -> Trajectory:
        bag = Counter(_tokenize(text))
        for name, args in actions:
            bag[_action_token(name, args)] += 1
        return cls(tokens=bag, raw_text=text, actions=list(actions))

    def is_empty(self) -> bool:
        return not self.tokens


def cosine_similarity(a: Trajectory, b: Trajectory) -> float:
    """Cosine similarity between two trajectory bag vectors. Range [0, 1].

    If either trajectory is empty the similarity is defined as 0.0 to avoid
    treating "no-output / no-output" as suspiciously identical behavior.
    """

    if a.is_empty() or b.is_empty():
        return 0.0
    common = set(a.tokens) & set(b.tokens)
    dot = sum(a.tokens[t] * b.tokens[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.tokens.values()))
    norm_b = math.sqrt(sum(v * v for v in b.tokens.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
