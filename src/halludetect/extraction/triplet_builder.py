"""Convert SRL frames into ``Triplet`` DTOs (RefChecker-style atomic facts)."""

from __future__ import annotations

from ..config import TripletsConfig
from ..types import Triplet
from .srl import SrlFrame


class TripletBuilder:
    def __init__(self, config: TripletsConfig | None = None) -> None:
        self.config = config or TripletsConfig()

    def build(self, frames: list[SrlFrame]) -> list[Triplet]:
        out: list[Triplet] = []
        seen: set[tuple[str, str, str]] = set()
        for f in frames:
            subj = f.subject.strip()
            rel = f.predicate.strip()
            obj = f.obj.strip()
            if not subj or not rel or not obj:
                continue
            if min(len(subj), len(rel), len(obj)) < self.config.min_token_length:
                continue
            key = (subj.lower(), rel.lower(), obj.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Triplet(subject=subj, relation=rel, object=obj, sentence=f.sentence, span=f.span)
            )
        return out
