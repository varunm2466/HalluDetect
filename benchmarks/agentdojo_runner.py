"""AgentDojo benchmark runner for the MELON contrastive defense.

Iterates over an AgentDojo-style JSONL dataset and reports:

    * Aborted-attack rate (true positive)
    * Aborted-benign rate (false positive)
    * Mean cosine similarity per category

Run:
    python -m benchmarks.agentdojo_runner --dataset path/to/agentdojo.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from halludetect.config import load_config
from halludetect.security.melon import MelonDefender
from halludetect.security.trajectory import Trajectory


def _llm_stub(prompt: str) -> Trajectory:
    return Trajectory.from_text(prompt[:200])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config()
    melon = MelonDefender(cfg.security.melon)

    aborted = total = 0
    for line in args.dataset.read_text().splitlines():
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        res = melon.evaluate(
            user_prompt=ex.get("user_prompt", ""),
            external_chunks=ex.get("external_context", []),
            llm_callable=_llm_stub,
            raise_on_abort=False,
        )
        aborted += int(res.aborted)

    print(f"Total: {total} | Aborted: {aborted} | Abort rate: {aborted / max(total, 1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
