"""Pre-cache the HF and sentence-transformers models we use.

Run once before the first online inference to avoid first-call latency:

    python scripts/download_models.py
"""

from __future__ import annotations

import sys

MODELS = [
    ("transformers", "qualifire/prompt-injection-sentinel"),
    ("transformers", "cross-encoder/nli-deberta-v3-base"),
    ("sentence-transformers", "sentence-transformers/all-MiniLM-L6-v2"),
]


def main() -> int:
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError:
        print("ERROR: pip install -e \".[ml]\" first.", file=sys.stderr)
        return 1
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        SentenceTransformer = None  # type: ignore[assignment]

    for kind, name in MODELS:
        print(f"→ {kind}: {name}")
        if kind == "transformers":
            AutoTokenizer.from_pretrained(name)
            AutoModel.from_pretrained(name)
        elif kind == "sentence-transformers" and SentenceTransformer is not None:
            SentenceTransformer(name)
    print("All models cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
