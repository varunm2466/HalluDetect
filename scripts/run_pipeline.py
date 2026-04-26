"""Convenience wrapper: ``python scripts/run_pipeline.py`` → demo run."""

from __future__ import annotations

from halludetect.cli import app

if __name__ == "__main__":
    app(["demo"])
