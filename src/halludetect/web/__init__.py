"""Web interface for halludetect — FastAPI backend + premium frontend."""

from .server import build_app, serve

__all__ = ["build_app", "serve"]
