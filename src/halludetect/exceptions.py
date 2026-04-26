"""Custom exceptions for the halludetect pipeline."""

from __future__ import annotations


class HallDetectError(Exception):
    """Base exception for the halludetect package."""


class AbortAgentError(HallDetectError):
    """Raised by MELON when contrastive divergence indicates an active IPI.

    Mirrors the kaijiezhu11/MELON ``AbortAgentError`` semantics: the agent must
    halt all subsequent tool calls and surface a safeguard message to the user.
    """

    def __init__(self, message: str = "MELON aborted execution due to suspected indirect prompt injection."):
        super().__init__(message)
        self.message = message


class InjectionDetected(HallDetectError):
    """Raised when Layer 1 Sentinel classifies the input as adversarial."""

    def __init__(self, label: str, confidence: float):
        super().__init__(f"Sentinel flagged input as '{label}' (confidence={confidence:.4f}).")
        self.label = label
        self.confidence = confidence


class RetrievalError(HallDetectError):
    """Raised when all retrieval sources fail or return non-recoverable errors."""


class LinkageError(HallDetectError):
    """Raised when record linkage cannot resolve a manifestation conflict safely."""


class PolicyViolation(HallDetectError):
    """Raised when a rewrite would violate the configured policy preset."""
