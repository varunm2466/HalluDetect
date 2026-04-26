"""Layer 3 — Multi-Agent Verification and Retrieval Engine."""

from .liveness_agent import LivenessAgent
from .parsing_agent import ParsingAgent
from .reasoning_agent import ReasoningAgent
from .retrieval_agent import RetrievalAgent
from .sources.arxiv import ArxivSource
from .sources.crossref import CrossrefSource
from .sources.pubmed import PubmedSource
from .sources.semantic_scholar import SemanticScholarSource

__all__ = [
    "ArxivSource",
    "CrossrefSource",
    "LivenessAgent",
    "ParsingAgent",
    "PubmedSource",
    "ReasoningAgent",
    "RetrievalAgent",
    "SemanticScholarSource",
]
