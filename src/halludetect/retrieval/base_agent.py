"""Base class + helpers for the retrieval source agents.

Includes:

* **Retry-After-aware backoff** for HTTP 429 responses (honors the header
  when present, capped at ``MAX_BACKOFF_S``).
* **Per-source circuit breaker** — after ``CIRCUIT_BREAK_THRESHOLD`` consecutive
  rate-limit responses, the source self-disables for the rest of the run so
  it stops contributing noise to the logs and burning latency.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod

import httpx

from ..config import HttpConfig
from ..logging_setup import get_logger
from ..types import Citation, RetrievedRecord

log = get_logger(__name__)


MAX_BACKOFF_S = 30.0
CIRCUIT_BREAK_THRESHOLD = 4  # consecutive 429s before a source is auto-disabled


class BaseSource(ABC):
    """A single external academic data source.

    Implementations override ``query`` to translate a normalized ``Citation``
    into one or more candidate ``RetrievedRecord`` objects, asynchronously.
    """

    name: str = "base"

    def __init__(self, http: HttpConfig | None = None) -> None:
        self.http = http or HttpConfig()
        self._consecutive_429s: int = 0
        self._circuit_open: bool = False

    @property
    def offline(self) -> bool:
        return os.environ.get("HALLUDETECT_OFFLINE", "0") == "1"

    @property
    def disabled(self) -> bool:
        return self._circuit_open

    # ------------------------------------------------------------------------

    @abstractmethod
    async def query(self, client: httpx.AsyncClient, citation: Citation) -> list[RetrievedRecord]:
        """Run a multi-pass query for ``citation`` and return candidates."""

    # ------------------------------------------------------------------------

    async def _safe_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response | None:
        if self._circuit_open:
            return None

        delay = self.http.backoff_seconds
        for attempt in range(1, self.http.max_retries + 1):
            try:
                resp = await client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                log.warning(
                    "source request failed",
                    source=self.name,
                    attempt=attempt,
                    url=url,
                    error=str(exc),
                )
                if attempt == self.http.max_retries:
                    return None
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_BACKOFF_S)
                continue

            if resp.status_code == 429:
                self._consecutive_429s += 1
                if self._consecutive_429s >= CIRCUIT_BREAK_THRESHOLD:
                    log.warning(
                        "circuit breaker tripped — disabling source for the rest of this run",
                        source=self.name,
                        consecutive_429s=self._consecutive_429s,
                    )
                    self._circuit_open = True
                    return None
                if attempt < self.http.max_retries:
                    wait = self._retry_after(resp, fallback=delay)
                    log.info(
                        "rate limited; backing off",
                        source=self.name,
                        attempt=attempt,
                        wait_s=round(wait, 2),
                    )
                    await asyncio.sleep(wait)
                    delay = min(delay * 2, MAX_BACKOFF_S)
                    continue
                return resp

            self._consecutive_429s = 0  # successful (non-429) response
            return resp
        return None

    @staticmethod
    def _retry_after(resp: httpx.Response, *, fallback: float) -> float:
        """Honor the standard ``Retry-After`` header (seconds or HTTP-date)."""

        raw = resp.headers.get("retry-after")
        if not raw:
            return min(fallback, MAX_BACKOFF_S)
        raw = raw.strip()
        try:
            return min(float(raw), MAX_BACKOFF_S)
        except ValueError:
            return min(fallback, MAX_BACKOFF_S)
