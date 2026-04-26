"""Liveness & anti-rot agent — distinguishes hallucinated URLs from link-rot.

Per the design doc:

    * HEAD/GET each candidate URL/DOI.
    * On 4xx/5xx, query the Wayback CDX API.
    * If neither resolves AND no archive exists → URL is hallucinated.
    * If unreachable now BUT archived → link-rot, not a hallucination.
"""

from __future__ import annotations

import re

import httpx

from ..config import LivenessConfig
from ..logging_setup import get_logger
from ..types import LivenessReport

log = get_logger(__name__)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s]+", re.I)


class LivenessAgent:
    def __init__(self, config: LivenessConfig | None = None) -> None:
        self.config = config or LivenessConfig()

    # ------------------------------------------------------------------------

    async def check(self, url: str, *, client: httpx.AsyncClient | None = None) -> LivenessReport:
        if not url:
            return LivenessReport(url=url, reachable=False, archived=False, error="empty url")
        target = self._normalize(url)
        owns = client is None
        client = client or httpx.AsyncClient(
            timeout=self.config.timeout_s,
            follow_redirects=self.config.follow_redirects,
            headers={"User-Agent": self.config.user_agent},
        )
        try:
            return await self._check_with(client, target)
        finally:
            if owns:
                await client.aclose()

    async def _check_with(self, client: httpx.AsyncClient, url: str) -> LivenessReport:
        status: int | None = None
        err: str | None = None
        try:
            resp = await client.head(url)
            status = resp.status_code
            if status >= 400:
                resp = await client.get(url)
                status = resp.status_code
        except httpx.HTTPError as exc:
            err = str(exc)

        reachable = status is not None and 200 <= status < 400
        if reachable:
            return LivenessReport(url=url, http_status=status, reachable=True, archived=False)

        archived, archive_url = await self._wayback_lookup(client, url)
        return LivenessReport(
            url=url,
            http_status=status,
            reachable=False,
            archived=archived,
            archive_url=archive_url,
            error=err,
        )

    async def _wayback_lookup(self, client: httpx.AsyncClient, url: str) -> tuple[bool, str | None]:
        params = {
            "url": url,
            "output": "json",
            "limit": "1",
            "filter": "statuscode:200",
            "fastLatest": "true",
        }
        try:
            resp = await client.get(self.config.wayback_cdx_url, params=params)
            if resp.status_code != 200:
                return False, None
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return False, None
        if not data or len(data) < 2:
            return False, None
        ts, original = data[1][1], data[1][2]
        return True, f"https://web.archive.org/web/{ts}/{original}"

    @staticmethod
    def _normalize(url: str) -> str:
        if _DOI_RE.fullmatch(url.strip()):
            return f"https://doi.org/{url.strip()}"
        return url.strip()
