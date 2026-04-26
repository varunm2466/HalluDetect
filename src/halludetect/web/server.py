"""FastAPI backend for the halludetect real-time web UI."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field

from ..config import HallDetectConfig, load_config
from ..logging_setup import get_logger, setup_logging
from ..pipeline import HallucinationDetectionPipeline
from ..security.melon import default_stub_llm
from ..types import InputDocument

log = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────


class ThreatRequest(BaseModel):
    user_prompt: str = Field(..., min_length=0, max_length=32_000)
    external_context: list[str] = Field(default_factory=list)
    preset: str = Field(default="default", pattern="^(default|strict|lenient)$")


class AuditRequest(BaseModel):
    user_prompt: str = Field(..., min_length=0, max_length=32_000)
    generated_text: str = Field(default="", max_length=200_000)
    external_context: list[str] = Field(default_factory=list)
    preset: str = Field(default="default", pattern="^(default|strict|lenient)$")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline cache (one instance per preset)
# ─────────────────────────────────────────────────────────────────────────────


_PIPELINE_CACHE: dict[str, HallucinationDetectionPipeline] = {}


def _config_for_preset(preset: str) -> HallDetectConfig:
    if preset == "strict":
        return load_config("configs/safety_strict.yaml")
    if preset == "lenient":
        return load_config("configs/safety_lenient.yaml")
    return load_config()


def _get_pipeline(preset: str) -> HallucinationDetectionPipeline:
    if preset not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[preset] = HallucinationDetectionPipeline(_config_for_preset(preset))
    return _PIPELINE_CACHE[preset]


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────


def build_app():
    """Construct the FastAPI app. Lazy import so non-web installs still work."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is not installed. Install with: pip install -e \".[web]\""
        ) from exc

    setup_logging()
    app = FastAPI(
        title="halludetect",
        description="Prompt-Aware Citation Hallucination Detection",
        version="0.1.0",
    )

    # ── routes ──────────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/healthz", include_in_schema=False)
    async def health():
        return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}

    @app.post("/api/threat-check")
    async def threat_check(req: ThreatRequest):
        """Fast Layer-1-only check: Sentinel + MELON contrastive eval."""

        t0 = time.perf_counter()
        pipe = _get_pipeline(req.preset)
        sentinel_result = pipe.sentinel.classify(req.user_prompt)

        from ..exceptions import AbortAgentError

        try:
            melon_result = pipe.melon.evaluate(
                user_prompt=req.user_prompt,
                external_chunks=req.external_context,
                llm_callable=default_stub_llm,
                raise_on_abort=False,
            )
        except AbortAgentError as exc:
            from ..security.melon import MelonResult
            from ..security.trajectory import Trajectory
            melon_result = MelonResult(
                cosine_similarity=1.0,
                aborted=True,
                reason=str(exc),
                origin_trajectory=Trajectory(),
                masked_trajectory=Trajectory(),
            )

        threat = pipe.threat_aggregator.aggregate(
            sentinel_result, melon_result, sentinel_blocked=sentinel_result.is_malicious
        )
        return {
            "threat": threat.model_dump(),
            "sentinel_hits": sentinel_result.hits,
            "sentinel_backend": sentinel_result.backend,
            "config": {
                "preset": req.preset,
                "melon_threshold": pipe.config.security.melon.cosine_threshold,
                "block_threshold": pipe.config.security.sentinel.block_threshold,
                "abort_p_inj": pipe.config.scoring.policy.block_on_injection_prob,
            },
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
        }

    @app.post("/api/audit")
    async def audit(req: AuditRequest):
        """Full pipeline run — returns the complete PipelineReport."""

        t0 = time.perf_counter()
        pipe = _get_pipeline(req.preset)
        doc = InputDocument(
            user_prompt=req.user_prompt,
            generated_text=req.generated_text or req.user_prompt,
            external_context=req.external_context,
        )
        try:
            report = await pipe.arun(doc)
        except Exception as exc:  # pragma: no cover
            log.error("audit failed", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc))
        return {
            "report": report.model_dump(mode="json"),
            "summary": pipe.summarize(report),
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
        }

    @app.post("/api/audit/stream")
    async def audit_stream(req: AuditRequest):
        """Server-Sent Events stream — emits per-layer progress events."""

        async def event_stream() -> AsyncGenerator[bytes, None]:
            pipe = _get_pipeline(req.preset)
            doc = InputDocument(
                user_prompt=req.user_prompt,
                generated_text=req.generated_text or req.user_prompt,
                external_context=req.external_context,
            )

            yield _sse("status", {"phase": "starting"})
            try:
                report = await pipe.arun(doc)
            except Exception as exc:  # pragma: no cover
                yield _sse("error", {"message": str(exc)})
                return

            for lr in report.layer_results:
                yield _sse(
                    "layer",
                    {
                        "layer": lr.layer,
                        "duration_ms": lr.duration_ms,
                        "succeeded": lr.succeeded,
                        "payload": lr.payload,
                    },
                )
                await asyncio.sleep(0.15)

            yield _sse(
                "complete",
                {
                    "report": report.model_dump(mode="json"),
                    "summary": pipe.summarize(report),
                },
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ── static assets ───────────────────────────────────────────────────────

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


def _sse(event: str, data: dict[str, Any]) -> bytes:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Launch the development server."""

    try:
        import uvicorn  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is not installed. Install with: pip install -e \".[web]\"") from exc

    if reload:
        uvicorn.run("halludetect.web.server:build_app", host=host, port=port, reload=True, factory=True)
    else:
        uvicorn.run(build_app(), host=host, port=port)
