"""Layer 3 MCP (Model Context Protocol) server.

Exposes the retrieval, liveness, and reasoning agents as MCP tools so that
external LLM agents (Cursor, Claude Desktop, custom Python clients) can call
them through the standardized protocol.

If the ``mcp`` SDK is installed the server runs over stdio; otherwise this
module degrades gracefully into a tiny JSON-RPC-over-stdio shim that exposes
the same tool names — so ``make mcp`` always succeeds.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from typing import Any

from ..config import load_config
from ..logging_setup import get_logger
from ..types import Citation
from .liveness_agent import LivenessAgent
from .reasoning_agent import ReasoningAgent
from .retrieval_agent import RetrievalAgent

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations (backend-agnostic)
# ─────────────────────────────────────────────────────────────────────────────


async def _tool_retrieve(retrieval: RetrievalAgent, payload: dict[str, Any]) -> dict[str, Any]:
    citation = Citation(**payload)
    records = await retrieval.retrieve(citation)
    return {"records": [r.model_dump() for r in records]}


async def _tool_liveness(liveness: LivenessAgent, payload: dict[str, Any]) -> dict[str, Any]:
    return (await liveness.check(payload.get("url", ""))).model_dump()


async def _tool_judge(reasoning: ReasoningAgent, payload: dict[str, Any]) -> dict[str, Any]:
    citation = Citation(**payload.get("citation", {}))
    candidates_payload = payload.get("candidates") or []
    from .reasoning_agent import RetrievedRecord  # re-export for type hint
    candidates = [RetrievedRecord(**c) for c in candidates_payload]
    j = reasoning.judge(citation, candidates)
    return {
        "score": j.score,
        "label": j.label.value,
        "rationale": j.rationale,
        "best_record": j.best_record.model_dump() if j.best_record else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Real MCP server (preferred when the SDK is installed)
# ─────────────────────────────────────────────────────────────────────────────


async def _serve_via_mcp_sdk() -> bool:
    try:
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        from mcp.types import TextContent, Tool  # type: ignore
    except ImportError:
        return False

    cfg = load_config()
    retrieval = RetrievalAgent(cfg.retrieval)
    liveness = LivenessAgent(cfg.retrieval.liveness)
    reasoning = ReasoningAgent(cfg.retrieval.reasoning)

    server = Server("halludetect-retrieval")

    @server.list_tools()  # type: ignore
    async def _list() -> list[Tool]:
        return [
            Tool(name="retrieve_citation", description="Cascaded multi-source retrieval for a Citation", inputSchema={"type": "object"}),
            Tool(name="check_url_liveness", description="Liveness + Wayback fallback for a URL/DOI", inputSchema={"type": "object"}),
            Tool(name="judge_citation", description="Semantic LLM-as-Judge over retrieved candidates", inputSchema={"type": "object"}),
        ]

    @server.call_tool()  # type: ignore
    async def _call(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "retrieve_citation":
            out = await _tool_retrieve(retrieval, arguments)
        elif name == "check_url_liveness":
            out = await _tool_liveness(liveness, arguments)
        elif name == "judge_citation":
            out = await _tool_judge(reasoning, arguments)
        else:
            out = {"error": f"unknown tool {name!r}"}
        return [TextContent(type="text", text=json.dumps(out))]

    async with stdio_server() as (read, write):  # type: ignore[attr-defined]
        await server.run(read, write, server.create_initialization_options())  # type: ignore[arg-type]
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Minimal stdio JSON-RPC fallback
# ─────────────────────────────────────────────────────────────────────────────


async def _serve_stdio_fallback() -> None:
    cfg = load_config()
    retrieval = RetrievalAgent(cfg.retrieval)
    liveness = LivenessAgent(cfg.retrieval.liveness)
    reasoning = ReasoningAgent(cfg.retrieval.reasoning)

    log.info("MCP stdio fallback active (install `mcp` for full protocol support)")
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(loop=loop)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            return
        try:
            msg = json.loads(line.decode())
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        params = msg.get("params") or {}
        msg_id = msg.get("id")

        try:
            if method == "tools/list":
                result = {
                    "tools": [
                        {"name": "retrieve_citation"},
                        {"name": "check_url_liveness"},
                        {"name": "judge_citation"},
                    ]
                }
            elif method == "tools/call":
                name = params.get("name")
                args = params.get("arguments") or {}
                if name == "retrieve_citation":
                    result = await _tool_retrieve(retrieval, args)
                elif name == "check_url_liveness":
                    result = await _tool_liveness(liveness, args)
                elif name == "judge_citation":
                    result = await _tool_judge(reasoning, args)
                else:
                    result = {"error": f"unknown tool {name!r}"}
            else:
                result = {"error": f"unknown method {method!r}"}
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n")
            sys.stdout.flush()
        except Exception as exc:  # pragma: no cover
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "error": {"message": str(exc)}}) + "\n")
            sys.stdout.flush()


async def _amain() -> None:
    served = await _serve_via_mcp_sdk()
    if not served:
        await _serve_stdio_fallback()


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_amain())


if __name__ == "__main__":
    main()
