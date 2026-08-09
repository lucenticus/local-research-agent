"""Sync bridge over `langchain-mcp-adapters` (async-only) — funnel/ingest
code in this project is synchronous throughout (§7 CLAUDE.md provider-seam
convention: agent logic never talks to an SDK directly, only through a thin
wrapper), this module is the one place that spins an event loop to reach an
MCP server.

Connect-per-call, not a persistent session: `MultiServerMCPClient.get_tools()`
already opens a fresh session per tool invocation internally (see its
docstring — "a new session will be created for each tool call"), so there is
no long-lived connection to manage here either. That matches this project's
scale (a single local user, occasional calls) — a persistent connection
would need a background event-loop thread for no real benefit at this QPS.

`langchain-mcp-adapters` returns tools whose `_run` raises
`NotImplementedError` (async-only `StructuredTool`s, confirmed by a real
`.invoke()` call against a local filesystem MCP server) — `get_mcp_tools`
re-wraps each one so `.invoke()` works from plain sync code.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def _wrap_sync(tool: BaseTool) -> StructuredTool:
    def _run(**kwargs: Any) -> Any:
        return asyncio.run(tool.ainvoke(kwargs))

    return StructuredTool.from_function(
        func=_run, name=tool.name, description=tool.description, args_schema=tool.args_schema
    )


def get_mcp_tools(connections: dict[str, Any], *, server_name: str | None = None) -> list[StructuredTool]:
    """Connects to the given MCP server(s) just long enough to list their
    tools, then returns them wrapped for sync `.invoke()`. `connections` is
    the same connection-config dict `MultiServerMCPClient` takes (e.g.
    `{"fs": {"transport": "stdio", "command": "npx", "args": [...]}}`)."""

    async def _fetch() -> list[BaseTool]:
        client = MultiServerMCPClient(connections)
        return await client.get_tools(server_name=server_name)

    return [_wrap_sync(t) for t in asyncio.run(_fetch())]


def get_single_tool(connections: dict[str, Any], tool_name: str) -> StructuredTool | None:
    """`get_mcp_tools` + pick one named tool, `None` on ANY failure (server
    not installed/reachable, or the server just doesn't have that tool) —
    the shape every lazily-cached single-tool MCP integration in this
    project needs (funnel.py's MCP fetch fallback, sources/github_mcp.py's
    `GitHubMCPSource`), extracted here instead of duplicated in both."""
    try:
        tools = get_mcp_tools(connections)
    except Exception:
        return None
    return next((t for t in tools if t.name == tool_name), None)


def content_to_text(result: Any) -> str:
    """MCP tool results come back as a list of LangChain content blocks
    (`{"type": "text", "text": ...}`, images, etc. — see
    `langchain_mcp_adapters.tools`), not a plain string. This pulls out and
    joins just the text blocks, which is all the ingest/funnel pipeline
    needs (§1 CLAUDE.md: no binary content held around)."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(
            block.get("text", "") for block in result if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(result)
