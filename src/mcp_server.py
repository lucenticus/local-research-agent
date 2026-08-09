"""Exposes this agent's own `ask`/`research` as MCP tools over stdio — a
third frontend alongside `cli.py` and `web/app.py`, calling the exact same
`agent/research_runner.py` functions those two already use (no separate
reimplementation of retrieval/funnel/synthesis).

Run with `python -m src.cli mcp-serve`; point any MCP client (Claude Code,
Claude Desktop, etc.) at that command over stdio to let it call this agent
as a tool.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import config
from .agent.research_runner import SourceRef, retrieve, run_research, unique_sources
from .agent.synthesize import synthesize
from .providers import tracing
from .store.lancedb_store import LanceDBStore

mcp = FastMCP("local-research-agent")


def _format_sources(sources: list[SourceRef]) -> str:
    lines = []
    for i, source in enumerate(sources, start=1):
        line = f"[{i}] {source.title}"
        if source.citation_count is not None:
            line += f" (citations: {source.citation_count})"
        if source.url:
            line += f"\n    {source.url}"
        lines.append(line)
    return "\n".join(lines)


@mcp.tool()
def ask(question: str) -> str:
    """Answer a question from the locally indexed corpus (build it first
    with the `index` CLI command) — no internet access, grounded only in
    what's already indexed."""
    store = LanceDBStore()
    hits = retrieve(store, question)
    answer = synthesize(question, hits)
    sources = unique_sources(hits)
    return f"{answer}\n\nSources:\n{_format_sources(sources)}" if sources else answer


@mcp.tool()
def research(question: str) -> str:
    """Deep research: searches arXiv, Semantic Scholar, CrossRef,
    Wikipedia, and the web, reads what it finds, and returns a cited
    answer with self-verification. Can take a couple of minutes — it's a
    real multi-source search, not a single retrieval call."""
    store = LanceDBStore(table_name=config.RESEARCH_INDEX_TABLE)
    result = run_research(question, store)
    parts = [result.answer, "", "Sources:", _format_sources(result.sources)]
    if result.gaps:
        parts.append("")
        parts.append("Uncovered subquestions (research budget exhausted):")
        parts.extend(f"- {gap}" for gap in result.gaps)
    return "\n".join(parts)


def serve() -> None:
    tracing.enable_if_configured()
    mcp.run(transport="stdio")
