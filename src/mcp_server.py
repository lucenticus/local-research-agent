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
from .agent.funnel import discover_candidates
from .agent.research_runner import SourceRef, retrieve, run_research, unique_sources
from .agent.synthesize import synthesize
from .providers import tracing
from .sources.arxiv import ArxivSource
from .sources.semantic_scholar import SemanticScholarSource
from .store.qdrant_store import QdrantStore

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
    store = QdrantStore()
    hits = retrieve(store, question)
    answer = synthesize(question, hits)
    sources = unique_sources(hits)
    return f"{answer}\n\nSources:\n{_format_sources(sources)}" if sources else answer


def _paper_sources() -> list:
    # Paper-only, deliberately narrower than default_sources(): this tool's
    # whole point is "candidate papers to reproduce", so a general web
    # source (blog posts, Nature news articles) would outrank the actual
    # arXiv preprint on relevance and crowd out the one result a caller can
    # act on — confirmed by a real run on "parameter-efficient fine-tuning
    # for LLMs" where all 3 top web hits were blog/journal write-ups, not a
    # single arXiv paper.
    return [ArxivSource(categories=config.ARXIV_AI_CATEGORIES), SemanticScholarSource()]


@mcp.tool()
def find_papers(topic: str, top_n: int = 5) -> str:
    """Find candidate arXiv/Semantic Scholar papers for a topic — discovery
    + relevance/citation/recency triage over arXiv and Semantic Scholar
    only (no general web search — see `_paper_sources`), no deep read, no
    synthesis. Returns a ranked shortlist (title, paper URL, citation
    count, id) so a caller can pick which paper to look at deeply next,
    without paying for a full `research` call on every candidate."""
    candidates = discover_candidates(topic, _paper_sources(), top_n)
    if not candidates:
        return "No candidates found."
    lines = []
    for i, c in enumerate(candidates, start=1):
        citations = c.meta.get("citation_count")
        year = c.meta.get("year")
        suffix = f" (citations: {citations})" if citations is not None else ""
        suffix += f", {year}" if year else ""
        lines.append(f"[{i}] {c.title}{suffix}\n    id: {c.id}\n    {c.meta.get('url') or ''}")
    return "\n".join(lines)


@mcp.tool()
def research(question: str) -> str:
    """Deep research: searches arXiv, Semantic Scholar, CrossRef,
    Wikipedia, and the web, reads what it finds, and returns a cited
    answer with self-verification. Can take a couple of minutes — it's a
    real multi-source search, not a single retrieval call."""
    store = QdrantStore(collection_name=config.QDRANT_RESEARCH_COLLECTION)
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
