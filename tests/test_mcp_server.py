"""Юнит-тесты src/mcp_server.py — retrieve/synthesize/run_research замокан
(офлайн, без реальных моделей/MCP-транспорта). Реальный stdio round-trip
через `mcp-serve` проверен вручную (см. README)."""

from __future__ import annotations

from src import mcp_server
from src.agent.research_runner import ResearchResult, SourceRef


def test_ask_returns_answer_with_sources(monkeypatch):
    monkeypatch.setattr(mcp_server, "retrieve", lambda store, question: [
        {"text": "whales are mammals", "source_title": "whales.md", "url": "", "citation_count": -1},
    ])
    monkeypatch.setattr(mcp_server, "synthesize", lambda question, hits: "A whale is a mammal [1].")
    monkeypatch.setattr(mcp_server, "QdrantStore", lambda: object())

    result = mcp_server.ask("What is a whale?")
    assert "A whale is a mammal [1]." in result
    assert "[1] whales.md" in result


def test_ask_without_hits_returns_bare_answer(monkeypatch):
    monkeypatch.setattr(mcp_server, "retrieve", lambda store, question: [])
    monkeypatch.setattr(mcp_server, "synthesize", lambda question, hits: "No information found.")
    monkeypatch.setattr(mcp_server, "QdrantStore", lambda: object())

    assert mcp_server.ask("q") == "No information found."


def test_research_formats_answer_sources_and_gaps(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "run_research",
        lambda question, store: ResearchResult(
            answer="KV-cache stores key/value tensors [1].",
            sources=[SourceRef(title="KV Cache Explained", url="https://x/y", citation_count=42)],
            candidates=[], gaps=["how does eviction work?"], iterations=2, read_count=3, candidates_count=5,
        ),
    )
    monkeypatch.setattr(mcp_server, "QdrantStore", lambda collection_name=None: object())

    result = mcp_server.research("What is a KV cache?")
    assert "KV-cache stores key/value tensors [1]." in result
    assert "[1] KV Cache Explained (citations: 42)" in result
    assert "https://x/y" in result
    assert "Uncovered subquestions" in result
    assert "how does eviction work?" in result


def test_research_without_gaps_omits_gaps_section(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "run_research",
        lambda question, store: ResearchResult(
            answer="ok", sources=[], gaps=[], candidates=[], iterations=1, read_count=1, candidates_count=1,
        ),
    )
    monkeypatch.setattr(mcp_server, "QdrantStore", lambda collection_name=None: object())

    result = mcp_server.research("q")
    assert "Uncovered subquestions" not in result
