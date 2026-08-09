"""Юнит-тесты sources/github_mcp.py — get_mcp_tools замокан офлайн (без
Docker/реального GitHub MCP-сервера)."""

from __future__ import annotations

import json

from src.providers import mcp_client
from src.sources import github_mcp
from src.sources.github_mcp import GitHubMCPSource


class _FakeTool:
    def __init__(self, text):
        self._text = text
        self.name = "search_repositories"
        self.calls = []

    def invoke(self, kwargs):
        self.calls.append(kwargs)
        return [{"type": "text", "text": self._text}]


def _repos_payload(items):
    return json.dumps({"total_count": len(items), "items": items})


def test_discover_returns_empty_list_without_token(monkeypatch):
    monkeypatch.setattr(
        mcp_client, "get_mcp_tools",
        lambda connections: (_ for _ in ()).throw(AssertionError("must not reach Docker without a token")),
    )
    assert GitHubMCPSource(token=None).discover("query", limit=5) == []


def test_discover_parses_repositories(monkeypatch):
    payload = _repos_payload(
        [
            {
                "full_name": "NVIDIA/kvpress", "html_url": "https://github.com/NVIDIA/kvpress",
                "description": "LLM KV cache compression made easy",
                "topics": ["kv-cache", "llm"], "stargazers_count": 1157,
            },
        ]
    )
    fake_tool = _FakeTool(payload)
    monkeypatch.setattr(mcp_client, "get_mcp_tools", lambda connections: [fake_tool])

    items = GitHubMCPSource(token="fake-token").discover("kv cache compression", limit=5)
    assert len(items) == 1
    item = items[0]
    assert item.title == "NVIDIA/kvpress"
    assert item.url == "https://github.com/NVIDIA/kvpress"
    assert item.id == "github:NVIDIA/kvpress"
    assert item.citation_count == 1157
    assert "LLM KV cache compression made easy" in item.abstract
    assert "kv-cache" in item.abstract
    assert item.source == "github"
    assert fake_tool.calls == [{"query": "kv cache compression", "perPage": 5}]


def test_discover_skips_items_without_full_name_or_url(monkeypatch):
    payload = _repos_payload([{"description": "no full_name/url"}])
    monkeypatch.setattr(mcp_client, "get_mcp_tools", lambda connections: [_FakeTool(payload)])

    assert GitHubMCPSource(token="fake-token").discover("q", limit=5) == []


def test_discover_returns_empty_list_on_invalid_json(monkeypatch):
    monkeypatch.setattr(mcp_client, "get_mcp_tools", lambda connections: [_FakeTool("not json")])
    assert GitHubMCPSource(token="fake-token").discover("q", limit=5) == []


def test_discover_returns_empty_list_when_get_mcp_tools_fails(monkeypatch):
    def _raise(connections):
        raise RuntimeError("docker not available")

    monkeypatch.setattr(mcp_client, "get_mcp_tools", _raise)
    assert GitHubMCPSource(token="fake-token").discover("q", limit=5) == []


def test_get_tool_only_calls_get_mcp_tools_once_per_instance(monkeypatch):
    calls = {"n": 0}

    def fake_get_mcp_tools(connections):
        calls["n"] += 1
        return [_FakeTool(_repos_payload([]))]

    monkeypatch.setattr(mcp_client, "get_mcp_tools", fake_get_mcp_tools)
    source = GitHubMCPSource(token="fake-token")

    source.discover("a", limit=1)
    source.discover("b", limit=1)
    assert calls["n"] == 1
