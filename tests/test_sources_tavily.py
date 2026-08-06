"""Юнит-тесты sources/tavily.py — HTTP замокан (офлайн, без реального ключа)."""

from __future__ import annotations

import io
import json
import urllib.error

from src.sources.tavily import TavilySource


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return io.BytesIO(self._body)

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_discover_returns_empty_list_without_api_key(monkeypatch):
    # config.py грузит .env при импорте - на этой машине там реальный ключ,
    # так что явный delenv обязателен, иначе тест тихо подхватит его из
    # окружения и сделает настоящий сетевой запрос вместо проверки short-circuit.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    source = TavilySource(api_key=None)
    assert source.discover("query", limit=5) == []


def test_discover_parses_results(monkeypatch):
    payload = {
        "results": [
            {"title": "A", "content": "about cats", "url": "https://example.com/a"},
            {"title": "B", "content": "about dogs", "url": "https://example.com/b"},
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )

    items = TavilySource(api_key="fake-key").discover("cats", limit=5)
    assert [i.title for i in items] == ["A", "B"]
    assert items[0].abstract == "about cats"
    assert items[0].url == "https://example.com/a"
    assert items[0].source == "web"


def test_discover_filters_out_arxiv_listing_pages(monkeypatch):
    payload = {
        "results": [
            {"title": "Artificial Intelligence - arXiv", "content": "recent papers",
             "url": "https://arxiv.org/list/cs.AI/recent"},
            {"title": "Real Paper", "content": "abstract", "url": "https://arxiv.org/abs/2508.11957"},
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )

    items = TavilySource(api_key="fake-key").discover("q", limit=5)
    assert [i.title for i in items] == ["Real Paper"]


def test_discover_returns_empty_list_on_network_error(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    items = TavilySource(api_key="fake-key").discover("q", limit=5)
    assert items == []


def test_uses_env_var_by_default(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    source = TavilySource()
    assert source._api_key == "from-env"
