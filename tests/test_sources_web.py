"""Юнит-тесты sources/web.py — HTTP замокан (офлайн, не требует поднятого
локального SearXNG)."""

from __future__ import annotations

import io
import json
import urllib.error

from src.sources.web import WebSource


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return io.BytesIO(self._body)

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


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

    items = WebSource(base_url="http://localhost:8888").discover("cats", limit=5)
    assert [i.title for i in items] == ["A", "B"]
    assert items[0].abstract == "about cats"
    assert items[0].url == "https://example.com/a"
    assert items[0].id == "web:https://example.com/a"
    assert items[0].source == "web"


def test_discover_respects_limit(monkeypatch):
    payload = {"results": [{"title": f"T{i}", "content": "x", "url": f"https://x/{i}"} for i in range(5)]}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )

    items = WebSource(base_url="http://localhost:8888").discover("q", limit=2)
    assert len(items) == 2


def test_discover_skips_results_without_url(monkeypatch):
    payload = {"results": [{"title": "no url", "content": "x", "url": ""}]}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )

    items = WebSource(base_url="http://localhost:8888").discover("q", limit=5)
    assert items == []


def test_discover_returns_empty_list_when_searxng_is_down(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    items = WebSource(base_url="http://localhost:8888").discover("q", limit=5)
    assert items == []


def test_uses_configured_base_url_by_default(monkeypatch):
    from src import config

    monkeypatch.setattr(config, "SEARXNG_BASE_URL", "http://example-searxng:1234")
    source = WebSource()
    assert source._base_url == "http://example-searxng:1234"


def test_discover_filters_out_arxiv_listing_pages(monkeypatch):
    """Регрессия на реальный баг (2026-08-06): arxiv.org/list/... — страница-
    листинг (список последних публикаций), а не отдельная статья, но
    проходила как полноценный источник в реальном ответе."""
    payload = {
        "results": [
            {"title": "Artificial Intelligence - arXiv", "content": "recent papers",
             "url": "https://arxiv.org/list/cs.AI/recent"},
            {"title": "Real Paper", "content": "abstract",
             "url": "https://arxiv.org/abs/2508.11957"},
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )

    items = WebSource(base_url="http://localhost:8888").discover("q", limit=5)
    assert [i.title for i in items] == ["Real Paper"]
