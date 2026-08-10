"""Юнит-тесты sources/arxiv.py — HTTP замокан (офлайн). Раньше не было ни
одного теста на этот источник."""

from __future__ import annotations

import io
import urllib.request

from src.sources.arxiv import ArxivSource

_ATOM_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2508.12345v1</id>
    <title>Attention Is All You Need Again</title>
    <summary>We revisit attention mechanisms for transformers.</summary>
    <published>2026-08-01T12:00:00Z</published>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_discover_parses_entry_including_published_date(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse(_ATOM_TEMPLATE))

    items = ArxivSource().discover("attention", limit=5)
    assert len(items) == 1
    item = items[0]
    assert item.id == "arxiv:2508.12345v1"
    assert item.title == "Attention Is All You Need Again"
    assert item.abstract == "We revisit attention mechanisms for transformers."
    assert item.url == "https://arxiv.org/abs/2508.12345v1"
    assert item.year == 2026
    assert item.published_date == "2026-08-01T12:00:00Z"
    assert item.citation_count is None
    assert item.meta["pdf_url"] == "https://arxiv.org/pdf/2508.12345v1"
    assert item.source == "arxiv"


def test_discover_without_categories_sends_plain_keyword_query(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(_ATOM_TEMPLATE)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource().discover("attention transformers", limit=5)
    assert "search_query=all%3Aattention+AND+all%3Atransformers" in captured["url"]
    assert "cat%3A" not in captured["url"]


def test_discover_with_categories_ands_category_filter_into_query(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(_ATOM_TEMPLATE)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource(categories=["cs.AI", "cs.LG"]).discover("attention", limit=5)
    url = captured["url"]
    assert "cat%3Acs.AI+OR+cat%3Acs.LG" in url
    assert "all%3Aattention" in url


def test_discover_handles_missing_published_date():
    from src.sources.arxiv import ArxivSource as _AS

    body = _ATOM_TEMPLATE.replace(
        "<published>2026-08-01T12:00:00Z</published>", ""
    ).encode("utf-8")
    items = list(_AS()._parse(body))
    assert items[0].year is None
    assert items[0].published_date is None
