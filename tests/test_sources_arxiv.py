"""Юнит-тесты sources/arxiv.py — HTTP замокан (офлайн). Раньше не было ни
одного теста на этот источник."""

from __future__ import annotations

import io
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

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


def test_parse_extracts_authors_and_categories():
    body = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2508.99999v1</id>
    <title>Some Paper</title>
    <summary>An abstract.</summary>
    <published>2026-08-01T12:00:00Z</published>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <author><name>Alice Example</name></author>
    <author><name>Bob Example</name></author>
  </entry>
</feed>
""".encode("utf-8")
    item = list(ArxivSource()._parse(body))[0]
    assert item.meta["authors"] == ["Alice Example", "Bob Example"]
    assert item.meta["categories"] == ["cs.CL", "cs.AI"]


def _atom_with_published(*dates: str) -> str:
    entries = "".join(
        f"""<entry>
    <id>http://arxiv.org/abs/2508.{i:05d}v1</id>
    <title>Paper {i}</title>
    <summary>Abstract {i}.</summary>
    <published>{date}</published>
  </entry>"""
        for i, date in enumerate(dates)
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'


def test_recent_requires_categories():
    with pytest.raises(ValueError):
        ArxivSource().recent(days=7, limit=10)


def test_recent_sorts_by_submitted_date_not_relevance(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(_atom_with_published(datetime.now(timezone.utc).isoformat()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10)
    assert "sortBy=submittedDate" in captured["url"]
    assert "sortOrder=descending" in captured["url"]
    assert "all%3A" not in captured["url"]  # без keyword-фильтра — только категории
    assert "cat%3Acs.AI" in captured["url"]


def test_recent_filters_out_items_older_than_days(monkeypatch):
    fresh = datetime.now(timezone.utc)
    old = fresh - timedelta(days=30)
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda request, timeout=None: _FakeResponse(_atom_with_published(fresh.isoformat(), old.isoformat())),
    )

    items = ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10)
    assert len(items) == 1
    assert items[0].title == "Paper 0"


def test_recent_with_query_ands_keyword_but_still_sorts_by_date(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(_atom_with_published(datetime.now(timezone.utc).isoformat()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10, query="diffusion models")
    url = captured["url"]
    assert "sortBy=submittedDate" in url
    assert "cat%3Acs.AI" in url
    assert "all%3Adiffusion+AND+all%3Amodels" in url


def test_recent_without_query_omits_keyword_clause(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(_atom_with_published(datetime.now(timezone.utc).isoformat()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10, query="   ")
    assert "all%3A" not in captured["url"]
