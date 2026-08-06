"""Юнит-тесты sources/crossref.py — HTTP замокан (офлайн)."""

from __future__ import annotations

import io
import json
import urllib.error

from src.sources.crossref import CrossrefSource


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return io.BytesIO(self._body)

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _payload(items):
    return {"status": "ok", "message": {"items": items}}


def test_discover_parses_results_with_abstract(monkeypatch):
    payload = _payload(
        [
            {
                "DOI": "10.1/abc",
                "title": ["A Paper"],
                "abstract": "<title>Abstract</title><p>About cats.</p>",
                "URL": "https://doi.org/10.1/abc",
                "is-referenced-by-count": 12,
                "published": {"date-parts": [[2023, 5]]},
            }
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload))

    items = CrossrefSource().discover("cats", limit=5)
    assert len(items) == 1
    item = items[0]
    assert item.title == "A Paper"
    assert item.abstract == "About cats."
    assert item.url == "https://doi.org/10.1/abc"
    assert item.id == "doi:10.1/abc"
    assert item.citation_count == 12
    assert item.year == 2023
    assert item.source == "crossref"


def test_discover_falls_back_to_empty_abstract_when_missing(monkeypatch):
    payload = _payload(
        [{"DOI": "10.1/xyz", "title": ["No Abstract Paper"], "URL": "https://doi.org/10.1/xyz"}]
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload))

    items = CrossrefSource().discover("q", limit=5)
    assert items[0].abstract == ""
    assert items[0].citation_count is None
    assert items[0].year is None


def test_discover_skips_items_without_doi_or_title(monkeypatch):
    payload = _payload([{"title": ["No DOI"]}, {"DOI": "10.1/x", "title": []}])
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload))

    assert CrossrefSource().discover("q", limit=5) == []


def test_discover_returns_empty_list_on_network_error(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    assert CrossrefSource().discover("q", limit=5) == []
