"""Юнит-тесты sources/wikipedia.py — HTTP замокан (офлайн)."""

from __future__ import annotations

import io
import json
import urllib.error

from src.sources.wikipedia import WikipediaSource


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
        "query": {
            "pages": {
                "1": {
                    "pageid": 1, "title": "Cats",
                    "fullurl": "https://en.wikipedia.org/wiki/Cats",
                    "extract": "The cat is a domestic species.",
                },
                "2": {
                    "pageid": 2, "title": "Dogs",
                    "fullurl": "https://en.wikipedia.org/wiki/Dogs",
                    "extract": "The dog is a domesticated canid.",
                },
            }
        }
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload))

    items = WikipediaSource().discover("cats and dogs", limit=5)
    assert {i.title for i in items} == {"Cats", "Dogs"}
    cats = next(i for i in items if i.title == "Cats")
    assert cats.abstract == "The cat is a domestic species."
    assert cats.url == "https://en.wikipedia.org/wiki/Cats"
    assert cats.id == "wikipedia:1"
    assert cats.source == "wikipedia"


def test_discover_skips_pages_without_title_or_url(monkeypatch):
    payload = {"query": {"pages": {"1": {"pageid": 1, "extract": "no title/url"}}}}
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload))

    assert WikipediaSource().discover("q", limit=5) == []


def test_discover_handles_missing_pages_key(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse({"query": {}}))
    assert WikipediaSource().discover("q", limit=5) == []


def test_discover_returns_empty_list_on_network_error(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    assert WikipediaSource().discover("q", limit=5) == []
