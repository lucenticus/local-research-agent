"""Юнит-тесты sources/citations.py — HTTP замокан (офлайн)."""

from __future__ import annotations

import io
import json
import urllib.error

from src.sources import citations


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return io.BytesIO(self._body)

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_lookup_returns_citation_count(monkeypatch):
    payload = {"results": [{"cited_by_count": 49, "title": "An Analysis of Fusion Functions"}]}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )
    assert citations.lookup_citation_count("An Analysis of Fusion Functions") == 49


def test_lookup_returns_none_when_no_results(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse({"results": []})
    )
    assert citations.lookup_citation_count("nonexistent paper title") is None


def test_lookup_returns_none_on_empty_title():
    assert citations.lookup_citation_count("   ") is None


def test_lookup_returns_none_on_network_error(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    assert citations.lookup_citation_count("some title") is None


def test_lookup_returns_none_when_count_field_missing(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeResponse({"results": [{"title": "x"}]}),
    )
    assert citations.lookup_citation_count("some title") is None


def test_lookup_rejects_mismatched_title_result(monkeypatch):
    """Регрессия на реальный баг (2026-08-06): на запрос про "risk of KV cache
    compression" OpenAlex топ-1 результатом отдал никак не связанную статью
    "XGBoost" (50k+ цитирований) — заголовок совсем другой, число неверное.
    """
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeResponse(
            {"results": [{"cited_by_count": 50424, "title": "XGBoost"}]}
        ),
    )
    assert citations.lookup_citation_count("The risk of KV cache compression") is None


def test_lookup_accepts_closely_matching_title(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: _FakeResponse(
            {"results": [{"cited_by_count": 1, "title": "LoRC: Low-Rank Compression for LLMs KV Cache"}]}
        ),
    )
    result = citations.lookup_citation_count(
        "LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy"
    )
    assert result == 1
