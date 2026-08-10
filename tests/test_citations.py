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


def _dispatch_by_url(responses: dict[str, dict]):
    """Роутит fake urlopen по подстроке в URL — works-запрос и authors-запрос
    идут на разные пути, оба через один и тот же _common.fetch_json."""

    def _urlopen(request, timeout=None):
        for marker, payload in responses.items():
            if marker in request.full_url:
                return _FakeResponse(payload)
        raise AssertionError(f"unexpected URL: {request.full_url}")

    return _urlopen


def test_lookup_paper_details_returns_none_when_work_not_found(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", _dispatch_by_url({"api.openalex.org/works": {"results": []}})
    )
    assert citations.lookup_paper_details("some very fresh preprint") is None


def test_lookup_paper_details_returns_citations_venue_and_authors(monkeypatch):
    work_payload = {
        "results": [
            {
                "title": "Attention Is All You Need",
                "cited_by_count": 6602,
                "primary_location": {"source": {"display_name": "NeurIPS"}},
                "authorships": [
                    {
                        "author": {"display_name": "Ashish Vaswani", "id": "https://openalex.org/A1"},
                        "institutions": [{"display_name": "Google"}],
                    },
                    {
                        "author": {"display_name": "Noam Shazeer", "id": "https://openalex.org/A2"},
                        "institutions": [],
                    },
                ],
            }
        ]
    }
    authors_payload = {
        "results": [
            {"id": "https://openalex.org/A1", "summary_stats": {"h_index": 29}},
            {"id": "https://openalex.org/A2", "summary_stats": {"h_index": 34}},
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _dispatch_by_url(
            {"api.openalex.org/works": work_payload, "api.openalex.org/authors": authors_payload}
        ),
    )

    result = citations.lookup_paper_details("Attention Is All You Need")
    assert result.citation_count == 6602
    assert result.venue == "NeurIPS"
    assert result.authors == [
        citations.AuthorDetails(name="Ashish Vaswani", institution="Google", h_index=29),
        citations.AuthorDetails(name="Noam Shazeer", institution=None, h_index=34),
    ]


def test_lookup_paper_details_handles_missing_venue_and_h_index(monkeypatch):
    work_payload = {
        "results": [
            {
                "title": "Some Paper",
                "cited_by_count": 3,
                "authorships": [
                    {"author": {"display_name": "A. Uthor", "id": "https://openalex.org/A9"}, "institutions": []}
                ],
            }
        ]
    }
    authors_payload = {"results": []}  # автор ещё не найден в OpenAlex Authors
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _dispatch_by_url(
            {"api.openalex.org/works": work_payload, "api.openalex.org/authors": authors_payload}
        ),
    )

    result = citations.lookup_paper_details("Some Paper")
    assert result.venue is None
    assert result.authors == [citations.AuthorDetails(name="A. Uthor", institution=None, h_index=None)]


def test_lookup_paper_details_skips_author_batch_call_when_no_author_ids(monkeypatch):
    work_payload = {
        "results": [
            {"title": "Anon Paper", "cited_by_count": 0, "authorships": [{"author": {"display_name": "Anon"}}]}
        ]
    }

    def _urlopen(request, timeout=None):
        if "api.openalex.org/authors" in request.full_url:
            raise AssertionError("must not call authors endpoint when no author has an id")
        return _FakeResponse(work_payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    result = citations.lookup_paper_details("Anon Paper")
    assert result.authors == [citations.AuthorDetails(name="Anon", institution=None, h_index=None)]
