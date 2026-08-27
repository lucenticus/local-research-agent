"""Юнит-тесты sources/semantic_scholar.py — HTTP замокан (офлайн)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from src.sources import semantic_scholar as s2
from src.sources._common import SourceUnavailable


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return io.BytesIO(self._body)

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _urlopen_returning(payload, seen: list | None = None):
    def _urlopen(request, timeout=None):
        if seen is not None:
            seen.append(json.loads(request.data))
        return _FakeResponse(payload)

    return _urlopen


def test_batch_maps_counts_back_to_the_ids_it_was_given(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _urlopen_returning([{"citationCount": 190309}, {"citationCount": 17586}]),
    )
    assert s2.lookup_citation_counts(["1706.03762", "2005.11401"]) == {
        "1706.03762": 190309,
        "2005.11401": 17586,
    }


def test_batch_strips_the_arxiv_version_suffix(monkeypatch):
    """arXiv отдаёт id с версией (`2203.16487v6`), S2 такие не принимает —
    `{"error": "No valid paper ids given"}`, проверено вживую. Ключ при этом
    возвращается в том виде, в каком пришёл: сопоставлять с кандидатами —
    задача вызывающего, и про нормализацию он знать не должен."""
    seen: list = []
    monkeypatch.setattr(
        "urllib.request.urlopen", _urlopen_returning([{"citationCount": 5}], seen)
    )

    assert s2.lookup_citation_counts(["2203.16487v6"]) == {"2203.16487v6": 5}
    assert seen == [{"ids": ["ARXIV:2203.16487"]}]


def test_unknown_paper_is_absent_not_zero(monkeypatch):
    """S2 отдаёт null на месте статьи, которой не знает. Записать это нулём
    значило бы утверждать «на неё никто не ссылается» — свежий препринт
    получил бы отрицательный сигнал вместо отсутствия сигнала."""
    monkeypatch.setattr(
        "urllib.request.urlopen", _urlopen_returning([{"citationCount": 7}, None])
    )
    assert s2.lookup_citation_counts(["1706.03762", "2605.00001"]) == {"1706.03762": 7}


def test_no_ids_means_no_request(monkeypatch):
    def _fail(request, timeout=None):
        raise AssertionError("пустой список не должен порождать запрос")

    monkeypatch.setattr("urllib.request.urlopen", _fail)
    assert s2.lookup_citation_counts([]) == {}


def test_ids_beyond_the_batch_limit_are_paged_not_dropped(monkeypatch):
    """Лимит S2 — 500 id на запрос. Молча потерять хвост списка хуже, чем
    сделать два вызова."""
    monkeypatch.setattr(s2, "_BATCH_MAX_IDS", 2)
    pages: list = []

    def _urlopen(request, timeout=None):
        ids = json.loads(request.data)["ids"]
        pages.append(ids)
        return _FakeResponse([{"citationCount": 1} for _ in ids])

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    assert s2.lookup_citation_counts(["a", "b", "c"]) == {"a": 1, "b": 1, "c": 1}
    assert pages == [["ARXIV:a", "ARXIV:b"], ["ARXIV:c"]]


def test_rate_limit_raises_instead_of_looking_like_no_data(monkeypatch):
    """429 — не «у статей нет цитирований». Молчаливый None здесь попал бы в
    фикстуры как факт и заморозил аварию навсегда."""
    def _429(request, timeout=None):
        raise urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _429)
    monkeypatch.setattr(s2, "_BACKOFF_BASE_SECONDS", 0)
    with pytest.raises(SourceUnavailable):
        s2.lookup_citation_counts(["1706.03762"])


def test_error_object_instead_of_a_list_is_not_silently_accepted(monkeypatch):
    """На невалидный запрос S2 отвечает объектом с "error", а не списком —
    распарсить его как ответ значило бы принять ошибку за пустую выдачу."""
    monkeypatch.setattr(
        "urllib.request.urlopen", _urlopen_returning({"error": "No valid paper ids given"})
    )
    with pytest.raises(SourceUnavailable):
        s2.lookup_citation_counts(["bad"])


def test_second_call_for_the_same_paper_makes_no_request(monkeypatch):
    """Ради этого кэш и заводился: популярные статьи находятся снова и снова
    на каждом прогоне, и платить за них запросом каждый раз незачем."""
    calls: list = []
    monkeypatch.setattr(
        "urllib.request.urlopen", _urlopen_returning([{"citationCount": 42}], calls)
    )
    assert s2.lookup_citation_counts(["1706.03762"]) == {"1706.03762": 42}
    assert s2.lookup_citation_counts(["1706.03762"]) == {"1706.03762": 42}
    assert len(calls) == 1


def test_only_the_uncached_ids_are_requested(monkeypatch):
    seen: list = []
    monkeypatch.setattr("urllib.request.urlopen", _urlopen_returning([{"citationCount": 1}], seen))
    s2.lookup_citation_counts(["a"])

    monkeypatch.setattr("urllib.request.urlopen", _urlopen_returning([{"citationCount": 2}], seen))
    assert s2.lookup_citation_counts(["a", "b"]) == {"a": 1, "b": 2}
    assert seen[-1] == {"ids": ["ARXIV:b"]}, "спрашивать надо только недостающее"
