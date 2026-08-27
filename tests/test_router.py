"""Юнит-тесты agent/router.py — эмбеддер, реранкер и хранилище замоканы."""

from __future__ import annotations

import pytest

from src.agent import router
from src.agent.router import ROUTE_ASK, ROUTE_CLARIFY, ROUTE_RESEARCH, route


class _FakeStore:
    def __init__(self, hits=None, raises=False):
        self._hits = hits if hits is not None else [{"text": "chunk"}]
        self._raises = raises

    def search_hybrid(self, query_text, query_vector, k):
        if self._raises:
            raise RuntimeError("Qdrant недоступен")
        return self._hits


@pytest.fixture
def probe(monkeypatch):
    """Подменяет эмбеддер и реранкер; возвращает сеттер лучшего скора."""
    monkeypatch.setattr(router.embed, "embed_texts", lambda texts: [[0.1, 0.2]])

    def _set(best: float):
        monkeypatch.setattr(router.rerank, "score",
                            lambda query, candidates: [(c, best) for c in candidates])

    return _set


@pytest.mark.parametrize("query", ["question?", "hello", "что это", "?", "   "])
def test_contentless_queries_ask_for_clarification(query):
    """Ровно тот вход, на котором модель не отказывается, а сочиняет."""
    assert route(query).route == ROUTE_CLARIFY


def test_clarify_reason_says_how_many_content_words_were_found():
    assert "0 содержательных слов" in route("hello?").reason


def test_substantive_query_without_store_goes_to_research():
    r = route("What approaches exist for KV-cache compression?")
    assert r.route == ROUTE_RESEARCH


def test_well_covered_query_goes_to_ask(probe):
    probe(0.95)
    r = route("Что такое Retrieval-Augmented Generation?", _FakeStore())
    assert r.route == ROUTE_ASK
    assert r.evidence_score == 0.95


def test_poorly_covered_query_goes_to_research(probe):
    probe(0.02)
    r = route("What approaches exist for KV-cache compression?", _FakeStore())
    assert r.route == ROUTE_RESEARCH
    assert r.evidence_score == 0.02


def test_threshold_is_the_configured_one(monkeypatch, probe):
    monkeypatch.setattr(router.config, "ROUTER_ASK_MIN_SCORE", 0.5)
    probe(0.6)
    assert route("Что такое Retrieval-Augmented Generation?", _FakeStore()).route == ROUTE_ASK


def test_empty_index_falls_back_to_research(probe):
    probe(0.99)
    r = route("Что такое Retrieval-Augmented Generation?", _FakeStore(hits=[]))
    assert r.route == ROUTE_RESEARCH
    assert r.evidence_score is None


def test_unavailable_store_falls_back_to_research_not_an_error(probe):
    """Отсутствие локального корпуса — штатная ситуация, а не сбой."""
    probe(0.99)
    r = route("Что такое Retrieval-Augmented Generation?", _FakeStore(raises=True))
    assert r.route == ROUTE_RESEARCH


def test_probe_can_be_disabled(monkeypatch):
    monkeypatch.setattr(router.config, "ROUTER_PROBE_LOCAL_INDEX", False)

    def _fail(*a, **kw):
        raise AssertionError("проба выключена — эмбеддер звать не должны")

    monkeypatch.setattr(router.embed, "embed_texts", _fail)
    assert route("Что такое Retrieval-Augmented Generation?", _FakeStore()).route == ROUTE_RESEARCH


def test_default_threshold_is_stricter_than_the_general_one():
    """Ошибка в сторону ask отдаёт плохой ответ молча, ошибка в сторону
    research — всего лишь дольше. Порог должен быть асимметричным."""
    assert router.config.ROUTER_ASK_MIN_SCORE > router.config.FUNNEL_MIN_RERANK_SCORE


def test_content_words_drops_filler_in_both_languages():
    assert router.content_words("What is the RAG?") == ["rag"]
    assert router.content_words("Что такое RAG?") == ["rag"]
