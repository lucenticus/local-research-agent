"""Юнит-тесты agent/loop.py.

test_loop_* — funnel.run, _is_covered и _draft_is_faithful замоканы (офлайн,
проверяем только логику контроллера: покрытие, дедуп по подвопросам,
расширение discovery_limit, остановку по бюджету). `_draft_is_faithful`
мокается на True в этих тестах — сама retry-логика по faithfulness
проверяется отдельно в test_loop_faithfulness_retry_*.

test_is_covered_* — сама gap-оценка (embed/store/rerank замоканы): порог
score + счёт различных источников (найдено реальным прогоном 2026-08-05, что
без порога score тематически смежный, но нерелевантный контент ложно
"закрывал" вопрос)."""

from __future__ import annotations

from src import config
from src.agent import loop
from src.agent.state import Budget, SubQuestion
from src.providers import embed, rerank


def test_loop_covers_in_one_pass_when_already_covered(monkeypatch):
    monkeypatch.setattr(loop, "_is_covered", lambda store, sq: True)
    monkeypatch.setattr(loop, "_draft_is_faithful", lambda question, store: True)
    monkeypatch.setattr(loop.funnel, "run", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("funnel.run не должен вызываться, если уже покрыто")
    ))

    state = loop.run(
        "question?", sources=[], store=object(),
        budget=Budget(max_iterations=5, max_deep_reads=100, max_seconds=None),
    )

    assert state.iterations == 1
    assert state.gaps == []
    assert state.open_sub_questions() == []


def test_loop_runs_multiple_iterations_until_covered(monkeypatch):
    calls = {"funnel_run": 0}

    def fake_is_covered(store, sq):
        return calls["funnel_run"] >= 2

    def fake_funnel_run(sq, sources, state, store, discovery_limit=None, on_progress=None):
        calls["funnel_run"] += 1

    monkeypatch.setattr(loop, "_is_covered", fake_is_covered)
    monkeypatch.setattr(loop, "_draft_is_faithful", lambda question, store: True)
    monkeypatch.setattr(loop.funnel, "run", fake_funnel_run)

    state = loop.run(
        "question?", sources=[], store=object(),
        budget=Budget(max_iterations=5, max_deep_reads=100, max_seconds=None),
    )

    assert calls["funnel_run"] == 2
    assert state.iterations == 2
    assert state.gaps == []


def test_loop_stops_by_budget_when_never_covered(monkeypatch):
    monkeypatch.setattr(loop, "_is_covered", lambda store, sq: False)
    monkeypatch.setattr(loop.funnel, "run", lambda *a, **kw: None)

    state = loop.run(
        "question?", sources=[], store=object(),
        budget=Budget(max_iterations=2, max_deep_reads=100, max_seconds=None),
    )

    assert state.iterations == 2
    assert state.gaps == ["question?"]


def test_loop_widens_discovery_limit_each_iteration(monkeypatch):
    seen_limits = []

    def fake_funnel_run(sq, sources, state, store, discovery_limit=None, on_progress=None):
        seen_limits.append(discovery_limit)

    monkeypatch.setattr(loop, "_is_covered", lambda store, sq: False)
    monkeypatch.setattr(loop.funnel, "run", fake_funnel_run)

    loop.run(
        "q?", sources=[], store=object(),
        budget=Budget(max_iterations=3, max_deep_reads=100, max_seconds=None),
    )

    assert seen_limits == [config.FUNNEL_DISCOVERY_LIMIT_PER_SOURCE * i for i in (1, 2, 3)]


def test_loop_only_calls_funnel_for_still_open_subquestions(monkeypatch):
    call_log = []

    def fake_is_covered(store, sq):
        return sq.text == "a?"  # "a?" покрыт сразу, "b?" не покрывается никогда

    def fake_funnel_run(sq, sources, state, store, discovery_limit=None, on_progress=None):
        call_log.append(sq.text)

    monkeypatch.setattr(loop, "_is_covered", fake_is_covered)
    monkeypatch.setattr(loop.funnel, "run", fake_funnel_run)

    state = loop.run(
        "a? b?", sources=[], store=object(),
        budget=Budget(max_iterations=2, max_deep_reads=100, max_seconds=None),
    )

    assert call_log == ["b?", "b?"]
    assert state.gaps == ["b?"]


class _FakeStore:
    def __init__(self, hits):
        self._hits = hits

    def search_hybrid(self, query_text, query_vector, k):
        return self._hits


def _hit(source_id, text="text"):
    return {"source_id": source_id, "text": text}


def test_is_covered_false_when_index_empty(monkeypatch):
    monkeypatch.setattr(embed, "embed_texts", lambda texts: [[0.0]])
    store = _FakeStore(hits=[])
    assert loop._is_covered(store, SubQuestion(text="q")) is False


def test_is_covered_false_when_hits_score_below_threshold(monkeypatch):
    monkeypatch.setattr(embed, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(config, "FUNNEL_MIN_SOURCES_TO_COVER", 2)
    monkeypatch.setattr(config, "FUNNEL_MIN_RERANK_SCORE", 0.5)
    hits = [_hit("a"), _hit("b")]
    # Оба источника разные, но score реранкера ниже порога - тематически
    # смежный, но нерелевантный контент не должен ложно "закрывать" вопрос.
    monkeypatch.setattr(rerank, "score", lambda q, cands: [(c, 0.1) for c in cands])
    store = _FakeStore(hits=hits)

    assert loop._is_covered(store, SubQuestion(text="q")) is False


def test_is_covered_true_when_enough_distinct_relevant_sources(monkeypatch):
    monkeypatch.setattr(embed, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(config, "FUNNEL_MIN_SOURCES_TO_COVER", 2)
    monkeypatch.setattr(config, "FUNNEL_MIN_RERANK_SCORE", 0.5)
    hits = [_hit("a"), _hit("b"), _hit("c")]
    monkeypatch.setattr(rerank, "score", lambda q, cands: [(c, 0.9) for c in cands])
    store = _FakeStore(hits=hits)

    assert loop._is_covered(store, SubQuestion(text="q")) is True


def test_is_covered_counts_only_distinct_sources_above_threshold(monkeypatch):
    monkeypatch.setattr(embed, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(config, "FUNNEL_MIN_SOURCES_TO_COVER", 2)
    monkeypatch.setattr(config, "FUNNEL_MIN_RERANK_SCORE", 0.5)
    # Один источник дважды (высокий score) + другой источник, но с низким score.
    hits = [_hit("a"), _hit("a"), _hit("b")]

    def fake_score(q, cands):
        return [(c, 0.9 if c["source_id"] == "a" else 0.1) for c in cands]

    monkeypatch.setattr(rerank, "score", fake_score)
    store = _FakeStore(hits=hits)

    assert loop._is_covered(store, SubQuestion(text="q")) is False


def test_loop_reopens_once_when_draft_is_not_faithful(monkeypatch):
    faithful_calls = []

    def fake_draft_is_faithful(question, store):
        faithful_calls.append(1)
        return False  # никогда не "честный" - должны увидеть ровно один retry

    monkeypatch.setattr(loop, "_is_covered", lambda store, sq: True)
    monkeypatch.setattr(loop, "_draft_is_faithful", fake_draft_is_faithful)
    monkeypatch.setattr(loop.funnel, "run", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("funnel.run не должен вызываться - подвопрос уже covered")
    ))

    state = loop.run(
        "question?", sources=[], store=object(),
        budget=Budget(max_iterations=10, max_deep_reads=100, max_seconds=None),
    )

    # Ровно один вызов _draft_is_faithful - после него retry использован, и
    # цикл завершается независимо от результата ВТОРОЙ проверки (её вообще
    # не будет - именно это здесь и проверяется).
    assert len(faithful_calls) == 1
    assert state.iterations == 2  # исходный проход + один retry-проход
    assert state.open_sub_questions() == []
    assert state.gaps == []


def test_loop_does_not_reopen_when_draft_is_already_faithful(monkeypatch):
    faithful_calls = []

    def fake_draft_is_faithful(question, store):
        faithful_calls.append(1)
        return True

    monkeypatch.setattr(loop, "_is_covered", lambda store, sq: True)
    monkeypatch.setattr(loop, "_draft_is_faithful", fake_draft_is_faithful)
    monkeypatch.setattr(loop.funnel, "run", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("funnel.run не должен вызываться - подвопрос уже covered")
    ))

    state = loop.run(
        "question?", sources=[], store=object(),
        budget=Budget(max_iterations=10, max_deep_reads=100, max_seconds=None),
    )

    assert len(faithful_calls) == 1
    assert state.iterations == 1  # без retry - сразу честный черновик


def test_loop_reports_progress_messages(monkeypatch):
    monkeypatch.setattr(loop, "_is_covered", lambda store, sq: True)
    monkeypatch.setattr(loop, "_draft_is_faithful", lambda question, store: True)
    monkeypatch.setattr(loop.funnel, "run", lambda *a, **kw: None)
    messages = []

    loop.run(
        "question?", sources=[], store=object(),
        budget=Budget(max_iterations=5, max_deep_reads=100, max_seconds=None),
        on_progress=messages.append,
    )

    assert any("Подвопросов" in m for m in messages)
    assert any("Синтезируем" in m for m in messages)


def test_loop_without_on_progress_does_not_raise(monkeypatch):
    monkeypatch.setattr(embed, "embed_texts", lambda texts: [[0.0]])
    loop.run(
        "question?", sources=[], store=_FakeStore(hits=[]),
        budget=Budget(max_iterations=1, max_deep_reads=1, max_seconds=None),
    )  # on_progress не передан - должен просто молчать
