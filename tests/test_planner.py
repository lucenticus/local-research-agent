"""Юнит-тесты agent/planner.py — LLM замокан (§7 CLAUDE.md: тесты офлайн и быстрые).

`plan()` делает bounded LLM-вызов для разбора вопроса на аспекты, поэтому мок
обязателен во ВСЕХ тестах: без него юнит-тест грузит 4B-модель.
"""

from __future__ import annotations

import pytest

from src.agent import planner
from src.agent.planner import plan


@pytest.fixture
def no_llm(monkeypatch):
    """LLM недоступна: любой вызов — ошибка теста."""
    def _fail(*a, **kw):
        raise AssertionError("LLM звать здесь не должны")

    monkeypatch.setattr(planner.llm, "build_chat_prompt", _fail)
    monkeypatch.setattr(planner.llm, "generate", _fail)


@pytest.fixture
def fake_llm(monkeypatch):
    """Подменяет ответ модели строкой и включает декомпозицию.

    По умолчанию она выключена (замер показал, что она ухудшает покрытие
    подтем — см. config.PLANNER_LLM_DECOMPOSE), но сам механизм должен
    оставаться проверенным: это отрицательный результат, а не мёртвый код.
    """
    def _set(raw: str):
        monkeypatch.setattr(planner.config, "PLANNER_LLM_DECOMPOSE", True)
        monkeypatch.setattr(planner.llm, "build_chat_prompt", lambda system, user: user)
        monkeypatch.setattr(planner.llm, "generate", lambda prompt, **kw: raw)

    return _set


def test_empty_question_returns_no_subquestions(no_llm):
    assert plan("") == []
    assert plan("   ") == []


def test_compound_question_splits_deterministically_without_the_llm(no_llm):
    """Явные разделители — предметное знание не нужно, модель не зовём."""
    result = plan("Что такое RAG? Зачем нужен hybrid retrieval?")
    assert [sq.text for sq in result] == ["Что такое RAG?", "Зачем нужен hybrid retrieval?"]


def test_splits_on_a_takzhe_separator(no_llm):
    result = plan("Что такое BM25, а также как работает RRF?")
    # каждый подвопрос гарантированно заканчивается "?", даже если в исходном
    # тексте разделитель его "съел"
    assert [sq.text for sq in result] == ["Что такое BM25?", "как работает RRF?"]


def test_single_question_stays_intact_when_decomposition_is_off(monkeypatch, no_llm):
    monkeypatch.setattr(planner.config, "PLANNER_LLM_DECOMPOSE", False)
    assert [sq.text for sq in plan("Что такое RAG?")] == ["Что такое RAG?"]


def test_broad_question_is_split_into_facets(fake_llm):
    """Ради чего всё затевалось: покрытие подтем было 0.42, потому что
    многогранный вопрос оставался одним подвопросом."""
    fake_llm("How does KV-cache quantization work?\n"
             "Which token eviction policies exist?\n"
             "How is attention state shared across layers?")
    result = plan("What approaches exist for KV-cache compression?")
    assert [sq.text for sq in result] == [
        "How does KV-cache quantization work?",
        "Which token eviction policies exist?",
        "How is attention state shared across layers?",
    ]


def test_bullets_and_numbering_are_stripped(fake_llm):
    fake_llm("1. How does quantization work?\n- What about eviction policies?\n• And layer sharing?")
    assert [sq.text for sq in plan("What approaches exist for KV-cache compression?")] == [
        "How does quantization work?", "What about eviction policies?", "And layer sharing?",
    ]


def test_single_line_answer_falls_back_to_the_original_question(fake_llm):
    """Модель сочла вопрос неделимым — делить нечего."""
    fake_llm("What is retrieval augmented generation?")
    assert [sq.text for sq in plan("Что такое RAG?")] == ["Что такое RAG?"]


def test_duplicate_facets_are_collapsed(fake_llm):
    fake_llm("How does quantization work?\nhow does quantization work\nWhat about eviction?")
    assert [sq.text for sq in plan("What approaches exist for KV-cache compression?")] == [
        "How does quantization work?", "What about eviction?",
    ]


def test_facet_count_is_capped(monkeypatch, fake_llm):
    monkeypatch.setattr(planner.config, "PLANNER_MAX_SUBQUESTIONS", 2)
    fake_llm("First aspect of the topic?\nSecond aspect of the topic?\nThird aspect of the topic?")
    assert len(plan("What approaches exist for KV-cache compression?")) == 2


def test_rambling_answer_is_rejected_entirely(monkeypatch, fake_llm):
    """Слишком длинная строка — модель ушла в рассуждения вместо списка.
    Гнать воронку по такой строке дороже, чем откатиться на оригинал."""
    monkeypatch.setattr(planner.config, "PLANNER_MAX_FACET_WORDS", 8)
    fake_llm("Sure! Here are the aspects you asked about, let me explain each one in detail below\n"
             "Second line?")
    assert [sq.text for sq in plan("What approaches exist for cache compression?")] == ["What approaches exist for cache compression?"]


def test_too_short_facet_is_rejected_entirely(monkeypatch, fake_llm):
    monkeypatch.setattr(planner.config, "PLANNER_MIN_FACET_WORDS", 3)
    fake_llm("Quantization?\nWhich eviction policies exist here?")
    assert [sq.text for sq in plan("What approaches exist for cache compression?")] == ["What approaches exist for cache compression?"]


def test_llm_failure_falls_back_to_the_original_question(monkeypatch):
    """Планирование не должно ронять запрос: без декомпозиции агент работает
    как раньше, просто уже."""
    monkeypatch.setattr(planner.llm, "build_chat_prompt", lambda system, user: user)

    def _boom(prompt, **kw):
        raise RuntimeError("модель не загрузилась")

    monkeypatch.setattr(planner.llm, "generate", _boom)
    assert [sq.text for sq in plan("What approaches exist for cache compression?")] == ["What approaches exist for cache compression?"]


def test_empty_llm_answer_falls_back(fake_llm):
    fake_llm("   \n\n  ")
    assert [sq.text for sq in plan("What approaches exist for cache compression?")] == ["What approaches exist for cache compression?"]


def test_short_question_is_not_sent_to_the_llm(monkeypatch):
    """На коротком бессмысленном входе модель не отказывается делить, а
    сочиняет: "question?" превращался в три подвопроса про цепочки поставок."""
    def _fail(*a, **kw):
        raise AssertionError("короткий вопрос не должен доходить до LLM")

    monkeypatch.setattr(planner.llm, "build_chat_prompt", _fail)
    monkeypatch.setattr(planner.llm, "generate", _fail)
    assert [sq.text for sq in plan("question?")] == ["question?"]
