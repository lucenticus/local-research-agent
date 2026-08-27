"""Юнит-тесты логики eval-харнесса (evals/) — офлайн, реранкер и research замоканы.

Сам харнесс тоже код, и ошибка в нём тише ошибки в агенте: он не падает, а
показывает неверные числа, которым потом верят.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals import run_discovery, run_quality


def test_golden_set_is_wellformed():
    """Схема золотого набора — контракт между вопросами и всеми раннерами."""
    cases = run_discovery.load_questions()
    assert len(cases) >= 10, "меньше 10 вопросов — набор не репрезентативен"

    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "id должны быть уникальны: по ним идёт дифф между прогонами"

    for c in cases:
        assert c["question"].strip(), c["id"]
        assert isinstance(c["expected_subtopics"], list), c["id"]
        assert isinstance(c["expected_gaps"], bool), c["id"]
        assert c["tags"], c["id"]
        # Неотвечаемый вопрос не должен иметь ожидаемых подтем, иначе метрика
        # покрытия требует раскрыть то, чего не существует.
        if c["expected_gaps"]:
            assert not c["expected_subtopics"], c["id"]


def test_golden_set_covers_the_planned_slices():
    tags = {t for c in run_discovery.load_questions() for t in c["tags"]}
    for expected in ("en", "ru", "recency", "adversarial", "multi-hop", "comparison"):
        assert expected in tags, f"нет ни одного вопроса со срезом {expected}"


def test_load_questions_filters_by_tag():
    ru = run_discovery.load_questions("ru")
    assert ru and all("ru" in c["tags"] for c in ru)


def test_subtopic_coverage_counts_confirmed_topics(monkeypatch):
    monkeypatch.setattr(run_quality.rerank, "score_pairs", lambda pairs: [0.9, 0.1, 0.8])
    covered, missed = run_quality.subtopic_coverage("ответ", ["a", "b", "c"])
    assert covered == round(2 / 3, 3) or abs(covered - 2 / 3) < 1e-9
    assert missed == ["b"]


def test_subtopic_coverage_scores_topic_against_answer(monkeypatch):
    """Порядок в паре важен: реранкер спрашивает «подтверждает ли документ
    запрос», значит подтема — запрос, ответ — документ."""
    captured = {}
    monkeypatch.setattr(run_quality.rerank, "score_pairs",
                        lambda pairs: (captured.setdefault("pairs", pairs), [0.9])[1])
    run_quality.subtopic_coverage("текст ответа", ["ожидаемая подтема"])
    assert captured["pairs"] == [("ожидаемая подтема", "текст ответа")]


def test_subtopic_coverage_without_topics_is_not_a_penalty(monkeypatch):
    """У неотвечаемых вопросов подтем нет — покрывать нечего, и штрафовать не за что."""
    def _fail(pairs):
        raise AssertionError("без подтем реранкер звать незачем")

    monkeypatch.setattr(run_quality.rerank, "score_pairs", _fail)
    assert run_quality.subtopic_coverage("ответ", []) == (1.0, [])


def test_aggregate_skips_failed_cases_but_counts_them():
    rows = [
        {"id": "a", "wall_seconds": 10.0, "citation_coverage": 1.0, "faithfulness": 0.5,
         "subtopic_coverage": 1.0, "gap_honest": None, "providers": {}},
        {"id": "b", "error": "RuntimeError: boom", "wall_seconds": 2.0, "providers": {}},
    ]
    agg = run_quality.aggregate(rows)
    assert agg["n_ok"] == 1 and agg["n_error"] == 1
    assert agg["faithfulness"] == 0.5  # упавший вопрос не тянет среднее
    assert agg["wall_seconds"]["max"] == 10.0


def test_aggregate_reports_gap_honesty_only_for_adversarial_cases():
    rows = [
        {"id": "ok", "wall_seconds": 1.0, "gap_honest": None, "providers": {}},
        {"id": "adv1", "wall_seconds": 1.0, "gap_honest": True, "providers": {}},
        {"id": "adv2", "wall_seconds": 1.0, "gap_honest": False, "providers": {}},
    ]
    assert run_quality.aggregate(rows)["gap_honesty"] == 0.5


def test_aggregate_sums_provider_cost_across_cases():
    rows = [
        {"id": "a", "wall_seconds": 1.0, "gap_honest": None,
         "providers": {"llm.generate": {"calls": 2, "seconds": 3.0, "items": 0,
                                        "prompt_tokens": 100, "completion_tokens": 20}}},
        {"id": "b", "wall_seconds": 1.0, "gap_honest": None,
         "providers": {"llm.generate": {"calls": 1, "seconds": 1.5, "items": 0,
                                        "prompt_tokens": 50, "completion_tokens": 10}}},
    ]
    total = run_quality.aggregate(rows)["providers_total"]["llm.generate"]
    assert total["calls"] == 3
    assert total["seconds"] == 4.5
    assert total["tokens"] == 180


def test_aggregate_on_empty_rows_does_not_crash():
    agg = run_quality.aggregate([])
    assert agg["n_ok"] == 0 and agg["wall_seconds"] is None


def test_discovery_aggregate_separates_empty_from_failed():
    """Ключевое различие всего харнесса: источник, вернувший ноль, и источник,
    который упал, — разные факты с разными причинами."""
    rows = [
        {"per_source": {"arxiv": 0, "s2": 0}, "per_errors": {"s2": ["HTTPError: 429"]}},
        {"per_source": {"arxiv": 0, "s2": 0}, "per_errors": {"s2": ["HTTPError: 429"]}},
    ]
    agg = run_discovery.aggregate(rows)
    assert agg["arxiv"]["hit_rate"] == 0.0 and agg["arxiv"]["error_rate"] == 0.0
    assert agg["s2"]["hit_rate"] == 0.0 and agg["s2"]["error_rate"] == 1.0


def test_each_case_starts_from_a_clean_index(monkeypatch):
    """Замер на накопительной коллекции врёт: второй прогон того же вопроса
    переиспользует скачанное первым и выглядит лучше без единой правки."""
    calls = []

    class _Store:
        def __init__(self, collection_name=None):
            calls.append(("init", collection_name))

        def rebuild(self, chunks):
            calls.append(("rebuild", tuple(chunks)))

    monkeypatch.setattr(run_quality, "QdrantStore", _Store)
    monkeypatch.setattr(run_quality, "run_research",
                        lambda q, store, **kw: (_ for _ in ()).throw(RuntimeError("stop here")))

    run_quality.evaluate_case({"id": "x", "question": "вопрос"})
    assert ("init", run_quality.config.QDRANT_EVAL_COLLECTION) in calls
    assert ("rebuild", ()) in calls, "коллекция должна очищаться перед вопросом"


def test_error_row_has_the_same_shape_as_a_successful_one(monkeypatch):
    """Строка с ошибкой должна быть той же формы: иначе на ней падает любой
    потребитель — дифф, отчёт, внешний анализ."""
    monkeypatch.setattr(run_quality, "QdrantStore", lambda collection_name=None: type(
        "S", (), {"rebuild": lambda self, chunks: None})())
    monkeypatch.setattr(run_quality, "run_research",
                        lambda q, store, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    row = run_quality.evaluate_case({"id": "x", "question": "вопрос", "expected_gaps": False})
    for key in ("id", "question", "citation_coverage", "faithfulness", "subtopic_coverage",
                "gap_honest", "wall_seconds", "providers"):
        assert key in row, key
    assert row["error"].startswith("RuntimeError")


# --- фикстуры внешних входов ---

def _cache(tmp_path, mode):
    from src.sources.replay import DiscoveryCache
    return DiscoveryCache(tmp_path / "fx.json", mode=mode)


def test_frozen_world_is_a_noop_when_disabled(tmp_path):
    from src.agent import funnel
    from src.sources.replay import MODE_OFF
    from evals.fixtures import frozen_world

    before = (funnel.fetch_pdf_sections, funnel.lookup_citation_counts, funnel._now)
    with frozen_world(_cache(tmp_path, MODE_OFF)):
        assert (funnel.fetch_pdf_sections, funnel.lookup_citation_counts, funnel._now) == before


def test_frozen_world_restores_the_originals_even_on_error(tmp_path):
    from src.agent import funnel
    from src.sources.replay import MODE_RECORD
    from evals.fixtures import frozen_world

    before = (funnel.fetch_pdf_sections, funnel.lookup_citation_counts, funnel._now)
    try:
        with frozen_world(_cache(tmp_path, MODE_RECORD)):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert (funnel.fetch_pdf_sections, funnel.lookup_citation_counts, funnel._now) == before


def test_pdf_and_citations_are_recorded_then_replayed(tmp_path, monkeypatch):
    from src.agent import funnel
    from src.ingest.extract import Section
    from src.sources.replay import MODE_RECORD, MODE_REPLAY
    from evals.fixtures import frozen_world

    calls = []
    monkeypatch.setattr(funnel, "fetch_pdf_sections",
                        lambda url: (calls.append(url), [Section(name="R", category="results", text="t")])[1])
    monkeypatch.setattr(funnel, "lookup_citation_counts",
                        lambda ids: (calls.append(ids), {"1706.03762": 42})[1])

    rec = _cache(tmp_path, MODE_RECORD)
    with frozen_world(rec):
        funnel.fetch_pdf_sections("http://x/p.pdf")
        funnel.lookup_citation_counts(["1706.03762"])
    rec.save()
    assert len(calls) == 2

    rep = _cache(tmp_path, MODE_REPLAY)
    with frozen_world(rep):
        sections = funnel.fetch_pdf_sections("http://x/p.pdf")
        counts = funnel.lookup_citation_counts(["1706.03762"])
    assert len(calls) == 2, "при replay настоящие функции звать нельзя"
    assert [s.text for s in sections] == ["t"]
    assert counts == {"1706.03762": 42}


def test_unknown_citation_is_replayed_as_absent_not_as_a_lookup(tmp_path, monkeypatch):
    """Статья, которой S2 не знает, отсутствует в ответе — это записываемый
    факт «не нашли», а не промах кэша: иначе при replay мы бы снова полезли
    в сеть именно там, где её нет."""
    from src.agent import funnel
    from src.sources.replay import MODE_RECORD, MODE_REPLAY
    from evals.fixtures import frozen_world

    monkeypatch.setattr(funnel, "lookup_citation_counts", lambda ids: {})
    rec = _cache(tmp_path, MODE_RECORD)
    with frozen_world(rec):
        funnel.lookup_citation_counts(["2605.00001"])
    rec.save()

    def _fail(ids):
        raise AssertionError("при replay сеть трогать нельзя")

    monkeypatch.setattr(funnel, "lookup_citation_counts", _fail)
    rep = _cache(tmp_path, MODE_REPLAY)
    with frozen_world(rep):
        assert funnel.lookup_citation_counts(["2605.00001"]) == {}


def test_citation_fixture_key_ignores_the_order_of_ids(tmp_path, monkeypatch):
    """Тот же набор статей в другом порядке — тот же вопрос к API. Источники
    отвечают не по расписанию, и порядок кандидатов между прогонами плавает."""
    from src.agent import funnel
    from src.sources.replay import MODE_RECORD, MODE_REPLAY
    from evals.fixtures import frozen_world

    monkeypatch.setattr(funnel, "lookup_citation_counts", lambda ids: {"a": 1, "b": 2})
    rec = _cache(tmp_path, MODE_RECORD)
    with frozen_world(rec):
        funnel.lookup_citation_counts(["a", "b"])
    rec.save()

    def _fail(ids):
        raise AssertionError("при replay сеть трогать нельзя")

    monkeypatch.setattr(funnel, "lookup_citation_counts", _fail)
    rep = _cache(tmp_path, MODE_REPLAY)
    with frozen_world(rep):
        assert funnel.lookup_citation_counts(["b", "a"]) == {"a": 1, "b": 2}


def test_time_is_frozen_to_the_recorded_moment(tmp_path):
    from src.agent import funnel
    from src.sources.replay import MODE_RECORD, MODE_REPLAY
    from evals.fixtures import frozen_world

    rec = _cache(tmp_path, MODE_RECORD)
    with frozen_world(rec):
        recorded = funnel._now()
    rec.save()

    rep = _cache(tmp_path, MODE_REPLAY)
    with frozen_world(rep):
        assert funnel._now() == recorded, "свежесть считается от «сейчас» — момент должен совпадать"


def test_eval_budget_drops_the_wall_clock_limit_but_keeps_the_others():
    """Лимит по секундам делает результат зависимым от загрузки машины;
    остальные лимиты остаются, иначе цикл может не завершиться."""
    from evals.fixtures import eval_budget

    b = eval_budget()
    assert b.max_seconds is None
    assert b.max_iterations > 0 and b.max_deep_reads > 0


def test_a_freshly_created_cache_is_not_treated_as_absent(tmp_path, monkeypatch):
    """DiscoveryCache имеет __len__, поэтому пустой кэш ложен — а пустой он
    ровно в начале записи. `cache or default` подменял его выключенным, и
    девятиминутный прогон записи вернул пустой файл, отчитавшись об успехе.
    """
    from src.sources.replay import MODE_RECORD, DiscoveryCache
    from evals import run_quality

    cache = DiscoveryCache(tmp_path / "f.json", mode=MODE_RECORD)
    assert not cache, "предпосылка теста: пустой кэш действительно ложен"

    seen: dict = {}

    def _fake_run_research(question, store, **kw):
        seen["sources_given"] = kw.get("sources")
        raise RuntimeError("дальше не нужно")

    monkeypatch.setattr(run_quality, "run_research", _fake_run_research)
    monkeypatch.setattr(run_quality, "QdrantStore", lambda **kw: _NoopStore())
    run_quality.evaluate_case({"id": "q", "question": "q?"}, cache)

    assert seen["sources_given"] is not None, "источники должны быть обёрнуты фикстурами"


class _NoopStore:
    def rebuild(self, docs):
        pass
