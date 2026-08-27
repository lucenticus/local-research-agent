"""Юнит-тесты providers/metrics.py — счётчики cost/latency (узел 22)."""

from __future__ import annotations

import pytest

from src.providers import metrics


@pytest.fixture(autouse=True)
def _clean():
    metrics.reset()
    yield
    metrics.reset()


def test_track_counts_calls_and_items():
    for n in (3, 5):
        with metrics.track("embed.embed_texts", items=n):
            pass

    snap = metrics.snapshot()["embed.embed_texts"]
    assert snap["calls"] == 2
    assert snap["items"] == 8
    assert snap["seconds"] >= 0


def test_track_records_time_even_when_the_call_raises():
    """Упавший вызов всё равно стоил времени — не считать его значит занижать
    стоимость ровно там, где что-то пошло не так."""
    with pytest.raises(RuntimeError):
        with metrics.track("llm.generate"):
            raise RuntimeError("boom")

    assert metrics.snapshot()["llm.generate"]["calls"] == 1


def test_tokens_are_accumulated_separately():
    with metrics.track("llm.generate"):
        pass
    metrics.add_tokens("llm.generate", prompt=100, completion=20)
    metrics.add_tokens("llm.generate", prompt=50, completion=10)

    snap = metrics.snapshot()["llm.generate"]
    assert snap["prompt_tokens"] == 150
    assert snap["completion_tokens"] == 30


def test_snapshot_omits_token_fields_for_non_llm_providers():
    with metrics.track("rerank.score", items=4):
        pass
    snap = metrics.snapshot()["rerank.score"]
    assert "prompt_tokens" not in snap and "completion_tokens" not in snap


def test_percentiles_are_taken_from_observed_values(monkeypatch):
    """p95 берётся из реально наблюдавшихся длительностей, а не интерполяцией:
    на малой выборке интерполяция придумывает значение, которого не было."""
    durations = [0.1, 0.2, 0.3, 0.4, 100.0]
    counter = metrics._counter("rerank.load")
    counter.durations.extend(durations)
    counter.calls = len(durations)

    snap = metrics.snapshot()["rerank.load"]
    assert snap["p50_seconds"] == 0.3
    assert snap["p95_seconds"] in {0.4, 100.0}  # одно из наблюдений, не среднее между ними


def test_reset_clears_everything():
    with metrics.track("llm.generate"):
        pass
    metrics.reset()
    assert metrics.snapshot() == {}


def test_snapshot_of_untouched_provider_is_absent():
    with metrics.track("llm.generate"):
        pass
    assert "embed.embed_texts" not in metrics.snapshot()
