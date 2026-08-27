"""Юнит-тесты sources/replay.py — фикстуры discovery (офлайн, без сети)."""

from __future__ import annotations

from src.sources.base import DiscoveredItem
from src.sources.replay import (
    MODE_OFF,
    MODE_RECORD,
    MODE_REPLAY,
    DiscoveryCache,
    wrap,
)


class _FakeSource:
    name = "fake"

    def __init__(self, items=None):
        self.calls: list[tuple[str, int]] = []
        self._items = items if items is not None else [
            DiscoveredItem(id="1", source="fake", title="T", abstract="A", published_date="2026-01-01")
        ]

    def discover(self, query, limit):
        self.calls.append((query, limit))
        return self._items


def test_off_mode_passes_through_without_touching_the_cache(tmp_path):
    inner = _FakeSource()
    cache = DiscoveryCache(tmp_path / "f.json", mode=MODE_OFF)
    sources = wrap([inner], cache)

    assert sources[0] is inner  # обёртки нет вовсе
    assert len(cache) == 0


def test_record_calls_the_source_and_saves_a_replayable_file(tmp_path):
    path = tmp_path / "f.json"
    inner = _FakeSource()
    cache = DiscoveryCache(path, mode=MODE_RECORD)
    items = wrap([inner], cache)[0].discover("query", 5)

    assert len(inner.calls) == 1
    assert len(items) == 1
    cache.save()

    replay_inner = _FakeSource()
    replayed = wrap([replay_inner], DiscoveryCache(path, mode=MODE_REPLAY))[0].discover("query", 5)
    assert replay_inner.calls == []  # источник не дёргался
    assert replayed == items  # включая published_date и meta


def test_replay_distinguishes_by_query_and_limit(tmp_path):
    path = tmp_path / "f.json"
    cache = DiscoveryCache(path, mode=MODE_RECORD)
    wrapped = wrap([_FakeSource()], cache)[0]
    wrapped.discover("a", 5)
    cache.save()

    replay = DiscoveryCache(path, mode=MODE_REPLAY)
    source = wrap([_FakeSource()], replay)[0]
    assert len(source.discover("a", 5)) == 1
    assert source.discover("a", 9) == []  # другой limit — другая запись
    assert source.discover("b", 5) == []


def test_replay_miss_is_counted_not_raised(tmp_path):
    """Новый вопрос в golden-наборе не должен ронять прогон — но и молча
    меряться на пустоте нельзя, поэтому промахи считаются."""
    replay = DiscoveryCache(tmp_path / "missing.json", mode=MODE_REPLAY)
    assert wrap([_FakeSource()], replay)[0].discover("unseen", 5) == []
    assert len(replay.misses) == 1
    assert "unseen" in replay.misses[0]


def test_record_appends_to_an_existing_file(tmp_path):
    """Записывать набор можно в несколько заходов — источники троттлят
    по-разному, и переснимать всё целиком слишком дорого."""
    path = tmp_path / "f.json"
    first = DiscoveryCache(path, mode=MODE_RECORD)
    wrap([_FakeSource()], first)[0].discover("first", 5)
    first.save()

    second = DiscoveryCache(path, mode=MODE_RECORD)
    wrap([_FakeSource()], second)[0].discover("second", 5)
    second.save()

    replay = DiscoveryCache(path, mode=MODE_REPLAY)
    source = wrap([_FakeSource()], replay)[0]
    assert len(source.discover("first", 5)) == 1
    assert len(source.discover("second", 5)) == 1


def test_empty_result_is_recorded_as_empty_not_as_a_miss(tmp_path):
    """«Источник ответил пусто» и «записи нет» — разные факты: первое
    воспроизводится как пустота, второе считается промахом."""
    path = tmp_path / "f.json"
    cache = DiscoveryCache(path, mode=MODE_RECORD)
    wrap([_FakeSource(items=[])], cache)[0].discover("q", 5)
    cache.save()

    replay = DiscoveryCache(path, mode=MODE_REPLAY)
    assert wrap([_FakeSource()], replay)[0].discover("q", 5) == []
    assert replay.misses == []
