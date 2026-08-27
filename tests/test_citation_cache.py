"""Юнит-тесты sources/citation_cache.py — файловый кэш, без сети."""

from __future__ import annotations

import json
import time

from src import config
from src.sources import citation_cache


def test_stores_and_returns_what_it_stored(monkeypatch):
    citation_cache.put_many({"1706.03762": 190309})
    hits, misses = citation_cache.get_many(["1706.03762"])
    assert hits == {"1706.03762": 190309}
    assert misses == []


def test_unknown_id_goes_to_the_ask_list(monkeypatch):
    citation_cache.put_many({"a": 1})
    hits, misses = citation_cache.get_many(["a", "b"])
    assert hits == {"a": 1}
    assert misses == ["b"], "спросить надо ровно недостающие, а не всё заново"


def test_stale_entry_is_asked_again(monkeypatch):
    """Цитируемость растёт. Бессрочный кэш заморозил бы свежую статью на
    нуле навсегда — ровно ту, ради которой в триаже есть буст по свежести."""
    old = time.time() - (config.CITATION_CACHE_TTL_DAYS + 1) * 86400
    config.CITATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CITATION_CACHE_PATH.write_text(json.dumps({"a": [5, old]}), encoding="utf-8")

    hits, misses = citation_cache.get_many(["a"])
    assert hits == {}
    assert misses == ["a"]


def test_disabled_cache_asks_for_everything(monkeypatch):
    monkeypatch.setattr(config, "CITATION_CACHE_ENABLED", False)
    citation_cache.put_many({"a": 1})
    assert citation_cache.get_many(["a"]) == ({}, ["a"])


def test_a_corrupt_cache_file_is_ignored_not_fatal(monkeypatch):
    """Кэш — ускоритель, а не источник истины: худшее последствие порчи —
    лишний запрос к API, а не упавший прогон."""
    config.CITATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CITATION_CACHE_PATH.write_text("{не json", encoding="utf-8")

    assert citation_cache.get_many(["a"]) == ({}, ["a"])
    citation_cache.put_many({"a": 1})  # не должно бросить
    assert citation_cache.get_many(["a"]) == ({"a": 1}, [])


def test_writing_does_not_drop_what_another_writer_stored(monkeypatch):
    """Веб гоняет джобы в потоках. Файл перечитывается под локом перед
    записью, иначе второй писатель затирал бы значения первого."""
    citation_cache.put_many({"a": 1})
    citation_cache.put_many({"b": 2})
    hits, misses = citation_cache.get_many(["a", "b"])
    assert hits == {"a": 1, "b": 2}
    assert misses == []


def test_cache_path_is_read_at_call_time_not_at_import(monkeypatch, tmp_path):
    """Путь, вычисленный на импорте, молча игнорировал бы подмену конфига —
    тот же класс ошибки, что уже ловили в providers/llm.py с temperature."""
    citation_cache.put_many({"a": 1})
    monkeypatch.setattr(config, "CITATION_CACHE_PATH", tmp_path / "elsewhere.json")
    assert citation_cache.get_many(["a"]) == ({}, ["a"])
